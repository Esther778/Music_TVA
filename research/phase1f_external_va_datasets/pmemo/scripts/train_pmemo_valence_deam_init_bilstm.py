#!/usr/bin/env python3
"""Fine-tune a PMEmo Valence BiLSTM initialized from the DEAM MERT teacher."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_SPLIT_ROOT = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_DEAM_CKPT = ROOT.parents[1] / "phase1b_bert_deam_va_training" / "outputs" / "models" / "bilstm" / "bilstm_baseline_best.pt"
DEFAULT_OUT = ROOT / "outputs" / "valence_deam_init_bilstm_screen_20260709"


@dataclass
class Config:
    trial_id: str = "deam_init_h128_l1_do020_lr15e5_ccc003"
    hidden_dim: int = 128
    dropout: float = 0.20
    learning_rate: float = 1.5e-4
    weight_decay: float = 2e-4
    batch_size: int = 24
    ccc_weight: float = 0.03
    patience: int = 10
    epochs: int = 80


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_norm(index: pd.DataFrame, mert_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        xs.append(np.load(mert_dir / f"{int(row['musicId'])}.npz")["embedding"].astype(np.float32))
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "mert_norm.npz", mean=mean, std=std)
    summary = {"feature_dim": int(concat.shape[1]), "train_frames": int(concat.shape[0])}
    (out_dir / "mert_norm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class PmemoValenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, mert_dir: Path, label_dir: Path, mean: np.ndarray, std: np.ndarray):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mert_dir = mert_dir
        self.label_dir = label_dir
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid = int(self.rows[idx]["musicId"])
        mert = np.load(self.mert_dir / f"{sid}.npz")
        label = np.load(self.label_dir / f"{sid}.npz")
        x = ((mert["embedding"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        y = label["Y"].astype(np.float32)[:, 0]
        time_sec = mert["time_sec"].astype(np.float32)
        n = min(len(x), len(y), len(time_sec))
        return torch.from_numpy(x[:n]), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim)
    y_out = torch.zeros(len(batch), max_len)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, song, time_sec) in enumerate(batch):
        x_out[i, : x.shape[0]] = x
        y_out[i, : y.shape[0]] = y
        song_out[i, : song.shape[0]] = song
        time_out[i, : time_sec.shape[0]] = time_sec
        mask[i, : x.shape[0]] = True
    return x_out, y_out, song_out, time_out, mask


class DeamShapeValenceBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(self.drop(y)).squeeze(-1)


def init_from_deam(model: DeamShapeValenceBiLSTM, ckpt_path: Path) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    model_state = model.state_dict()
    loaded = []
    for key, tensor in state.items():
        if key.startswith("lstm.") and key in model_state and tuple(model_state[key].shape) == tuple(tensor.shape):
            model_state[key] = tensor
            loaded.append(key)
    if "out.weight" in state and tuple(state["out.weight"].shape) == (2, model.out.weight.shape[1]):
        model_state["out.weight"] = state["out.weight"][0:1].clone()
        model_state["out.bias"] = state["out.bias"][0:1].clone()
        loaded.extend(["out.weight[valence_row]", "out.bias[valence_row]"])
    model.load_state_dict(model_state)
    return {"checkpoint": str(ckpt_path), "deam_trial": ckpt.get("trial", {}), "loaded_keys": loaded}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, song, time_sec, mask in loader:
        pred = model(x.to(device), mask.to(device)).detach().cpu()
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return {"metrics": metrics(pred_np, target_np), "pred": pred_np, "target": target_np, "song_id": np.concatenate(songs), "time_sec": np.concatenate(times)}


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.03 * max(0.0, 0.80 - m["ccc"]) + 0.02 * max(0.0, 0.80 - m["pcc"]) + 0.005 * m["mae"]


def run_split(args: argparse.Namespace, config: Config) -> dict[str, Any]:
    set_seed(args.seed)
    split_name = f"split_{args.split_id:02d}_seed_{args.seed}"
    split_dir = args.out_dir / config.trial_id / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    if args.index:
        index = pd.read_csv(args.index)
        index = index[index["split"].isin(["train", "val"])].copy()
        split_name = f"{args.index.parent.name}_seed_{args.seed}"
        split_dir = args.out_dir / config.trial_id / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        eval_splits = ["train", "val"]
    else:
        index = pd.read_csv(args.split_root / split_name / "sequence_index.csv")
        eval_splits = ["train", "val", "test"]
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    norm_summary = build_norm(index, args.mert_dir, split_dir / "dataset")
    norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    loaders = {
        split: DataLoader(
            PmemoValenceDataset(index, split, args.mert_dir, args.label_dir, norm["mean"], norm["std"]),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in eval_splits
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = DeamShapeValenceBiLSTM(int(sample_x.shape[1]), config.hidden_dim, config.dropout)
    init_info = init_from_deam(model, args.deam_checkpoint) if args.use_deam_init else {"checkpoint": "", "loaded_keys": []}
    device = resolve_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = ((pred_valid - y_valid) ** 2).mean() + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_selection_score": score}
        history.append(row)
        print(
            f"{config.trial_id} {split_name} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "init_info": init_info, "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device) for split in eval_splits}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "model_id": f"pmemo_valence_{config.trial_id}",
        "purpose": "screen whether DEAM MERT BiLSTM initialization improves PMEmo Valence generalization",
        "split_name": split_name,
        "feature_type": "PMEmo MERT hidden6 768 only",
        "label_scale": "minus1_1",
        "config": asdict(config),
        "init_info": init_info,
        "norm_summary": norm_summary,
        "best_epoch": best_epoch,
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "val_gap_rows": threshold_rows(split_evals["val"]["metrics"]),
        "current_best_reference_split_02": {
            "rmse": 0.19592354741413673,
            "mae": 0.1524825593125003,
            "pcc": 0.7963035520602079,
            "ccc": 0.7785656675399162,
        },
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--split-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--deam-checkpoint", type=Path, default=DEFAULT_DEAM_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-deam-init", action="store_true")
    parser.add_argument("--trial-id", default=Config.trial_id)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--ccc-weight", type=float, default=Config.ccc_weight)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    args = parser.parse_args()
    config = Config(
        trial_id=args.trial_id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        ccc_weight=args.ccc_weight,
        batch_size=args.batch_size,
        patience=args.patience,
        epochs=args.epochs,
    )
    run_split(args, config)


if __name__ == "__main__":
    main()
