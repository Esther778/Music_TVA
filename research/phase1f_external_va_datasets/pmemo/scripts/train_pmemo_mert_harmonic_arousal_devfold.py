#!/usr/bin/env python3
"""Arousal MERT hidden6 + harmonic BiLSTM development-fold trial."""

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

from search_pmemo_valence_dev_val_postprocess import moving_average_by_song
from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM, build_harmonic_norm, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import build_mert_norm


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_trainval_devfolds_20260709" / "fold_03" / "sequence_index.csv"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "arousal_mert_harmonic_devfolds_20260709" / "fold_03"


@dataclass
class Config:
    trial_id: str = "a_mert_harm_h80_do030_lr4e4"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 3e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.04
    clip_mse_weight: float = 0.08
    extreme_weight: float = 1.10
    clip_std_weight: float = 0.0
    epochs: int = 80
    patience: int = 10


class ArousalFusionDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        mert_dir: Path,
        label_dir: Path,
        librosa_dataset: Path,
        mert_mean: np.ndarray,
        mert_std: np.ndarray,
        harm_mean: np.ndarray,
        harm_std: np.ndarray,
        feature_idx: np.ndarray,
    ):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mert_dir = mert_dir
        self.label_dir = label_dir
        self.librosa_dataset = librosa_dataset
        self.mert_mean = mert_mean.astype(np.float32)
        self.mert_std = mert_std.astype(np.float32)
        self.harm_mean = harm_mean.astype(np.float32)
        self.harm_std = harm_std.astype(np.float32)
        self.feature_idx = feature_idx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid = int(self.rows[idx]["musicId"])
        mert = np.load(self.mert_dir / f"{sid}.npz")
        label = np.load(self.label_dir / f"{sid}.npz")
        lib = np.load(self.librosa_dataset / "sequence" / f"{sid}.npz")
        x_mert = ((mert["embedding"].astype(np.float32) - self.mert_mean) / self.mert_std).astype(np.float32)
        x_harm = lib["X"].astype(np.float32)[:, self.feature_idx]
        x_harm = ((x_harm - self.harm_mean) / self.harm_std).astype(np.float32)
        y = label["Y"].astype(np.float32)[:, 1]
        time_sec = mert["time_sec"].astype(np.float32)
        n = min(len(x_mert), len(x_harm), len(y), len(time_sec))
        x = np.concatenate([x_mert[:n], x_harm[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return m["rmse"] + 0.02 * max(0, m["mae"] - 0.15) + 0.05 * max(0, 0.80 - m["ccc"])


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": selection_score(val_eval["metrics"])}
    for window in [1, 7, 11, 15, 21, 31, 41, 51, 71, 91]:
        sm = moving_average_by_song(pred, song, window)
        for blend in [0.25, 0.50, 0.75, 1.0]:
            pp = (1 - blend) * pred + blend * sm
            m = metrics(pp, target)
            score = selection_score(m)
            row = {"postprocess": f"smooth_w{window}_b{blend:g}", "window": window, "blend": blend, **m, "score": score}
            rows.append(row)
            if score < best["score"]:
                best = {"postprocess": row["postprocess"], "metrics": m, "score": score, "window": window, "blend": blend}
    return best, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--proj-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--frame-ccc-weight", type=float, default=None)
    parser.add_argument("--clip-mse-weight", type=float, default=None)
    parser.add_argument("--extreme-weight", type=float, default=None)
    parser.add_argument("--clip-std-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    for name in [
        "trial_id",
        "proj_dim",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "frame_ccc_weight",
        "clip_mse_weight",
        "extreme_weight",
        "clip_std_weight",
        "epochs",
        "patience",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    set_seed(args.seed)
    index = pd.read_csv(args.index)
    build_mert_norm(index, args.mert_dir, args.out_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), args.out_dir / "dataset", force=True)
    mert_norm = np.load(args.out_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(args.out_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            ArousalFusionDataset(index, split, args.mert_dir, args.label_dir, args.librosa_dataset, mert_norm["mean"], mert_norm["std"], harm_norm["mean"], harm_norm["std"], harm_norm["feature_idx"]),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = args.out_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            opt.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]; y_valid = y[mask]
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = selection_score(m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score}
        history.append(row)
        print(f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}", flush=True)
        if score < best_score:
            best_score = score; best_epoch = epoch; bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample_x.shape[1]), "config": asdict(config)}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_arousal": data["target"], "pred_arousal": data["pred"]}).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {"created_at": "2026-07-09", "purpose": "Arousal MERT+harmony fold trial; holdout/test excluded", "best_epoch": best_epoch, "config": asdict(config), **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()}, "best_val_postprocess": pp, "val_gap_rows_best_postprocess": threshold_rows(pp["metrics"])}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
