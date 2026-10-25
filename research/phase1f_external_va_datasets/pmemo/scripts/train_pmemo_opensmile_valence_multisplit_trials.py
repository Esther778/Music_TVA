#!/usr/bin/env python3
"""Train PMEmo OpenSMILE Valence-only BiLSTM trials on fixed song splits.

This is a Valence-focused counterpart to the Arousal OpenSMILE trials. It uses
the existing PMEmo OpenSMILE sequence cache, but evaluates on the same three
song-level splits used by the current Valence teacher search.
"""

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

from train_pmemo_mert_valence_baseline import ROOT, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


DEFAULT_DATASET = ROOT / "outputs" / "opensmile_baseline" / "dataset"
BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_valence_multisplit_trials_20260708"
SPLIT_SEEDS = {1: 101, 2: 202, 3: 303}

CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_index(split_id: int, dataset_dir: Path) -> pd.DataFrame:
    seed = SPLIT_SEEDS[int(split_id)]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{split_id:02d}_seed_{seed}" / "sequence_index.csv")
    index = index.copy()
    index["path"] = index["musicId"].map(lambda sid: str(dataset_dir / "sequence" / f"{int(sid)}.npz"))
    return index


def build_norm(index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        xs.append(np.load(row["path"])["X"].astype(np.float32))
    x = np.concatenate(xs, axis=0)
    return x.mean(axis=0).astype(np.float32), (x.std(axis=0) + 1e-8).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, mean: np.ndarray, std: np.ndarray):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        data = np.load(self.rows[idx]["path"])
        x = ((data["X"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        y = data["Y"].astype(np.float32)[:, 0]
        n = min(len(x), len(y), len(data["time_sec"]))
        sid = int(data["music_id"])
        return (
            torch.from_numpy(x[:n]),
            torch.from_numpy(y[:n]),
            torch.full((n,), sid, dtype=torch.int32),
            torch.from_numpy(data["time_sec"].astype(np.float32)[:n]),
        )


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    y_out = torch.zeros(len(batch), max_len, dtype=torch.float32)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, song, time_sec) in enumerate(batch):
        x_out[i, : x.shape[0]] = x
        y_out[i, : y.shape[0]] = y
        song_out[i, : song.shape[0]] = song
        time_out[i, : time_sec.shape[0]] = time_sec
        mask[i, : x.shape[0]] = True
    return x_out, y_out, song_out, time_out, mask


class ValenceBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float, head_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y).squeeze(-1)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    denom = pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    return 1.0 - (2.0 * cov / denom)


def clip_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(pred.shape[0]):
        valid = mask[i]
        if valid.any():
            losses.append((pred[i][valid].mean() - target[i][valid].mean()) ** 2)
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if not valid.any():
        return pred.new_tensor(0.0)
    return ((pred[:, 1:] - pred[:, :-1])[valid] ** 2).mean()


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
    return {
        "metrics": metrics(pred_np, target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


@dataclass
class TrialConfig:
    trial_id: str
    hidden_dim: int
    layers: int
    dropout: float
    head_hidden: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    ccc_weight: float
    clip_mean_weight: float
    smooth_weight: float
    patience: int = 12
    epochs: int = 90


TRIALS = [
    TrialConfig("v_smile_h96_l1_do010_rmse", 96, 1, 0.10, 64, 8e-4, 1e-4, 36, 0.00, 0.00, 0.00),
    TrialConfig("v_smile_h128_l1_do020_ccc_mean", 128, 1, 0.20, 96, 6e-4, 2e-4, 28, 0.12, 0.08, 0.00),
    TrialConfig("v_smile_h160_l2_do030_ccc_mean_smooth", 160, 2, 0.30, 128, 5e-4, 2e-4, 20, 0.18, 0.12, 0.01),
]


def train_one_split(
    split_id: int,
    seed: int,
    config: TrialConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{SPLIT_SEEDS[split_id]}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index = load_split_index(split_id, args.data_dir)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    mean, std = build_norm(index)
    np.savez(split_dir / "feature_norm.npz", mean=mean, std=std)
    loaders = {
        split: DataLoader(
            SequenceDataset(index, split, mean, std),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    input_dim = int(loaders["train"].dataset[0][0].shape[1])
    model = ValenceBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
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
            loss = ((pred_valid - y_valid) ** 2).mean()
            if config.ccc_weight:
                loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            if config.clip_mean_weight:
                loss = loss + config.clip_mean_weight * clip_mean_loss(pred, y, mask)
            if config.smooth_weight:
                loss = loss + config.smooth_weight * smoothness_loss(pred, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_select_score": score}
        history.append(row)
        print(
            f"{config.trial_id} split={split_id:02d} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "input_dim": input_dim}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
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
        "split_id": split_id,
        "source_split_seed": SPLIT_SEEDS[split_id],
        "seed": seed,
        "best_epoch": best_epoch,
        "feature_type": "PMEmo OpenSMILE dynamic 260, renormalized per fixed train split",
        "target": "valence",
        "label_scale": "[-1, 1]",
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def macro(rows: list[dict[str, Any]], prefix: str = "test") -> dict[str, float]:
    return {m: float(np.mean([float(row[f"{prefix}_{m}"]) for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def summarize_trial(config: TrialConfig, rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    test_macro = macro(rows, "test")
    val_macro = macro(rows, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": f"pmemo_opensmile_valence_multisplit_{config.trial_id}",
        "purpose": "Valence-only OpenSMILE BiLSTM multisplit trial",
        "current_best_reference": CURRENT_BEST,
        "selection_rule": "checkpoint selected by validation selection_score in each split; test reported after selection",
        "config": asdict(config),
        "val_macro": val_macro,
        "test_macro": test_macro,
        "delta_vs_current_best": {m: float(test_macro[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "test_gap_rows": threshold_rows(test_macro),
        "per_split": rows,
    }
    trial_dir = out_dir / config.trial_id
    pd.DataFrame(rows).to_csv(trial_dir / "comparison.csv", index=False)
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--trial", choices=[t.trial_id for t in TRIALS] + ["all"], default=TRIALS[0].trial_id)
    parser.add_argument("--splits", nargs="*", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configs = TRIALS if args.trial == "all" else [t for t in TRIALS if t.trial_id == args.trial]
    summaries = []
    for config in configs:
        rows = [train_one_split(int(split_id), args.seed, config, args, device) for split_id in args.splits]
        summaries.append(summarize_trial(config, rows, args.out_dir))
    comparison = []
    for summary in summaries:
        row = {"trial_id": summary["config"]["trial_id"], **{f"test_{m}": summary["test_macro"][m] for m in ["rmse", "mae", "pcc", "ccc"]}}
        row.update({f"delta_vs_best_{m}": summary["delta_vs_current_best"][m] for m in ["rmse", "mae", "pcc", "ccc"]})
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(args.out_dir / "trial_comparison.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps({"created_at": "2026-07-08", "current_best_reference": CURRENT_BEST, "trials": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
