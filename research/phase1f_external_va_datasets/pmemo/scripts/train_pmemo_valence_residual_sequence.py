#!/usr/bin/env python3
"""Train a small sequence residual corrector on top of the current Valence best."""

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


DEFAULT_BASE_PRED_DIR = ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "valence_residual_sequence_20260709"
SPLIT_SEEDS = {1: 101, 2: 202, 3: 303}
CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


@dataclass
class ResidualConfig:
    trial_id: str = "h6_base_resid_h48_do045_cap006"
    hidden_dim: int = 48
    proj_dim: int = 128
    dropout: float = 0.45
    residual_cap: float = 0.06
    learning_rate: float = 2.0e-4
    weight_decay: float = 8.0e-4
    batch_size: int = 24
    ccc_weight: float = 0.02
    residual_l2_weight: float = 0.20
    patience: int = 10
    epochs: int = 80


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column in {list(df.columns)}")


def build_norm(base_pred_dir: Path, mert_dir: Path, split_name: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(base_pred_dir / split_name / "train_predictions.csv")
    xs = []
    for sid in sorted(train["musicId"].astype(int).unique()):
        xs.append(np.load(mert_dir / f"{sid}.npz")["embedding"].astype(np.float32))
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "mert_norm.npz", mean=mean, std=std)
    summary = {"feature_dim": int(concat.shape[1]), "train_frames": int(concat.shape[0])}
    (out_dir / "mert_norm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class ResidualDataset(Dataset):
    def __init__(self, base_pred_dir: Path, mert_dir: Path, split_name: str, data_split: str, mean: np.ndarray, std: np.ndarray):
        df = pd.read_csv(base_pred_dir / split_name / f"{data_split}_predictions.csv")
        self.groups = [g.sort_values("time_sec").reset_index(drop=True) for _, g in df.groupby("musicId", sort=True)]
        self.mert_dir = mert_dir
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        sid = int(g["musicId"].iloc[0])
        mert = np.load(self.mert_dir / f"{sid}.npz")
        x_mert = ((mert["embedding"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        time_sec = mert["time_sec"].astype(np.float32)
        base = g[pred_col(g)].to_numpy(dtype=np.float32)
        target = g[target_col(g)].to_numpy(dtype=np.float32)
        n = min(len(x_mert), len(time_sec), len(base), len(target))
        base = base[:n]
        target = target[:n]
        local_time = g["time_sec"].to_numpy(dtype=np.float32)[:n]
        denom = max(float(local_time.max() - local_time.min()), 1e-6) if n > 1 else 1.0
        time_norm = ((local_time - float(local_time.min())) / denom).astype(np.float32)
        base_mean = np.full(n, float(base.mean()), dtype=np.float32)
        base_std = np.full(n, float(base.std()), dtype=np.float32)
        base_centered = (base - base_mean).astype(np.float32)
        base_delta = np.diff(base, prepend=base[0]).astype(np.float32)
        aux = np.stack([base, base_mean, base_std, base_centered, base_delta, time_norm], axis=1).astype(np.float32)
        x = np.concatenate([x_mert[:n], aux], axis=1).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(target),
            torch.from_numpy(base),
            torch.full((n,), sid, dtype=torch.int32),
            torch.from_numpy(local_time),
        )


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim)
    y_out = torch.zeros(len(batch), max_len)
    base_out = torch.zeros(len(batch), max_len)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, base, song, time_sec) in enumerate(batch):
        n = x.shape[0]
        x_out[i, :n] = x
        y_out[i, :n] = y
        base_out[i, :n] = base
        song_out[i, :n] = song
        time_out[i, :n] = time_sec
        mask[i, :n] = True
    return x_out, y_out, base_out, song_out, time_out, mask


class ResidualBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: ResidualConfig):
        super().__init__()
        self.cap = float(config.residual_cap)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.cap * torch.tanh(self.out(y).squeeze(-1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, bases, songs, times, residuals = [], [], [], [], [], []
    for x, y, base, song, time_sec, mask in loader:
        x = x.to(device)
        mask_d = mask.to(device)
        resid = model(x, mask_d).detach().cpu()
        pred = torch.clamp(base + resid, -1.0, 1.0)
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        bases.append(base[mask].numpy())
        residuals.append(resid[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    base_np = np.concatenate(bases)
    resid_np = np.concatenate(residuals)
    return {
        "metrics": metrics(pred_np, target_np),
        "base_metrics": metrics(base_np, target_np),
        "pred": pred_np,
        "base": base_np,
        "residual": resid_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.02 * max(0.0, 0.80 - m["ccc"]) + 0.01 * m["mae"]


def run(args: argparse.Namespace, config: ResidualConfig) -> dict[str, Any]:
    set_seed(args.seed)
    split_name = f"split_{args.split_id:02d}_seed_{SPLIT_SEEDS[args.split_id]}"
    trial_dir = args.out_dir / config.trial_id / split_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    norm_summary = build_norm(args.base_pred_dir, args.mert_dir, split_name, trial_dir / "dataset")
    norm = np.load(trial_dir / "dataset" / "mert_norm.npz")
    loaders = {
        split: DataLoader(
            ResidualDataset(args.base_pred_dir, args.mert_dir, split_name, split, norm["mean"], norm["std"]),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, *_ = loaders["train"].dataset[0]
    model = ResidualBiLSTM(int(sample_x.shape[1]), config).to(resolve_device(args.device))
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, base, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            base = base.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            resid = model(x, mask)
            pred = torch.clamp(base + resid, -1.0, 1.0)
            pred_valid = pred[mask]
            y_valid = y[mask]
            resid_valid = resid[mask]
            loss = ((pred_valid - y_valid) ** 2).mean()
            loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + config.residual_l2_weight * (resid_valid**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_score": score}
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
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
    for split, data in evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "base_pred_valence": data["base"],
                "residual_pred": data["residual"],
                "pred_valence": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "model_id": f"pmemo_valence_residual_sequence_{config.trial_id}",
        "split_name": split_name,
        "purpose": "sequence residual correction on current Valence best",
        "base_model": CURRENT_BEST,
        "feature_type": "current best prediction + MERT hidden6 + prediction/time context",
        "config": asdict(config),
        "norm_summary": norm_summary,
        "best_epoch": best_epoch,
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_base_{k}": v for split, e in evals.items() for k, v in e["base_metrics"].items()},
        "test_gap_rows": threshold_rows(evals["test"]["metrics"]),
        "delta_vs_base_test": {
            k: float(evals["test"]["metrics"][k] - evals["test"]["base_metrics"][k])
            for k in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--base-pred-dir", type=Path, default=DEFAULT_BASE_PRED_DIR)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", default=ResidualConfig.trial_id)
    parser.add_argument("--hidden-dim", type=int, default=ResidualConfig.hidden_dim)
    parser.add_argument("--proj-dim", type=int, default=ResidualConfig.proj_dim)
    parser.add_argument("--dropout", type=float, default=ResidualConfig.dropout)
    parser.add_argument("--residual-cap", type=float, default=ResidualConfig.residual_cap)
    parser.add_argument("--learning-rate", type=float, default=ResidualConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ResidualConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=ResidualConfig.batch_size)
    parser.add_argument("--ccc-weight", type=float, default=ResidualConfig.ccc_weight)
    parser.add_argument("--residual-l2-weight", type=float, default=ResidualConfig.residual_l2_weight)
    parser.add_argument("--patience", type=int, default=ResidualConfig.patience)
    parser.add_argument("--epochs", type=int, default=ResidualConfig.epochs)
    args = parser.parse_args()
    config = ResidualConfig(
        trial_id=args.trial_id,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        residual_cap=args.residual_cap,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        ccc_weight=args.ccc_weight,
        residual_l2_weight=args.residual_l2_weight,
        patience=args.patience,
        epochs=args.epochs,
    )
    run(args, config)


if __name__ == "__main__":
    main()
