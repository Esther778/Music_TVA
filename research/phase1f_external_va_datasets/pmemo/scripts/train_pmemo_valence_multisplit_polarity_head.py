#!/usr/bin/env python3
"""Multi-split Valence BiLSTM with an auxiliary song-level polarity head."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionDataset,
    build_harmonic_norm,
    collate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR, aggregate_metrics, build_mert_norm


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_polarity_head_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_frame_calibrator_song_bias_20260628",
    "rmse": 0.19748015854593182,
    "mae": 0.15352887995061668,
    "pcc": 0.7932350928356282,
    "ccc": 0.7800520380365018,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


@dataclass
class PolarityConfig:
    trial_id: str = "bilstm_polarity_h80_do035_song020_clip015"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.08
    clip_mse_weight: float = 0.15
    clip_std_weight: float = 0.01
    song_head_weight: float = 0.20
    frame_song_consistency_weight: float = 0.05
    extreme_weight: float = 1.25
    patience: int = 10
    epochs: int = 70


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection_score(m: dict[str, float], song_rmse: float) -> float:
    return (
        2.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.3 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.15 * song_rmse
        + 0.05 * m["rmse"]
    )


class PolarityBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: PolarityConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        hidden2 = config.hidden_dim * 2
        self.frame_out = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(hidden2, 1))
        self.song_head = nn.Sequential(
            nn.LayerNorm(hidden2 * 3),
            nn.Linear(hidden2 * 3, hidden2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.proj(x))
        frame = self.frame_out(h).squeeze(-1)
        mask_f = mask.float().unsqueeze(-1)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (h * mask_f).sum(dim=1) / count
        var = (((h - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
        std = torch.sqrt(var + 1e-6)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        song = self.song_head(torch.cat([mean, std, max_pool], dim=1)).squeeze(-1)
        return frame, song


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def song_mean_metrics(song_ids: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    df = pd.DataFrame({"musicId": song_ids, "target": target, "pred": pred})
    grouped = df.groupby("musicId", sort=False).mean(numeric_only=True)
    return metrics(grouped["pred"].to_numpy(dtype=np.float64), grouped["target"].to_numpy(dtype=np.float64))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times, song_head_preds = [], [], [], [], []
    for x, y, song, time_sec, mask in loader:
        frame_pred, song_pred = model(x.to(device), mask.to(device))
        frame_pred = frame_pred.detach().cpu()
        song_pred = song_pred.detach().cpu()
        preds.append(frame_pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
        repeated = torch.zeros_like(frame_pred)
        for i in range(frame_pred.shape[0]):
            repeated[i, mask[i]] = song_pred[i]
        song_head_preds.append(repeated[mask].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    song_np = np.concatenate(songs)
    song_head_np = np.concatenate(song_head_preds)
    return {
        "metrics": metrics(pred_np, target_np),
        "song_mean_metrics": song_mean_metrics(song_np, target_np, pred_np),
        "song_head_metrics": song_mean_metrics(song_np, target_np, song_head_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": song_np,
        "time_sec": np.concatenate(times),
        "song_head_pred": song_head_np,
    }


def train_one_split(split_dir_in: Path, split_id: int, seed: int, args: argparse.Namespace, device: torch.device, config: PolarityConfig) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(split_dir_in / "sequence_index.csv")
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.mert_dir, split_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset", force=True)
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            FusionDataset(
                index,
                split,
                args.mert_dir,
                args.label_dir,
                args.librosa_dataset,
                mert_norm["mean"],
                mert_norm["std"],
                harm_norm["mean"],
                harm_norm["std"],
                harm_norm["feature_idx"],
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = PolarityBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            frame_pred, song_pred = model(x, mask)
            pred_valid = frame_pred[mask]
            y_valid = y[mask]
            y_song = masked_mean(y, mask)
            frame_song = masked_mean(frame_pred, mask)
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(frame_pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(frame_pred, y, mask, config.clip_std_weight)
            loss = loss + config.song_head_weight * ((song_pred - y_song) ** 2).mean()
            loss = loss + config.frame_song_consistency_weight * ((frame_song - song_pred.detach()) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        val_song = val_eval["song_mean_metrics"]
        score = selection_score(val_m, val_song["rmse"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"val_song_mean_{k}": v for k, v in val_song.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"polarity split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f} "
            f"song_rmse={val_song['rmse']:.5f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
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
                "song_head_pred": data["song_head_pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "MERT hidden_6 + harmonic BiLSTM with auxiliary song-level polarity head",
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_song_mean_{k}": v for split, e in split_evals.items() for k, v in e["song_mean_metrics"].items()},
        **{f"{split}_song_head_{k}": v for split, e in split_evals.items() for k, v in e["song_head_metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate(rows: list[dict[str, Any]], out_dir: Path, config: PolarityConfig) -> dict[str, Any]:
    pd.DataFrame(rows).to_csv(out_dir / config.trial_id / "comparison.csv", index=False)
    aggregate_data: dict[str, Any] = {
        "created_at": "2026-06-28",
        "current_best_reference": CURRENT_BEST,
        "config": asdict(config),
        "num_splits": len(rows),
        "aggregate": {**aggregate_metrics(rows, "val"), **aggregate_metrics(rows, "test")},
        "per_split": rows,
    }
    test_mean = {k: aggregate_data["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate_data["mean_test_gap_rows"] = threshold_rows(test_mean)
    aggregate_data["mean_test_delta_vs_current_best"] = {k: float(test_mean[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]}
    (out_dir / config.trial_id / "summary.json").write_text(json.dumps(aggregate_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_data, ensure_ascii=False, indent=2), flush=True)
    return aggregate_data


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = PolarityConfig()
    train_args = SimpleNamespace(out_dir=args.out_dir, mert_dir=args.mert_dir, label_dir=args.label_dir, librosa_dataset=args.librosa_dataset)
    rows = []
    for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
        split_id, seed = split_seed(split_dir)
        rows.append(train_one_split(split_dir, split_id, seed, train_args, device, config))
    aggregate(rows, args.out_dir, config)


if __name__ == "__main__":
    main()
