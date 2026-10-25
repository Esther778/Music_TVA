#!/usr/bin/env python3
"""Two-head PMEmo Valence model with explicit dynamic song-mean supervision."""

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
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionDataset,
    build_harmonic_norm,
    collate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import (
    clip_mean_loss,
    clip_std_loss,
    weighted_frame_mse,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR, aggregate_metrics, build_mert_norm


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_two_head_song_mean_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}


@dataclass
class TwoHeadConfig:
    trial_id: str = "two_head_h80_songw008_blend015_cap004"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    song_mean_weight: float = 0.08
    song_consistency_weight: float = 0.02
    mean_blend_weight: float = 0.15
    mean_shift_cap: float = 0.04
    patience: int = 7
    epochs: int = 45


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def shifted_prediction(frame_pred: torch.Tensor, song_mean_pred: torch.Tensor, mask: torch.Tensor, config: TwoHeadConfig) -> torch.Tensor:
    frame_mean = masked_mean(frame_pred, mask)
    shift = song_mean_pred - frame_mean
    if config.mean_shift_cap > 0:
        shift = torch.clamp(shift, -float(config.mean_shift_cap), float(config.mean_shift_cap))
    return frame_pred + float(config.mean_blend_weight) * shift.unsqueeze(1)


class TwoHeadBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: TwoHeadConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        hidden2 = config.hidden_dim * 2
        self.frame_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(hidden2, 1))
        self.song_head = nn.Sequential(
            nn.LayerNorm(hidden2 * 3),
            nn.Linear(hidden2 * 3, hidden2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.proj(x))
        frame = self.frame_head(h).squeeze(-1)
        mask_f = mask.float().unsqueeze(-1)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (h * mask_f).sum(dim=1) / count
        var = (((h - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
        std = torch.sqrt(var + 1e-6)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        song_mean = self.song_head(torch.cat([mean, std, max_pool], dim=1)).squeeze(-1)
        return frame, song_mean


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.05 * max(0.0, 0.80 - m["ccc"]) + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.01 * m["mae"]


def song_level_metrics(song_ids: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    df = pd.DataFrame({"musicId": song_ids, "target": target, "pred": pred})
    grouped = df.groupby("musicId", sort=False).mean(numeric_only=True)
    return metrics(grouped["pred"].to_numpy(dtype=np.float64), grouped["target"].to_numpy(dtype=np.float64))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: TwoHeadConfig) -> dict[str, Any]:
    model.eval()
    preds, raw_preds, targets, songs, times, repeated_song = [], [], [], [], [], []
    for x, y, song, time_sec, mask in loader:
        x = x.to(device)
        mask_dev = mask.to(device)
        frame_pred, song_mean_pred = model(x, mask_dev)
        pred = shifted_prediction(frame_pred, song_mean_pred, mask_dev, config).detach().cpu()
        raw = frame_pred.detach().cpu()
        song_cpu = song_mean_pred.detach().cpu()
        repeated = torch.zeros_like(raw)
        for i in range(raw.shape[0]):
            repeated[i, mask[i]] = song_cpu[i]
        preds.append(pred[mask].numpy())
        raw_preds.append(raw[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
        repeated_song.append(repeated[mask].numpy())
    pred_np = np.concatenate(preds)
    raw_np = np.concatenate(raw_preds)
    target_np = np.concatenate(targets)
    song_np = np.concatenate(songs)
    return {
        "metrics": metrics(pred_np, target_np),
        "raw_metrics": metrics(raw_np, target_np),
        "song_mean_metrics": song_level_metrics(song_np, target_np, pred_np),
        "pred": pred_np,
        "raw_pred": raw_np,
        "target": target_np,
        "song_id": song_np,
        "time_sec": np.concatenate(times),
        "song_mean_pred": np.concatenate(repeated_song),
    }


def train_one_split(
    split_dir_in: Path,
    split_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    config: TwoHeadConfig,
) -> dict[str, Any]:
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
    model = TwoHeadBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            frame_pred, song_mean_pred = model(x, mask)
            pred = shifted_prediction(frame_pred, song_mean_pred, mask, config)
            pred_valid = pred[mask]
            frame_valid = frame_pred[mask]
            y_valid = y[mask]
            song_target = masked_mean(y, mask)
            frame_mean = masked_mean(frame_pred, mask)
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + float(config.frame_ccc_weight) * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss = loss + float(config.song_mean_weight) * nn.functional.mse_loss(song_mean_pred, song_target)
            loss = loss + float(config.song_consistency_weight) * nn.functional.mse_loss(song_mean_pred, frame_mean)
            loss = loss + 0.01 * weighted_frame_mse(frame_valid, y_valid, config.extreme_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device, config)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"val_raw_{k}": v for k, v in val_eval["raw_metrics"].items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"trial={config.trial_id} split={split_id:02d} seed={seed} epoch={epoch:03d} "
            f"loss={row['train_loss']:.5f} val_rmse={val_m['rmse']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "config": asdict(config),
                    "input_dim": int(sample_x.shape[1]),
                },
                ckpt,
            )
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device, config) for split in ["train", "val", "test"]}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
                "raw_pred_valence": data["raw_pred"],
                "song_mean_pred": data["song_mean_pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_raw_{k}": v for split, e in split_evals.items() for k, v in e["raw_metrics"].items()},
        **{f"{split}_song_mean_{k}": v for split, e in split_evals.items() for k, v in e["song_mean_metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize(rows: list[dict[str, Any]], out_dir: Path, config: TwoHeadConfig) -> dict[str, Any]:
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / config.trial_id / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-06-28",
        "model_id": f"pmemo_valence_{config.trial_id}_20260628",
        "current_best": CURRENT_BEST,
        "purpose": "Add explicit dynamic song-mean supervision to reduce Valence global mean/polarity failures.",
        "selection_rule": "checkpoint selected by validation frame-level selection_score; test held out per split",
        "num_splits": len(rows),
        "config": asdict(config),
        "aggregate": {
            **aggregate_metrics(rows, "val"),
            **aggregate_metrics(rows, "test"),
            **aggregate_metrics(rows, "test_raw"),
            **aggregate_metrics(rows, "test_song_mean"),
        },
        "per_split": rows,
    }
    test_mean = {k: aggregate["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["mean_test_gap_rows"] = threshold_rows(test_mean)
    aggregate["delta_vs_current_best"] = {
        metric: float(test_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    (out_dir / config.trial_id / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split-dir", type=Path, default=BASE_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--song-mean-weight", type=float, default=None)
    parser.add_argument("--song-consistency-weight", type=float, default=None)
    parser.add_argument("--mean-blend-weight", type=float, default=None)
    parser.add_argument("--mean-shift-cap", type=float, default=None)
    parser.add_argument("--split-limit", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    config = TwoHeadConfig()
    if args.trial_id is not None:
        config.trial_id = str(args.trial_id)
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.patience is not None:
        config.patience = int(args.patience)
    if args.song_mean_weight is not None:
        config.song_mean_weight = float(args.song_mean_weight)
    if args.song_consistency_weight is not None:
        config.song_consistency_weight = float(args.song_consistency_weight)
    if args.mean_blend_weight is not None:
        config.mean_blend_weight = float(args.mean_blend_weight)
    if args.mean_shift_cap is not None:
        config.mean_shift_cap = float(args.mean_shift_cap)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(trial_dir.glob("split_*/summary.json"))]
        summarize(rows, args.out_dir, config)
        return
    device = resolve_device(args.device)
    rows = []
    split_dirs = sorted(p for p in args.base_split_dir.glob("split_*_seed_*") if p.is_dir())
    if args.split_limit is not None:
        split_dirs = split_dirs[: int(args.split_limit)]
    for split_id, split_dir in enumerate(split_dirs, start=1):
        seed = int(split_dir.name.split("_seed_")[-1])
        rows.append(train_one_split(split_dir, split_id, seed, args, device, config))
    summarize(rows, args.out_dir, config)


if __name__ == "__main__":
    main()
