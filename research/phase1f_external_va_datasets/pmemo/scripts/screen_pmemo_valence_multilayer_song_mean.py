#!/usr/bin/env python3
"""Screen multi-layer MERT Valence model with a training-time song-mean head."""

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

from screen_pmemo_valence_multilayer_mert import (
    BASE_SPLIT_DIR,
    DEFAULT_LAYER_DIRS,
    DEFAULT_LIBROSA_DATASET,
    SPLIT_REFERENCES,
    MultiLayerDataset,
    build_harmonic_norm,
    build_layer_norm,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_song_mean_screen_20260629"


@dataclass
class MultiLayerSongMeanConfig:
    trial_id: str = "h468_harm_songmean_proj384_h80_do035_sw008_blend015_cap004"
    proj_dim: int = 384
    hidden_dim: int = 80
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    batch_size: int = 18
    frame_ccc_weight: float = 0.08
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.25
    song_mean_weight: float = 0.08
    song_consistency_weight: float = 0.02
    mean_blend_weight: float = 0.15
    mean_shift_cap: float = 0.04
    raw_frame_aux_weight: float = 0.01
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def shifted_prediction(frame_pred: torch.Tensor, song_mean_pred: torch.Tensor, mask: torch.Tensor, config: MultiLayerSongMeanConfig) -> torch.Tensor:
    frame_mean = masked_mean(frame_pred, mask)
    shift = song_mean_pred - frame_mean
    if config.mean_shift_cap > 0:
        shift = torch.clamp(shift, -float(config.mean_shift_cap), float(config.mean_shift_cap))
    return frame_pred + float(config.mean_blend_weight) * shift.unsqueeze(1)


class MultiLayerSongMeanBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: MultiLayerSongMeanConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        hidden2 = config.hidden_dim * 2
        self.frame_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
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


def song_level_metrics(song_ids: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    df = pd.DataFrame({"musicId": song_ids, "target": target, "pred": pred})
    grouped = df.groupby("musicId", sort=False).mean(numeric_only=True)
    return metrics(grouped["pred"].to_numpy(dtype=np.float64), grouped["target"].to_numpy(dtype=np.float64))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: MultiLayerSongMeanConfig) -> dict[str, Any]:
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


def train_one_split(split_id: int, seed: int, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, config: MultiLayerSongMeanConfig) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    layer_summary = build_layer_norm(index, args.layer_dirs, split_dir / "dataset")
    harm_summary = build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset")
    layer_norms = {}
    for name in args.layer_dirs:
        n = np.load(split_dir / "dataset" / f"{name}_norm.npz")
        layer_norms[name] = (n["mean"], n["std"])
    hn = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            MultiLayerDataset(
                index,
                split,
                args.layer_dirs,
                layer_norms,
                args.label_dir,
                args.librosa_dataset,
                hn["mean"],
                hn["std"],
                hn["feature_idx"],
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = MultiLayerSongMeanBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss = loss + config.song_mean_weight * nn.functional.mse_loss(song_mean_pred, song_target)
            loss = loss + config.song_consistency_weight * nn.functional.mse_loss(song_mean_pred, frame_mean)
            loss = loss + config.raw_frame_aux_weight * weighted_frame_mse(frame_valid, y_valid, config.extreme_weight)
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
            f"ml_songmean split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
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
        "model_family": "MERT hidden_4+hidden_6+hidden_8 + harmonic BiLSTM with dynamic song-mean head",
        "deployability": "audio-only features; usable for arbitrary songs after extracting the same MERT layers and harmonic features",
        "config": asdict(config),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "feature_dim": int(sample_x.shape[1]),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_raw_{k}": v for split, e in split_evals.items() for k, v in e["raw_metrics"].items()},
        **{f"{split}_song_mean_{k}": v for split, e in split_evals.items() for k, v in e["song_mean_metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
        "delta_vs_split_reference": {
            k: float(split_evals["test"]["metrics"][k] - SPLIT_REFERENCES[split_id][k]) for k in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-id", type=int, default=2)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--trial-id", default=MultiLayerSongMeanConfig.trial_id)
    parser.add_argument("--song-mean-weight", type=float, default=MultiLayerSongMeanConfig.song_mean_weight)
    parser.add_argument("--song-consistency-weight", type=float, default=MultiLayerSongMeanConfig.song_consistency_weight)
    parser.add_argument("--mean-blend-weight", type=float, default=MultiLayerSongMeanConfig.mean_blend_weight)
    parser.add_argument("--mean-shift-cap", type=float, default=MultiLayerSongMeanConfig.mean_shift_cap)
    parser.add_argument("--patience", type=int, default=MultiLayerSongMeanConfig.patience)
    parser.add_argument("--epochs", type=int, default=MultiLayerSongMeanConfig.epochs)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    device = resolve_device(args.device)
    config = MultiLayerSongMeanConfig(
        trial_id=args.trial_id,
        song_mean_weight=args.song_mean_weight,
        song_consistency_weight=args.song_consistency_weight,
        mean_blend_weight=args.mean_blend_weight,
        mean_shift_cap=args.mean_shift_cap,
        patience=args.patience,
        epochs=args.epochs,
    )
    row = train_one_split(args.split_id, args.seed, index, args, device, config)
    summary = {
        "created_at": "2026-06-29",
        "purpose": "screen whether dynamic song-mean supervision improves the multi-layer MERT Valence bottleneck split",
        "split_reference": SPLIT_REFERENCES[args.split_id],
        "result": row,
    }
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
