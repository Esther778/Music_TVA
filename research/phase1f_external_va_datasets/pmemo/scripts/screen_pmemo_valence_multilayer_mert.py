#!/usr/bin/env python3
"""Screen multi-layer MERT fusion for PMEmo Valence.

The current Valence bottleneck is split_02. This experiment keeps the model
deployable for arbitrary songs by using only audio-derived features:
MERT hidden_4, hidden_6, hidden_8, plus harmonic Librosa features.
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

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    collate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_mert_screen_20260629"
DEFAULT_LAYER_DIRS = {
    "h4": ROOT / "outputs" / "mert_pmemo_hidden4" / "features",
    "h6": ROOT / "outputs" / "mert_pmemo_hidden6" / "features",
    "h8": ROOT / "outputs" / "mert_pmemo_hidden8" / "features",
}

CURRENT_STRICTISH = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}

SPLIT_REFERENCES = {
    1: {"rmse": 0.1815586949125916, "mae": 0.1419195070813537, "pcc": 0.8449925768337622, "ccc": 0.835335379286839},
    2: {"rmse": 0.2093862469953028, "mae": 0.1624014423698784, "pcc": 0.7622912308938166, "ccc": 0.7429679473144445},
    3: {"rmse": 0.1950039682105851, "mae": 0.1510577558657, "pcc": 0.7860129711134074, "ccc": 0.77619944804177},
}


@dataclass
class MultiLayerConfig:
    trial_id: str = "h468_harm_concat_proj384_h80_do035"
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
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_layer_norm(index: pd.DataFrame, layer_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, layer_dir in layer_dirs.items():
        xs = []
        for row in index[index["split"] == "train"].to_dict("records"):
            sid = int(row["musicId"])
            xs.append(np.load(layer_dir / f"{sid}.npz")["embedding"].astype(np.float32))
        concat = np.concatenate(xs, axis=0)
        mean = concat.mean(axis=0).astype(np.float32)
        std = (concat.std(axis=0) + 1e-8).astype(np.float32)
        np.savez(out_dir / f"{name}_norm.npz", mean=mean, std=std)
        summary[name] = {"dim": int(concat.shape[1]), "train_frames": int(concat.shape[0])}
    (out_dir / "layer_norm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_harmonic_norm(index: pd.DataFrame, librosa_dataset: Path, feature_idx: list[int], out_dir: Path) -> dict[str, Any]:
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        xs.append(np.load(librosa_dataset / "sequence" / f"{sid}.npz")["X"].astype(np.float32)[:, feature_idx])
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "harmonic_norm.npz", mean=mean, std=std, feature_idx=np.asarray(feature_idx, dtype=np.int32))
    return {"dim": int(len(feature_idx)), "train_frames": int(concat.shape[0])}


class MultiLayerDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        layer_dirs: dict[str, Path],
        layer_norms: dict[str, tuple[np.ndarray, np.ndarray]],
        label_dir: Path,
        librosa_dataset: Path,
        harm_mean: np.ndarray,
        harm_std: np.ndarray,
        feature_idx: np.ndarray,
    ):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.layer_dirs = layer_dirs
        self.layer_norms = layer_norms
        self.label_dir = label_dir
        self.librosa_dataset = librosa_dataset
        self.harm_mean = harm_mean.astype(np.float32)
        self.harm_std = harm_std.astype(np.float32)
        self.feature_idx = feature_idx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid = int(self.rows[idx]["musicId"])
        parts = []
        time_sec = None
        for name, layer_dir in self.layer_dirs.items():
            data = np.load(layer_dir / f"{sid}.npz")
            x = data["embedding"].astype(np.float32)
            mean, std = self.layer_norms[name]
            parts.append(((x - mean) / std).astype(np.float32))
            if time_sec is None:
                time_sec = data["time_sec"].astype(np.float32)
        lib = np.load(self.librosa_dataset / "sequence" / f"{sid}.npz")
        x_harm = lib["X"].astype(np.float32)[:, self.feature_idx]
        x_harm = ((x_harm - self.harm_mean) / self.harm_std).astype(np.float32)
        y = np.load(self.label_dir / f"{sid}.npz")["Y"].astype(np.float32)[:, 0]
        n = min(*[len(p) for p in parts], len(x_harm), len(y), len(time_sec))
        x = np.concatenate([*[p[:n] for p in parts], x_harm[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


class MultiLayerBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: MultiLayerConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.out(y).squeeze(-1)


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


def train_one_split(split_id: int, seed: int, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, config: MultiLayerConfig) -> dict[str, Any]:
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
    model = MultiLayerBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
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
            f"multilayer split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
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
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "MERT hidden_4+hidden_6+hidden_8 + harmonic BiLSTM",
        "deployability": "audio-only features; usable for arbitrary songs after extracting the same MERT layers and harmonic features",
        "config": asdict(config),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "feature_dim": int(sample_x.shape[1]),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
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
    parser.add_argument("--trial-id", default=MultiLayerConfig.trial_id)
    parser.add_argument("--proj-dim", type=int, default=MultiLayerConfig.proj_dim)
    parser.add_argument("--hidden-dim", type=int, default=MultiLayerConfig.hidden_dim)
    parser.add_argument("--dropout", type=float, default=MultiLayerConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=MultiLayerConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=MultiLayerConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=MultiLayerConfig.batch_size)
    parser.add_argument("--frame-ccc-weight", type=float, default=MultiLayerConfig.frame_ccc_weight)
    parser.add_argument("--clip-mse-weight", type=float, default=MultiLayerConfig.clip_mse_weight)
    parser.add_argument("--clip-std-weight", type=float, default=MultiLayerConfig.clip_std_weight)
    parser.add_argument("--extreme-weight", type=float, default=MultiLayerConfig.extreme_weight)
    parser.add_argument("--patience", type=int, default=MultiLayerConfig.patience)
    parser.add_argument("--epochs", type=int, default=MultiLayerConfig.epochs)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    device = resolve_device(args.device)
    config = MultiLayerConfig(
        trial_id=args.trial_id,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        frame_ccc_weight=args.frame_ccc_weight,
        clip_mse_weight=args.clip_mse_weight,
        clip_std_weight=args.clip_std_weight,
        extreme_weight=args.extreme_weight,
        patience=args.patience,
        epochs=args.epochs,
    )
    row = train_one_split(args.split_id, args.seed, index, args, device, config)
    summary = {
        "created_at": "2026-06-29",
        "purpose": "screen multi-layer MERT feature fusion on the Valence bottleneck split",
        "current_strictish_reference": CURRENT_STRICTISH,
        "split_reference": SPLIT_REFERENCES[args.split_id],
        "result": row,
    }
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
