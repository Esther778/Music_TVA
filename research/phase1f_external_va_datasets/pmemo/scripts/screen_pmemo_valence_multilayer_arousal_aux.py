#!/usr/bin/env python3
"""Screen Valence-focused multi-layer MERT model with Arousal auxiliary loss.

This is still an audio-only model: both Valence and Arousal targets are used
only during training. Inference for arbitrary songs only needs the same MERT
layers and harmonic Librosa features.
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

from screen_pmemo_valence_multilayer_mert import (
    BASE_SPLIT_DIR,
    DEFAULT_LAYER_DIRS,
    DEFAULT_LIBROSA_DATASET,
    SPLIT_REFERENCES,
    build_harmonic_norm,
    build_layer_norm,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, collate as _single_collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_arousal_aux_screen_20260629"


@dataclass
class AuxConfig:
    trial_id: str = "h468_harm_auxA_proj384_h80_do035_aux020"
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
    arousal_aux_weight: float = 0.20
    arousal_ccc_weight: float = 0.05
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiTargetDataset(Dataset):
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
        y = np.load(self.label_dir / f"{sid}.npz")["Y"].astype(np.float32)[:, :2]
        n = min(*[len(p) for p in parts], len(x_harm), len(y), len(time_sec))
        x = np.concatenate([*[p[:n] for p in parts], x_harm[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    target_dim = batch[0][1].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim)
    y_out = torch.zeros(len(batch), max_len, target_dim)
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


class AuxBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: AuxConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.head(y)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds_v, preds_a, targets_v, targets_a, songs, times = [], [], [], [], [], []
    for x, y, song, time_sec, mask in loader:
        pred = model(x.to(device), mask.to(device)).detach().cpu()
        preds_v.append(pred[..., 0][mask].numpy())
        preds_a.append(pred[..., 1][mask].numpy())
        targets_v.append(y[..., 0][mask].numpy())
        targets_a.append(y[..., 1][mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred_v = np.concatenate(preds_v)
    pred_a = np.concatenate(preds_a)
    target_v = np.concatenate(targets_v)
    target_a = np.concatenate(targets_a)
    return {
        "valence_metrics": metrics(pred_v, target_v),
        "arousal_metrics": metrics(pred_a, target_a),
        "pred_valence": pred_v,
        "pred_arousal": pred_a,
        "target_valence": target_v,
        "target_arousal": target_a,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


def train_one_split(split_id: int, seed: int, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, config: AuxConfig) -> dict[str, Any]:
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
            MultiTargetDataset(
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
    model = AuxBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            pred_v = pred[..., 0]
            pred_a = pred[..., 1]
            y_v = y[..., 0]
            y_a = y[..., 1]
            pred_v_valid = pred_v[mask]
            y_v_valid = y_v[mask]
            pred_a_valid = pred_a[mask]
            y_a_valid = y_a[mask]
            loss = weighted_frame_mse(pred_v_valid, y_v_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_v_valid, y_v_valid)
            loss = loss + clip_mean_loss(pred_v, y_v, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred_v, y_v, mask, config.clip_std_weight)
            aux = ((pred_a_valid - y_a_valid) ** 2).mean()
            aux = aux + config.arousal_ccc_weight * ccc_loss(pred_a_valid, y_a_valid)
            loss = loss + config.arousal_aux_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["valence_metrics"]
        score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_select_score": score}
        history.append(row)
        print(
            f"auxA split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
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
                "target_valence": data["target_valence"],
                "pred_valence": data["pred_valence"],
                "target_arousal": data["target_arousal"],
                "pred_arousal": data["pred_arousal"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "MERT hidden_4+hidden_6+hidden_8 + harmonic BiLSTM with Arousal auxiliary head",
        "deployability": "audio-only features; usable for arbitrary songs after extracting the same MERT layers and harmonic features",
        "config": asdict(config),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "feature_dim": int(sample_x.shape[1]),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["valence_metrics"].items()},
        "aux_arousal_metrics": {split: e["arousal_metrics"] for split, e in split_evals.items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["valence_metrics"]),
        "delta_vs_split_reference": {
            k: float(split_evals["test"]["valence_metrics"][k] - SPLIT_REFERENCES[split_id][k]) for k in ["rmse", "mae", "pcc", "ccc"]
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
    parser.add_argument("--trial-id", default=AuxConfig.trial_id)
    parser.add_argument("--arousal-aux-weight", type=float, default=AuxConfig.arousal_aux_weight)
    parser.add_argument("--arousal-ccc-weight", type=float, default=AuxConfig.arousal_ccc_weight)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    device = resolve_device(args.device)
    config = AuxConfig(
        trial_id=args.trial_id,
        arousal_aux_weight=args.arousal_aux_weight,
        arousal_ccc_weight=args.arousal_ccc_weight,
    )
    row = train_one_split(args.split_id, args.seed, index, args, device, config)
    summary = {
        "created_at": "2026-06-29",
        "purpose": "screen whether Arousal auxiliary supervision improves the Valence bottleneck split",
        "split_reference": SPLIT_REFERENCES[args.split_id],
        "result": row,
    }
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
