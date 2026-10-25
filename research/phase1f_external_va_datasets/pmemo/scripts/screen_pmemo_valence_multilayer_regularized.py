#!/usr/bin/env python3
"""Screen a more regularized direct-concat multi-layer MERT Valence model.

Direct MERT hidden_4+6+8 concatenation improved split_02 but hurt split_01.
The layer-gated low-dimensional model lost the useful split_02 signal. This
experiment keeps direct concatenation but reduces head capacity and strengthens
regularization.
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
from torch.utils.data import DataLoader

from screen_pmemo_valence_multilayer_mert import (
    BASE_SPLIT_DIR,
    CURRENT_STRICTISH,
    DEFAULT_LAYER_DIRS,
    SPLIT_REFERENCES,
    MultiLayerDataset,
    build_harmonic_norm,
    build_layer_norm,
    evaluate,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import selection_score


DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_regularized_screen_20260629"


@dataclass
class RegularizedMultiLayerConfig:
    trial_id: str = "h468_harm_proj256_h64_do045_wd001_smooth002"
    proj_dim: int = 256
    hidden_dim: int = 64
    dropout: float = 0.45
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-3
    batch_size: int = 16
    frame_ccc_weight: float = 0.08
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    extreme_weight: float = 1.15
    smooth_weight: float = 0.002
    patience: int = 10
    epochs: int = 70


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temporal_smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    diff = pred[:, 1:] - pred[:, :-1]
    return (diff[valid] ** 2).mean()


class RegularizedMultiLayerBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: RegularizedMultiLayerConfig):
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
            nn.Linear(config.hidden_dim * 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.out(y).squeeze(-1)


def train_one_split(
    split_id: int,
    seed: int,
    index: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    config: RegularizedMultiLayerConfig,
) -> dict[str, Any]:
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
    model = RegularizedMultiLayerBiLSTM(int(sample_x.shape[1]), config).to(device)
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
            loss = loss + config.smooth_weight * temporal_smoothness_loss(pred, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"regularized_multilayer split={split_id:02d} seed={seed} epoch={epoch:03d} "
            f"loss={row['train_loss']:.5f} val_rmse={val_m['rmse']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
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
        "model_family": "Regularized direct-concat MERT hidden_4+hidden_6+hidden_8 + harmonic BiLSTM",
        "deployability": "audio-only features; usable for arbitrary songs after extracting the same MERT layers and harmonic features",
        "config": asdict(config),
        "layer_summary": layer_summary,
        "harmonic_summary": harm_summary,
        "feature_dim": int(sample_x.shape[1]),
        "effective_sequence_dim": int(config.proj_dim),
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
    parser.add_argument("--seed", type=int, default=808)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    source_seed = {1: 101, 2: 202, 3: 303}[args.split_id]
    index = pd.read_csv(BASE_SPLIT_DIR / f"split_{args.split_id:02d}_seed_{source_seed}" / "sequence_index.csv")
    device = resolve_device(args.device)
    config = RegularizedMultiLayerConfig()
    row = train_one_split(args.split_id, args.seed, index, args, device, config)
    summary = {
        "created_at": "2026-06-29",
        "purpose": "screen stronger regularization on direct-concat multi-layer MERT for Valence",
        "current_strictish_reference": CURRENT_STRICTISH,
        "split_reference": SPLIT_REFERENCES[args.split_id],
        "result": row,
    }
    out_path = args.out_dir / config.trial_id / "screen_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
