#!/usr/bin/env python3
"""Multi-split lightweight TCN training for PMEmo Valence."""

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
from train_pmemo_valence_multisplit_best import (
    DEFAULT_MERT_DIR,
    HISTORICAL_SINGLE_SPLIT_BEST,
    aggregate_metrics,
    build_mert_norm,
    selection_score,
)


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_tcn_20260627"


@dataclass
class TCNConfig:
    trial_id: str = "h6_harm_tcn_c128_k5_dilated_ext125"
    proj_dim: int = 128
    channels: int = 128
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.25
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    extreme_weight: float = 1.25
    clip_std_weight: float = 0.0
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm(x + y)
        return out * mask.float().unsqueeze(-1)


class FusionTCN(nn.Module):
    def __init__(self, input_dim: int, config: TCNConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.proj_dim, config.channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(config.channels, config.kernel_size, dilation, config.dropout) for dilation in config.dilations]
        )
        self.out = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.channels, config.channels // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.proj(x) * mask.float().unsqueeze(-1)
        for block in self.blocks:
            y = block(y, mask)
        return self.out(y).squeeze(-1)


@torch.no_grad()
def evaluate_tcn(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
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


def train_one_split(split_dir_in: Path, split_id: int, seed: int, args: argparse.Namespace, device: torch.device, config: TCNConfig) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(split_dir_in if split_dir_in.is_file() else split_dir_in / "sequence_index.csv")
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.mert_dir, split_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset", force=True)
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    split_names = [split for split in ["train", "val", "test"] if split in set(index["split"])]
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
        for split in split_names
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionTCN(int(sample_x.shape[1]), config).to(device)
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
            loss = loss + float(config.frame_ccc_weight) * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate_tcn(model, loaders["val"], device)
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
            f"TCN split={split_id:02d} seed={seed} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
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
    split_evals = {split: evaluate_tcn(model, loaders[split], device) for split in split_names}
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
        "model_family": "MERT hidden_6 + harmonic lightweight TCN",
        "config": asdict(config),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in split_names},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in split_names},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "val_gap_rows": threshold_rows(split_evals["val"]["metrics"]),
    }
    if "test" in split_evals:
        summary["test_gap_rows"] = threshold_rows(split_evals["test"]["metrics"])
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def aggregate(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    aggregate_data: dict[str, Any] = {
        "created_at": "2026-06-27",
        "selection_rule": "same fixed splits as valence_multisplit_best_20260627; checkpoint selected by validation selection_score",
        "historical_single_split_best": HISTORICAL_SINGLE_SPLIT_BEST,
        "num_splits": len(rows),
        "aggregate": {
            **aggregate_metrics(rows, "val"),
            **aggregate_metrics(rows, "test"),
        },
        "per_split": rows,
    }
    test_mean = {k: aggregate_data["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate_data["mean_test_gap_rows"] = threshold_rows(test_mean)
    (out_dir / "summary.json").write_text(json.dumps(aggregate_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_data, ensure_ascii=False, indent=2), flush=True)
    return aggregate_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split-dir", type=Path, default=BASE_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--split-id", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.out_dir.glob("split_*/summary.json"))]
        aggregate(rows, args.out_dir)
        return
    device = resolve_device(args.device)
    config = TCNConfig()
    if args.index is not None:
        rows = [train_one_split(args.index, args.split_id, 20260713, args, device, config)]
        summary = {
            "created_at": "2026-07-13",
            "selection_boundary": "Development folds only; final locked test excluded",
            "per_split": rows,
        }
        (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    rows = []
    for split_id, split_dir in enumerate(sorted(args.base_split_dir.glob("split_*")), start=1):
        if not (split_dir / "sequence_index.csv").exists():
            continue
        seed = int(split_dir.name.rsplit("_", 1)[-1])
        rows.append(train_one_split(split_dir, split_id, seed, args, device, config))
    aggregate(rows, args.out_dir)


if __name__ == "__main__":
    main()
