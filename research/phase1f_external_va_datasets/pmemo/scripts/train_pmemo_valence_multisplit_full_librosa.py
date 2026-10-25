#!/usr/bin/env python3
"""Multi-split Valence BiLSTM using MERT hidden_6 + all Librosa features."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
    FusionBiLSTM,
    FusionDataset,
    build_harmonic_norm,
    collate,
    evaluate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR, BestConfig, aggregate_metrics, build_mert_norm, selection_score, set_seed


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_full_librosa_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}

DEFAULT_TRIAL_ID = "h6_alllib_h80_ext125_clip010_ccc005"
DEFAULT_FEATURE_GROUPS = ["mfcc", "spectral", "harmonic", "energy", "tempo"]


def make_config(trial_id: str) -> BestConfig:
    return BestConfig(
    trial_id=trial_id,
    proj_dim=352,
    hidden_dim=80,
    dropout=0.30,
    learning_rate=4e-4,
    weight_decay=2e-4,
    batch_size=20,
    frame_ccc_weight=0.05,
    clip_mse_weight=0.10,
    extreme_weight=1.25,
    clip_std_weight=0.0,
    patience=8,
    epochs=60,
)


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def summarize(rows: list[dict[str, Any]], out_dir: Path, config: BestConfig) -> dict[str, Any]:
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    data = {
        "created_at": "2026-06-28",
        "model_family": "MERT hidden_6 + all Librosa feature BiLSTM",
        "feature_groups": ["mfcc", "spectral", "harmonic", "energy", "tempo"],
        "fixed_split_source": str(BASE_SPLIT_DIR),
        "current_best_reference": CURRENT_BEST,
        "config": asdict(config),
        "num_splits": len(rows),
        "aggregate": {**aggregate_metrics(rows, "val"), **aggregate_metrics(rows, "test")},
        "per_split": rows,
    }
    test_mean = {k: data["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    data["mean_test_gap_rows"] = threshold_rows(test_mean)
    data["mean_test_delta_vs_current_best"] = {k: float(test_mean[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]}
    (out_dir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    return data


def train_one_split_full(
    split_id: int,
    seed: int,
    index: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    config: BestConfig,
) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.mert_dir, split_dir / "dataset")
    all_idx = selected_indices(args.feature_groups)
    build_harmonic_norm(index, args.librosa_dataset, all_idx, split_dir / "dataset", force=True)
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    lib_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
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
                lib_norm["mean"],
                lib_norm["std"],
                lib_norm["feature_idx"],
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
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
            f"alllib split={split_id:02d} seed={seed} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
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
        "model_family": "MERT hidden_6 + all Librosa feature BiLSTM",
        "config": asdict(config),
        "feature_dim": int(sample_x.shape[1]),
        "librosa_feature_dim": int(len(all_idx)),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", default=DEFAULT_TRIAL_ID)
    parser.add_argument("--feature-groups", nargs="+", default=DEFAULT_FEATURE_GROUPS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = make_config(args.trial_id)
    rows = []
    for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
        split_id, seed = split_seed(split_dir)
        index = pd.read_csv(split_dir / "sequence_index.csv")
        rows.append(train_one_split_full(split_id, seed, index, args, device, config))
    summarize(rows, args.out_dir, config)


if __name__ == "__main__":
    main()
