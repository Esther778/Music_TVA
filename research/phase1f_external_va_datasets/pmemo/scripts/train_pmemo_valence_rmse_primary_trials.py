#!/usr/bin/env python3
"""RMSE-primary Valence training trials near the historical best family."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import ROOT, resolve_device, threshold_rows
import train_pmemo_valence_multisplit_best as best
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR, BestConfig, train_one_split


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_rmse_primary_training_trials_20260708"

CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}

TRIALS = {
    "h80_rmse_ext110_clip005_ccc002_do025": BestConfig(
        trial_id="h80_rmse_ext110_clip005_ccc002_do025",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.25,
        learning_rate=4e-4,
        weight_decay=2e-4,
        batch_size=20,
        frame_ccc_weight=0.02,
        clip_mse_weight=0.05,
        extreme_weight=1.10,
        clip_std_weight=0.0,
        patience=8,
        epochs=60,
    ),
    "h96_rmse_ext100_clip005_ccc003_do030": BestConfig(
        trial_id="h96_rmse_ext100_clip005_ccc003_do030",
        proj_dim=384,
        hidden_dim=96,
        dropout=0.30,
        learning_rate=3e-4,
        weight_decay=3e-4,
        batch_size=20,
        frame_ccc_weight=0.03,
        clip_mse_weight=0.05,
        extreme_weight=1.00,
        clip_std_weight=0.0,
        patience=8,
        epochs=60,
    ),
}


def rmse_primary_score(m: dict[str, float]) -> float:
    """Select checkpoints by validation RMSE first, with tiny trend penalties."""
    return (
        float(m["rmse"])
        + 0.015 * max(0.0, 0.80 - float(m["ccc"]))
        + 0.010 * max(0.0, 0.80 - float(m["pcc"]))
        + 0.003 * float(m["mae"])
    )


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def macro(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[f"{split}_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", choices=list(TRIALS) + ["all"], default="all")
    parser.add_argument("--split-id", type=int, default=2)
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best.selection_score = rmse_primary_score
    device = resolve_device(args.device)
    configs = list(TRIALS.values()) if args.trial_id == "all" else [TRIALS[args.trial_id]]
    split_dirs = sorted(BASE_SPLIT_DIR.glob("split_*"))
    if not args.all_splits:
        split_dirs = [p for p in split_dirs if split_seed(p)[0] == args.split_id]

    summaries = []
    for config in configs:
        trial_out = args.out_dir / config.trial_id
        trial_out.mkdir(parents=True, exist_ok=True)
        train_args = SimpleNamespace(
            out_dir=trial_out,
            mert_dir=args.mert_dir,
            label_dir=args.label_dir,
            librosa_dataset=args.librosa_dataset,
        )
        rows = []
        for split_dir in split_dirs:
            split_id, seed = split_seed(split_dir)
            index = pd.read_csv(split_dir / "sequence_index.csv")
            rows.append(train_one_split(split_id, seed, index, train_args, device, config))
        test_macro = macro(rows, "test")
        summary = {
            "created_at": "2026-07-08",
            "purpose": "RMSE-primary checkpoint/loss screen for PMEmo Valence",
            "selection_rule": "checkpoint selected by validation RMSE-primary score; test held out",
            "current_best_reference": CURRENT_BEST,
            "config": asdict(config),
            "split_scope": "all_splits" if args.all_splits else f"split_{args.split_id:02d}",
            "per_split": rows,
            "test_macro": test_macro,
            "delta_vs_current_best": {
                metric: float(test_macro[metric] - CURRENT_BEST[metric])
                for metric in ["rmse", "mae", "pcc", "ccc"]
            },
            "threshold_gaps": threshold_rows(test_macro),
        }
        (trial_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        summaries.append(summary)

    comparison = []
    for summary in summaries:
        row = {
            "trial_id": summary["config"]["trial_id"],
            **{f"test_{k}": v for k, v in summary["test_macro"].items()},
            **{f"delta_vs_best_{k}": v for k, v in summary["delta_vs_current_best"].items()},
        }
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(args.out_dir / "trial_comparison.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-08",
                "current_best_reference": CURRENT_BEST,
                "split_scope": "all_splits" if args.all_splits else f"split_{args.split_id:02d}",
                "trials": comparison,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
