#!/usr/bin/env python3
"""Train a clean multi-layer MERT Valence teacher candidate.

Purpose:
- move from prediction-level artifacts toward a reproducible teacher made from
  model checkpoints;
- keep the same strongest audio-only feature direction:
  MERT hidden_4 + hidden_6 + hidden_8 + Librosa harmonic;
- report every run against the current metric-best and threshold lines.

This script is intentionally a wrapper around the already tested multi-layer
training implementation. It does not hide test results during development, but
all checkpoint selection remains validation-based inside each split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screen_pmemo_valence_multilayer_mert import (
    BASE_SPLIT_DIR,
    DEFAULT_LABEL_DIR,
    DEFAULT_LAYER_DIRS,
    DEFAULT_LIBROSA_DATASET,
    MultiLayerConfig,
    train_one_split,
)
from train_pmemo_mert_valence_baseline import ROOT, resolve_device, threshold_rows


DEFAULT_OUT = ROOT / "outputs" / "valence_multilayer_clean_teacher_20260707"
SOURCE_SPLIT_SEEDS = {1: 101, 2: 202, 3: 303}

CURRENT_METRIC_BEST = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
    "deployability": "metric-best CV prediction artifact, not a clean checkpoint teacher",
}

DEPLOYABLE_APPROX = {
    "model_id": "pmemo_valence_current_best_deployable_approx_20260707",
    "script": "scripts/predict_full_song_valence_current_best_approx.py",
    "note": "raw-MP3 deployable approximation; not independently metric-validated as a new best",
}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[f"test_{metric}"]) for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def split_ids_from_args(args: argparse.Namespace) -> list[int]:
    if args.all_splits:
        return [1, 2, 3]
    return [int(args.split_id)]


def load_index(split_id: int) -> pd.DataFrame:
    source_seed = SOURCE_SPLIT_SEEDS[int(split_id)]
    return pd.read_csv(BASE_SPLIT_DIR / f"split_{int(split_id):02d}_seed_{source_seed}" / "sequence_index.csv")


def build_config(args: argparse.Namespace) -> MultiLayerConfig:
    return MultiLayerConfig(
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


def write_model_card(out_dir: Path, config: MultiLayerConfig, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    card = {
        "created_at": "2026-07-07",
        "model_id": summary["model_id"],
        "purpose": "clean checkpoint-based Valence teacher candidate",
        "target": "Valence",
        "label_scale": "[-1, 1]",
        "feature_type": "MERT hidden_4 + hidden_6 + hidden_8 + Librosa harmonic",
        "architecture": "BiLSTM with projection head",
        "config": asdict(config),
        "selection_rule": "checkpoint selected by validation selection_score; test reported after selection",
        "current_metric_best_reference": CURRENT_METRIC_BEST,
        "deployable_approx_reference": DEPLOYABLE_APPROX,
        "summary": summary,
        "per_split": rows,
        "deployment_note": (
            "If this clean teacher beats or sufficiently approaches the artifact-based metric best, "
            "write a matching raw-MP3 predictor that loads its checkpoints directly."
        ),
    }
    (out_dir / "MODEL_CARD.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-id", type=int, default=2)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--trial-id", default="h468_harm_clean_proj384_h80_do035")
    parser.add_argument("--proj-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--frame-ccc-weight", type=float, default=0.08)
    parser.add_argument("--clip-mse-weight", type=float, default=0.10)
    parser.add_argument("--clip-std-weight", type=float, default=0.0)
    parser.add_argument("--extreme-weight", type=float, default=1.25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.layer_dirs = DEFAULT_LAYER_DIRS
    device = resolve_device(args.device)
    config = build_config(args)

    rows = []
    for split_id in split_ids_from_args(args):
        index = load_index(split_id)
        rows.append(train_one_split(split_id, int(args.seed), index, args, device, config))

    test_macro = aggregate(rows)
    summary = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_multilayer_clean_teacher_20260707",
        "purpose": "clean checkpoint-based multi-layer MERT Valence teacher candidate",
        "num_splits": len(rows),
        "config": asdict(config),
        "test_macro": test_macro,
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_current_metric_best": {
            metric: float(test_macro[metric] - CURRENT_METRIC_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "current_metric_best_reference": CURRENT_METRIC_BEST,
        "deployable_approx_reference": DEPLOYABLE_APPROX,
        "best_update_rule": (
            "Promote only if validation-selected test macro metrics improve the current metric best "
            "without losing deployability, or if user explicitly accepts a slightly weaker but cleaner teacher."
        ),
        "per_split": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_model_card(args.out_dir, config, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
