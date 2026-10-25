#!/usr/bin/env python3
"""Screen a weak static-Valence auxiliary head for PMEmo Valence.

This is a deliberately conservative experiment:
- keep the current MERT hidden_6 + harmonic BiLSTM frame-level predictor;
- add PMEmo static Valence only as a very weak auxiliary head;
- do not blend static predictions into frame-level output;
- run split_01 first before considering full multi-split training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from train_pmemo_mert_valence_baseline import ROOT, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_static_aux import (
    BASE_SPLIT_DIR,
    CURRENT_BEST,
    DEFAULT_MERT_DIR,
    PASS,
    STATIC_ANNOTATIONS,
    StaticAuxConfig,
    aggregate_metrics,
    load_static_valence,
    split_seed,
    train_one_split,
)
from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET


DEFAULT_OUT = ROOT / "outputs" / "valence_weak_static_aux_screen_20260628"

SPLIT01_BASELINE = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "split_id": 1,
    "val_rmse": 0.215691,
    "val_mae": 0.167271,
    "val_pcc": 0.717712,
    "val_ccc": 0.707375,
    "test_rmse": 0.181559,
    "test_mae": 0.141920,
    "test_pcc": 0.844993,
    "test_ccc": 0.835335,
}


def metric_gaps(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "rmse_gap_to_pass": float(metrics["rmse"] - PASS["rmse"]),
        "mae_gap_to_pass": float(metrics["mae"] - PASS["mae"]),
        "pcc_gap_to_pass": float(PASS["pcc"] - metrics["pcc"]),
        "ccc_gap_to_pass": float(PASS["ccc"] - metrics["ccc"]),
    }


def screen_decision(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no_result"
    row = rows[0]
    val_ok = row["val_rmse"] <= SPLIT01_BASELINE["val_rmse"] + 0.005 and row["val_ccc"] >= SPLIT01_BASELINE["val_ccc"] - 0.015
    test_ok = row["test_rmse"] <= SPLIT01_BASELINE["test_rmse"] + 0.005 and row["test_ccc"] >= SPLIT01_BASELINE["test_ccc"] - 0.015
    if val_ok and test_ok:
        return "continue_to_all_splits"
    return "reject_after_split01_screen"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-limit", type=int, default=1)
    parser.add_argument("--static-head-weight", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial_suffix = str(args.static_head_weight).replace(".", "p")
    config = StaticAuxConfig(
        trial_id=f"weak_static_aux_no_output_sw{trial_suffix}",
        static_head_weight=args.static_head_weight,
        static_frame_consistency_weight=0.0,
        blend_static_weight=0.0,
        epochs=args.epochs,
        patience=args.patience,
    )
    static_map = load_static_valence(STATIC_ANNOTATIONS)
    device = resolve_device(args.device)
    train_args = SimpleNamespace(
        out_dir=args.out_dir,
        mert_dir=args.mert_dir,
        label_dir=args.label_dir,
        librosa_dataset=args.librosa_dataset,
    )

    rows = []
    split_dirs = sorted(BASE_SPLIT_DIR.glob("split_*"))[: args.split_limit]
    for split_dir in split_dirs:
        split_id, seed = split_seed(split_dir)
        rows.append(train_one_split(split_dir, split_id, seed, train_args, device, config, static_map))

    trial_dir = args.out_dir / config.trial_id
    aggregate = {**aggregate_metrics(rows, "val"), **aggregate_metrics(rows, "test")} if rows else {}
    test_mean = {k: aggregate[f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]} if rows else {}
    summary = {
        "created_at": "2026-06-28",
        "purpose": "screen weak static Valence auxiliary supervision without using static output at inference",
        "universal_prediction_note": "static head is training-only; final frame-level output remains audio-only and can be used for arbitrary songs",
        "current_best_reference": CURRENT_BEST,
        "split01_baseline_reference": SPLIT01_BASELINE,
        "config": asdict(config),
        "static_annotation_source": str(STATIC_ANNOTATIONS),
        "num_splits": len(rows),
        "aggregate": aggregate,
        "per_split": rows,
        "mean_test_gap_rows": threshold_rows(test_mean) if rows else [],
        "mean_test_gap_to_pass": metric_gaps(test_mean) if rows else {},
        "screen_decision": screen_decision(rows),
    }
    (trial_dir / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
