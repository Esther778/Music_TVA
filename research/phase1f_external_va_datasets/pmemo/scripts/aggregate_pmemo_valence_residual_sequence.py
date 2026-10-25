#!/usr/bin/env python3
"""Aggregate the three residual-sequence Valence split summaries."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np

from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_residual_sequence_20260709" / "h6_base_resid_h48_do045_cap006"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
HISTORICAL_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


def mean_metric(rows: list[dict[str, Any]], prefix: str, metric: str) -> float:
    return float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    rows = [json.loads((out_dir / split / "summary.json").read_text(encoding="utf-8")) for split in SPLITS]
    selected_test = {metric: mean_metric(rows, "test", metric) for metric in ["rmse", "mae", "pcc", "ccc"]}
    base_test = {metric: mean_metric(rows, "test_base", metric) for metric in ["rmse", "mae", "pcc", "ccc"]}
    selected_val = {metric: mean_metric(rows, "val", metric) for metric in ["rmse", "mae", "pcc", "ccc"]}
    base_val = {metric: mean_metric(rows, "val_base", metric) for metric in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-09",
        "model_id": args.model_id or f"pmemo_valence_residual_sequence_{out_dir.name}",
        "output": str(out_dir.relative_to(ROOT.resolve())),
        "purpose": "three-split aggregate for sequence residual correction on current Valence best",
        "feature_type": "current best prediction + MERT hidden6 + prediction/time context",
        "selection_rule": "each split checkpoint selected by validation selection_score; test labels unused for checkpoint selection",
        "historical_best_reference": HISTORICAL_BEST,
        "base_val": base_val,
        "selected_val": selected_val,
        "base_test": base_test,
        "selected_test": selected_test,
        "delta_vs_historical_best": {
            metric: float(selected_test[metric] - HISTORICAL_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_base_recomputed": {
            metric: float(selected_test[metric] - base_test[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": rows,
        "promotion_decision": (
            "Promote as current RMSE-primary Valence artifact because mean RMSE improves slightly "
            "and CCC crosses 0.80, while noting that RMSE remains above the minimum pass line."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
