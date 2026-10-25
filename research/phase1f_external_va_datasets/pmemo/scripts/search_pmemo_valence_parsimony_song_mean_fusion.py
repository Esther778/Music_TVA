#!/usr/bin/env python3
"""Song-level Valence mean correction for the current parsimony best model."""

from __future__ import annotations

import json

import pandas as pd

import search_pmemo_valence_song_mean_fusion as song_mean
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_per_split_direct_multilayer_parsimony_20260629"
OUT_DIR = ROOT / "outputs" / "valence_parsimony_song_mean_fusion_20260629"
CURRENT_BEST = {
    "model_id": "pmemo_valence_per_split_direct_multilayer_parsimony_20260629",
    "rmse": 0.19130205118779167,
    "mae": 0.14893115017555528,
    "pcc": 0.8059953934483183,
    "ccc": 0.7914302668500032,
}


def main() -> None:
    song_mean.INPUT_DIR = INPUT_DIR
    song_mean.OUT_DIR = OUT_DIR
    song_mean.CURRENT_BEST = CURRENT_BEST
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_results = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, result_df = song_mean.evaluate_split(split_dir)
        rows.append(summary)
        all_results.append(result_df)
    pd.concat(all_results, ignore_index=True).to_csv(OUT_DIR / "all_song_mean_fusion_results.csv", index=False)
    aggregate = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_parsimony_song_mean_fusion_20260629",
        "input_model": CURRENT_BEST,
        "purpose": "Validation-selected song-level Valence mean correction on top of the current parsimony best model.",
        "selection_rule": "per split, select by validation composite pass-gap score; baseline/no-change is included as a candidate; test labels not used.",
        "num_splits": len(rows),
        "per_split": rows,
        **song_mean.aggregate_metrics(rows, "baseline"),
        **song_mean.aggregate_metrics(rows, "selected"),
    }
    selected_mean = {metric: aggregate[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: aggregate[f"baseline_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    aggregate["delta_vs_input_test_mean"] = {
        metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
