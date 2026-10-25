#!/usr/bin/env python3
"""Static-target song-mean correction on the current Valence global policy best."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import search_pmemo_valence_static_target_song_mean as static_mean
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_song_mean_global_policy_20260707"
OUT_DIR = ROOT / "outputs" / "valence_current_global_static_target_20260707"
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_song_mean_global_policy_20260707",
    "rmse": 0.19078885880411342,
    "mae": 0.1486967345561368,
    "pcc": 0.8074956914596779,
    "ccc": 0.7951366852830063,
}


def main() -> None:
    static_mean.INPUT_DIR = INPUT_DIR
    static_mean.CURRENT_BEST = CURRENT_BEST
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    results = []
    for split_dir in sorted(INPUT_DIR.glob("split_*_seed_*")):
        summary, result_df = static_mean.evaluate_split(split_dir, OUT_DIR, eps=0.0)
        rows.append(summary)
        results.append(result_df)
    pd.concat(results, ignore_index=True).to_csv(OUT_DIR / "all_static_target_song_mean_results.csv", index=False)
    baseline_mean = static_mean.aggregate(rows, "baseline")
    selected_mean = static_mean.aggregate(rows, "selected")
    selected_metrics = {
        "rmse": selected_mean["selected_test_rmse_mean"],
        "mae": selected_mean["selected_test_mae_mean"],
        "pcc": selected_mean["selected_test_pcc_mean"],
        "ccc": selected_mean["selected_test_ccc_mean"],
    }
    summary = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_current_global_static_target_20260707",
        "input_model": CURRENT_BEST,
        "target_source": "PMEmo static Valence, converted from 0-1 to -1..1",
        "selection_rule": "Per split, candidate must not degrade validation RMSE/MAE/PCC/CCC; then select by validation score. Test labels are not used.",
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_input_test_mean": {
            metric: float(selected_metrics[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
