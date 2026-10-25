#!/usr/bin/env python3
"""Refined prediction-only Valence rescaling search.

This refines the positive coarse result from
search_pmemo_valence_prediction_only_rescale.py. The coarse run selected
std_gain=0.8 on all splits, so this script searches a narrower mean_gain range
and a lower std_gain range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_prediction_only_rescale import (
    CURRENT_BEST,
    INPUT_DIR,
    apply_rescale,
    eval_df,
    metrics,
    prediction_arrays,
    selection_score,
    threshold_rows,
    transform_prediction,
)
from train_pmemo_mert_valence_baseline import ROOT


OUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_refine_20260628"


def evaluate_split(split_dir: Path) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    arrays = {split: prediction_arrays(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    for mean_gain in np.round(np.arange(0.85, 1.151, 0.025), 3):
        for std_gain in np.round(np.arange(0.50, 1.001, 0.025), 3):
            for blend in np.round(np.arange(0.70, 1.001, 0.05), 2):
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    pred, target, song_mean = arrays[split]
                    corrected_pred = transform_prediction(pred, song_mean, float(mean_gain), float(std_gain), float(blend), None)
                    split_metrics[split] = metrics(corrected_pred, target)
                rows.append(
                    {
                        "split_name": split_dir.name,
                        "mean_gain": float(mean_gain),
                        "std_gain": float(std_gain),
                        "blend": float(blend),
                        "clip": "none",
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        apply_rescale(
            frames[split],
            float(selected["mean_gain"]),
            float(selected["std_gain"]),
            float(selected["blend"]),
            None,
        ).to_csv(out_split / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_prediction_only_rescale_refine_20260628",
        "input_model": CURRENT_BEST,
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            vals = np.asarray(
                [
                    r["selected"][f"test_{metric}"] if variant == "selected" else r["baseline"]["test"][metric]
                    for r in rows
                ],
                dtype=np.float64,
            )
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_rescale_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
