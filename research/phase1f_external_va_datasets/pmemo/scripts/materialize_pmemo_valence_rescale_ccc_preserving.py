#!/usr/bin/env python3
"""Materialize the CCC-preserving prediction-only Valence rescale candidate."""

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
    threshold_rows,
)
from train_pmemo_mert_valence_baseline import ROOT


CANDIDATE_TABLE = ROOT / "outputs" / "valence_prediction_only_rescale_refine_20260628" / "all_rescale_results.csv"
OUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
CCC_EPS = 0.001


def selection_score(row: pd.Series) -> float:
    return (
        2.20 * max(0.0, float(row["val_rmse"]) - 0.18) / 0.18
        + 2.20 * max(0.0, float(row["val_mae"]) - 0.15) / 0.15
        + 1.25 * max(0.0, 0.80 - float(row["val_pcc"])) / 0.80
        + 2.50 * max(0.0, 0.80 - float(row["val_ccc"])) / 0.80
        + 0.05 * float(row["val_rmse"])
    )


def materialize_split(split_name: str, candidates: pd.DataFrame) -> dict:
    split_dir = INPUT_DIR / split_name
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    allowed = candidates[candidates["val_ccc"] >= baseline["val"]["ccc"] - CCC_EPS].copy()
    if allowed.empty:
        selected = {
            "split_name": split_name,
            "source": "baseline",
            "mean_gain": 1.0,
            "std_gain": 1.0,
            "blend": 0.0,
            "clip": "none",
            **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
            "val_selection_score": None,
        }
    else:
        allowed["selection_score"] = allowed.apply(selection_score, axis=1)
        selected = allowed.sort_values(["selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False]).iloc[0].to_dict()
        selected["source"] = "rescale"

    out_split = OUT_DIR / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        apply_rescale(
            frames[split],
            float(selected["mean_gain"]),
            float(selected["std_gain"]),
            float(selected["blend"]),
            None,
        ).to_csv(out_split / f"{split}_predictions.csv", index=False)
    return {
        "split_name": split_name,
        "ccc_eps": CCC_EPS,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }


def aggregate(rows: list[dict]) -> dict:
    out = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
        "input_model": CURRENT_BEST,
        "selection_rule": f"select rescale candidate only if val_ccc >= baseline_val_ccc - {CCC_EPS}",
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
    candidates = pd.read_csv(CANDIDATE_TABLE)
    rows = []
    for split_name, split_candidates in candidates.groupby("split_name"):
        rows.append(materialize_split(str(split_name), split_candidates))
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([r["selected"] for r in rows]).to_csv(OUT_DIR / "selected_by_split.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
