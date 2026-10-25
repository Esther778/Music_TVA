#!/usr/bin/env python3
"""Policy-level selection for current-best Valence song-mean candidates.

The underlying candidate search can overfit small validation splits. This
script re-selects from the saved candidate table with explicit guards:

- strict_no_degrade: only select a candidate if validation RMSE/MAE do not
  degrade and CCC improves.
- ccc_priority_tiny_cost: allow tiny validation RMSE/MAE cost if CCC improves
  clearly, intended as a secondary diagnostic candidate, not an automatic best.

Test metrics are reported only after validation-policy selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


SOURCE_DIR = ROOT / "outputs" / "valence_current_best_song_mean_fusion_20260707"
SOURCE_CANDIDATES = SOURCE_DIR / "all_song_mean_fusion_results.csv"
OUT_DIR = ROOT / "outputs" / "valence_current_best_song_mean_guarded_policy_20260707"

CURRENT_BEST = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}


def mean_metrics(rows: list[dict[str, Any]], prefix: str = "selected") -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([float(r[f"test_{metric}"]) for r in rows], dtype=np.float64)
        out[f"{prefix}_test_{metric}_mean"] = float(vals.mean())
        out[f"{prefix}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def select_for_split(df: pd.DataFrame, policy: str) -> dict[str, Any]:
    base = df[df["model_id"].eq("no_change")].iloc[0]
    candidates = df.copy()
    if policy == "strict_no_degrade":
        mask = (
            (candidates["val_rmse"] <= base["val_rmse"] + 1e-12)
            & (candidates["val_mae"] <= base["val_mae"] + 1e-12)
            & (candidates["val_ccc"] >= base["val_ccc"] + 0.001)
        )
        sort_cols = ["val_ccc", "val_rmse", "val_mae", "val_pcc"]
        ascending = [False, True, True, False]
    elif policy == "ccc_priority_tiny_cost":
        mask = (
            (candidates["val_rmse"] <= base["val_rmse"] + 0.0010)
            & (candidates["val_mae"] <= base["val_mae"] + 0.0010)
            & (candidates["val_ccc"] >= base["val_ccc"] + 0.004)
        )
        sort_cols = ["val_ccc", "val_rmse", "val_mae", "val_pcc"]
        ascending = [False, True, True, False]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    valid = candidates[mask]
    if valid.empty:
        selected = base.copy()
        reason = "identity retained; no candidate passed policy guard"
    else:
        selected = valid.sort_values(sort_cols, ascending=ascending).iloc[0]
        reason = "candidate selected by validation policy guard"

    out = selected.to_dict()
    out["selection_reason"] = reason
    out["baseline_val_rmse"] = float(base["val_rmse"])
    out["baseline_val_mae"] = float(base["val_mae"])
    out["baseline_val_pcc"] = float(base["val_pcc"])
    out["baseline_val_ccc"] = float(base["val_ccc"])
    out["baseline_test_rmse"] = float(base["test_rmse"])
    out["baseline_test_mae"] = float(base["test_mae"])
    out["baseline_test_pcc"] = float(base["test_pcc"])
    out["baseline_test_ccc"] = float(base["test_ccc"])
    out["delta_test_rmse"] = float(out["test_rmse"] - base["test_rmse"])
    out["delta_test_mae"] = float(out["test_mae"] - base["test_mae"])
    out["delta_test_pcc"] = float(out["test_pcc"] - base["test_pcc"])
    out["delta_test_ccc"] = float(out["test_ccc"] - base["test_ccc"])
    return out


def main() -> None:
    df = pd.read_csv(SOURCE_CANDIDATES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_current_best_song_mean_guarded_policy_20260707",
        "source_candidates": str(SOURCE_CANDIDATES.relative_to(ROOT)),
        "current_best_reference": CURRENT_BEST,
        "policies": {},
    }
    for policy in ["strict_no_degrade", "ccc_priority_tiny_cost"]:
        rows = [select_for_split(g, policy) for _, g in df.groupby("split_name", sort=True)]
        result = {
            "policy": policy,
            "selection_rule": (
                "strict: val RMSE/MAE no degradation and CCC +0.001"
                if policy == "strict_no_degrade"
                else "CCC priority: val RMSE/MAE <= +0.001 and CCC +0.004"
            ),
            "per_split": rows,
            **mean_metrics(rows),
        }
        selected_mean = {metric: result[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
        result["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
        result["delta_vs_current_best"] = {
            metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        }
        summaries["policies"][policy] = result
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{policy}_selected_rows.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
