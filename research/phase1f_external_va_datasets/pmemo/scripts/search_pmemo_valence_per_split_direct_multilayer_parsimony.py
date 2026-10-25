#!/usr/bin/env python3
"""Parsimony variant of per-split direct multi-layer MERT ensemble.

When validation scores are effectively tied, choose the smaller direct
multi-layer weight. This is a common anti-overfitting rule: prefer the simpler
model when validation evidence is not meaningfully different.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows
from search_pmemo_valence_per_split_direct_multilayer_ensemble import (
    BASE_DIR,
    CURRENT_BEST,
    DIRECT_MULTILAYER_DIRS,
    PASS,
    combine,
    eval_df,
    selection_score,
)


OUT_DIR = ROOT / "outputs" / "valence_per_split_direct_multilayer_parsimony_20260629"
TIE_TOLERANCE = 1e-4


def evaluate_split(split_name: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    candidate_dir = DIRECT_MULTILAYER_DIRS[split_name]
    frames = {
        split: {
            "base": pd.read_csv(BASE_DIR / split_name / f"{split}_predictions.csv"),
            "candidate": pd.read_csv(candidate_dir / f"{split}_predictions.csv"),
        }
        for split in ["train", "val", "test"]
    }
    base_metrics = {split: eval_df(frames[split]["base"]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float], pd.DataFrame] = {}
    for weight in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        metrics_by_split = {}
        for split in ["train", "val", "test"]:
            out = combine(frames[split]["base"], frames[split]["candidate"], float(weight))
            cache[(split, float(weight))] = out
            metrics_by_split[split] = eval_df(out)
        row = {
            "split_name": split_name,
            "weight_direct_multilayer": float(weight),
            **{f"{split}_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
        }
        row["val_selection_score"] = selection_score(metrics_by_split["val"])
        rows.append(row)
    table = pd.DataFrame(rows)
    base_val = base_metrics["val"]
    viable = table[
        (table["val_rmse"] <= base_val["rmse"] + 0.0008)
        & (table["val_mae"] <= base_val["mae"] + 0.0008)
        & (table["val_pcc"] >= base_val["pcc"] - 0.0015)
        & (table["val_ccc"] >= base_val["ccc"] - 0.0015)
    ].copy()
    if viable.empty:
        viable = table.copy()
    best_score = float(viable["val_selection_score"].min())
    tied = viable[viable["val_selection_score"] <= best_score + TIE_TOLERANCE].copy()
    selected = tied.sort_values(
        ["weight_direct_multilayer", "val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_weight = float(selected["weight_direct_multilayer"])
    selected_frames = {split: cache[(split, selected_weight)] for split in ["train", "val", "test"]}
    summary = {
        "split_name": split_name,
        "candidate_dir": str(candidate_dir.relative_to(ROOT)),
        "tie_tolerance": TIE_TOLERANCE,
        "best_validation_score": best_score,
        "tied_candidate_count": int(len(tied)),
        "base_metrics": base_metrics,
        "selected": selected,
        "delta_vs_base_test": {
            metric: float(selected[f"test_{metric}"] - base_metrics["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, table, selected_frames


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row["selected"][f"test_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    candidate_tables = []
    for split_name in sorted(DIRECT_MULTILAYER_DIRS):
        summary, table, selected_frames = evaluate_split(split_name)
        summaries.append(summary)
        candidate_tables.append(table)
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for split, df in selected_frames.items():
            df.to_csv(split_out / f"{split}_predictions.csv", index=False)
        (split_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(candidate_tables, ignore_index=True).to_csv(OUT_DIR / "all_weight_candidates.csv", index=False)
    test_macro = aggregate(summaries)
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_per_split_direct_multilayer_parsimony_20260629",
        "purpose": "per-split direct multi-layer MERT ensemble with validation-score tie parsimony",
        "selection_rule": "within each CV split, candidates first pass no-degradation guard; if validation selection score is within 1e-4 of the best, choose the smaller direct-multilayer weight; test labels not used",
        "base_model": CURRENT_BEST,
        "candidate_model": "direct MERT hidden_4+6+8 concat with validation-selected rescale from screen outputs",
        "tie_tolerance": TIE_TOLERANCE,
        "per_split": summaries,
        "test_macro": test_macro,
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_base": {metric: float(test_macro[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
