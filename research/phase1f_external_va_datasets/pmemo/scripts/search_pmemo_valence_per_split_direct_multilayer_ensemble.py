#!/usr/bin/env python3
"""Per-split validation-selected ensemble with direct multi-layer MERT Valence.

This formalizes the useful finding that direct hidden_4+6+8 MERT concat helps
the difficult split_02 while hurting other splits. In cross-validation, each
fold may select its own hyperparameters from its validation set; test labels are
not used for selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BASE_DIR = ROOT / "outputs" / "valence_targeted_multilayer_postprocess_20260629"
OUT_DIR = ROOT / "outputs" / "valence_per_split_direct_multilayer_ensemble_20260629"

DIRECT_MULTILAYER_DIRS = {
    "split_01_seed_101": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629_split01",
    "split_02_seed_202": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629",
    "split_03_seed_303": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629_split03",
}

CURRENT_BEST = {
    "model_id": "pmemo_valence_targeted_multilayer_postprocess_20260629",
    "rmse": 0.19270108142080475,
    "mae": 0.15030960055717216,
    "pcc": 0.8029485501074519,
    "ccc": 0.7870809597955878,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        2.2 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.6 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.4 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.04 * m["rmse"]
    )


def combine(base: pd.DataFrame, candidate: pd.DataFrame, weight_candidate: float) -> pd.DataFrame:
    right = candidate[["musicId", "time_sec", "pred_valence"]].rename(columns={"pred_valence": "candidate_pred"})
    merged = base.merge(right, on=["musicId", "time_sec"], how="inner")
    if len(merged) != len(base):
        raise ValueError(f"Alignment mismatch: base={len(base)} merged={len(merged)}")
    merged["pred_valence"] = (
        (1.0 - weight_candidate) * merged["pred_valence"].to_numpy(dtype=np.float64)
        + weight_candidate * merged["candidate_pred"].to_numpy(dtype=np.float64)
    )
    merged["direct_multilayer_weight"] = weight_candidate
    return merged.drop(columns=["candidate_pred"])


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
    selected = viable.sort_values(
        ["val_selection_score", "val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_weight = float(selected["weight_direct_multilayer"])
    selected_frames = {split: cache[(split, selected_weight)] for split in ["train", "val", "test"]}
    summary = {
        "split_name": split_name,
        "candidate_dir": str(candidate_dir.relative_to(ROOT)),
        "base_metrics": base_metrics,
        "selected": selected,
        "delta_vs_base_test": {
            metric: float(selected[f"test_{metric}"] - base_metrics["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, table, selected_frames


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        out[metric] = float(np.mean([row["selected"][f"test_{metric}"] for row in rows]))
    return out


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
        "model_id": "pmemo_valence_per_split_direct_multilayer_ensemble_20260629",
        "purpose": "per-split validation-selected ensemble with direct hidden_4+6+8 MERT concat candidate",
        "selection_rule": "each CV split selects direct-multilayer weight on its validation set with no-degradation guard; test labels not used",
        "base_model": CURRENT_BEST,
        "candidate_model": "direct MERT hidden_4+6+8 concat with validation-selected rescale from screen outputs",
        "per_split": summaries,
        "test_macro": test_macro,
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_base": {metric: float(test_macro[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
