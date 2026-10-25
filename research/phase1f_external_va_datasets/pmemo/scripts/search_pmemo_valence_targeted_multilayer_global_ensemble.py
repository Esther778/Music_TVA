#!/usr/bin/env python3
"""Validation-selected global ensemble of two Valence candidates.

Base candidate:
- pmemo_valence_targeted_rescale_valsafe_20260628

Diversity candidate:
- pmemo_valence_multilayer_global_ensemble_20260629

The script searches one global weight on validation macro metrics and reports
test only after selection. It is intentionally lightweight and does not train a
new model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BASE_DIR = ROOT / "outputs" / "valence_targeted_rescale_valsafe_20260628"
CANDIDATE_DIR = ROOT / "outputs" / "valence_multilayer_global_ensemble_20260629"
OUT_DIR = ROOT / "outputs" / "valence_targeted_multilayer_global_ensemble_20260629"

BASE_REFERENCE = {
    "model_id": "pmemo_valence_targeted_rescale_valsafe_20260628",
    "rmse": 0.19298069016594951,
    "mae": 0.15064551164858095,
    "pcc": 0.8024746777503741,
    "ccc": 0.785846536609028,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(
        df[pred_col(df)].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def selection_score(m: dict[str, float]) -> float:
    return (
        2.2 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.2 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def combine(base: pd.DataFrame, cand: pd.DataFrame, weight_candidate: float) -> pd.DataFrame:
    left = base.copy()
    right = cand[["musicId", "time_sec", pred_col(cand)]].copy()
    right = right.rename(columns={pred_col(cand): "candidate_pred"})
    merged = left.merge(right, on=["musicId", "time_sec"], how="inner")
    if len(merged) != len(left):
        raise ValueError(f"Alignment mismatch: base={len(left)} merged={len(merged)}")
    pcol = pred_col(merged)
    merged[pcol] = (
        (1.0 - weight_candidate) * merged[pcol].to_numpy(dtype=np.float64)
        + weight_candidate * merged["candidate_pred"].to_numpy(dtype=np.float64)
    )
    merged["weight_candidate"] = weight_candidate
    return merged.drop(columns=["candidate_pred"])


def macro(rows: list[dict[str, Any]], split: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        out[metric] = float(np.mean([r[f"{split}_{metric}"] for r in rows]))
    return out


def evaluate_weight(weight: float) -> tuple[dict[str, Any], dict[tuple[str, str], pd.DataFrame]]:
    split_rows = []
    cache = {}
    for split_dir in sorted(BASE_DIR.glob("split_*_seed_*")):
        split_name = split_dir.name
        row = {"split_name": split_name}
        for split in ["train", "val", "test"]:
            base = pd.read_csv(BASE_DIR / split_name / f"{split}_predictions.csv")
            cand = pd.read_csv(CANDIDATE_DIR / split_name / f"{split}_predictions.csv")
            out = combine(base, cand, weight)
            cache[(split_name, split)] = out
            m = eval_df(out)
            row.update({f"{split}_{k}": v for k, v in m.items()})
        split_rows.append(row)
    val_macro = macro(split_rows, "val")
    test_macro = macro(split_rows, "test")
    result = {
        "weight_candidate": weight,
        **{f"val_macro_{k}": v for k, v in val_macro.items()},
        **{f"test_macro_{k}": v for k, v in test_macro.items()},
        "val_selection_score": selection_score(val_macro),
        "per_split": split_rows,
    }
    return result, cache


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    caches = {}
    for weight in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        result, cache = evaluate_weight(float(weight))
        rows.append(result)
        caches[float(weight)] = cache
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "per_split"} for r in rows])
    base_val = table[table["weight_candidate"] == 0.0].iloc[0]
    viable = table[
        (table["val_macro_rmse"] <= base_val["val_macro_rmse"] + 0.0005)
        & (table["val_macro_mae"] <= base_val["val_macro_mae"] + 0.0005)
        & (table["val_macro_pcc"] >= base_val["val_macro_pcc"] - 0.001)
        & (table["val_macro_ccc"] >= base_val["val_macro_ccc"] - 0.001)
    ].copy()
    if viable.empty:
        viable = table.copy()
    selected_row = viable.sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_weight = float(selected_row["weight_candidate"])
    selected_full = next(r for r in rows if float(r["weight_candidate"]) == selected_weight)
    cache = caches[selected_weight]
    for split_name in sorted({k[0] for k in cache}):
        out_split = OUT_DIR / split_name
        out_split.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            cache[(split_name, split)].to_csv(out_split / f"{split}_predictions.csv", index=False)
    table.to_csv(OUT_DIR / "all_weight_candidates.csv", index=False)
    test_metrics = {k: float(selected_row[f"test_macro_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_targeted_multilayer_global_ensemble_20260629",
        "purpose": "one global validation-selected weight between targeted_rescale_valsafe and multilayer_global_ensemble",
        "selection_rule": "single global weight selected on validation macro metrics with a small no-degradation guard against the base candidate; test labels not used",
        "base_model": BASE_REFERENCE,
        "candidate_model": "pmemo_valence_multilayer_global_ensemble_20260629",
        "selected": selected_row,
        "test_macro": {"per_split": selected_full["per_split"], **test_metrics},
        "test_gap_rows": threshold_rows(test_metrics),
        "delta_vs_base": {metric: float(test_metrics[metric] - BASE_REFERENCE[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
