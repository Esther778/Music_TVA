#!/usr/bin/env python3
"""Blend RMSE-primary Ridge output with CCC-guarded smoothed Ridge output."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


RMSE_DIR = ROOT / "outputs" / "valence_latest_ridge_song_mean_fast_20260709"
CCC_DIR = ROOT / "outputs" / "valence_latest_ridge_ccc_guard_smoothing_20260709"
OUT_DIR = ROOT / "outputs" / "valence_ridge_rmse_ccc_blend_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
REFERENCE = {
    "model_id": "pmemo_valence_latest_ridge_ccc_guard_smoothing_20260709",
    "rmse": 0.187623279893854,
    "mae": 0.14601650974911215,
    "pcc": 0.8141844446811181,
    "ccc": 0.8002325331205622,
}


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column: {list(df.columns)}")


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def load_pair(split: str, data_split: str) -> pd.DataFrame:
    a = pd.read_csv(RMSE_DIR / split / f"{data_split}_predictions.csv")
    b = pd.read_csv(CCC_DIR / split / f"{data_split}_predictions.csv")
    left = a[["musicId", "time_sec", "target_valence", pred_col(a)]].copy()
    left.columns = ["musicId", "time_sec", "target_valence", "rmse_pred"]
    right = b[["musicId", "time_sec", pred_col(b)]].copy()
    right.columns = ["musicId", "time_sec", "ccc_pred"]
    return left.merge(right, on=["musicId", "time_sec"], how="inner")


def apply_blend(df: pd.DataFrame, rmse_weight: float) -> pd.DataFrame:
    out = df.copy()
    out["pred_valence"] = np.clip(
        float(rmse_weight) * out["rmse_pred"].to_numpy(dtype=np.float64)
        + (1.0 - float(rmse_weight)) * out["ccc_pred"].to_numpy(dtype=np.float64),
        -1.0,
        1.0,
    )
    out["rmse_weight"] = float(rmse_weight)
    return out


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.10 * max(0.0, 0.80 - m["ccc"]) + 0.005 * m["mae"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = {(s, d): load_pair(s, d) for s in SPLITS for d in DATA_SPLITS}
    candidates = []
    cache: dict[float, dict[tuple[str, str], pd.DataFrame]] = {}
    for rmse_weight in np.linspace(0.0, 1.0, 41):
        rows = []
        frames = {}
        for split in SPLITS:
            row: dict[str, Any] = {"split_name": split}
            for data_split in DATA_SPLITS:
                df = apply_blend(base[(split, data_split)], float(rmse_weight))
                frames[(split, data_split)] = df
                row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
            rows.append(row)
        val_m = macro(rows, "val")
        test_m = macro(rows, "test")
        candidates.append(
            {
                "rmse_weight": float(rmse_weight),
                **{f"val_{m}": v for m, v in val_m.items()},
                **{f"test_{m}": v for m, v in test_m.items()},
                "val_score": score(val_m),
                "passes_guard": (
                    val_m["rmse"] <= 0.20025
                    and val_m["mae"] <= 0.15770
                    and val_m["ccc"] >= 0.77768
                ),
            }
        )
        cache[float(rmse_weight)] = frames
    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_blend_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = result.sort_values(["val_score", "val_rmse"], ascending=[True, True]).iloc[0].to_dict()
    else:
        selected = guarded.sort_values(["val_score", "val_rmse"], ascending=[True, True]).iloc[0].to_dict()
    weight = float(selected["rmse_weight"])
    per_split = []
    for split in SPLITS:
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"split_name": split}
        for data_split in DATA_SPLITS:
            df = cache[weight][(split, data_split)]
            df.to_csv(split_dir / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_ridge_rmse_ccc_blend_20260709",
        "input_models": {
            "rmse_primary": str(RMSE_DIR.relative_to(ROOT)),
            "ccc_guard_smoothing": str(CCC_DIR.relative_to(ROOT)),
        },
        "reference": REFERENCE,
        "selection_rule": "blend weight selected on validation RMSE+CCC guard; test labels unused",
        "selected_config": selected,
        "selected_test": selected_test,
        "delta_vs_reference": {m: float(selected_test[m] - REFERENCE[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": per_split,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
