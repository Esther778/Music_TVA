#!/usr/bin/env python3
"""Validation-safe global ensemble of current Valence teacher and multi-layer MERT.

Uses one fixed ensemble weight selected on pooled validation predictions across
the three existing song-level splits, then reports test metrics. This is a
diagnostic for whether multi-layer MERT can be used safely without split-specific
test-driven switching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


CURRENT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
MULTILAYER_DIRS = {
    "split_01_seed_101": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629_split01",
    "split_02_seed_202": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629",
    "split_03_seed_303": ROOT / "outputs" / "valence_multilayer_mert_rescale_screen_20260629_split03",
}
OUT_DIR = ROOT / "outputs" / "valence_multilayer_global_ensemble_20260629"

CURRENT_STRICTISH = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.1953163033728265,
    "mae": 0.15179290177231072,
    "pcc": 0.7977655929469954,
    "ccc": 0.7848342582143512,
}
EXPLORATORY_BEST = {
    "model_id": "pmemo_valence_universal_metric_optimized_exploratory_20260628",
    "rmse": 0.19182218120147795,
    "mae": 0.14919783550199786,
    "pcc": 0.8044833131619238,
    "ccc": 0.7881895323604796,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def read_pair(split_name: str, split: str) -> pd.DataFrame:
    cur = pd.read_csv(CURRENT_DIR / split_name / f"{split}_predictions.csv")
    ml = pd.read_csv(MULTILAYER_DIRS[split_name] / f"{split}_predictions.csv")
    for df in (cur, ml):
        df["musicId"] = df["musicId"].astype(str)
        df["time_key"] = df["time_sec"].round(6)
        df["target_key"] = df[target_col(df)].round(6)
    merged = cur.loc[:, ["musicId", "time_key", "target_key", "time_sec", target_col(cur), pred_col(cur)]].merge(
        ml.loc[:, ["musicId", "time_key", "target_key", pred_col(ml)]],
        on=["musicId", "time_key", "target_key"],
        how="inner",
        suffixes=("_current", "_multilayer"),
    )
    if len(merged) != len(cur) or len(merged) != len(ml):
        raise ValueError(f"Alignment mismatch {split_name} {split}: current={len(cur)} multilayer={len(ml)} merged={len(merged)}")
    return pd.DataFrame(
        {
            "split_name": split_name,
            "musicId": merged["musicId"],
            "time_sec": merged["time_sec"],
            "target_valence": merged[target_col(cur)].to_numpy(dtype=np.float64),
            "pred_current": merged[f"{pred_col(cur)}_current"].to_numpy(dtype=np.float64),
            "pred_multilayer": merged[f"{pred_col(ml)}_multilayer"].to_numpy(dtype=np.float64),
        }
    )


def apply_weight(df: pd.DataFrame, weight_multilayer: float) -> np.ndarray:
    return (1.0 - weight_multilayer) * df["pred_current"].to_numpy(dtype=np.float64) + weight_multilayer * df["pred_multilayer"].to_numpy(dtype=np.float64)


def eval_df(df: pd.DataFrame, weight_multilayer: float) -> dict[str, float]:
    return metrics(apply_weight(df, weight_multilayer), df["target_valence"].to_numpy(dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        2.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.5 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def macro_metrics(frames: dict[str, pd.DataFrame], weight: float, split: str) -> dict[str, float]:
    rows = []
    for split_name, df in frames.items():
        m = eval_df(df, weight)
        rows.append({"split_name": split_name, **m})
    out = {k: float(np.mean([r[k] for r in rows])) for k in ["rmse", "mae", "pcc", "ccc"]}
    out["per_split"] = rows
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        split: {split_name: read_pair(split_name, split) for split_name in MULTILAYER_DIRS}
        for split in ["train", "val", "test"]
    }
    rows = []
    for weight in np.round(np.arange(0.0, 1.001, 0.01), 2):
        val_concat = pd.concat(frames["val"].values(), ignore_index=True)
        val_pooled = eval_df(val_concat, float(weight))
        val_macro = macro_metrics(frames["val"], float(weight), "val")
        test_macro = macro_metrics(frames["test"], float(weight), "test")
        rows.append(
            {
                "weight_multilayer": float(weight),
                **{f"val_pooled_{k}": v for k, v in val_pooled.items()},
                **{f"val_macro_{k}": v for k, v in val_macro.items() if k != "per_split"},
                **{f"test_macro_{k}": v for k, v in test_macro.items() if k != "per_split"},
                "val_selection_score": selection_score(val_macro),
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    )
    selected = candidates.iloc[0].to_dict()
    weight = float(selected["weight_multilayer"])
    for split in ["train", "val", "test"]:
        for split_name, df in frames[split].items():
            out_dir = OUT_DIR / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out = df.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
            out["pred_valence"] = apply_weight(df, weight)
            out["weight_multilayer"] = weight
            out.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    test_macro = macro_metrics(frames["test"], weight, "test")
    val_macro = macro_metrics(frames["val"], weight, "val")
    summary: dict[str, Any] = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_multilayer_global_ensemble_20260629",
        "purpose": "one global validation-selected weight between current strict-ish Valence teacher and rescaled direct-concat multi-layer MERT",
        "selection_rule": "single weight selected by pooled three-split validation macro score; test labels are not used for selection",
        "input_current": str(CURRENT_DIR),
        "input_multilayer_by_split": {k: str(v) for k, v in MULTILAYER_DIRS.items()},
        "current_strictish_reference": CURRENT_STRICTISH,
        "exploratory_best_reference": EXPLORATORY_BEST,
        "selected": selected,
        "val_macro": val_macro,
        "test_macro": test_macro,
        "test_gap_rows": threshold_rows({k: test_macro[k] for k in ["rmse", "mae", "pcc", "ccc"]}),
        "delta_vs_current_strictish": {k: float(test_macro[k] - CURRENT_STRICTISH[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
        "delta_vs_exploratory_best": {k: float(test_macro[k] - EXPLORATORY_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    candidates.to_csv(OUT_DIR / "all_weight_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
