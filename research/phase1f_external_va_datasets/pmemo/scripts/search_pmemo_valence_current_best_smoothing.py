#!/usr/bin/env python3
"""Validation-selected smoothing on the current best Valence curve."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_wide_20260708"
OUT_DIR = ROOT / "outputs" / "valence_current_best_smoothing_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_guardtopccc_temporal_shift_wide_20260708",
    "rmse": 0.1896185697242768,
    "mae": 0.14745982046674147,
    "pcc": 0.8105040088472681,
    "ccc": 0.797385312909266,
}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 2:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) <= 1:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def apply_smoothing(df: pd.DataFrame, method: str, param: float, blend: float) -> pd.DataFrame:
    pieces = []
    for _, g in df.groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        if method == "ma":
            sm = moving_average(pred, int(param))
        elif method == "exp":
            sm = exp_smooth(pred, float(param))
        else:
            sm = pred
        out = g.copy()
        out["base_pred_valence"] = pred
        out["pred_valence"] = np.clip((1.0 - blend) * pred + blend * sm, -1.0, 1.0)
        out["smooth_method"] = method
        out["smooth_param"] = float(param)
        out["smooth_blend"] = float(blend)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv") for s in SPLITS for d in DATA_SPLITS}
    rows = []
    cache: dict[tuple[str, float, float], dict[tuple[str, str], pd.DataFrame]] = {}
    candidates = [("identity", 0.0)]
    candidates += [("ma", float(w)) for w in [3, 5, 7, 9, 11, 15, 21]]
    candidates += [("exp", a) for a in [0.20, 0.35, 0.50, 0.65, 0.80]]
    for method, param in candidates:
        for blend in ([0.0] if method == "identity" else [0.15, 0.30, 0.50, 0.70, 1.0]):
            per_split = []
            smoothed = {}
            for split in SPLITS:
                row = {"split_name": split}
                for data in DATA_SPLITS:
                    df = apply_smoothing(frames[(split, data)], method, param, float(blend))
                    smoothed[(split, data)] = df
                    row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
                per_split.append(row)
            val_m = macro(per_split, "val")
            test_m = macro(per_split, "test")
            rows.append(
                {
                    "method": method,
                    "param": float(param),
                    "blend": float(blend),
                    **{f"val_{m}": v for m, v in val_m.items()},
                    **{f"test_{m}": v for m, v in test_m.items()},
                    "val_score": score(val_m),
                }
            )
            cache[(method, float(param), float(blend))] = smoothed
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_smoothing_candidates.csv", index=False)
    selected = result.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    selected_key = (str(selected["method"]), float(selected["param"]), float(selected["blend"]))
    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split}
        for data in DATA_SPLITS:
            df = cache[selected_key][(split, data)]
            df.to_csv(split_out / f"{data}_predictions.csv", index=False)
            row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_current_best_smoothing_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "smoothing parameters selected by validation macro pass-gap score",
        "selected_config": selected,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
