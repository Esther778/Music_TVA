#!/usr/bin/env python3
"""Validation-selected postprocess for targeted+multilayer Valence ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_targeted_multilayer_global_ensemble_20260629"
OUT_DIR = ROOT / "outputs" / "valence_targeted_multilayer_postprocess_20260629"

INPUT_REFERENCE = {
    "model_id": "pmemo_valence_targeted_multilayer_global_ensemble_20260629",
    "rmse": 0.19271916888346244,
    "mae": 0.15031439671721836,
    "pcc": 0.802897399767356,
    "ccc": 0.7870243969577443,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(
        df["pred_valence"].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


def moving_average_by_song(df: pd.DataFrame, pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    tmp = df[["musicId", "time_sec"]].copy()
    tmp["pred"] = pred
    tmp["_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["musicId", "time_sec"])
    smoothed = (
        tmp.groupby("musicId", sort=False)["pred"]
        .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
        .to_numpy(dtype=np.float64)
    )
    out = np.empty_like(smoothed)
    out[tmp["_order"].to_numpy(dtype=np.int64)] = smoothed
    return out


def fit_params(val_df: pd.DataFrame, method: str) -> dict[str, Any]:
    pred = val_df["pred_valence"].to_numpy(dtype=np.float64)
    target = val_df["target_valence"].to_numpy(dtype=np.float64)
    if method == "identity":
        return {"method": method, "a": 1.0, "b": 0.0, "pm": 0.0, "ps": 1.0, "tm": 0.0, "ts": 1.0}
    if method == "bias":
        return {"method": method, "a": 1.0, "b": float(np.mean(target - pred)), "pm": 0.0, "ps": 1.0, "tm": 0.0, "ts": 1.0}
    if method == "linear":
        a, b = fit_linear(pred, target)
        return {"method": method, "a": a, "b": b, "pm": 0.0, "ps": 1.0, "tm": 0.0, "ts": 1.0}
    if method == "mean_std":
        return {
            "method": method,
            "a": 1.0,
            "b": 0.0,
            "pm": float(pred.mean()),
            "ps": float(pred.std() + 1e-12),
            "tm": float(target.mean()),
            "ts": float(target.std() + 1e-12),
        }
    raise ValueError(method)


def apply(df: pd.DataFrame, params: dict[str, Any], strength: float, smooth_window: int) -> pd.DataFrame:
    out = df.copy()
    pred = out["pred_valence"].to_numpy(dtype=np.float64)
    method = str(params["method"])
    if method in {"identity", "bias", "linear"}:
        cal = float(params["a"]) * pred + float(params["b"])
    elif method == "mean_std":
        cal = (pred - float(params["pm"])) / float(params["ps"]) * float(params["ts"]) + float(params["tm"])
    else:
        raise ValueError(method)
    final = (1.0 - strength) * pred + strength * cal
    final = moving_average_by_song(out, final, smooth_window)
    out["pred_valence"] = np.clip(final, -1.0, 1.0)
    out["postprocess_method"] = method
    out["postprocess_strength"] = strength
    out["smooth_window"] = smooth_window
    return out


def macro(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    rows = [eval_df(df) for df in frames.values()]
    return {k: float(np.mean([m[k] for m in rows])) for k in ["rmse", "mae", "pcc", "ccc"]}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 3.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.04 * m["rmse"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_names = sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir())
    frames = {
        split: {name: pd.read_csv(INPUT_DIR / name / f"{split}_predictions.csv") for name in split_names}
        for split in ["train", "val", "test"]
    }
    val_pooled = pd.concat(frames["val"].values(), ignore_index=True)
    baseline = {split: macro(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float, int], dict[str, dict[str, pd.DataFrame]]] = {}
    for method in ["identity", "bias", "linear", "mean_std"]:
        params = fit_params(val_pooled, method)
        for strength in [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
            for window in [1, 3, 5, 7, 9, 11]:
                corrected = {
                    split: {name: apply(df, params, strength, window) for name, df in frames[split].items()}
                    for split in ["train", "val", "test"]
                }
                metrics_by_split = {split: macro(corrected[split]) for split in ["train", "val", "test"]}
                row = {
                    "method": method,
                    "strength": float(strength),
                    "smooth_window": int(window),
                    "params": json.dumps(params, ensure_ascii=True),
                    **{f"{split}_macro_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
                }
                row["val_selection_score"] = selection_score(metrics_by_split["val"])
                rows.append(row)
                cache[(method, float(strength), int(window))] = corrected
    cand = pd.DataFrame(rows)
    base_val = baseline["val"]
    allowed = cand[
        (cand["val_macro_rmse"] <= base_val["rmse"] + 0.0005)
        & (cand["val_macro_mae"] <= base_val["mae"] + 0.0005)
        & (cand["val_macro_pcc"] >= base_val["pcc"] - 0.001)
        & (cand["val_macro_ccc"] >= base_val["ccc"] - 0.001)
    ].copy()
    if allowed.empty:
        allowed = cand.copy()
    selected = allowed.sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    key = (str(selected["method"]), float(selected["strength"]), int(selected["smooth_window"]))
    corrected = cache[key]
    for split in ["train", "val", "test"]:
        for name, df in corrected[split].items():
            out_dir = OUT_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    cand.to_csv(OUT_DIR / "all_postprocess_candidates.csv", index=False)
    test_macro = macro(corrected["test"])
    val_macro = macro(corrected["val"])
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_targeted_multilayer_postprocess_20260629",
        "purpose": "validation-selected calibration/smoothing on targeted+multilayer Valence ensemble",
        "selection_rule": "postprocess selected on validation macro metrics with no-degradation guard; test labels not used",
        "input_model": INPUT_REFERENCE,
        "baseline_macro": baseline,
        "selected": selected,
        "selected_macro": {"val": val_macro, "test": test_macro},
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_input": {k: float(test_macro[k] - INPUT_REFERENCE[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
