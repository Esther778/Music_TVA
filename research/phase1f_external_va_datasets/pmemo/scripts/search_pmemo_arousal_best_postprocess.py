#!/usr/bin/env python3
"""Validation-selected postprocess for the best PMEmo Arousal model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "opensmile_arousal_trials" / "a_only_h96_l1_do015_mse"
OUT_DIR = ROOT / "outputs" / "opensmile_arousal_best_postprocess_20260629"

CURRENT_BEST = {
    "model_id": "a_only_h96_l1_do015_mse",
    "rmse": 0.18095970153808594,
    "mae": 0.14009205996990204,
    "pcc": 0.8526338558393907,
    "ccc": 0.8481215815055151,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


def moving_average_by_song(df: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    tmp = df.loc[:, ["musicId", "time_sec"]].copy()
    tmp["pred"] = values
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
    pred = val_df["pred_arousal"].to_numpy(dtype=np.float64)
    target = val_df["target_arousal"].to_numpy(dtype=np.float64)
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


def apply(df: pd.DataFrame, params: dict[str, Any], strength: float, smooth_window: int, clip: float | None) -> pd.DataFrame:
    out = df.copy()
    pred = out["pred_arousal"].to_numpy(dtype=np.float64)
    method = str(params["method"])
    if method in {"identity", "bias", "linear"}:
        cal = float(params["a"]) * pred + float(params["b"])
    elif method == "mean_std":
        cal = (pred - float(params["pm"])) / float(params["ps"]) * float(params["ts"]) + float(params["tm"])
    else:
        raise ValueError(method)
    final = (1.0 - strength) * pred + strength * cal
    final = moving_average_by_song(out, final, smooth_window)
    if clip is not None:
        final = np.clip(final, -float(clip), float(clip))
    out["pred_arousal"] = final
    out["postprocess_method"] = method
    out["postprocess_strength"] = strength
    out["smooth_window"] = smooth_window
    return out


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_arousal"].to_numpy(dtype=np.float64), df["target_arousal"].to_numpy(dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 0.8 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.2 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {split: pd.read_csv(INPUT_DIR / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float, int, str], dict[str, pd.DataFrame]] = {}
    for method in ["identity", "bias", "linear", "mean_std"]:
        params = fit_params(frames["val"], method)
        for strength in [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
            for smooth_window in [1, 3, 5, 7, 9]:
                for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90)]:
                    corrected = {split: apply(frames[split], params, strength, smooth_window, clip) for split in ["train", "val", "test"]}
                    metrics_by_split = {split: eval_df(corrected[split]) for split in ["train", "val", "test"]}
                    row = {
                        "method": method,
                        "strength": strength,
                        "smooth_window": smooth_window,
                        "clip": clip_name,
                        "params": json.dumps(params, ensure_ascii=True),
                        **{f"{split}_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
                    }
                    row["val_selection_score"] = selection_score(metrics_by_split["val"])
                    rows.append(row)
                    cache[(method, float(strength), int(smooth_window), clip_name)] = corrected
    cand = pd.DataFrame(rows)
    base_val = baseline["val"]
    allowed = cand[
        (cand["val_pcc"] >= base_val["pcc"] - 0.001)
        & (cand["val_ccc"] >= base_val["ccc"] - 0.001)
    ].copy()
    if allowed.empty:
        allowed = cand
    selected = allowed.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False]).iloc[0].to_dict()
    selected_key = (str(selected["method"]), float(selected["strength"]), int(selected["smooth_window"]), str(selected["clip"]))
    corrected = cache[selected_key]
    for split, df in corrected.items():
        df.to_csv(OUT_DIR / f"{split}_predictions.csv", index=False)
    test_m = eval_df(corrected["test"])
    val_m = eval_df(corrected["val"])
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_arousal_best_postprocess_20260629",
        "purpose": "validation-selected postprocess for best Arousal model",
        "selection_rule": "parameters selected on validation only; test labels not used",
        "input_model": CURRENT_BEST,
        "baseline": baseline,
        "selected": selected,
        "selected_metrics": {"val": val_m, "test": test_m},
        "test_gap_rows": threshold_rows(test_m),
        "delta_vs_input_model": {k: float(test_m[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    cand.to_csv(OUT_DIR / "all_postprocess_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
