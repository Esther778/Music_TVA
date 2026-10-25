#!/usr/bin/env python3
"""Fold-03 diagnostic: test whether MERT+harmony helps Arousal ensembles.

This is a development-fold screening script only. It reads fold_03 validation
predictions and compares five-way blends against the current four-way
development reference. It must not be reported as blind/test performance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "arousal_mert_harmonic_ensemble_screen_20260709"
FOLD = "fold_03"

INPUTS = {
    "h96": ROOT
    / "outputs"
    / "arousal_devfolds_best_20260709"
    / FOLD
    / "a_dev_h96_l1_do015_lr8e4_mse"
    / "val_predictions.csv",
    "h64": ROOT
    / "outputs"
    / "arousal_devfolds_smallreg_20260709"
    / FOLD
    / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003"
    / "val_predictions.csv",
    "h128": ROOT
    / "outputs"
    / "arousal_devfolds_h128_20260709"
    / FOLD
    / "a_dev_h128_l1_do020_lr6e4_mse"
    / "val_predictions.csv",
    "ridge": ROOT
    / "outputs"
    / "opensmile_ridge_devfolds_20260709"
    / "arousal"
    / FOLD
    / "val_predictions.csv",
    "mert_harm": ROOT
    / "outputs"
    / "arousal_mert_harmonic_devfolds_20260709"
    / FOLD
    / "val_predictions.csv",
}

CURRENT_REFERENCES = {
    "fourway_fold03": {
        "rmse": 0.2137737514864081,
        "mae": 0.16780796544976187,
        "pcc": 0.8121220840156693,
        "ccc": 0.794848420018425,
    },
    "amplitude_shift_fold03": {
        "rmse": 0.20554045101653848,
        "mae": 0.16046544506865418,
        "pcc": 0.8278055942489234,
        "ccc": 0.8132409802718632,
    },
}


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_arousal", "pred_arousal_raw"]:
        if col in df.columns:
            return col
    raise ValueError(f"No arousal prediction column in {df.columns.tolist()}")


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


def smooth_by_song(df: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    tmp = df[["musicId", "time_sec"]].copy()
    tmp["value"] = values
    tmp["_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["musicId", "time_sec"])
    smoothed = (
        tmp.groupby("musicId", sort=False)["value"]
        .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
        .to_numpy(dtype=np.float64)
    )
    out = np.empty_like(smoothed)
    out[tmp["_order"].to_numpy(dtype=np.int64)] = smoothed
    return out


def calibrate(pred: np.ndarray, target: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "identity":
        return pred.copy(), {"method": method}
    if method == "bias":
        b = float(np.mean(target - pred))
        return pred + b, {"method": method, "b": b}
    if method == "linear":
        a, b = fit_linear(pred, target)
        return a * pred + b, {"method": method, "a": a, "b": b}
    if method == "mean_std":
        pm = float(pred.mean())
        ps = float(pred.std() + 1e-12)
        tm = float(target.mean())
        ts = float(target.std() + 1e-12)
        return (pred - pm) / ps * ts + tm, {
            "method": method,
            "pred_mean": pm,
            "pred_std": ps,
            "target_mean": tm,
            "target_std": ts,
        }
    raise ValueError(method)


def shift_by_song(df: pd.DataFrame, values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0:
        return values.copy()
    tmp = df[["musicId", "time_sec"]].copy()
    tmp["value"] = values
    tmp["_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["musicId", "time_sec"])
    shifted = tmp.groupby("musicId", sort=False)["value"].shift(frames)
    tmp["shifted"] = shifted.fillna(tmp["value"]).to_numpy(dtype=np.float64)
    out = np.empty(len(tmp), dtype=np.float64)
    out[tmp["_order"].to_numpy(dtype=np.int64)] = tmp["shifted"].to_numpy(dtype=np.float64)
    return out


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.01 * max(0.0, 0.80 - m["ccc"])
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(path) for name, path in INPUTS.items()}
    base = frames["h96"][["musicId", "time_sec", "target_arousal"]].copy()
    for name, frame in frames.items():
        pcol = pred_col(frame)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    used_rows = int(len(base))
    reference_rows = int(len(frames["h96"]))
    preds: dict[str, np.ndarray] = {}
    for name in frames:
        preds[name] = base[f"pred_{name}"].to_numpy(dtype=np.float64)

    target = base["target_arousal"].to_numpy(dtype=np.float64)
    base = base[["musicId", "time_sec", "target_arousal"]].copy()
    names = ["h96", "h64", "h128", "ridge", "mert_harm"]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None

    # Local search around the current fold_03 four-way ensemble. The goal is
    # not a large hyperparameter search, only to test whether adding up to 20%
    # MERT+harmony weight is useful.
    base_weights = np.array([0.0, 0.3, 0.4, 0.3, 0.0], dtype=np.float64)
    candidate_weights: list[np.ndarray] = []
    for mert_weight in [0.0, 0.05, 0.10, 0.15, 0.20]:
        non_mert = base_weights[:4] * (1.0 - mert_weight)
        candidate_weights.append(np.r_[non_mert, mert_weight])
        for dh64 in [-0.1, 0.0, 0.1]:
            for dh128 in [-0.1, 0.0, 0.1]:
                for dridge in [-0.1, 0.0, 0.1]:
                    raw4 = np.array([0.0, 0.3 + dh64, 0.4 + dh128, 0.3 + dridge], dtype=np.float64)
                    if np.any(raw4 < 0):
                        continue
                    if raw4.sum() <= 1e-12:
                        continue
                    raw4 = raw4 / raw4.sum() * (1.0 - mert_weight)
                    candidate_weights.append(np.r_[raw4, mert_weight])
    # Deduplicate numerical repeats.
    unique_weights = []
    seen = set()
    for weights in candidate_weights:
        key = tuple(np.round(weights, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_weights.append(weights)
    for weights in unique_weights:
        raw = sum(float(w) * preds[name] for w, name in zip(weights, names))
        for shift_frames in [0, 4, 8, 12, 16]:
            shifted = shift_by_song(base, raw, shift_frames)
            for window in [31, 51, 61, 71, 91, 111]:
                smoothed = smooth_by_song(base, shifted, window)
                for method in ["identity", "bias", "linear", "mean_std"]:
                    calibrated, params = calibrate(smoothed, target, method)
                    final = np.clip(calibrated, -1.0, 1.0)
                    m = metrics(final, target)
                    row = {
                        "fold": FOLD,
                        "shift_frames": shift_frames,
                        "shift_sec": shift_frames * 0.5,
                        "window": window,
                        "method": method,
                        "params": json.dumps(params, ensure_ascii=True),
                        **{f"w_{name}": float(w) for w, name in zip(weights, names)},
                        **m,
                        "score": score(m),
                    }
                    rows.append(row)
                    if best is None or (row["score"], row["rmse"], -row["ccc"]) < (
                        best["score"],
                        best["rmse"],
                        -best["ccc"],
                    ):
                        best = row
                        best_pred = final

    assert best is not None and best_pred is not None
    rows_df = pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False])
    rows_df.to_csv(OUT_DIR / "candidates.csv", index=False)
    pred_out = base.copy()
    pred_out["pred_arousal"] = best_pred
    pred_out.to_csv(OUT_DIR / "fold_03_val_predictions.csv", index=False)
    best_public = {k: v for k, v in best.items() if k != "params"}
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Arousal fold_03 diagnostic screen for adding MERT+harmony to the current OpenSMILE ensemble",
        "selection_boundary": "Development fold validation only; holdout/test excluded; diagnostic, not final model proof",
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "row_alignment": {
            "reference_rows": reference_rows,
            "used_intersection_rows": used_rows,
            "dropped_rows": reference_rows - used_rows,
        },
        "current_references": CURRENT_REFERENCES,
        "selected": best_public,
        "selected_params": json.loads(str(best["params"])),
        "gap_rows": threshold_rows({k: float(best[k]) for k in ["rmse", "mae", "pcc", "ccc"]}),
        "delta_vs_current_shift_fold03": {
            k: float(best[k] - CURRENT_REFERENCES["amplitude_shift_fold03"][k])
            for k in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
