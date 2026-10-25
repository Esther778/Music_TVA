#!/usr/bin/env python3
"""Three-fold Arousal ensemble screen with MERT+harmony diversity.

This keeps the strict train+val development-fold protocol. It searches only a
small neighborhood around the current four-way OpenSMILE ensemble and temporal
shift settings, then reports fold-local validation metrics. Holdout/test data
are not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "arousal_mert_harmonic_fiveway_shift_20260709"

FOLD_CONFIGS = {
    "fold_01": {
        "base_weights": {"h96": 0.3, "h64": 0.4, "h128": 0.0, "ridge": 0.3},
        "shift_frames": 13,
        "windows": [91, 111, 131],
    },
    "fold_02": {
        "base_weights": {"h96": 0.5, "h64": 0.1, "h128": 0.0, "ridge": 0.4},
        "shift_frames": 7,
        "windows": [21, 31, 51, 71],
    },
    "fold_03": {
        "base_weights": {"h96": 0.0, "h64": 0.3, "h128": 0.4, "ridge": 0.3},
        "shift_frames": 16,
        "windows": [51, 61, 71, 91],
    },
}

CURRENT_BEST = {
    "model_id": "pmemo_arousal_amplitude_shift_20260709",
    "mean": {
        "rmse": 0.1986657668875332,
        "mae": 0.15556935128109076,
        "pcc": 0.8251076819873374,
        "ccc": 0.8100892405045655,
    },
}


def input_paths(fold: str) -> dict[str, Path]:
    return {
        "h96": ROOT / "outputs" / "arousal_devfolds_best_20260709" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / "val_predictions.csv",
        "h64": ROOT / "outputs" / "arousal_devfolds_smallreg_20260709" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003" / "val_predictions.csv",
        "h128": ROOT / "outputs" / "arousal_devfolds_h128_20260709" / fold / "a_dev_h128_l1_do020_lr6e4_mse" / "val_predictions.csv",
        "ridge": ROOT / "outputs" / "opensmile_ridge_devfolds_20260709" / "arousal" / fold / "val_predictions.csv",
        "mert_harm": ROOT / "outputs" / "arousal_mert_harmonic_devfolds_20260709" / fold / "val_predictions.csv",
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


def scale_amplitude_by_song(df: pd.DataFrame, values: np.ndarray, amp: float) -> np.ndarray:
    if abs(amp - 1.0) < 1e-12:
        return values.copy()
    tmp = df[["musicId"]].copy()
    tmp["value"] = values
    song_mean = tmp.groupby("musicId", sort=False)["value"].transform("mean").to_numpy(dtype=np.float64)
    return song_mean + amp * (values - song_mean)


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


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.008 * max(0.0, 0.80 - m["ccc"])
    )


def load_fold(fold: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    paths = input_paths(fold)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    base = frames["h96"][["musicId", "time_sec", "target_arousal"]].copy()
    for name, frame in frames.items():
        pcol = pred_col(frame)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    preds = {name: base[f"pred_{name}"].to_numpy(dtype=np.float64) for name in frames}
    return base[["musicId", "time_sec", "target_arousal"]].copy(), preds


def candidate_weights(fold: str) -> list[dict[str, float]]:
    cfg = FOLD_CONFIGS[fold]
    base = cfg["base_weights"]
    rows: list[dict[str, float]] = []
    for mert_weight in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        raw_base = np.array([base["h96"], base["h64"], base["h128"], base["ridge"]], dtype=np.float64)
        rows.append({
            "h96": raw_base[0] * (1.0 - mert_weight),
            "h64": raw_base[1] * (1.0 - mert_weight),
            "h128": raw_base[2] * (1.0 - mert_weight),
            "ridge": raw_base[3] * (1.0 - mert_weight),
            "mert_harm": mert_weight,
        })
    out = []
    seen = set()
    for row in rows:
        key = tuple(round(row[name], 6) for name in ["h96", "h64", "h128", "ridge", "mert_harm"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def search_fold(fold: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base, preds = load_fold(fold)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    shifts = sorted(set([max(0, FOLD_CONFIGS[fold]["shift_frames"] + d) for d in [-2, 0, 2]]))
    for weights in candidate_weights(fold):
        raw = sum(weights[name] * preds[name] for name in ["h96", "h64", "h128", "ridge", "mert_harm"])
        for shift_frames in shifts:
            shifted = shift_by_song(base, raw, shift_frames)
            for amp in [0.7, 0.85, 1.0, 1.15, 1.3]:
                amplified = scale_amplitude_by_song(base, shifted, amp)
                for window in FOLD_CONFIGS[fold]["windows"]:
                    smoothed = smooth_by_song(base, amplified, window)
                    for method in ["bias", "linear", "mean_std"]:
                        final, params = calibrate(smoothed, target, method)
                        final = np.clip(final, -1.0, 1.0)
                        m = metrics(final, target)
                        row = {
                            "fold": fold,
                            "shift_frames": shift_frames,
                            "shift_sec": shift_frames * 0.5,
                            "amp": amp,
                            "window": window,
                            "method": method,
                            "params": json.dumps(params, ensure_ascii=True),
                            **{f"w_{name}": float(weights[name]) for name in ["h96", "h64", "h128", "ridge", "mert_harm"]},
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
    pred_out = base.copy()
    pred_out["pred_arousal"] = best_pred
    return best, pd.DataFrame(rows), pred_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        best, rows, pred_out = search_fold(fold)
        rows.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(OUT_DIR / f"{fold}_candidates.csv", index=False)
        pred_out.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({k: v for k, v in best.items() if k != "params"})
    mean = {
        metric: float(np.mean([float(row[metric]) for row in per_fold]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Arousal five-way ensemble with MERT+harmony, local fold search; dev folds only",
        "selection_boundary": "Development-fold validation only; holdout/test excluded",
        "current_best_reference": CURRENT_BEST,
        "per_fold": per_fold,
        "mean": mean,
        "gap_rows": threshold_rows(mean),
        "delta_vs_current_best_mean": {
            k: float(mean[k] - CURRENT_BEST["mean"][k])
            for k in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per_fold).to_csv(OUT_DIR / "comparison.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
