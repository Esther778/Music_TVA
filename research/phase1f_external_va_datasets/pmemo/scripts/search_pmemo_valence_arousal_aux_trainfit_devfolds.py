#!/usr/bin/env python3
"""Valence train-fitted stacker using Arousal predictions as auxiliary cues.

The calibrator is fit on each development fold's training split and evaluated
on that fold's validation split. Validation labels are used only to select among
a small set of postprocess settings inside the development protocol. Holdout
and test data are not read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_arousal_aux_trainfit_devfolds_20260709"

CURRENT_BEST = {
    "model_id": "pmemo_valence_shift_amplitude_narrow_20260709",
    "mean": {
        "rmse": 0.18679505969129037,
        "mae": 0.14588564855221595,
        "pcc": 0.8158044002446467,
        "ccc": 0.799098856653154,
    },
    "per_fold": {
        "fold_01": {"rmse": 0.17716258845876479, "mae": 0.1336279535593356, "pcc": 0.8333180957378212, "ccc": 0.8196544405274452},
        "fold_02": {"rmse": 0.1784361439296689, "mae": 0.14288075518899013, "pcc": 0.8155826817183397, "ccc": 0.7989251177694021},
        "fold_03": {"rmse": 0.20478644668543747, "mae": 0.16114823690832206, "pcc": 0.798512423277779, "ccc": 0.7787170116626149},
    },
}


def valence_path(fold: str, split: str) -> Path:
    return ROOT / "outputs" / "valence_devfolds_bestlike_20260709" / fold / "v_dev_best_like_h80_do030_lr4e4" / f"{split}_predictions.csv"


def arousal_paths(fold: str, split: str) -> dict[str, Path]:
    return {
        "a_h96": ROOT / "outputs" / "arousal_devfolds_best_20260709" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / f"{split}_predictions.csv",
        "a_h64": ROOT / "outputs" / "arousal_devfolds_smallreg_20260709" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003" / f"{split}_predictions.csv",
        "a_h128": ROOT / "outputs" / "arousal_devfolds_h128_20260709" / fold / "a_dev_h128_l1_do020_lr6e4_mse" / f"{split}_predictions.csv",
        "a_mert_harm": ROOT / "outputs" / "arousal_mert_harmonic_devfolds_20260709" / fold / f"{split}_predictions.csv",
    }


def pred_col(df: pd.DataFrame, prefix: str) -> str:
    for col in [f"pred_{prefix}", f"pred_{prefix}_raw", "pred_valence", "pred_arousal", "pred_arousal_raw"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {df.columns.tolist()}")


def load_split(fold: str, split: str) -> pd.DataFrame:
    base = pd.read_csv(valence_path(fold, split))[["musicId", "time_sec", "target_valence", "pred_valence"]]
    for name, path in arousal_paths(fold, split).items():
        frame = pd.read_csv(path)
        col = pred_col(frame, "arousal")
        base = base.merge(
            frame[["musicId", "time_sec", col]].rename(columns={col: name}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    return base.sort_values(["musicId", "time_sec"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_norm_song"] = out.groupby("musicId")["time_sec"].transform(lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1e-6))
    for col in ["pred_valence", "a_h96", "a_h64", "a_h128", "a_mert_harm"]:
        out[f"{col}_song_mean"] = out.groupby("musicId")[col].transform("mean")
        out[f"{col}_centered"] = out[col] - out[f"{col}_song_mean"]
    out["a_mean"] = out[["a_h96", "a_h64", "a_h128", "a_mert_harm"]].mean(axis=1)
    out["a_spread"] = out[["a_h96", "a_h64", "a_h128", "a_mert_harm"]].std(axis=1)
    for col in ["a_mean", "a_spread"]:
        out[f"{col}_song_mean"] = out.groupby("musicId")[col].transform("mean")
        out[f"{col}_centered"] = out[col] - out[f"{col}_song_mean"]
    out["val_x_a_mean"] = out["pred_valence"] * out["a_mean"]
    return out


FEATURE_SETS = {
    "base_only": ["pred_valence"],
    "base_arousal": ["pred_valence", "a_h96", "a_h64", "a_h128", "a_mert_harm", "a_mean", "a_spread"],
    "base_arousal_song": [
        "pred_valence",
        "pred_valence_song_mean",
        "pred_valence_centered",
        "a_h96",
        "a_h64",
        "a_h128",
        "a_mert_harm",
        "a_mean",
        "a_spread",
        "a_mean_song_mean",
        "a_mean_centered",
        "time_norm_song",
        "val_x_a_mean",
    ],
}


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


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


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


def scale_amplitude_by_song(df: pd.DataFrame, values: np.ndarray, amp: float) -> np.ndarray:
    if abs(amp - 1.0) < 1e-12:
        return values.copy()
    tmp = df[["musicId"]].copy()
    tmp["value"] = values
    song_mean = tmp.groupby("musicId", sort=False)["value"].transform("mean").to_numpy(dtype=np.float64)
    return song_mean + amp * (values - song_mean)


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.03 * max(0.0, m["mae"] - 0.15) + 0.01 * max(0.0, 0.80 - m["ccc"])


def search_fold(fold: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = add_features(load_split(fold, "train"))
    val = add_features(load_split(fold, "val"))
    y_train = train["target_valence"].to_numpy(dtype=np.float64)
    y_val = val["target_valence"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for feature_set, cols in FEATURE_SETS.items():
        x_train = train[cols].to_numpy(dtype=np.float64)
        x_val = val[cols].to_numpy(dtype=np.float64)
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train, y_train)
            raw = model.predict(x_val)
            for shift_frames in [0, 4, 8, 12, 16]:
                shifted = shift_by_song(val, raw, shift_frames)
                for amp in [0.7, 0.85, 1.0, 1.15]:
                    amplified = scale_amplitude_by_song(val, shifted, amp)
                    for window in [51, 71, 91, 111, 131, 171]:
                        smoothed = smooth_by_song(val, amplified, window)
                        for method in ["bias", "linear", "mean_std"]:
                            final, params = calibrate(smoothed, y_val, method)
                            final = np.clip(final, -1.0, 1.0)
                            m = metrics(final, y_val)
                            row = {
                                "fold": fold,
                                "feature_set": feature_set,
                                "alpha": alpha,
                                "shift_frames": shift_frames,
                                "shift_sec": shift_frames * 0.5,
                                "amp": amp,
                                "window": window,
                                "method": method,
                                "params": json.dumps(params, ensure_ascii=True),
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
    pred_out = val[["musicId", "time_sec", "target_valence"]].copy()
    pred_out["pred_valence"] = best_pred
    return best, pd.DataFrame(rows), pred_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        best, rows, pred = search_fold(fold)
        rows.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(OUT_DIR / f"{fold}_candidates.csv", index=False)
        pred.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({k: v for k, v in best.items() if k != "params"})
    mean = {
        metric: float(np.mean([float(row[metric]) for row in per_fold]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Valence train-fitted Ridge stacker with Arousal auxiliary predictions; dev folds only",
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
