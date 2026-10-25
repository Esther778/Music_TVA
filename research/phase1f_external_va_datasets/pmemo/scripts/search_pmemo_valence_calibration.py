#!/usr/bin/env python3
"""Validation-selected calibration/postprocessing for PMEmo Valence predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "valence_calibration_20260626"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}

CURRENT_BEST = {
    "id": "current_best_extreme",
    "rmse": 0.205672949552536,
    "mae": 0.16456688940525055,
    "pcc": 0.7834003513415063,
    "ccc": 0.7673044285835935,
}

MODELS = {
    "current_best_extreme": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext125_clip010_ccc005",
    "best_refine_rmse": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted_refine/h6_harm_h80_ext100_clip008_ccc005",
    "best_refine_ccc": ROOT
    / "outputs/mert_hidden6_harmonic_extreme_weighted_refine/h6_harm_h80_ext150_clip010_ccc008",
    "previous_clip_aux": ROOT
    / "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_ccc005",
    "mert_opensmile_fusion": ROOT
    / "outputs/mert_hidden6_opensmile_valence_fusion_fixed/h6_smile_h80_proj320_do030_ccc005",
}


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    pcc = 0.0 if np.std(pred) < 1e-12 or np.std(target) < 1e-12 else float(np.corrcoef(pred, target)[0, 1])
    mean_x = float(np.mean(pred))
    mean_y = float(np.mean(target))
    var_x = float(np.var(pred))
    var_y = float(np.var(target))
    cov = float(np.mean((pred - mean_x) * (target - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    ccc = float(2 * cov / denom) if denom > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "ccc": ccc}


def selection_score(m: dict[str, float]) -> float:
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.10 * m["rmse"]
    )


def read_split(model_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(model_dir / f"{split}_predictions.csv")
    pred_col = "pred_valence"
    if pred_col not in df.columns and "pred_valence_raw" in df.columns:
        pred_col = "pred_valence_raw"
    out = df.loc[:, ["musicId", "time_sec", "target_valence", pred_col]].copy()
    out["musicId"] = out["musicId"].astype(str)
    out["time_sec"] = out["time_sec"].astype(float)
    return out.rename(columns={pred_col: "pred_valence"})


def moving_average_by_song(df: pd.DataFrame, pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    out = np.zeros_like(pred, dtype=np.float64)
    tmp = df.loc[:, ["musicId", "time_sec"]].copy()
    tmp["pred"] = pred
    for _, idx in tmp.groupby("musicId", sort=False).groups.items():
        values = tmp.loc[idx, "pred"].to_numpy(dtype=np.float64)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        out[np.asarray(idx)] = np.convolve(padded, kernel, mode="valid")
    return out


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(pred.astype(np.float64), target.astype(np.float64), 1)
    return float(a), float(b)


def transform_predictions(
    df: pd.DataFrame,
    pred: np.ndarray,
    config: dict[str, float | int | str | bool],
    fit_stats: dict[str, float],
) -> np.ndarray:
    y = moving_average_by_song(df, pred.astype(np.float64), int(config["window"]))
    kind = str(config["kind"])
    if kind == "identity":
        out = y
    elif kind == "bias":
        out = y + float(fit_stats["bias"])
    elif kind == "linear":
        out = float(fit_stats["a"]) * y + float(fit_stats["b"])
    elif kind == "mean_std":
        out = (y - float(fit_stats["pred_mean"])) / float(fit_stats["pred_std"]) * float(fit_stats["target_std"]) + float(fit_stats["target_mean"])
    elif kind == "scale_about_mean":
        out = (y - float(fit_stats["pred_mean"])) * float(config["scale"]) + float(fit_stats["target_mean"])
    else:
        raise ValueError(kind)
    if bool(config.get("clip", False)):
        out = np.clip(out, -1.0, 1.0)
    return out


def fit_stats_for(kind: str, val_df: pd.DataFrame, val_pred_base: np.ndarray, val_target: np.ndarray) -> dict[str, float]:
    if kind == "bias":
        return {"bias": float(np.mean(val_target - val_pred_base))}
    if kind == "linear":
        a, b = fit_linear(val_pred_base, val_target)
        return {"a": a, "b": b}
    if kind in {"mean_std", "scale_about_mean"}:
        return {
            "pred_mean": float(np.mean(val_pred_base)),
            "pred_std": float(np.std(val_pred_base) + 1e-12),
            "target_mean": float(np.mean(val_target)),
            "target_std": float(np.std(val_target) + 1e-12),
        }
    return {}


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = [1, 3, 5, 9, 15]
    scales = [0.90, 1.00, 1.05, 1.10, 1.15, 1.20]
    kinds = ["identity", "bias", "linear", "mean_std", "scale_about_mean"]
    rows = []
    for model_id, model_dir in MODELS.items():
        if not (model_dir / "val_predictions.csv").exists():
            continue
        val_df = read_split(model_dir, "val")
        test_df = read_split(model_dir, "test")
        val_raw = val_df["pred_valence"].to_numpy(dtype=np.float64)
        val_target = val_df["target_valence"].to_numpy(dtype=np.float64)
        test_raw = test_df["pred_valence"].to_numpy(dtype=np.float64)
        test_target = test_df["target_valence"].to_numpy(dtype=np.float64)
        for window in windows:
            val_smoothed = moving_average_by_song(val_df, val_raw, window)
            for kind in kinds:
                scale_values = scales if kind == "scale_about_mean" else [1.0]
                for scale in scale_values:
                    for clip in [False, True]:
                        config = {"kind": kind, "window": window, "scale": scale, "clip": clip}
                        stats = fit_stats_for(kind, val_df, val_smoothed, val_target)
                        val_pred = transform_predictions(val_df, val_raw, config, stats)
                        val_m = metrics(val_pred, val_target)
                        test_pred = transform_predictions(test_df, test_raw, config, stats)
                        test_m = metrics(test_pred, test_target)
                        rows.append(
                            {
                                "model_id": model_id,
                                "config": json.dumps(config, ensure_ascii=True),
                                "fit_stats": json.dumps(stats, ensure_ascii=True),
                                **{f"val_{k}": v for k, v in val_m.items()},
                                **{f"test_{k}": v for k, v in test_m.items()},
                                "selection_score": selection_score(val_m),
                            }
                        )
    ranked = sorted(rows, key=lambda r: (r["selection_score"], r["val_rmse"], -r["val_ccc"]))
    top = ranked[:200]
    pd.DataFrame(rows).to_csv(OUT_DIR / "all_calibration_results.csv", index=False)
    pd.DataFrame(top).to_csv(OUT_DIR / "top_calibration_results.csv", index=False)
    selected = top[0]
    delta = {
        "rmse": float(selected["test_rmse"] - CURRENT_BEST["rmse"]),
        "mae": float(selected["test_mae"] - CURRENT_BEST["mae"]),
        "pcc": float(selected["test_pcc"] - CURRENT_BEST["pcc"]),
        "ccc": float(selected["test_ccc"] - CURRENT_BEST["ccc"]),
    }
    threshold_gap = {
        "rmse_to_pass": float(selected["test_rmse"] - PASS["rmse"]),
        "mae_to_pass": float(selected["test_mae"] - PASS["mae"]),
        "pcc_to_pass": float(PASS["pcc"] - selected["test_pcc"]),
        "ccc_to_pass": float(PASS["ccc"] - selected["test_ccc"]),
        "rmse_to_good": float(selected["test_rmse"] - GOOD["rmse"]),
        "mae_to_good": float(selected["test_mae"] - GOOD["mae"]),
        "pcc_to_good": float(GOOD["pcc"] - selected["test_pcc"]),
        "ccc_to_good": float(GOOD["ccc"] - selected["test_ccc"]),
        "rmse_to_excellent": float(selected["test_rmse"] - EXCELLENT["rmse"]),
        "mae_to_excellent": float(selected["test_mae"] - EXCELLENT["mae"]),
        "pcc_to_excellent": float(EXCELLENT["pcc"] - selected["test_pcc"]),
        "ccc_to_excellent": float(EXCELLENT["ccc"] - selected["test_ccc"]),
    }
    summary = {
        "created_at": "2026-06-26",
        "selection_rule": "choose transform by validation pass-gap composite; test not used for selection",
        "current_best": CURRENT_BEST,
        "selected": selected,
        "test_delta_vs_current_best": delta,
        "threshold_gap_selected_test": threshold_gap,
        "top_test_rmse": min(rows, key=lambda r: r["test_rmse"]),
        "top_test_ccc": max(rows, key=lambda r: r["test_ccc"]),
    }
    (OUT_DIR / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
