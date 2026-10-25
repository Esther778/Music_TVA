#!/usr/bin/env python3
"""Search validation-selected ensembles for PMEmo Valence predictions."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "valence_ensemble_search"

PASS_TARGETS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD_TARGETS = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT_TARGETS = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}


CANDIDATES = {
    "opensmile_v_only_h96": ROOT / "outputs" / "opensmile_valence_trials" / "v_only_h96_l1_do015_mse",
    "selected_harmonic_h96": ROOT / "outputs" / "valence_opensmile_selected_fusion" / "v_sel_harmonic_h96_l1",
    "selected_harmonic_spectral": ROOT / "outputs" / "valence_opensmile_selected_fusion" / "v_sel_harmonic_spectral_h128_l2_ccc",
    "selected_harmonic_mfcc": ROOT / "outputs" / "valence_opensmile_selected_fusion" / "v_sel_harmonic_mfcc_h96_l1",
    "fusion_full_h128": ROOT / "outputs" / "valence_opensmile_librosa_fusion" / "v_fusion_h128_l2_do030_mse_ccc",
    "harmonic_focus_ccc005": ROOT / "outputs" / "valence_opensmile_harmonic_focus" / "v_harm_h96_l1_do025_lr6e4_ccc005",
}


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def ccc(pred: np.ndarray, target: np.ndarray) -> float:
    mean_x = float(np.mean(pred))
    mean_y = float(np.mean(target))
    var_x = float(np.var(pred))
    var_y = float(np.var(target))
    cov = float(np.mean((pred - mean_x) * (target - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(pred, target),
        "ccc": ccc(pred, target),
    }


def target_gap(value: float, metric_name: str, target: float) -> float:
    return float(value - target) if metric_name in {"rmse", "mae"} else float(target - value)


def threshold_rows(m: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for key in ["rmse", "mae", "pcc", "ccc"]:
        pass_gap = target_gap(float(m[key]), key, PASS_TARGETS[key])
        rows.append(
            {
                "metric": key,
                "current": float(m[key]),
                "minimum_pass": PASS_TARGETS[key],
                "gap_to_pass": pass_gap,
                "good": GOOD_TARGETS[key],
                "gap_to_good": target_gap(float(m[key]), key, GOOD_TARGETS[key]),
                "excellent": EXCELLENT_TARGETS[key],
                "gap_to_excellent": target_gap(float(m[key]), key, EXCELLENT_TARGETS[key]),
                "status": "pass" if pass_gap <= 0 else "not_pass",
            }
        )
    return rows


def load_prediction_dir(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path / f"{split}_predictions.csv")
    pred_col = "pred_valence_raw" if "pred_valence_raw" in df.columns else "pred_valence"
    return df[["musicId", "time_sec", "target_valence", pred_col]].rename(columns={pred_col: path.name})


def load_split(split: str) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    names = []
    for name, path in CANDIDATES.items():
        if not (path / f"{split}_predictions.csv").exists():
            continue
        df = load_prediction_dir(path, split).rename(columns={path.name: name})
        names.append(name)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["musicId", "time_sec", "target_valence"], how="inner")
    if merged is None:
        raise RuntimeError(f"No predictions found for split={split}")
    return merged.sort_values(["musicId", "time_sec"]).reset_index(drop=True), names


def weight_grid(n: int, step: float = 0.05):
    units = int(round(1.0 / step))
    if n == 1:
        yield (1.0,)
        return
    for cuts in itertools.combinations_with_replacement(range(units + 1), n - 1):
        points = (0,) + cuts + (units,)
        weights = [(points[i + 1] - points[i]) / units for i in range(n)]
        if sum(w > 0 for w in weights) >= 1:
            yield tuple(weights)


def ensemble_pred(df: pd.DataFrame, names: list[str], weights: tuple[float, ...]) -> np.ndarray:
    mat = df[names].to_numpy(dtype=np.float64)
    return (mat * np.asarray(weights, dtype=np.float64)[None, :]).sum(axis=1).astype(np.float32)


def fit_linear_calibration(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    var = float(np.var(x))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    b = float(y.mean() - a * x.mean())
    return a, b


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_df, names = load_split("val")
    test_df, _ = load_split("test")
    y_val = val_df["target_valence"].to_numpy(dtype=np.float32)
    y_test = test_df["target_valence"].to_numpy(dtype=np.float32)
    searches = []
    for size in range(1, min(4, len(names)) + 1):
        for subset in itertools.combinations(names, size):
            for weights in weight_grid(len(subset), 0.05):
                pred = ensemble_pred(val_df, list(subset), weights)
                m = metrics(pred, y_val)
                searches.append({"subset": list(subset), "weights": list(weights), "val_metrics": m, "score": m["rmse"]})
    searches.sort(key=lambda row: row["score"])
    best = searches[0]
    subset = best["subset"]
    weights = tuple(float(w) for w in best["weights"])
    val_pred = ensemble_pred(val_df, subset, weights)
    test_pred = ensemble_pred(test_df, subset, weights)
    a, b = fit_linear_calibration(val_pred, y_val)
    test_cal = (a * test_pred + b).astype(np.float32)
    result = {
        "selection_rule": "min validation RMSE, convex weights step 0.05, subset size up to 4",
        "candidate_names": names,
        "best_subset": subset,
        "best_weights": list(weights),
        "val_metrics": metrics(val_pred, y_val),
        "test_metrics": metrics(test_pred, y_test),
        "linear_calibration_from_val": {"a": a, "b": b},
        "test_calibrated_metrics": metrics(test_cal, y_test),
    }
    result["test_gap_rows"] = threshold_rows(result["test_metrics"])
    result["test_calibrated_gap_rows"] = threshold_rows(result["test_calibrated_metrics"])
    (OUT_DIR / "ensemble_search_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(searches[:100]).to_csv(OUT_DIR / "top100_validation_search.csv", index=False)
    pd.DataFrame(
        {
            "musicId": test_df["musicId"],
            "time_sec": test_df["time_sec"],
            "target_valence": y_test,
            "pred_valence_ensemble": test_pred,
            "pred_valence_ensemble_calibrated": test_cal,
        }
    ).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
