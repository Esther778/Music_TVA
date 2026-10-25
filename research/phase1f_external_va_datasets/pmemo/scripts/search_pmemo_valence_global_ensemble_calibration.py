#!/usr/bin/env python3
"""Validation-only global calibration for the current Valence ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_multilayer_global_ensemble_20260629"
OUT_DIR = ROOT / "outputs" / "valence_global_ensemble_calibration_20260629"

CURRENT_STRICTISH = {
    "model_id": "pmemo_valence_multilayer_global_ensemble_20260629",
    "rmse": 0.19411262662053563,
    "mae": 0.15142593782241862,
    "pcc": 0.7998908886855055,
    "ccc": 0.7851966731300748,
}
EXPLORATORY_BEST = {
    "model_id": "pmemo_valence_universal_metric_optimized_exploratory_20260628",
    "rmse": 0.19182218120147795,
    "mae": 0.14919783550199786,
    "pcc": 0.8044833131619238,
    "ccc": 0.7881895323604796,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


def fit_calibration(df: pd.DataFrame, method: str, strength: float, clip: float | None) -> dict[str, Any]:
    pred = df["pred_valence"].to_numpy(dtype=np.float64)
    target = df["target_valence"].to_numpy(dtype=np.float64)
    if method == "identity":
        params = {"a": 1.0, "b": 0.0, "pred_mean": 0.0, "pred_std": 1.0, "target_mean": 0.0, "target_std": 1.0}
    elif method == "bias":
        params = {"a": 1.0, "b": float(np.mean(target - pred)), "pred_mean": 0.0, "pred_std": 1.0, "target_mean": 0.0, "target_std": 1.0}
    elif method == "linear":
        a, b = fit_linear(pred, target)
        params = {"a": a, "b": b, "pred_mean": 0.0, "pred_std": 1.0, "target_mean": 0.0, "target_std": 1.0}
    elif method == "mean_std":
        params = {
            "a": 1.0,
            "b": 0.0,
            "pred_mean": float(pred.mean()),
            "pred_std": float(pred.std() + 1e-12),
            "target_mean": float(target.mean()),
            "target_std": float(target.std() + 1e-12),
        }
    else:
        raise ValueError(method)
    params["method"] = method
    params["strength"] = float(strength)
    params["clip"] = clip
    return params


def apply_calibration(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    pred = out["pred_valence"].to_numpy(dtype=np.float64)
    method = str(params["method"])
    if method in {"identity", "bias", "linear"}:
        calibrated = float(params["a"]) * pred + float(params["b"])
    elif method == "mean_std":
        calibrated = (pred - float(params["pred_mean"])) / float(params["pred_std"]) * float(params["target_std"]) + float(params["target_mean"])
    else:
        raise ValueError(method)
    strength = float(params["strength"])
    final = (1.0 - strength) * pred + strength * calibrated
    if params["clip"] is not None:
        final = np.clip(final, -float(params["clip"]), float(params["clip"]))
    out["pred_valence"] = final
    out["calibration_method"] = method
    out["calibration_strength"] = strength
    return out


def macro_metrics(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    rows = [eval_df(df) for df in frames.values()]
    return {k: float(np.mean([m[k] for m in rows])) for k in ["rmse", "mae", "pcc", "ccc"]}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.50 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.25 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.50 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.75 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_names = sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir())
    frames = {
        split: {split_name: pd.read_csv(INPUT_DIR / split_name / f"{split}_predictions.csv") for split_name in split_names}
        for split in ["train", "val", "test"]
    }
    val_pooled = pd.concat(frames["val"].values(), ignore_index=True)
    baseline = {split: macro_metrics(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache: dict[tuple[str, float, str], dict[str, dict[str, pd.DataFrame]]] = {}
    for method in ["identity", "bias", "linear", "mean_std"]:
        for strength in [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
            for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90)]:
                params = fit_calibration(val_pooled, method, strength, clip)
                corrected = {
                    split: {split_name: apply_calibration(df, params) for split_name, df in frames[split].items()}
                    for split in ["train", "val", "test"]
                }
                metrics_by_split = {split: macro_metrics(corrected[split]) for split in ["train", "val", "test"]}
                row = {
                    "method": method,
                    "strength": float(strength),
                    "clip": clip_name,
                    "params": json.dumps(params, ensure_ascii=True),
                    **{f"{split}_macro_{k}": v for split, m in metrics_by_split.items() for k, v in m.items()},
                }
                row["val_selection_score"] = selection_score(metrics_by_split["val"])
                rows.append(row)
                cache[(method, float(strength), clip_name)] = corrected
    cand = pd.DataFrame(rows)
    baseline_val = baseline["val"]
    allowed = cand[
        (cand["val_macro_pcc"] >= baseline_val["pcc"] - 0.001)
        & (cand["val_macro_ccc"] >= baseline_val["ccc"] - 0.001)
    ].copy()
    if allowed.empty:
        allowed = cand
    selected = allowed.sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_key = (str(selected["method"]), float(selected["strength"]), str(selected["clip"]))
    corrected = cache[selected_key]
    for split in ["train", "val", "test"]:
        for split_name, df in corrected[split].items():
            out_dir = OUT_DIR / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_dir / f"{split}_predictions.csv", index=False)
    test_macro = macro_metrics(corrected["test"])
    val_macro = macro_metrics(corrected["val"])
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_global_ensemble_calibration_20260629",
        "purpose": "validation-only global output calibration for current Valence ensemble",
        "selection_rule": "one global calibration selected on macro validation; test labels are not used",
        "input_model": CURRENT_STRICTISH,
        "exploratory_best_reference": EXPLORATORY_BEST,
        "baseline_macro": baseline,
        "selected": selected,
        "selected_macro": {"val": val_macro, "test": test_macro},
        "test_gap_rows": threshold_rows(test_macro),
        "delta_vs_input_model": {k: float(test_macro[k] - CURRENT_STRICTISH[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
        "delta_vs_exploratory_best": {k: float(test_macro[k] - EXPLORATORY_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    cand.to_csv(OUT_DIR / "all_calibration_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
