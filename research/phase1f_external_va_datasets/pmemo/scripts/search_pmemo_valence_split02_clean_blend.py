#!/usr/bin/env python3
"""Validation-guarded split_02 blend between historical and clean Valence teachers.

This experiment is intentionally narrow:
- it only uses already generated prediction CSVs;
- blend/calibration parameters are selected on validation predictions;
- test predictions are reported only after validation selection.

The purpose is to check whether the clean checkpoint's stronger correlation can
be used without sacrificing the historical best's lower RMSE/MAE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "valence_split02_clean_blend_20260707"

HISTORICAL_DIR = ROOT / "outputs" / "valence_parsimony_output_guarded_20260707" / "split_02_seed_202"
CLEAN_DIR = (
    ROOT
    / "outputs"
    / "valence_multilayer_clean_teacher_20260707"
    / "h468_harm_clean_proj384_h80_do035_ccc012_clip012_std003"
    / "split_02_seed_606"
)

DATA_SPLITS = ["train", "val", "test"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column found in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column found in {list(df.columns)}")


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.25 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def gap_rows(m: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        lower_is_better = metric in {"rmse", "mae"}
        current = float(m[metric])
        pass_v = PASS[metric]
        good_v = GOOD[metric]
        excellent_v = EXCELLENT[metric]
        if lower_is_better:
            gap_pass = current - pass_v
            gap_good = current - good_v
            gap_excellent = current - excellent_v
            status = "pass" if current <= pass_v else "not_pass"
        else:
            gap_pass = current - pass_v
            gap_good = current - good_v
            gap_excellent = current - excellent_v
            status = "pass" if current >= pass_v else "not_pass"
        rows.append(
            {
                "metric": metric,
                "current": current,
                "minimum_pass": pass_v,
                "gap_to_pass": float(gap_pass),
                "good": good_v,
                "gap_to_good": float(gap_good),
                "excellent": excellent_v,
                "gap_to_excellent": float(gap_excellent),
                "status": status,
            }
        )
    return rows


def moving_average_by_song(df: pd.DataFrame, pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    out = np.empty(len(df), dtype=np.float64)
    kernel = np.ones(int(window), dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        loc = np.asarray(idx)
        vals = pred[loc]
        start = (int(window) - 1) // 2
        summed = np.convolve(vals, kernel, mode="full")[start : start + len(vals)]
        counts = np.convolve(np.ones_like(vals), kernel, mode="full")[start : start + len(vals)]
        out[loc] = summed / np.maximum(counts, 1.0)
    return out


def load_pair(data_split: str) -> pd.DataFrame:
    hist = pd.read_csv(HISTORICAL_DIR / f"{data_split}_predictions.csv")
    clean = pd.read_csv(CLEAN_DIR / f"{data_split}_predictions.csv")
    h = hist[["musicId", "time_sec", target_col(hist), pred_col(hist)]].rename(
        columns={target_col(hist): "target_valence", pred_col(hist): "historical_pred"}
    )
    c = clean[["musicId", "time_sec", target_col(clean), pred_col(clean)]].rename(
        columns={target_col(clean): "target_valence_clean", pred_col(clean): "clean_pred"}
    )
    merged = h.merge(c, on=["musicId", "time_sec"], how="inner")
    if len(merged) != len(h) or len(merged) != len(c):
        raise ValueError(f"Frame mismatch for {data_split}: historical={len(h)} clean={len(c)} merged={len(merged)}")
    target_diff = np.max(np.abs(merged["target_valence"].to_numpy() - merged["target_valence_clean"].to_numpy()))
    if target_diff > 1e-7:
        raise ValueError(f"Target mismatch for {data_split}: max_diff={target_diff}")
    return merged.drop(columns=["target_valence_clean"])


def fit_affine(x: np.ndarray, y: np.ndarray, strength: float) -> tuple[float, float]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.var(x) + 1e-8)
    slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
    intercept = y_mean - slope * x_mean
    return (1.0 - strength) + strength * slope, strength * intercept


def main() -> None:
    frames = {split: load_pair(split) for split in DATA_SPLITS}
    y_train = frames["train"]["target_valence"].to_numpy(dtype=np.float64)
    base_preds = {split: frames[split]["historical_pred"].to_numpy(dtype=np.float64) for split in DATA_SPLITS}
    clean_preds = {split: frames[split]["clean_pred"].to_numpy(dtype=np.float64) for split in DATA_SPLITS}
    base_metrics = {
        split: eval_arrays(base_preds[split], frames[split]["target_valence"].to_numpy(dtype=np.float64))
        for split in DATA_SPLITS
    }
    clean_metrics = {
        split: eval_arrays(clean_preds[split], frames[split]["target_valence"].to_numpy(dtype=np.float64))
        for split in DATA_SPLITS
    }
    base_score = selection_score(base_metrics["val"])

    rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[float, float, int], dict[str, np.ndarray]] = {}
    for clean_weight in np.linspace(0.0, 0.50, 21):
        raw_train = (1.0 - clean_weight) * base_preds["train"] + clean_weight * clean_preds["train"]
        for affine_strength in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            slope, intercept = fit_affine(raw_train, y_train, affine_strength)
            for smooth_window in [1, 3, 5, 7, 9, 11, 15]:
                split_preds: dict[str, np.ndarray] = {}
                split_metrics: dict[str, dict[str, float]] = {}
                for split in DATA_SPLITS:
                    raw = (1.0 - clean_weight) * base_preds[split] + clean_weight * clean_preds[split]
                    pred = slope * raw + intercept
                    pred = moving_average_by_song(frames[split], pred, smooth_window)
                    split_preds[split] = pred
                    split_metrics[split] = eval_arrays(
                        pred,
                        frames[split]["target_valence"].to_numpy(dtype=np.float64),
                    )
                val_score = selection_score(split_metrics["val"])
                passes_guard = (
                    val_score < base_score - 0.00025
                    and split_metrics["val"]["rmse"] <= base_metrics["val"]["rmse"] + 0.00075
                    and split_metrics["val"]["mae"] <= base_metrics["val"]["mae"] + 0.00075
                    and split_metrics["val"]["ccc"] >= base_metrics["val"]["ccc"] - 0.00075
                )
                key = (float(clean_weight), float(affine_strength), int(smooth_window))
                pred_cache[key] = split_preds
                rows.append(
                    {
                        "clean_weight": float(clean_weight),
                        "affine_strength": float(affine_strength),
                        "affine_slope": float(slope),
                        "affine_intercept": float(intercept),
                        "smooth_window": int(smooth_window),
                        **{
                            f"{split}_{metric}": float(value)
                            for split, vals in split_metrics.items()
                            for metric, value in vals.items()
                        },
                        "val_selection_score": float(val_score),
                        "passes_guard": bool(passes_guard),
                    }
                )

    candidates = pd.DataFrame(rows)
    guarded = candidates[candidates["passes_guard"]].sort_values(
        ["val_selection_score", "val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, True, False],
    )
    if guarded.empty:
        selected = {
            "clean_weight": 0.0,
            "affine_strength": 0.0,
            "affine_slope": 1.0,
            "affine_intercept": 0.0,
            "smooth_window": 1,
            **{
                f"{split}_{metric}": float(value)
                for split, vals in base_metrics.items()
                for metric, value in vals.items()
            },
            "val_selection_score": float(base_score),
            "passes_guard": True,
            "selected_reason": "identity retained because no candidate passed validation guard",
        }
        selected_preds = base_preds
    else:
        selected = guarded.iloc[0].to_dict()
        selected["selected_reason"] = "validation guard passed"
        key = (
            float(selected["clean_weight"]),
            float(selected["affine_strength"]),
            int(selected["smooth_window"]),
        )
        selected_preds = pred_cache[key]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_DIR / "all_candidates.csv", index=False)
    for split in DATA_SPLITS:
        out = frames[split][["musicId", "time_sec", "target_valence"]].copy()
        out["pred_valence"] = selected_preds[split]
        out["historical_pred"] = base_preds[split]
        out["clean_pred"] = clean_preds[split]
        out["clean_weight"] = selected["clean_weight"]
        out["affine_strength"] = selected["affine_strength"]
        out["smooth_window"] = selected["smooth_window"]
        out.to_csv(OUT_DIR / f"{split}_predictions.csv", index=False)

    selected_test = {metric: float(selected[f"test_{metric}"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_split02_clean_blend_20260707",
        "purpose": "split_02 validation-guarded blend of historical Valence best and clean checkpoint candidate",
        "historical_input": str(HISTORICAL_DIR.relative_to(ROOT)),
        "clean_input": str(CLEAN_DIR.relative_to(ROOT)),
        "selection_rule": "grid search clean weight, train-fit affine strength and smoothing; select by validation score with RMSE/MAE/CCC guard; test only reported after selection",
        "baseline": base_metrics,
        "clean_checkpoint": clean_metrics,
        "selected": selected,
        "selected_test": selected_test,
        "delta_vs_split_historical_test": {
            metric: float(selected_test[metric] - base_metrics["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_current_macro_best_reference": {
            metric: float(selected_test[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "test_gap_rows": gap_rows(selected_test),
        "candidate_count": int(len(candidates)),
        "guarded_candidate_count": int(len(guarded)),
        "current_macro_best_reference": CURRENT_BEST,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
