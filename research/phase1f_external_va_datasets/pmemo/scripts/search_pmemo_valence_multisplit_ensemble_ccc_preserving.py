#!/usr/bin/env python3
"""CCC-preserving BiLSTM + TCN ensemble search for PMEmo Valence.

This script reuses the already trained multi-split BiLSTM and TCN predictions.
It does not train a new neural network. The goal is to select fusion and
post-processing by validation metrics with a stronger penalty on PCC/CCC loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BILSTM_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
TCN_DIR = ROOT / "outputs" / "valence_multisplit_tcn_20260627"
OUT_DIR = ROOT / "outputs" / "valence_multisplit_ensemble_ccc_preserving_20260627"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    """Validation objective: prioritize correlation/concordance, then error."""
    return (
        0.60 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 0.60 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.80 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 3.00 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def read_pair(split_dir: str, split: str) -> pd.DataFrame:
    b = pd.read_csv(BILSTM_DIR / split_dir / f"{split}_predictions.csv")
    t = pd.read_csv(TCN_DIR / split_dir / f"{split}_predictions.csv")
    b = b.copy()
    t = t.copy()
    for df in [b, t]:
        df["musicId"] = df["musicId"].astype(str)
        df["time_key"] = df["time_sec"].round(6)
        df["target_key"] = df["target_valence"].round(6)
    merged = b.loc[:, ["musicId", "time_key", "target_key", "time_sec", "target_valence", "pred_valence"]].merge(
        t.loc[:, ["musicId", "time_key", "target_key", "pred_valence"]],
        on=["musicId", "time_key", "target_key"],
        how="inner",
        suffixes=("_bilstm", "_tcn"),
    )
    if len(merged) != len(b) or len(merged) != len(t):
        raise ValueError(
            f"Prediction alignment mismatch: {split_dir} {split}: "
            f"bilstm={len(b)} tcn={len(t)} merged={len(merged)}"
        )
    out = merged.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
    out["pred_bilstm"] = merged["pred_valence_bilstm"].to_numpy(dtype=np.float64)
    out["pred_tcn"] = merged["pred_valence_tcn"].to_numpy(dtype=np.float64)
    return out


def moving_average_by_song(df: pd.DataFrame, pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred
    tmp = df.loc[:, ["musicId", "time_sec"]].copy()
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


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if np.std(pred) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(np.float64), target.astype(np.float64), 1)
    return float(a), float(b)


def raw_prediction(df: pd.DataFrame, weight_tcn: float, smooth_window: int) -> np.ndarray:
    pred = (1.0 - weight_tcn) * df["pred_bilstm"].to_numpy(dtype=np.float64) + weight_tcn * df["pred_tcn"].to_numpy(dtype=np.float64)
    return moving_average_by_song(df, pred, smooth_window)


def fit_stats(val_df: pd.DataFrame, weight_tcn: float, calibration: str, smooth_window: int) -> dict[str, float]:
    pred = raw_prediction(val_df, weight_tcn, smooth_window)
    return fit_stats_from_pred(pred, val_df["target_valence"].to_numpy(dtype=np.float64), calibration)


def fit_stats_from_pred(pred: np.ndarray, target: np.ndarray, calibration: str) -> dict[str, float]:
    if calibration == "identity":
        return {}
    if calibration == "bias":
        return {"bias": float(np.mean(target - pred))}
    if calibration == "linear":
        a, b = fit_linear(pred, target)
        return {"a": a, "b": b}
    if calibration == "mean_std":
        return {
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred) + 1e-12),
            "target_mean": float(np.mean(target)),
            "target_std": float(np.std(target) + 1e-12),
        }
    raise ValueError(calibration)


def apply_calibration_to_pred(pred: np.ndarray, calibration: str, stats: dict[str, float]) -> np.ndarray:
    if calibration == "identity":
        out = pred
    elif calibration == "bias":
        out = pred + stats["bias"]
    elif calibration == "linear":
        out = stats["a"] * pred + stats["b"]
    elif calibration == "mean_std":
        out = (pred - stats["pred_mean"]) / stats["pred_std"] * stats["target_std"] + stats["target_mean"]
    else:
        raise ValueError(calibration)
    return np.clip(out, -1.0, 1.0)


def apply_prediction(df: pd.DataFrame, weight_tcn: float, calibration: str, smooth_window: int, stats: dict[str, float]) -> np.ndarray:
    pred = raw_prediction(df, weight_tcn, smooth_window)
    return apply_calibration_to_pred(pred, calibration, stats)


def eval_candidate(df: pd.DataFrame, weight_tcn: float, calibration: str, smooth_window: int, stats: dict[str, float]) -> dict[str, float]:
    pred = apply_prediction(df, weight_tcn, calibration, smooth_window, stats)
    target = df["target_valence"].to_numpy(dtype=np.float64)
    return metrics(pred, target)


def eval_pred(pred: np.ndarray, target: np.ndarray, calibration: str, stats: dict[str, float]) -> dict[str, float]:
    return metrics(apply_calibration_to_pred(pred, calibration, stats), target)


def evaluate_split(split_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: read_pair(split_dir, split) for split in ["train", "val", "test"]}
    baseline = {
        "bilstm": {
            split: metrics(frames[split]["pred_bilstm"].to_numpy(dtype=np.float64), frames[split]["target_valence"].to_numpy(dtype=np.float64))
            for split in ["train", "val", "test"]
        },
        "tcn": {
            split: metrics(frames[split]["pred_tcn"].to_numpy(dtype=np.float64), frames[split]["target_valence"].to_numpy(dtype=np.float64))
            for split in ["train", "val", "test"]
        },
    }
    rows = []
    weights = [round(x, 2) for x in np.linspace(0.0, 1.0, 101)]
    calibrations = ["identity", "bias", "linear", "mean_std"]
    smooth_windows = [1, 3, 5, 9, 15]
    targets = {split: frames[split]["target_valence"].to_numpy(dtype=np.float64) for split in ["train", "val", "test"]}
    for weight in weights:
        for smooth_window in smooth_windows:
            pred_cache = {split: raw_prediction(frames[split], weight, smooth_window) for split in ["train", "val", "test"]}
            for calibration in calibrations:
                stats = fit_stats_from_pred(pred_cache["val"], targets["val"], calibration)
                split_metrics = {
                    split: eval_pred(pred_cache[split], targets[split], calibration, stats)
                    for split in ["train", "val", "test"]
                }
                rows.append(
                    {
                        "split_name": split_dir,
                        "weight_tcn": weight,
                        "weight_bilstm": round(1.0 - weight, 2),
                        "calibration": calibration,
                        "smooth_window": smooth_window,
                        "stats": json.dumps(stats, ensure_ascii=True),
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows).sort_values(
        ["val_selection_score", "val_ccc", "val_pcc", "val_rmse", "val_mae"],
        ascending=[True, False, False, True, True],
    )
    selected = df.iloc[0].to_dict()
    stats = json.loads(selected["stats"])
    split_out_dir = OUT_DIR / selected["split_name"]
    split_out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        out = frames[split].copy()
        out["pred_valence"] = apply_prediction(
            out,
            float(selected["weight_tcn"]),
            str(selected["calibration"]),
            int(selected["smooth_window"]),
            stats,
        )
        out.to_csv(split_out_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_bilstm_test": {
            metric: float(selected[f"test_{metric}"] - baseline["bilstm"]["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_tcn_test": {
            metric: float(selected[f"test_{metric}"] - baseline["tcn"]["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"created_at": "2026-06-27", "num_splits": len(rows), "per_split": rows}
    variants = ["bilstm", "tcn", "selected"]
    for variant in variants:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"][variant]["test"][metric] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    bilstm_mean = {metric: out[f"bilstm_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    tcn_mean = {metric: out[f"tcn_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_bilstm_mean"] = {
        metric: float(selected_mean[metric] - bilstm_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    out["selected_delta_vs_tcn_mean"] = {
        metric: float(selected_mean[metric] - tcn_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p.name for p in BILSTM_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_ensemble_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
