#!/usr/bin/env python3
"""Fuse current best Valence ensemble with a regularized BiLSTM candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


CURRENT_DIR = ROOT / "outputs" / "valence_multisplit_ensemble_ccc_preserving_20260627"
REG_DIR = ROOT / "outputs" / "valence_multisplit_regularized_bilstm_20260628" / "bilstm_reg_h64_do045_wd001_ccc010_clip010_ext115"
OUT_DIR = ROOT / "outputs" / "valence_current_regularized_ensemble_20260628"

PREVIOUS_BEST = {
    "model_id": "pmemo_valence_multisplit_ensemble_ccc_preserving_20260627",
    "rmse": 0.20306577173557291,
    "mae": 0.1565756349725939,
    "pcc": 0.7772862928172749,
    "ccc": 0.7603571428432958,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        0.60 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 0.60 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.80 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 3.00 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def read_pair(split_dir: str, split: str) -> pd.DataFrame:
    cur = pd.read_csv(CURRENT_DIR / split_dir / f"{split}_predictions.csv")
    reg = pd.read_csv(REG_DIR / split_dir / f"{split}_predictions.csv")
    for df in [cur, reg]:
        df["musicId"] = df["musicId"].astype(str)
        df["time_key"] = df["time_sec"].round(6)
        df["target_key"] = df["target_valence"].round(6)
    merged = cur.loc[:, ["musicId", "time_key", "target_key", "time_sec", "target_valence", "pred_valence"]].merge(
        reg.loc[:, ["musicId", "time_key", "target_key", "pred_valence"]],
        on=["musicId", "time_key", "target_key"],
        how="inner",
        suffixes=("_current", "_regularized"),
    )
    if len(merged) != len(cur) or len(merged) != len(reg):
        raise ValueError(f"Alignment mismatch: {split_dir} {split}: current={len(cur)} reg={len(reg)} merged={len(merged)}")
    out = merged.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
    out["pred_current"] = merged["pred_valence_current"].to_numpy(dtype=np.float64)
    out["pred_regularized"] = merged["pred_valence_regularized"].to_numpy(dtype=np.float64)
    return out


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if np.std(pred) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(np.float64), target.astype(np.float64), 1)
    return float(a), float(b)


def raw_pred(df: pd.DataFrame, weight_regularized: float) -> np.ndarray:
    return (
        (1.0 - weight_regularized) * df["pred_current"].to_numpy(dtype=np.float64)
        + weight_regularized * df["pred_regularized"].to_numpy(dtype=np.float64)
    )


def fit_stats(pred: np.ndarray, target: np.ndarray, calibration: str) -> dict[str, float]:
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


def apply_calibration(pred: np.ndarray, calibration: str, stats: dict[str, float]) -> np.ndarray:
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


def evaluate_pred(df: pd.DataFrame, weight_regularized: float, calibration: str, stats: dict[str, float]) -> dict[str, float]:
    pred = apply_calibration(raw_pred(df, weight_regularized), calibration, stats)
    return metrics(pred, df["target_valence"].to_numpy(dtype=np.float64))


def evaluate_split(split_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: read_pair(split_dir, split) for split in ["train", "val", "test"]}
    baselines = {
        "current": {
            split: metrics(frames[split]["pred_current"].to_numpy(dtype=np.float64), frames[split]["target_valence"].to_numpy(dtype=np.float64))
            for split in ["train", "val", "test"]
        },
        "regularized": {
            split: metrics(frames[split]["pred_regularized"].to_numpy(dtype=np.float64), frames[split]["target_valence"].to_numpy(dtype=np.float64))
            for split in ["train", "val", "test"]
        },
    }
    rows = []
    for weight_regularized in [round(x, 2) for x in np.linspace(0.0, 1.0, 101)]:
        val_raw = raw_pred(frames["val"], weight_regularized)
        val_target = frames["val"]["target_valence"].to_numpy(dtype=np.float64)
        for calibration in ["identity", "bias", "linear", "mean_std"]:
            stats = fit_stats(val_raw, val_target, calibration)
            split_metrics = {
                split: evaluate_pred(frames[split], weight_regularized, calibration, stats)
                for split in ["train", "val", "test"]
            }
            rows.append(
                {
                    "split_name": split_dir,
                    "weight_current": round(1.0 - weight_regularized, 2),
                    "weight_regularized": weight_regularized,
                    "calibration": calibration,
                    "stats": json.dumps(stats, ensure_ascii=True),
                    **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                    "val_selection_score": selection_score(split_metrics["val"]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_ccc", "val_pcc", "val_rmse"], ascending=[True, False, False, True])
    selected = df.iloc[0].to_dict()
    stats = json.loads(selected["stats"])
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        out = frames[split].copy()
        out["pred_valence"] = apply_calibration(
            raw_pred(out, float(selected["weight_regularized"])),
            str(selected["calibration"]),
            stats,
        )
        out.to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baselines": baselines,
        "selected": selected,
        "delta_vs_current_test": {
            metric: float(selected[f"test_{metric}"] - baselines["current"]["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "delta_vs_regularized_test": {
            metric: float(selected[f"test_{metric}"] - baselines["regularized"]["test"][metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"created_at": "2026-06-28", "previous_best": PREVIOUS_BEST, "num_splits": len(rows), "per_split": rows}
    for variant in ["current", "regularized", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baselines"][variant]["test"][metric] for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_previous_best"] = {
        metric: float(selected_mean[metric] - PREVIOUS_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidate_rows = []
    for split_dir in sorted(p.name for p in CURRENT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidate_rows.append(df)
    pd.concat(candidate_rows, ignore_index=True).to_csv(OUT_DIR / "all_ensemble_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
