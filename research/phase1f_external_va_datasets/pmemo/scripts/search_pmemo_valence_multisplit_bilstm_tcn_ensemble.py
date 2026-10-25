#!/usr/bin/env python3
"""Validation-selected BiLSTM + TCN ensemble for PMEmo Valence multi-split outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BILSTM_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
TCN_DIR = ROOT / "outputs" / "valence_multisplit_tcn_20260627"
OUT_DIR = ROOT / "outputs" / "valence_multisplit_bilstm_tcn_ensemble_20260627"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.10 * m["rmse"]
    )


def read_pair(split_dir: str, split: str) -> pd.DataFrame:
    b = pd.read_csv(BILSTM_DIR / split_dir / f"{split}_predictions.csv")
    t = pd.read_csv(TCN_DIR / split_dir / f"{split}_predictions.csv")
    key_cols = ["musicId", "time_sec", "target_valence"]
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
        raise ValueError(f"Prediction alignment mismatch: {split_dir} {split}: bilstm={len(b)} tcn={len(t)} merged={len(merged)}")
    out = merged.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
    out["pred_bilstm"] = merged["pred_valence_bilstm"].to_numpy(dtype=np.float64)
    out["pred_tcn"] = merged["pred_valence_tcn"].to_numpy(dtype=np.float64)
    return out


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if np.std(pred) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(np.float64), target.astype(np.float64), 1)
    return float(a), float(b)


def apply_prediction(df: pd.DataFrame, weight_tcn: float, kind: str, stats: dict[str, float]) -> np.ndarray:
    pred = (1.0 - weight_tcn) * df["pred_bilstm"].to_numpy(dtype=np.float64) + weight_tcn * df["pred_tcn"].to_numpy(dtype=np.float64)
    if kind == "identity":
        out = pred
    elif kind == "bias":
        out = pred + stats["bias"]
    elif kind == "linear":
        out = stats["a"] * pred + stats["b"]
    elif kind == "mean_std":
        out = (pred - stats["pred_mean"]) / stats["pred_std"] * stats["target_std"] + stats["target_mean"]
    else:
        raise ValueError(kind)
    return np.clip(out, -1.0, 1.0)


def fit_stats(val_df: pd.DataFrame, weight_tcn: float, kind: str) -> dict[str, float]:
    pred = (1.0 - weight_tcn) * val_df["pred_bilstm"].to_numpy(dtype=np.float64) + weight_tcn * val_df["pred_tcn"].to_numpy(dtype=np.float64)
    target = val_df["target_valence"].to_numpy(dtype=np.float64)
    if kind == "identity":
        return {}
    if kind == "bias":
        return {"bias": float(np.mean(target - pred))}
    if kind == "linear":
        a, b = fit_linear(pred, target)
        return {"a": a, "b": b}
    if kind == "mean_std":
        return {
            "pred_mean": float(np.mean(pred)),
            "pred_std": float(np.std(pred) + 1e-12),
            "target_mean": float(np.mean(target)),
            "target_std": float(np.std(target) + 1e-12),
        }
    raise ValueError(kind)


def eval_candidate(df: pd.DataFrame, weight_tcn: float, kind: str, stats: dict[str, float]) -> dict[str, float]:
    pred = apply_prediction(df, weight_tcn, kind, stats)
    target = df["target_valence"].to_numpy(dtype=np.float64)
    return metrics(pred, target)


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
    weights = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
    kinds = ["identity", "bias", "linear", "mean_std"]
    for weight in weights:
        for kind in kinds:
            stats = fit_stats(frames["val"], weight, kind)
            split_metrics = {split: eval_candidate(frames[split], weight, kind, stats) for split in ["train", "val", "test"]}
            rows.append(
                {
                    "split_name": split_dir,
                    "weight_tcn": weight,
                    "weight_bilstm": round(1.0 - weight, 2),
                    "calibration": kind,
                    "stats": json.dumps(stats, ensure_ascii=True),
                    **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                    "val_selection_score": selection_score(split_metrics["val"]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    stats = json.loads(selected["stats"])
    split_out_dir = OUT_DIR / selected["split_name"]
    split_out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        out = frames[split].copy()
        out["pred_valence"] = apply_prediction(out, float(selected["weight_tcn"]), str(selected["calibration"]), stats)
        out.to_csv(split_out_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_bilstm_test": {
            "rmse": float(selected["test_rmse"] - baseline["bilstm"]["test"]["rmse"]),
            "mae": float(selected["test_mae"] - baseline["bilstm"]["test"]["mae"]),
            "pcc": float(selected["test_pcc"] - baseline["bilstm"]["test"]["pcc"]),
            "ccc": float(selected["test_ccc"] - baseline["bilstm"]["test"]["ccc"]),
        },
        "delta_vs_tcn_test": {
            "rmse": float(selected["test_rmse"] - baseline["tcn"]["test"]["rmse"]),
            "mae": float(selected["test_mae"] - baseline["tcn"]["test"]["mae"]),
            "pcc": float(selected["test_pcc"] - baseline["tcn"]["test"]["pcc"]),
            "ccc": float(selected["test_ccc"] - baseline["tcn"]["test"]["ccc"]),
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
