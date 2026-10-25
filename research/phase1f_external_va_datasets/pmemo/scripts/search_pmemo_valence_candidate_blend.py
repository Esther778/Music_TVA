#!/usr/bin/env python3
"""Validation-selected convex blending among strong Valence prediction artifacts."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_candidate_blend_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "rmse": 0.1878117801132452,
    "mae": 0.14589902068622534,
    "pcc": 0.8140948778047831,
    "ccc": 0.8013555689217714,
}

CANDIDATES = [
    {
        "name": "hgb_shift_smoothing",
        "dir": ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708",
    },
    {
        "name": "residual_smoothing",
        "dir": ROOT / "outputs" / "valence_residual_sequence_smoothing_20260709",
    },
    {
        "name": "second_residual_smoothing",
        "dir": ROOT / "outputs" / "valence_second_residual_smoothing_20260709",
    },
    {
        "name": "hgb_arousal_residual",
        "dir": ROOT / "outputs" / "valence_hgb_arousal_residual_20260709",
    },
    {
        "name": "hgb_shift_frame_residual",
        "dir": ROOT / "outputs" / "valence_hgb_shift_frame_residual_20260708",
    },
    {
        "name": "best_after_smoothing_frame_residual",
        "dir": ROOT / "outputs" / "valence_best_after_smoothing_frame_residual_20260708",
    },
]


def pred_column(df: pd.DataFrame) -> str:
    for col in ["pred_valence", "pred", "prediction", "calibrated_pred"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {list(df.columns)}")


def target_column(df: pd.DataFrame) -> str:
    for col in ["target_valence", "target", "valence"]:
        if col in df.columns:
            return col
    raise ValueError(f"No target column in {list(df.columns)}")


def load_split(split: str, data_split: str) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    dfs = []
    for cand in CANDIDATES:
        path = cand["dir"] / split / f"{data_split}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        keep = ["musicId", "time_sec", target_column(df), pred_column(df)]
        tmp = df[keep].copy()
        tmp.columns = ["musicId", "time_sec", "target", cand["name"]]
        dfs.append(tmp)
    merged = dfs[0]
    for cand, df in zip(CANDIDATES[1:], dfs[1:]):
        merged = merged.merge(df[["musicId", "time_sec", cand["name"]]], on=["musicId", "time_sec"], how="inner")
    merged = merged.sort_values(["musicId", "time_sec"]).reset_index(drop=True)
    preds = [merged[cand["name"]].to_numpy(dtype=np.float64) for cand in CANDIDATES]
    target = merged["target"].to_numpy(dtype=np.float64)
    return merged[["musicId", "time_sec", "target"]], target, preds


def candidate_weights() -> list[np.ndarray]:
    n = len(CANDIDATES)
    rows: list[np.ndarray] = []
    for i in range(n):
        w = np.zeros(n)
        w[i] = 1.0
        rows.append(w)
    for i, j in itertools.combinations(range(n), 2):
        for a in np.linspace(0.0, 1.0, 41):
            w = np.zeros(n)
            w[i] = float(a)
            w[j] = float(1.0 - a)
            rows.append(w)
    for i, j, k in itertools.combinations(range(n), 3):
        for a in np.linspace(0.0, 1.0, 11):
            for b in np.linspace(0.0, 1.0 - a, int(round((1.0 - a) * 10)) + 1):
                c = 1.0 - a - b
                w = np.zeros(n)
                w[i] = float(a)
                w[j] = float(b)
                w[k] = float(c)
                rows.append(w)
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for w in rows:
        key = tuple(np.round(w, 4))
        unique[key] = w
    return list(unique.values())


def selection_score(row: dict[str, float]) -> float:
    return (
        row["rmse"]
        + 0.03 * max(0.0, 0.80 - row["ccc"])
        + 0.02 * max(0.0, 0.80 - row["pcc"])
        + 0.005 * row["mae"]
    )


def apply_weights(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    arr = np.stack(preds, axis=1)
    return np.clip(arr @ weights, -1.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = candidate_weights()
    per_split: list[dict[str, Any]] = []
    split_frames = []
    for split in SPLITS:
        _, val_target, val_preds = load_split(split, "val")
        base_df, test_target, test_preds = load_split(split, "test")
        scored = []
        for w in weights:
            val_m = metrics(apply_weights(val_preds, w), val_target)
            scored.append({"weights": w, **val_m, "score": selection_score(val_m)})
        selected = min(scored, key=lambda x: x["score"])
        selected_weights = selected["weights"]
        test_pred = apply_weights(test_preds, selected_weights)
        test_m = metrics(test_pred, test_target)
        out_split = OUT_DIR / split
        out_split.mkdir(parents=True, exist_ok=True)
        pred_df = base_df.copy()
        pred_df["pred_valence"] = test_pred
        pred_df.to_csv(out_split / "test_predictions.csv", index=False)
        pd.DataFrame(
            [
                {
                    "candidate": cand["name"],
                    "weight": float(weight),
                }
                for cand, weight in zip(CANDIDATES, selected_weights)
            ]
        ).to_csv(out_split / "selected_weights.csv", index=False)
        per_split.append(
            {
                "split_name": split,
                "selected_weights": {
                    cand["name"]: float(weight)
                    for cand, weight in zip(CANDIDATES, selected_weights)
                    if abs(float(weight)) > 1e-8
                },
                **{f"val_{k}": float(selected[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
                "val_score": float(selected["score"]),
                **{f"test_{k}": float(v) for k, v in test_m.items()},
            }
        )
        split_frames.append((base_df.assign(pred_valence=test_pred), test_target, test_pred))

    selected_test = {
        metric: float(np.mean([row[f"test_{metric}"] for row in per_split]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    selected_val = {
        metric: float(np.mean([row[f"val_{metric}"] for row in per_split]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_candidate_blend_20260709",
        "purpose": "validation-selected convex blending among strong Valence artifacts",
        "selection_rule": "per-split blend weights selected on validation only; test labels unused for selection",
        "candidates": [
            {"name": cand["name"], "dir": str(cand["dir"].relative_to(ROOT))}
            for cand in CANDIDATES
        ],
        "current_best_reference": CURRENT_BEST,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "delta_vs_current_best": {
            metric: float(selected_test[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "threshold_gaps": threshold_rows(selected_test),
        "per_split": per_split,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
