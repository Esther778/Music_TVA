#!/usr/bin/env python3
"""Validation-selected time-shift postprocess for PMEmo Valence predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_regularized_stack_ensemble_20260628"
OUT_DIR = ROOT / "outputs" / "valence_shift_postprocess_20260628"
PREVIOUS_BEST = {
    "model_id": "pmemo_valence_regularized_stack_ensemble_20260628",
    "rmse": 0.20213179319130736,
    "mae": 0.1561975245353329,
    "pcc": 0.7793220265608104,
    "ccc": 0.76463269438581,
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


def shift_array_edge(pred: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return pred.copy()
    idx = np.arange(len(pred)) + shift_frames
    idx = np.clip(idx, 0, len(pred) - 1)
    return pred[idx]


def apply_shift(df: pd.DataFrame, shift_frames: int) -> pd.DataFrame:
    rows = []
    for _, group in df.sort_values(["musicId", "time_sec"]).groupby("musicId", sort=False):
        out = group.copy()
        out["pred_valence"] = shift_array_edge(out["pred_valence"].to_numpy(dtype=np.float64), shift_frames)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def eval_shift(df: pd.DataFrame, shift_frames: int) -> tuple[dict[str, float], pd.DataFrame]:
    shifted = apply_shift(df, shift_frames)
    m = metrics(shifted["pred_valence"].to_numpy(dtype=np.float64), shifted["target_valence"].to_numpy(dtype=np.float64))
    return m, shifted


def evaluate_split(split_dir: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = {split: pd.read_csv(INPUT_DIR / split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: metrics(frames[split]["pred_valence"].to_numpy(dtype=np.float64), frames[split]["target_valence"].to_numpy(dtype=np.float64)) for split in ["train", "val", "test"]}
    rows = []
    shifted_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for shift_frames in range(-8, 9):
        split_metrics = {}
        for split in ["train", "val", "test"]:
            m, shifted = eval_shift(frames[split], shift_frames)
            split_metrics[split] = m
            shifted_cache[(split, shift_frames)] = shifted
        rows.append(
            {
                "split_name": split_dir,
                "shift_frames": shift_frames,
                "shift_sec": shift_frames * 0.5,
                **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                "val_selection_score": selection_score(split_metrics["val"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_ccc", "val_pcc", "val_rmse"], ascending=[True, False, False, True])
    selected = df.iloc[0].to_dict()
    split_out = OUT_DIR / selected["split_name"]
    split_out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        shifted_cache[(split, int(selected["shift_frames"]))].to_csv(split_out / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {
            metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    return summary, df


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"created_at": "2026-06-28", "previous_best": PREVIOUS_BEST, "num_splits": len(rows), "per_split": rows}
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
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
    candidates = []
    for split_dir in sorted(p.name for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_shift_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
