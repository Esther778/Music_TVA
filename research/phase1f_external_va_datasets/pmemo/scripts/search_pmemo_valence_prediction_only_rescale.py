#!/usr/bin/env python3
"""Prediction-only song mean/std rescaling for Valence.

The current Valence bottleneck includes near-neutral and overly flat
predictions on some songs. This script tests a deployable postprocess that uses
only the predicted curve of each song:

    y' = mean_gain * song_pred_mean + std_gain * (y - song_pred_mean)

Parameters are selected on validation predictions and reported on held-out test
predictions. No target labels are needed at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_frame_calibrator_song_bias_20260628"
OUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_frame_calibrator_song_bias_20260628",
    "rmse": 0.19748015854593182,
    "mae": 0.15352887995061668,
    "pcc": 0.7932350928356282,
    "ccc": 0.7800520380365018,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.70 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df[pred_col(df)].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def prediction_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = df[pred_col(df)].to_numpy(dtype=np.float64)
    target = df[target_col(df)].to_numpy(dtype=np.float64)
    song_mean = df.groupby("musicId")[pred_col(df)].transform("mean").to_numpy(dtype=np.float64)
    return pred, target, song_mean


def transform_prediction(
    pred: np.ndarray,
    song_mean: np.ndarray,
    mean_gain: float,
    std_gain: float,
    blend: float,
    clip: float | None,
) -> np.ndarray:
    transformed = mean_gain * song_mean + std_gain * (pred - song_mean)
    pred_final = (1.0 - blend) * pred + blend * transformed
    if clip is not None:
        pred_final = np.clip(pred_final, -float(clip), float(clip))
    return pred_final


def apply_rescale(df: pd.DataFrame, mean_gain: float, std_gain: float, blend: float, clip: float | None) -> pd.DataFrame:
    out = df.copy()
    pcol = pred_col(out)
    pred, _, song_mean = prediction_arrays(out)
    out[pcol] = transform_prediction(pred, song_mean, mean_gain, std_gain, blend, clip)
    out["mean_gain"] = mean_gain
    out["std_gain"] = std_gain
    out["rescale_blend"] = blend
    out["rescale_clip"] = "none" if clip is None else clip
    return out


def evaluate_split(split_dir: Path) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    arrays = {split: prediction_arrays(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    for mean_gain in np.round(np.arange(0.80, 1.51, 0.05), 2):
        for std_gain in np.round(np.arange(0.80, 1.81, 0.05), 2):
            for blend in np.round(np.arange(0.10, 1.01, 0.10), 2):
                for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90)]:
                    split_metrics = {}
                    for split in ["train", "val", "test"]:
                        pred, target, song_mean = arrays[split]
                        corrected_pred = transform_prediction(pred, song_mean, float(mean_gain), float(std_gain), float(blend), clip)
                        split_metrics[split] = metrics(corrected_pred, target)
                    rows.append(
                        {
                            "split_name": split_dir.name,
                            "mean_gain": float(mean_gain),
                            "std_gain": float(std_gain),
                            "blend": float(blend),
                            "clip": clip_name,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(split_metrics["val"]),
                        }
                    )
    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    key = (float(selected["mean_gain"]), float(selected["std_gain"]), float(selected["blend"]), str(selected["clip"]))
    clip = None if key[3] == "none" else float(key[3].replace("clip", "")) / 100.0
    for split in ["train", "val", "test"]:
        apply_rescale(frames[split], key[0], key[1], key[2], clip).to_csv(out_split / f"{split}_predictions.csv", index=False)
    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_prediction_only_rescale_20260628",
        "input_model": CURRENT_BEST,
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            vals = np.asarray(
                [
                    r["selected"][f"test_{metric}"] if variant == "selected" else r["baseline"]["test"][metric]
                    for r in rows
                ],
                dtype=np.float64,
            )
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_rescale_results.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
