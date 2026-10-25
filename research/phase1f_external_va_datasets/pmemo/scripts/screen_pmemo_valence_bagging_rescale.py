#!/usr/bin/env python3
"""Apply prediction-only rescale to a screened Valence bagging ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_seed_bagging_screen_20260628" / "split_01_ensemble_3seeds"
DEFAULT_OUT = ROOT / "outputs" / "valence_seed_bagging_rescale_screen_20260628"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.70 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def prediction_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = df["pred_valence"].to_numpy(dtype=np.float64)
    target = df["target_valence"].to_numpy(dtype=np.float64)
    song_mean = df.groupby("musicId")["pred_valence"].transform("mean").to_numpy(dtype=np.float64)
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
    out = (1.0 - blend) * pred + blend * transformed
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    return out


def eval_split(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def apply_to_df(df: pd.DataFrame, selected: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    clip = None if selected["clip"] == "none" else float(str(selected["clip"]).replace("clip", "")) / 100.0
    pred, _, song_mean = prediction_arrays(out)
    out["pred_valence"] = transform_prediction(
        pred,
        song_mean,
        float(selected["mean_gain"]),
        float(selected["std_gain"]),
        float(selected["blend"]),
        clip,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: pd.read_csv(args.input_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    baseline = {split: eval_split(df) for split, df in frames.items()}
    arrays = {split: prediction_arrays(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    for mean_gain in np.round(np.arange(0.80, 1.51, 0.05), 2):
        for std_gain in np.round(np.arange(0.80, 1.81, 0.05), 2):
            for blend in np.round(np.arange(0.10, 1.01, 0.10), 2):
                for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90)]:
                    split_metrics = {}
                    for split in ["train", "val", "test"]:
                        pred, target, song_mean = arrays[split]
                        corrected = transform_prediction(pred, song_mean, float(mean_gain), float(std_gain), float(blend), clip)
                        split_metrics[split] = metrics(corrected, target)
                    rows.append(
                        {
                            "mean_gain": float(mean_gain),
                            "std_gain": float(std_gain),
                            "blend": float(blend),
                            "clip": clip_name,
                            **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                            "val_selection_score": selection_score(split_metrics["val"]),
                        }
                    )
    candidates = pd.DataFrame(rows).sort_values(
        ["val_selection_score", "val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, True, False],
    )
    selected = candidates.iloc[0].to_dict()
    candidates.to_csv(args.out_dir / "all_rescale_candidates.csv", index=False)
    for split, df in frames.items():
        apply_to_df(df, selected).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    selected_metrics = {split: eval_split(pd.read_csv(args.out_dir / f"{split}_predictions.csv")) for split in ["train", "val", "test"]}
    summary = {
        "created_at": "2026-06-28",
        "purpose": "fair split_01 screen of seed-bagged Valence ensemble with prediction-only rescale",
        "input_dir": str(args.input_dir),
        "baseline": baseline,
        "selected": selected,
        "selected_metrics": selected_metrics,
        "test_gap_rows": threshold_rows(selected_metrics["test"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
