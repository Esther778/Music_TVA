from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from train_valence_harmonic_context import scalar_metrics


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)


def song_slices() -> list[slice]:
    split = pd.read_csv(ROOT / "data/shadow_split.csv")
    rows = (
        split[split["shadow_split"] == "shadow_val"]
        .sort_values("song_id")
    )
    result = []
    start = 0
    for length in rows["frames"].astype(int):
        result.append(slice(start, start + length))
        start += length
    return result


def transform(
    prediction: np.ndarray,
    target: np.ndarray,
    slices: list[slice],
    shift: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, targets = [], []
    for song_slice in slices:
        song_prediction = prediction[song_slice]
        song_target = target[song_slice]
        if sigma > 0:
            song_prediction = gaussian_filter1d(
                song_prediction, sigma=sigma, mode="nearest"
            )
        if shift > 0:
            song_prediction = song_prediction[:-shift]
            song_target = song_target[shift:]
        elif shift < 0:
            song_prediction = song_prediction[-shift:]
            song_target = song_target[:shift]
        predictions.append(song_prediction)
        targets.append(song_target)
    return np.concatenate(predictions), np.concatenate(targets)


def affine_calibration(prediction: np.ndarray, target: np.ndarray):
    covariance = np.mean(
        (prediction - prediction.mean()) * (target - target.mean())
    )
    variance = prediction.var()
    slope = max(float(covariance / max(variance, 1e-8)), 0.0)
    intercept = float(target.mean() - slope * prediction.mean())
    return slope, intercept, slope * prediction + intercept


def main() -> None:
    baseline = np.load(ROOT / "outputs/shadow_valence_baseline/shadow_pred.npy").reshape(-1)
    trend = np.load(ROOT / "outputs/shadow_valence_trend/shadow_pred.npy").reshape(-1)
    target = np.load(ROOT / "outputs/shadow_valence_baseline/shadow_target.npy").reshape(-1)
    trend_target = np.load(ROOT / "outputs/shadow_valence_trend/shadow_target.npy").reshape(-1)
    if not np.allclose(target, trend_target, atol=1e-7):
        raise ValueError("Baseline and trend shadow targets differ.")
    slices = song_slices()
    if slices[-1].stop != len(target):
        raise ValueError("Song slices do not cover shadow predictions.")
    scale = float(target.std())
    rows = []
    for trend_weight in np.arange(0.0, 1.0001, 0.05):
        prediction = (1 - trend_weight) * baseline + trend_weight * trend
        for shift in range(-8, 9):
            for sigma in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
                aligned_prediction, aligned_target = transform(
                    prediction, target, slices, shift, sigma
                )
                for calibrated in (False, True):
                    slope, intercept = 1.0, 0.0
                    values = aligned_prediction
                    if calibrated:
                        slope, intercept, values = affine_calibration(
                            aligned_prediction, aligned_target
                        )
                    result = scalar_metrics(values, aligned_target)
                    balanced = (
                        0.35 * result["rmse"] / scale
                        + 0.15 * result["mae"] / scale
                        + 0.25 * (1 - result["pcc"])
                        + 0.25 * (1 - result["ccc"])
                    )
                    rows.append(
                        {
                            "trend_weight": float(trend_weight),
                            "baseline_weight": float(1 - trend_weight),
                            "shift_frames": int(shift),
                            "sigma_frames": float(sigma),
                            "calibrated": calibrated,
                            "slope": slope,
                            "intercept": intercept,
                            "frames": int(len(values)),
                            "balanced_objective": float(balanced),
                            **result,
                        }
                    )
    result = {
        "selection_split": "shadow_val",
        "frozen_validation_used": False,
        "test_used": False,
        "shift_definition": "positive means prediction is matched to later target",
        "best_rmse": min(rows, key=lambda row: row["rmse"]),
        "best_ccc": max(rows, key=lambda row: row["ccc"]),
        "best_balanced": min(rows, key=lambda row: row["balanced_objective"]),
        "raw_baseline": scalar_metrics(baseline, target),
        "raw_trend": scalar_metrics(trend, target),
    }
    output = ROOT / "outputs/shadow_valence_postprocess.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
