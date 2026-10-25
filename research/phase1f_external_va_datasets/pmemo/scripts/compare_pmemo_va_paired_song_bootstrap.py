#!/usr/bin/env python3
"""Paired song bootstrap comparison for locked-development VA predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FOLDS = ["fold_01", "fold_02", "fold_03"]


def sufficient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = prediction - target
    return np.array([
        len(target),
        target.sum(),
        prediction.sum(),
        np.square(target).sum(),
        np.square(prediction).sum(),
        (target * prediction).sum(),
        np.abs(error).sum(),
        np.square(error).sum(),
    ], dtype=np.float64)


def metric_from_stats(stats: np.ndarray) -> np.ndarray:
    n, sy, sp, sy2, sp2, syp, sae, sse = stats
    my, mp = sy / n, sp / n
    vy = max(sy2 / n - my * my, 0.0)
    vp = max(sp2 / n - mp * mp, 0.0)
    cov = syp / n - my * mp
    pcc = cov / max(np.sqrt(vy * vp), 1e-12)
    ccc = 2.0 * cov / max(vy + vp + (my - mp) ** 2, 1e-12)
    return np.array([np.sqrt(sse / n), sae / n, pcc, ccc], dtype=np.float64)


def load_fold(
    reference_dir: Path,
    candidate_dir: Path,
    fold: str,
    dimension: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_col = f"target_{dimension}"
    pred_col = f"pred_{dimension}"
    reference = pd.read_csv(reference_dir / f"{fold}_val_predictions.csv")
    candidate = pd.read_csv(candidate_dir / f"{fold}_val_predictions.csv")
    aligned = reference[["musicId", "time_sec", target_col, pred_col]].merge(
        candidate[["musicId", "time_sec", pred_col]],
        on=["musicId", "time_sec"],
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(aligned) != len(reference) or len(aligned) != len(candidate):
        raise ValueError(f"Incomplete alignment for {fold}")
    reference_stats, candidate_stats = [], []
    for _, song in aligned.groupby("musicId", sort=False):
        target = song[target_col].to_numpy(dtype=np.float64)
        reference_stats.append(sufficient(target, song[f"{pred_col}_reference"].to_numpy(dtype=np.float64)))
        candidate_stats.append(sufficient(target, song[f"{pred_col}_candidate"].to_numpy(dtype=np.float64)))
    return np.stack(reference_stats), np.stack(candidate_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=["valence", "arousal"], required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fold_stats = {
        fold: load_fold(args.reference_dir, args.candidate_dir, fold, args.dimension)
        for fold in FOLDS
    }
    rng = np.random.default_rng(args.seed)
    deltas = np.zeros((args.resamples, 4), dtype=np.float64)
    for sample in range(args.resamples):
        reference_metrics, candidate_metrics = [], []
        for fold in FOLDS:
            reference, candidate = fold_stats[fold]
            indices = rng.integers(0, len(reference), size=len(reference))
            reference_metrics.append(metric_from_stats(reference[indices].sum(axis=0)))
            candidate_metrics.append(metric_from_stats(candidate[indices].sum(axis=0)))
        deltas[sample] = np.mean(candidate_metrics, axis=0) - np.mean(reference_metrics, axis=0)

    comparison = {}
    for index, name in enumerate(["rmse", "mae", "pcc", "ccc"]):
        values = deltas[:, index]
        lower_is_better = name in {"rmse", "mae"}
        probability = float(np.mean(values < 0.0) if lower_is_better else np.mean(values > 0.0))
        comparison[name] = {
            "mean_delta_candidate_minus_reference": float(values.mean()),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "improvement_probability": probability,
        }
    summary = {
        "created_at": "2026-07-15",
        "dimension": args.dimension,
        "unit": "song-stratified within each of three development folds",
        "reference_dir": str(args.reference_dir),
        "candidate_dir": str(args.candidate_dir),
        "resamples": args.resamples,
        "seed": args.seed,
        "comparison": comparison,
    }
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
