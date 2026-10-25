#!/usr/bin/env python3
"""Paired song bootstrap comparison for two locked-development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_REFERENCE = ROOT / "outputs/valence_locked_musicnn_dynamic_tag_past_dense_monotonic_20260714"
DEFAULT_CANDIDATE = ROOT / "outputs/valence_locked_discogs_effnet_dynamic_fine_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_discogs_effnet_dynamic_fine_20260714/bootstrap_vs_previous.json"


def sufficient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = prediction - target
    return np.array([
        len(target), target.sum(), prediction.sum(), np.square(target).sum(),
        np.square(prediction).sum(), (target * prediction).sum(),
        np.abs(error).sum(), np.square(error).sum(),
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


def load_fold(reference_dir: Path, candidate_dir: Path, fold: str) -> tuple[np.ndarray, np.ndarray]:
    reference = pd.read_csv(reference_dir / f"{fold}_val_predictions.csv")
    candidate = pd.read_csv(candidate_dir / f"{fold}_val_predictions.csv")
    aligned = reference[["musicId", "time_sec", "target_valence", "pred_valence"]].merge(
        candidate[["musicId", "time_sec", "pred_valence"]],
        on=["musicId", "time_sec"], how="inner", validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(aligned) != len(reference):
        raise ValueError(f"Incomplete alignment for {fold}")
    reference_stats, candidate_stats = [], []
    for _, song in aligned.groupby("musicId", sort=False):
        target = song["target_valence"].to_numpy(dtype=np.float64)
        reference_stats.append(sufficient(target, song["pred_valence_reference"].to_numpy(dtype=np.float64)))
        candidate_stats.append(sufficient(target, song["pred_valence_candidate"].to_numpy(dtype=np.float64)))
    return np.stack(reference_stats), np.stack(candidate_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resamples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fold_stats = {fold: load_fold(args.reference_dir, args.candidate_dir, fold) for fold in FOLDS}
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

    names = ["rmse", "mae", "pcc", "ccc"]
    comparison = {}
    for index, name in enumerate(names):
        values = deltas[:, index]
        lower_is_better = name in {"rmse", "mae"}
        probability = float(np.mean(values < 0.0) if lower_is_better else np.mean(values > 0.0))
        comparison[name] = {
            "mean_delta_candidate_minus_reference": float(values.mean()),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "improvement_probability": probability,
        }
    summary = {
        "created_at": "2026-07-14",
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
