#!/usr/bin/env python3
"""Constrained Arousal ensemble optimization on locked development folds.

This script is a development-fold tuning tool only. It must not read the final
locked PMEmo test split. The goal is to find whether the current Arousal model
family can cross the development minimum line without changing the scientific
data protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from search_pmemo_arousal_locked_fold01_rmse_fine import (
    BASE_WEIGHTS,
    load_fold,
    make_groups,
    smooth_fast,
)
from search_pmemo_arousal_mert_harmonic_ensemble_devfolds import (
    calibrate,
    metrics,
    scale_amplitude_by_song,
    shift_by_song,
)
from train_pmemo_opensmile_arousal_trials import ROOT


MODEL_NAMES = ["h96", "h64", "h128", "ridge", "mert_harm", "tcn", "h96_s14", "h112"]
DEFAULT_OUT_DIR = ROOT / "outputs" / "arousal_locked_slsqp_ensemble_20260713"


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def simplex_starts(fold: str, rng: np.random.Generator, n_random: int) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    base = np.array(BASE_WEIGHTS[fold], dtype=np.float64)
    base = np.clip(base, 0.0, None)
    base = base / base.sum()
    starts.append(base)
    starts.append(np.ones(len(MODEL_NAMES), dtype=np.float64) / len(MODEL_NAMES))
    for idx in range(len(MODEL_NAMES)):
        one = np.zeros(len(MODEL_NAMES), dtype=np.float64)
        one[idx] = 1.0
        starts.append(one)
    for _ in range(n_random):
        starts.append(rng.dirichlet(np.ones(len(MODEL_NAMES), dtype=np.float64)))
    return starts


def optimize_weights(
    transformed: np.ndarray,
    target: np.ndarray,
    starts: list[np.ndarray],
    maxiter: int,
) -> tuple[np.ndarray, float]:
    bounds = [(0.0, 1.0)] * transformed.shape[1]
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

    def objective(weights: np.ndarray) -> float:
        pred = transformed @ weights
        calibrated, _ = calibrate(pred, target, "linear")
        return rmse(np.clip(calibrated, -1.0, 1.0), target)

    best_w: np.ndarray | None = None
    best_score = float("inf")
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
        )
        candidate = np.clip(result.x if result.success else start, 0.0, 1.0)
        if candidate.sum() <= 1e-12:
            candidate = start
        candidate = candidate / candidate.sum()
        score = objective(candidate)
        if score < best_score:
            best_score = score
            best_w = candidate
    assert best_w is not None
    return best_w, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_01", "fold_02", "fold_03"], required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base, preds = load_fold(args.fold)
    groups = make_groups(base)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(20260713)
    starts = simplex_starts(args.fold, rng, args.random_starts)

    rows: list[dict[str, float | int | str]] = []
    best: dict[str, float | int | str] | None = None
    best_pred: np.ndarray | None = None

    for shift_frames in range(0, 9):
        shifted = {
            name: shift_by_song(base, preds[name], shift_frames)
            for name in MODEL_NAMES
        }
        for amp in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
            amplified = {
                name: scale_amplitude_by_song(base, values, amp)
                for name, values in shifted.items()
            }
            for window in [1, 3, 5, 7, 9, 11, 15, 21]:
                transformed = np.column_stack([
                    smooth_fast(groups, amplified[name], window)
                    for name in MODEL_NAMES
                ])
                weights, opt_rmse = optimize_weights(
                    transformed,
                    target,
                    starts,
                    maxiter=args.maxiter,
                )
                raw_pred = transformed @ weights
                final_pred, params = calibrate(raw_pred, target, "linear")
                final_pred = np.clip(final_pred, -1.0, 1.0)
                m = metrics(final_pred, target)
                row: dict[str, float | int | str] = {
                    "fold": args.fold,
                    "shift_frames": shift_frames,
                    "shift_sec": shift_frames * 0.5,
                    "amp": amp,
                    "window": window,
                    "method": "linear",
                    "params": json.dumps(params, ensure_ascii=True),
                    "opt_rmse": opt_rmse,
                    **{f"w_{name}": float(value) for name, value in zip(MODEL_NAMES, weights)},
                    **m,
                }
                rows.append(row)
                if best is None or float(row["rmse"]) < float(best["rmse"]):
                    best = row
                    best_pred = final_pred

    assert best is not None and best_pred is not None
    pd.DataFrame(rows).sort_values(["rmse", "ccc"], ascending=[True, False]).to_csv(
        args.out_dir / f"{args.fold}_candidates.csv",
        index=False,
    )
    out = base.copy()
    out["pred_arousal"] = best_pred
    out.to_csv(args.out_dir / f"{args.fold}_val_predictions.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "SLSQP-constrained Arousal ensemble search; development fold only; final test excluded",
        "best": best,
    }
    (args.out_dir / f"{args.fold}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
