#!/usr/bin/env python3
"""Focused Valence ensemble search on locked development folds.

This script is intentionally local around an existing locked-dev ensemble
summary. It never reads the final locked test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import (
    calibrate,
    load_fold,
    scale_amplitude_by_song,
    shift_by_song,
    smooth_by_song,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = (
    ROOT
    / "outputs"
    / "valence_locked_audio_ensemble_postprocess_tflstm_allfolds_20260713"
    / "summary.json"
)
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_audio_ensemble_focused_local_20260713"


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["pcc"])
        + 0.05 * max(0.0, 0.80 - m["ccc"])
    )


def normalize_weights(row: dict[str, float], names: list[str]) -> np.ndarray:
    values = np.array([float(row.get(f"w_{name}", 0.0)) for name in names], dtype=np.float64)
    if values.sum() <= 0:
        values[:] = 1.0 / len(values)
    return values / values.sum()


def focused_weights(base: np.ndarray, samples: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = [base]
    # Include simple shrinkage variants around the current solution.
    uniform = np.ones_like(base) / len(base)
    for blend in [0.05, 0.10, 0.20, 0.35]:
        rows.append((1.0 - blend) * base + blend * uniform)
    concentration_grid = [80.0, 160.0, 320.0]
    for i in range(samples):
        concentration = concentration_grid[i % len(concentration_grid)]
        alpha = np.maximum(base * concentration, 0.05)
        rows.append(rng.dirichlet(alpha))
    return rows


def local_choices(center: int, radius: int, lo: int, hi: int) -> list[int]:
    return list(range(max(lo, center - radius), min(hi, center + radius) + 1))


def search_fold(
    base_row: dict[str, float],
    samples: int,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    fold = str(base_row["fold"])
    frame, preds = load_fold(fold)
    names = list(preds.keys())
    target = frame["target_valence"].to_numpy(dtype=np.float64)
    base_weight = normalize_weights(base_row, names)
    shift_center = int(base_row["shift_frames"])
    amp_center = float(base_row["amp"])
    window_center = int(base_row["window"])

    shift_choices = local_choices(shift_center, radius=4, lo=-16, hi=16)
    amp_choices = sorted(
        set(
            round(max(0.55, min(1.20, amp_center + delta)), 2)
            for delta in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        )
    )
    window_choices = sorted(set([1, 3, 5, 7, 11, 15, window_center]))
    method_choices = ["linear", "mean_std", "bias"]

    best = None
    best_pred = None
    for weights in focused_weights(base_weight, samples=samples, seed=seed):
        raw = sum(float(weight) * preds[name] for weight, name in zip(weights, names))
        for shift_frames in shift_choices:
            shifted = shift_by_song(frame, raw, shift_frames)
            for amp in amp_choices:
                amplified = scale_amplitude_by_song(frame, shifted, amp)
                for window in window_choices:
                    smoothed = smooth_by_song(frame, amplified, window)
                    for method in method_choices:
                        final, params = calibrate(smoothed, target, method)
                        final = np.clip(final, -1.0, 1.0)
                        m = metrics(final, target)
                        row = {
                            "fold": fold,
                            "shift_frames": shift_frames,
                            "shift_sec": shift_frames * 0.5,
                            "amp": amp,
                            "window": window,
                            "method": method,
                            "params": json.dumps(params, ensure_ascii=True),
                            **{f"w_{name}": float(weight) for weight, name in zip(weights, names)},
                            **m,
                        }
                        row["score"] = score(m)
                        key = (row["score"], row["rmse"], -row["ccc"])
                        if best is None or key < (best["score"], best["rmse"], -best["ccc"]):
                            best = row
                            best_pred = final
    assert best is not None and best_pred is not None
    out = frame.copy()
    out["pred_valence"] = best_pred
    return best, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    base = json.loads(args.base_summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    frames = []
    for idx, row in enumerate(base["per_fold"]):
        best, pred = search_fold(row, samples=args.samples, seed=args.seed + idx)
        per_fold.append(best)
        frames.append(pred)
        pred.to_csv(args.out_dir / f"{best['fold']}_val_predictions.csv", index=False)

    macro = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(frames, ignore_index=True)
    global_frame = metrics(
        merged["pred_valence"].to_numpy(dtype=np.float64),
        merged["target_valence"].to_numpy(dtype=np.float64),
    )
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Focused local Valence ensemble/postprocess search on locked development folds; final locked test excluded",
        "base_summary": str(args.base_summary),
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "per_fold": per_fold,
        "macro_mean": macro,
        "global_frame": global_frame,
        "gap_rows_macro": threshold_rows(macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
