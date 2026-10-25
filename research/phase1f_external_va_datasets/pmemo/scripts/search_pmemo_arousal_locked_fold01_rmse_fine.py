#!/usr/bin/env python3
"""RMSE-focused fold_01 ensemble search for locked-dev Arousal."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_arousal_mert_harmonic_ensemble_devfolds import (
    calibrate,
    metrics,
    pred_col,
    scale_amplitude_by_song,
    shift_by_song,
)
from search_pmemo_arousal_locked_eightway_h112_refine import input_paths as eightway_paths
from train_pmemo_opensmile_arousal_trials import ROOT


DEFAULT_OUT_DIR = ROOT / "outputs" / "arousal_locked_rmse_fine_20260713"
BASE_WEIGHTS = {
    "fold_01": [0.16628662418507945, 0.0, 0.0, 0.3981262439841684, 0.4355871318307522, 0.0, 0.0, 0.0],
    "fold_02": [0.22232774845665926, 0.007631981245436543, 0.0, 0.08307017680935504, 0.48837290910698594, 0.0, 0.1290192945269845, 0.06957788985457876],
    "fold_03": [0.07178304931074855, 0.0, 0.0021658375573599107, 0.2734454375249162, 0.44075566359022583, 0.14086147991749526, 0.0, 0.07098853209925444],
}


def load_fold(fold: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    paths = eightway_paths(fold)
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    base = frames["h96"][["musicId", "time_sec", "target_arousal"]].copy()
    for name, frame in frames.items():
        pcol = pred_col(frame)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    return base[["musicId", "time_sec", "target_arousal"]].copy(), {
        name: base[f"pred_{name}"].to_numpy(dtype=np.float64) for name in paths
    }


def normalized_weights(fold: str) -> list[dict[str, float]]:
    rng = np.random.default_rng(20260713)
    base = np.array(BASE_WEIGHTS[fold], dtype=np.float64)
    names = ["h96", "h64", "h128", "ridge", "mert_harm", "tcn", "h96_s14", "h112"]
    rows = [{name: float(value) for name, value in zip(names, base)}]
    for _ in range(800):
        noise = rng.normal(0.0, [0.025, 0.01, 0.012, 0.025, 0.025, 0.012, 0.012, 0.012])
        vec = np.clip(base + noise, 0.0, None)
        if vec.sum() <= 1e-9:
            continue
        vec = vec / vec.sum()
        rows.append({name: float(value) for name, value in zip(names, vec)})
    return rows


def make_groups(base: pd.DataFrame) -> list[np.ndarray]:
    return [group.index.to_numpy(dtype=np.int64) for _, group in base.groupby("musicId", sort=False)]


def smooth_fast(groups: list[np.ndarray], values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    for idx in groups:
        n = len(idx)
        seg = values[idx]
        if n == 0:
            continue
        if n < window:
            sm = np.full(n, float(seg.mean()), dtype=np.float64)
        else:
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded = np.pad(seg, (pad_left, pad_right), mode="edge")
            sm = np.convolve(padded, kernel, mode="valid")
        out[idx] = sm
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_01", "fold_02", "fold_03"], default="fold_01")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base, preds = load_fold(args.fold)
    groups = make_groups(base)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    rows = []
    best = None
    best_pred = None
    names = ["h96", "h64", "h128", "ridge", "mert_harm", "tcn", "h96_s14", "h112"]
    for weights in normalized_weights(args.fold):
        raw = sum(weights[name] * preds[name] for name in names)
        if args.fold == "fold_01":
            shift_choices = [2, 3, 4]
            amp_choices = [0.25, 0.30, 0.35, 0.40]
            window_choices = [7, 11, 15]
        elif args.fold == "fold_03":
            shift_choices = [5, 6, 7]
            amp_choices = [0.40, 0.45, 0.50, 0.55]
            window_choices = [11, 15, 21]
        else:
            shift_choices = [6, 7, 8]
            amp_choices = [0.35, 0.40, 0.45]
            window_choices = [11, 15]
        for shift_frames in shift_choices:
            shifted = shift_by_song(base, raw, shift_frames)
            for amp in amp_choices:
                amplified = scale_amplitude_by_song(base, shifted, amp)
                for window in window_choices:
                    smoothed = smooth_fast(groups, amplified, window)
                    final, params = calibrate(smoothed, target, "linear")
                    final = np.clip(final, -1.0, 1.0)
                    m = metrics(final, target)
                    row = {
                        "fold": args.fold,
                        "shift_frames": shift_frames,
                        "shift_sec": shift_frames * 0.5,
                        "amp": amp,
                        "window": window,
                        "method": "linear",
                        "params": json.dumps(params, ensure_ascii=True),
                        **{f"w_{name}": weights[name] for name in names},
                        **m,
                    }
                    rows.append(row)
                    if best is None or row["rmse"] < best["rmse"]:
                        best = row
                        best_pred = final
    assert best is not None and best_pred is not None
    pd.DataFrame(rows).sort_values(["rmse", "ccc"], ascending=[True, False]).to_csv(args.out_dir / f"{args.fold}_candidates.csv", index=False)
    out = base.copy()
    out["pred_arousal"] = best_pred
    out.to_csv(args.out_dir / f"{args.fold}_val_predictions.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "RMSE-focused fold_01 fine search; development fold only; final test excluded",
        "best": best,
    }
    (args.out_dir / f"{args.fold}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
