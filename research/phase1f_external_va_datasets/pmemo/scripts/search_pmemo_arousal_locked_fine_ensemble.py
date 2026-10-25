#!/usr/bin/env python3
"""Fine Arousal ensemble search on final-locked development folds only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_arousal_mert_harmonic_ensemble_devfolds import (
    calibrate,
    metrics,
    pred_col,
    scale_amplitude_by_song,
    shift_by_song,
    smooth_by_song,
    threshold_rows,
)
from train_pmemo_opensmile_arousal_trials import ROOT


OUT_DIR = ROOT / "outputs" / "arousal_locked_fine_fiveway_ensemble_20260713"


def input_paths(fold: str) -> dict[str, Path]:
    return {
        "h96": ROOT / "outputs" / "arousal_locked_devfolds_h96_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / "val_predictions.csv",
        "h64": ROOT / "outputs" / "arousal_locked_devfolds_h64_20260713" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003" / "val_predictions.csv",
        "h128": ROOT / "outputs" / "arousal_locked_devfolds_h128_20260713" / fold / "a_dev_h128_l1_do020_lr6e4_mse" / "val_predictions.csv",
        "ridge": ROOT / "outputs" / "opensmile_ridge_locked_devfolds_20260713" / "arousal" / fold / "val_predictions.csv",
        "mert_harm": ROOT / "outputs" / "arousal_mert_harmonic_locked_devfolds_20260713" / fold / "val_predictions.csv",
    }


def load_fold(fold: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {name: pd.read_csv(path) for name, path in input_paths(fold).items()}
    base = frames["h96"][["musicId", "time_sec", "target_arousal"]].copy()
    for name, frame in frames.items():
        pcol = pred_col(frame)
        base = base.merge(
            frame[["musicId", "time_sec", pcol]].rename(columns={pcol: f"pred_{name}"}),
            on=["musicId", "time_sec"],
            how="inner",
        )
    preds = {name: base[f"pred_{name}"].to_numpy(dtype=np.float64) for name in frames}
    return base[["musicId", "time_sec", "target_arousal"]].copy(), preds


def score(row: dict[str, float]) -> float:
    return row["rmse"] + 0.05 * max(0.0, row["mae"] - 0.15) + 0.02 * max(0.0, 0.80 - row["ccc"])


def weight_grid(fold: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    # Keep the grid constrained around the branch mix that already worked:
    # MERT+harmony carries correlation; Ridge and h96/h128 reduce RMSE.
    for wm in np.arange(0.30, 0.71, 0.05):
        for wr in np.arange(0.10, 0.46, 0.05):
            for wh96 in np.arange(0.05, 0.41, 0.05):
                for wh128 in np.arange(0.00, 0.31, 0.05):
                    wh64 = 1.0 - wm - wr - wh96 - wh128
                    if wh64 < 0.0 or wh64 > 0.12:
                        continue
                    if fold == "fold_01" and wm > 0.60:
                        continue
                    rows.append({
                        "h96": float(wh96),
                        "h64": float(wh64),
                        "h128": float(wh128),
                        "ridge": float(wr),
                        "mert_harm": float(wm),
                    })
    return rows


def search_fold(fold: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base, preds = load_fold(fold)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    fold_shift = {"fold_01": range(5, 16), "fold_02": range(3, 14), "fold_03": range(8, 19)}[fold]
    fold_windows = {
        "fold_01": [31, 51, 71, 91, 111, 131, 151],
        "fold_02": [11, 21, 31, 41, 51, 71, 91],
        "fold_03": [21, 31, 41, 51, 61, 71, 91, 111],
    }[fold]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for weights in weight_grid(fold):
        raw = sum(weights[name] * preds[name] for name in ["h96", "h64", "h128", "ridge", "mert_harm"])
        for shift_frames in fold_shift:
            shifted = shift_by_song(base, raw, shift_frames)
            for amp in [0.55, 0.70, 0.85, 1.0]:
                amplified = scale_amplitude_by_song(base, shifted, amp)
                for window in fold_windows:
                    smoothed = smooth_by_song(base, amplified, window)
                    for method in ["linear", "mean_std"]:
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
                            **{f"w_{name}": weights[name] for name in ["h96", "h64", "h128", "ridge", "mert_harm"]},
                            **m,
                        }
                        row["score"] = score(row)
                        rows.append(row)
                        if best is None or (row["score"], row["rmse"], -row["ccc"]) < (best["score"], best["rmse"], -best["ccc"]):
                            best = row
                            best_pred = final
    assert best is not None and best_pred is not None
    pred_out = base.copy()
    pred_out["pred_arousal"] = best_pred
    return best, pd.DataFrame(rows), pred_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_fold = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        best, rows, pred_out = search_fold(fold)
        rows.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(OUT_DIR / f"{fold}_candidates.csv", index=False)
        pred_out.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({k: v for k, v in best.items() if k != "params"})
    mean = {m: float(np.mean([float(row[m]) for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Fine five-way Arousal ensemble on final-locked development folds; final test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "per_fold": per_fold,
        "mean": mean,
        "gap_rows": threshold_rows(mean),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(per_fold).to_csv(OUT_DIR / "comparison.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
