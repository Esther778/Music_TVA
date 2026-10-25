#!/usr/bin/env python3
"""Blend low-error and high-CCC calibration for locked-dev Valence.

This is a development-only diagnostic. It starts from a locked-dev prediction
artifact and tests whether a small blend toward mean/std calibration can improve
CCC without destroying RMSE/MAE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import calibrate, smooth_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_locked_crossfold_stacker_20260713"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_ccc_blend_calibration_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.04 * max(0.0, 0.80 - m["pcc"])
        + 0.13 * max(0.0, 0.80 - m["ccc"])
    )


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_valence", "pred", "prediction"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    for col in ["target_valence", "target", "valence"]:
        if col in df.columns:
            return col
    raise ValueError(f"No target column in {list(df.columns)}")


def calibrated_pair(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    base_pred = df[pred_col(df)].to_numpy(dtype=np.float64)
    target = df[target_col(df)].to_numpy(dtype=np.float64)
    smoothed = smooth_by_song(df, base_pred, window)
    low_err, _ = calibrate(smoothed, target, "linear")
    high_ccc, _ = calibrate(smoothed, target, "mean_std")
    return np.clip(low_err, -1.0, 1.0), np.clip(high_ccc, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_frames = {fold: pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    best = None
    best_preds: dict[str, np.ndarray] | None = None
    for window in [1, 3, 5, 7, 11]:
        for blend in [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]:
            per_fold = []
            preds: dict[str, np.ndarray] = {}
            for fold, df in fold_frames.items():
                key = (fold, window)
                if key not in cache:
                    cache[key] = calibrated_pair(df, window)
                low_err, high_ccc = cache[key]
                pred = np.clip((1.0 - blend) * low_err + blend * high_ccc, -1.0, 1.0)
                preds[fold] = pred
                per_fold.append(metrics(pred, df[target_col(df)].to_numpy(dtype=np.float64)))
            m = macro(per_fold)
            row = {"window": window, "mean_std_blend": float(blend), **m, "score": score(m)}
            rows.append(row)
            key_score = (row["score"], row["rmse"], -row["ccc"])
            if best is None or key_score < (best["score"], best["rmse"], -best["ccc"]):
                best = row
                best_preds = preds

    assert best is not None and best_preds is not None
    pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(
        args.out_dir / "all_ccc_blend_candidates.csv",
        index=False,
    )
    per_fold_out = []
    frames_out = []
    for fold, df in fold_frames.items():
        out = df.copy()
        out["base_pred_valence"] = df[pred_col(df)].to_numpy(dtype=np.float64)
        out["pred_valence"] = best_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        m = metrics(out["pred_valence"].to_numpy(dtype=np.float64), out[target_col(out)].to_numpy(dtype=np.float64))
        per_fold_out.append({"fold": fold, **m})
        frames_out.append(out)
    selected_macro = macro(per_fold_out)
    merged = pd.concat(frames_out, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged[target_col(merged)].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Locked-dev blend between low-error linear calibration and high-CCC mean/std calibration; final locked test excluded",
        "selection_boundary": "Development-fold calibration search only; not final locked test performance",
        "input_dir": str(args.input_dir),
        "selected_config": best,
        "per_fold": per_fold_out,
        "macro_mean": selected_macro,
        "global_frame": global_frame,
        "gap_rows_macro": threshold_rows(selected_macro),
        "gap_rows_global_frame": threshold_rows(global_frame),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
