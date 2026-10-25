#!/usr/bin/env python3
"""Generic locked-dev shift/smooth/calibration search for Valence predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import calibrate, smooth_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_locked_arousal_assisted_crossfold_20260713"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_arousal_assisted_shift_postprocess_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]


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


def shift_by_song(df: pd.DataFrame, values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0:
        return values.copy()
    out = np.empty_like(values, dtype=np.float64)
    for _, idx in df.groupby("musicId", sort=False).groups.items():
        ii = np.fromiter(idx, dtype=np.int64)
        seg = values[ii]
        if len(seg) <= 1:
            out[ii] = seg
            continue
        if frames > 0:
            k = min(frames, len(seg) - 1)
            shifted = np.empty_like(seg)
            shifted[:-k] = seg[k:]
            shifted[-k:] = seg[-1]
        else:
            k = min(-frames, len(seg) - 1)
            shifted = np.empty_like(seg)
            shifted[:k] = seg[0]
            shifted[k:] = seg[:-k]
        out[ii] = shifted
    return out


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.05 * max(0.0, 0.80 - m["pcc"])
        + 0.10 * max(0.0, 0.80 - m["ccc"])
    )


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selection", choices=["score", "rmse"], default="score")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_frames = {fold: pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}

    rows = []
    best = None
    best_key = None
    best_preds: dict[str, np.ndarray] | None = None
    for shift in [-32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12]:
        for window in [1, 3, 5, 7, 11, 21]:
            for method in ["identity", "bias", "linear", "mean_std"]:
                for blend_mean_std in [0.0, 0.1, 0.2] if method == "linear" else [0.0]:
                    per_fold = []
                    preds: dict[str, np.ndarray] = {}
                    for fold, df in fold_frames.items():
                        base = df[pred_col(df)].to_numpy(dtype=np.float64)
                        target = df[target_col(df)].to_numpy(dtype=np.float64)
                        shifted = shift_by_song(df, base, shift)
                        smoothed = smooth_by_song(df, shifted, window)
                        pp, _ = calibrate(smoothed, target, method)
                        if method == "linear" and blend_mean_std > 0:
                            ms, _ = calibrate(smoothed, target, "mean_std")
                            pp = (1.0 - blend_mean_std) * pp + blend_mean_std * ms
                        pp = np.clip(pp, -1.0, 1.0)
                        preds[fold] = pp
                        per_fold.append(metrics(pp, target))
                    m = macro(per_fold)
                    row = {
                        "shift_frames": int(shift),
                        "shift_sec": float(shift * 0.5),
                        "window": int(window),
                        "method": method,
                        "mean_std_blend": float(blend_mean_std),
                        **m,
                        "score": score(m),
                    }
                    rows.append(row)
                    if args.selection == "rmse":
                        key = (row["rmse"], row["mae"], -row["pcc"], -row["ccc"])
                    else:
                        key = (row["score"], row["rmse"], -row["ccc"])
                    if best_key is None or key < best_key:
                        best = row
                        best_key = key
                        best_preds = preds
    assert best is not None and best_preds is not None
    pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(
        args.out_dir / "all_shift_postprocess_candidates.csv",
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
        "purpose": "Locked-dev shift/smooth/calibration search for Valence; final locked test excluded",
        "selection_boundary": "Development-fold postprocess search only; not final locked test performance",
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
