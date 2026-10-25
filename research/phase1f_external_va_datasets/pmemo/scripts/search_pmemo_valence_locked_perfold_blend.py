#!/usr/bin/env python3
"""Per-fold blend diagnostic for locked-dev Valence calibration.

This is a development-only postprocess search. It tests whether fold-specific
mean/std blend strengths can close CCC gaps without excessive RMSE loss.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import calibrate, smooth_by_song
from search_pmemo_valence_locked_shift_postprocess import shift_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_locked_arousal_assisted_crossfold_20260713"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_perfold_blend_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: float(np.mean([row[m] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.05 * max(0.0, 0.80 - m["pcc"])
        + 0.13 * max(0.0, 0.80 - m["ccc"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_data = {}
    for fold in FOLDS:
        df = pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv")
        pred = df[pred_col(df)].to_numpy(dtype=np.float64)
        target = df[target_col(df)].to_numpy(dtype=np.float64)
        shifted = shift_by_song(df, pred, -12)
        smoothed = smooth_by_song(df, shifted, 1)
        low, _ = calibrate(smoothed, target, "linear")
        high, _ = calibrate(smoothed, target, "mean_std")
        fold_data[fold] = {
            "df": df,
            "target": target,
            "low": np.clip(low, -1.0, 1.0),
            "high": np.clip(high, -1.0, 1.0),
        }

    blend_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    rows = []
    pred_cache = {}
    for b1, b2, b3 in itertools.product(blend_grid, repeat=3):
        blends = {"fold_01": b1, "fold_02": b2, "fold_03": b3}
        per_fold = []
        preds = {}
        for fold in FOLDS:
            data = fold_data[fold]
            blend = blends[fold]
            pred = (1.0 - blend) * data["low"] + blend * data["high"]
            pred = np.clip(pred, -1.0, 1.0)
            preds[fold] = pred
            per_fold.append(metrics(pred, data["target"]))
        m = macro(per_fold)
        row = {
            "blend_fold_01": b1,
            "blend_fold_02": b2,
            "blend_fold_03": b3,
            **m,
            "score": score(m),
        }
        rows.append(row)
        pred_cache[(b1, b2, b3)] = preds

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "all_perfold_blend_candidates.csv", index=False)

    score_selected = result.sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    rmse_selected = result.sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    ccc_selected = result.sort_values(["ccc", "rmse"], ascending=[False, True]).iloc[0].to_dict()

    def materialize(name: str, selected: dict[str, Any]) -> dict[str, Any]:
        sub = args.out_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        key = (selected["blend_fold_01"], selected["blend_fold_02"], selected["blend_fold_03"])
        preds = pred_cache[key]
        frames = []
        per_fold = []
        for fold in FOLDS:
            df = fold_data[fold]["df"].copy()
            df["base_pred_valence"] = df[pred_col(df)].to_numpy(dtype=np.float64)
            df["pred_valence"] = preds[fold]
            df.to_csv(sub / f"{fold}_val_predictions.csv", index=False)
            m = metrics(df["pred_valence"].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))
            per_fold.append({"fold": fold, **m})
            frames.append(df)
        selected_macro = macro(per_fold)
        merged = pd.concat(frames, ignore_index=True)
        global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged[target_col(merged)].to_numpy(dtype=np.float64))
        return {
            "selected_config": selected,
            "per_fold": per_fold,
            "macro_mean": selected_macro,
            "global_frame": global_frame,
            "gap_rows_macro": threshold_rows(selected_macro),
            "gap_rows_global_frame": threshold_rows(global_frame),
        }

    summary = {
        "created_at": "2026-07-13",
        "purpose": "Development-only per-fold Valence mean/std blend diagnostic; final locked test excluded",
        "selection_boundary": "Development-fold postprocess search only; not final locked test performance",
        "input_dir": str(args.input_dir),
        "fixed_preprocess": "shift_frames=-12, window=1, linear low-error calibration + mean/std high-CCC calibration",
        "score_primary": materialize("score_primary", score_selected),
        "rmse_primary": materialize("rmse_primary", rmse_selected),
        "ccc_primary": materialize("ccc_primary", ccc_selected),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
