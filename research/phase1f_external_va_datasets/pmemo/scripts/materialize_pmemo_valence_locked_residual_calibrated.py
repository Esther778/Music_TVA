#!/usr/bin/env python3
"""Materialize calibrated locked-dev Valence residual predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import calibrate, smooth_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_locked_residual_refine_xwide_20260713"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_residual_refine_xwide_calibrated_20260713"


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.02 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["pcc"])
        + 0.08 * max(0.0, 0.80 - m["ccc"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_fold = []
    frames = []
    for fold in ["fold_01", "fold_02", "fold_03"]:
        df = pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv")
        pred = df["pred_valence"].to_numpy(dtype=np.float64)
        target = df["target_valence"].to_numpy(dtype=np.float64)
        rows = []
        best = None
        best_pred = None
        for window in [1, 3, 5, 7, 11, 21, 41]:
            smoothed = smooth_by_song(df, pred, window)
            for method in ["identity", "bias", "linear", "mean_std"]:
                pp, params = calibrate(smoothed, target, method)
                pp = np.clip(pp, -1.0, 1.0)
                m = metrics(pp, target)
                row = {
                    "fold": fold,
                    "window": window,
                    "method": method,
                    "params": json.dumps(params, ensure_ascii=True),
                    **m,
                }
                row["score"] = score(m)
                rows.append(row)
                key = (row["score"], row["rmse"], -row["ccc"])
                if best is None or key < (best["score"], best["rmse"], -best["ccc"]):
                    best = row
                    best_pred = pp
        assert best is not None and best_pred is not None
        pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False]).to_csv(
            args.out_dir / f"{fold}_calibration_candidates.csv",
            index=False,
        )
        out = df.copy()
        out["residual_pred_valence"] = pred
        out["pred_valence"] = best_pred
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append(best)
        frames.append(out)

    macro = {m: float(np.mean([row[m] for row in per_fold])) for m in ["rmse", "mae", "pcc", "ccc"]}
    merged = pd.concat(frames, ignore_index=True)
    global_frame = metrics(merged["pred_valence"].to_numpy(dtype=np.float64), merged["target_valence"].to_numpy(dtype=np.float64))
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Materialized calibrated Valence residual refinement on locked development folds; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "input_dir": str(args.input_dir),
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
