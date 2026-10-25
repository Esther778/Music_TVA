#!/usr/bin/env python3
"""Refine locked-dev Arousal ensemble after adding MERT+OpenSMILE branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_arousal_locked_fold01_rmse_fine import load_fold, make_groups, smooth_fast
from search_pmemo_arousal_mert_harmonic_ensemble_devfolds import (
    calibrate,
    metrics,
    pred_col,
    scale_amplitude_by_song,
    shift_by_song,
)
from train_pmemo_opensmile_arousal_trials import ROOT


OUT_DIR = ROOT / "outputs" / "arousal_locked_mert_smile_refine_20260713"
MERT_SMILE_DIR = ROOT / "outputs" / "arousal_locked_mert_opensmile_fusion_20260713"
NAMES = ["h96", "h64", "h128", "ridge", "mert_harm", "tcn", "h96_s14", "h112", "mert_smile"]
BASE = {
    "fold_01": [0.1261358598210049, 0.0003456818733913758, 0.0, 0.4721255856119412, 0.39376744013735737, 0.007625432556305254, 0.0, 0.0, 0.0],
    "fold_02": [0.22232774845665926, 0.007631981245436543, 0.0, 0.08307017680935504, 0.48837290910698594, 0.0, 0.1290192945269845, 0.06957788985457876, 0.0],
    "fold_03": [0.02721153856113269, 0.0, 0.0, 0.26064391695299943, 0.4329653819232755, 0.16839949363741827, 0.0, 0.11077966892517409, 0.0],
}


def load_with_mert_smile(fold: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base, preds = load_fold(fold)
    branch = pd.read_csv(MERT_SMILE_DIR / fold / "val_predictions.csv")
    pcol = pred_col(branch)
    merged = base.merge(
        branch[["musicId", "time_sec", pcol]].rename(columns={pcol: "pred_mert_smile"}),
        on=["musicId", "time_sec"],
        how="inner",
    )
    preds = {name: values[: len(merged)] for name, values in preds.items()}
    # Re-merge all original predictions to preserve row alignment after inner join.
    aligned = base[["musicId", "time_sec", "target_arousal"]].merge(
        branch[["musicId", "time_sec", pcol]].rename(columns={pcol: "pred_mert_smile"}),
        on=["musicId", "time_sec"],
        how="inner",
    )
    original_base, original_preds = load_fold(fold)
    aligned = aligned[["musicId", "time_sec", "target_arousal"]].copy()
    for name, values in original_preds.items():
        tmp = original_base[["musicId", "time_sec"]].copy()
        tmp[f"pred_{name}"] = values
        aligned = aligned.merge(tmp, on=["musicId", "time_sec"], how="inner")
    aligned = aligned.merge(
        branch[["musicId", "time_sec", pcol]].rename(columns={pcol: "pred_mert_smile"}),
        on=["musicId", "time_sec"],
        how="inner",
    )
    return aligned[["musicId", "time_sec", "target_arousal"]].copy(), {
        name: aligned[f"pred_{name}"].to_numpy(dtype=np.float64) for name in NAMES
    }


def weight_rows(fold: str, samples: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(20260713)
    base = np.array(BASE[fold], dtype=np.float64)
    seed_w = {"fold_01": 0.214, "fold_02": 0.111, "fold_03": 0.123}[fold]
    base[:-1] = base[:-1] / max(base[:-1].sum(), 1e-12) * (1.0 - seed_w)
    base[-1] = seed_w
    rows = [{name: float(value) for name, value in zip(NAMES, base)}]
    for _ in range(samples):
        noise = rng.normal(0.0, [0.02, 0.008, 0.01, 0.025, 0.025, 0.015, 0.015, 0.015, 0.035])
        vec = np.clip(base + noise, 0.0, None)
        if vec.sum() <= 1e-12:
            continue
        vec = vec / vec.sum()
        rows.append({name: float(value) for name, value in zip(NAMES, vec)})
    return rows


def choices(fold: str) -> tuple[list[int], list[float], list[int]]:
    if fold == "fold_01":
        return [2, 3, 4], [0.25, 0.30, 0.35, 0.40], [7, 11, 15]
    if fold == "fold_02":
        return [6, 7, 8], [0.35, 0.40, 0.45], [11, 15, 21]
    return [5, 6, 7], [0.40, 0.45, 0.50, 0.55], [11, 15, 21]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_01", "fold_02", "fold_03"], required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--samples", type=int, default=700)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base, preds = load_with_mert_smile(args.fold)
    groups = make_groups(base)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    shift_choices, amp_choices, window_choices = choices(args.fold)
    rows = []
    best = None
    best_pred = None
    for weights in weight_rows(args.fold, args.samples):
        raw = sum(weights[name] * preds[name] for name in NAMES)
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
                        **{f"w_{name}": weights[name] for name in NAMES},
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
        "purpose": "MERT+OpenSMILE branch added to raw Arousal ensemble search; development folds only; final test excluded",
        "best": best,
    }
    (args.out_dir / f"{args.fold}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
