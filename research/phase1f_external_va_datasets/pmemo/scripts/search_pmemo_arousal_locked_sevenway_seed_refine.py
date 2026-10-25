#!/usr/bin/env python3
"""Small seven-way Arousal ensemble refinement adding h96 seed14."""

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
from search_pmemo_arousal_locked_sixway_tcn_refine import input_paths as sixway_paths
from train_pmemo_opensmile_arousal_trials import ROOT


BASE_DIR = ROOT / "outputs" / "arousal_locked_sixway_tcn_refine_20260713"
OUT_DIR = ROOT / "outputs" / "arousal_locked_sevenway_seed_refine_20260713"
NAMES = ["h96", "h64", "h128", "ridge", "mert_harm", "tcn", "h96_s14"]


def input_paths(fold: str) -> dict[str, Path]:
    paths = sixway_paths(fold)
    paths["h96_s14"] = ROOT / "outputs" / "arousal_locked_devfolds_h96_seed14_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / "val_predictions.csv"
    return paths


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
    return base[["musicId", "time_sec", "target_arousal"]].copy(), {
        name: base[f"pred_{name}"].to_numpy(dtype=np.float64) for name in NAMES
    }


def score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.03 * max(0.0, m["mae"] - 0.15) + 0.01 * max(0.0, 0.80 - m["ccc"])


def search_fold(fold: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base, preds = load_fold(fold)
    target = base["target_arousal"].to_numpy(dtype=np.float64)
    seeds = pd.read_csv(BASE_DIR / f"{fold}_candidates.csv").head(5)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None
    for _, seed in seeds.iterrows():
        old_names = [name for name in NAMES if name != "h96_s14"]
        old = {name: float(seed[f"w_{name}"]) for name in old_names}
        old_total = sum(old.values())
        old = {name: value / old_total for name, value in old.items()}
        for ws14 in [0.0, 0.03, 0.06, 0.10, 0.15, 0.20]:
            weights = {name: old[name] * (1.0 - ws14) for name in old}
            weights["h96_s14"] = ws14
            raw = sum(weights[name] * preds[name] for name in NAMES)
            for shift_frames in range(max(0, int(seed["shift_frames"]) - 1), int(seed["shift_frames"]) + 2):
                shifted = shift_by_song(base, raw, shift_frames)
                for amp in [0.55, 0.60, 0.65]:
                    amplified = scale_amplitude_by_song(base, shifted, amp)
                    for window in sorted({max(1, int(seed["window"]) + d) for d in [-10, 0, 10]}):
                        smoothed = smooth_by_song(base, amplified, window)
                        final, params = calibrate(smoothed, target, "linear")
                        final = np.clip(final, -1.0, 1.0)
                        m = metrics(final, target)
                        item = {
                            "fold": fold,
                            "shift_frames": shift_frames,
                            "shift_sec": shift_frames * 0.5,
                            "amp": amp,
                            "window": window,
                            "method": "linear",
                            "params": json.dumps(params, ensure_ascii=True),
                            **{f"w_{name}": float(weights[name]) for name in NAMES},
                            **m,
                        }
                        item["score"] = score(item)
                        rows.append(item)
                        if best is None or (item["score"], item["rmse"], -item["ccc"]) < (best["score"], best["rmse"], -best["ccc"]):
                            best = item
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
        "purpose": "Small seven-way Arousal ensemble refinement with h96 seed14; final test excluded",
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
