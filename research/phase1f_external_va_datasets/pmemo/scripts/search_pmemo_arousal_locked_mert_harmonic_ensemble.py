#!/usr/bin/env python3
"""Locked-development-fold Arousal ensemble search.

This wrapper reuses the 2026-07-09 five-way Arousal ensemble search logic, but
redirects every input to the 2026-07-13 final-locked development folds. It must
not read the locked final test split.
"""

from __future__ import annotations

from pathlib import Path

import search_pmemo_arousal_mert_harmonic_ensemble_devfolds as base


ROOT = base.ROOT

base.OUT_DIR = ROOT / "outputs" / "arousal_locked_mert_harmonic_fiveway_shift_20260713"

base.FOLD_CONFIGS = {
    "fold_01": {
        "base_weights": {"h96": 0.45, "h64": 0.05, "h128": 0.15, "ridge": 0.35},
        "shift_frames": 13,
        "windows": [71, 91, 111, 131],
    },
    "fold_02": {
        "base_weights": {"h96": 0.45, "h64": 0.05, "h128": 0.10, "ridge": 0.40},
        "shift_frames": 7,
        "windows": [21, 31, 51, 71, 91],
    },
    "fold_03": {
        "base_weights": {"h96": 0.25, "h64": 0.05, "h128": 0.35, "ridge": 0.35},
        "shift_frames": 16,
        "windows": [51, 61, 71, 91, 111],
    },
}

base.CURRENT_BEST = {
    "model_id": "locked_devfold_best_single_branch_before_ensemble",
    "mean": {
        "rmse": 0.213675,
        "mae": 0.168158,
        "pcc": 0.793739,
        "ccc": 0.782024,
    },
}


def input_paths(fold: str) -> dict[str, Path]:
    return {
        "h96": ROOT / "outputs" / "arousal_locked_devfolds_h96_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse" / "val_predictions.csv",
        "h64": ROOT / "outputs" / "arousal_locked_devfolds_h64_20260713" / fold / "a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003" / "val_predictions.csv",
        "h128": ROOT / "outputs" / "arousal_locked_devfolds_h128_20260713" / fold / "a_dev_h128_l1_do020_lr6e4_mse" / "val_predictions.csv",
        "ridge": ROOT / "outputs" / "opensmile_ridge_locked_devfolds_20260713" / "arousal" / fold / "val_predictions.csv",
        "mert_harm": ROOT / "outputs" / "arousal_mert_harmonic_locked_devfolds_20260713" / fold / "val_predictions.csv",
    }


base.input_paths = input_paths


if __name__ == "__main__":
    base.main()
