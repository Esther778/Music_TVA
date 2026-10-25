#!/usr/bin/env python3
"""Affine calibration after song-mean correction on the closest Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_affine_calibration.py"

spec = importlib.util.spec_from_file_location("affine_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_best_after_smoothing_song_mean_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_best_after_smoothing_song_mean_affine_20260708"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_best_after_smoothing_song_mean_20260708",
    "rmse": 0.1887237192520319,
    "mae": 0.14659872590934805,
    "pcc": 0.8122448827043954,
    "ccc": 0.7965267808300491,
}


if __name__ == "__main__":
    mod.main()
