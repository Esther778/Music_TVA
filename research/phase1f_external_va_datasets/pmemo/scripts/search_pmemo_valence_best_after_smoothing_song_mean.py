#!/usr/bin/env python3
"""Song-level residual correction after the current closest Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_correlated_song_mean.py"

spec = importlib.util.spec_from_file_location("correlated_song_mean", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_best_after_smoothing_song_mean_20260708"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


if __name__ == "__main__":
    mod.main()
