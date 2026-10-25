#!/usr/bin/env python3
"""Run the HGB song-mean correction search on the latest Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_extra_wide_hgb_song_mean.py"

spec = importlib.util.spec_from_file_location("hgb_song_mean", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_second_residual_smoothing_20260709"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_latest_hgb_song_mean_20260709"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "rmse": 0.1878117801132452,
    "mae": 0.14589902068622534,
    "pcc": 0.8140948778047831,
    "ccc": 0.8013555689217714,
}


if __name__ == "__main__":
    mod.main()
