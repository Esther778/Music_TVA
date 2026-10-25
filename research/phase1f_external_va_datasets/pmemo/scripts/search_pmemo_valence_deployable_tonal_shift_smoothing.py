#!/usr/bin/env python3
"""Smoothing on the deployable tonal temporal-shift Valence candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_smoothing.py"

spec = importlib.util.spec_from_file_location("smoothing_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

smoothing_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoothing_base)

smoothing_base.INPUT_DIR = (
    smoothing_base.ROOT / "outputs" / "valence_deployable_tonal_temporal_shift_extra_wide_20260708"
)
smoothing_base.OUT_DIR = (
    smoothing_base.ROOT / "outputs" / "valence_deployable_tonal_shift_smoothing_20260708"
)
smoothing_base.CURRENT_BEST = {
    "model_id": "pmemo_valence_deployable_tonal_temporal_shift_extra_wide_20260708",
    "rmse": 0.18894203430927717,
    "mae": 0.1466059369028961,
    "pcc": 0.8116413627342448,
    "ccc": 0.7964590302137723,
}


if __name__ == "__main__":
    smoothing_base.main()
