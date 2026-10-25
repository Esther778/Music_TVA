#!/usr/bin/env python3
"""Monotonic calibration wrapper for the extra-wide shifted+smoothing Valence best."""

from __future__ import annotations

import search_pmemo_valence_monotonic_calibration as mod


mod.BASE_DIR = mod.ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_extra_wide_monotonic_calibration_20260708"
mod.MODEL_ID = "pmemo_valence_extra_wide_monotonic_calibration_20260708"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


if __name__ == "__main__":
    mod.main()
