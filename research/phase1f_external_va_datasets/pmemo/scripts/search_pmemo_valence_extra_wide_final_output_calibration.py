#!/usr/bin/env python3
"""Run final output calibration on the current extra-wide Valence best."""

from __future__ import annotations

import search_pmemo_valence_final_output_calibration as base


base.INPUT_DIR = base.ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
base.OUT_DIR = base.ROOT / "outputs" / "valence_extra_wide_final_output_calibration_20260708"
base.CREATED_AT = "2026-07-08"
base.MODEL_ID = "pmemo_valence_extra_wide_final_output_calibration_20260708"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


if __name__ == "__main__":
    base.run()
