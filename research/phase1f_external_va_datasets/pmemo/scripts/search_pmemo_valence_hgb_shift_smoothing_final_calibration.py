#!/usr/bin/env python3
"""Final output calibration after HGB song-mean, temporal shift, and smoothing."""

from __future__ import annotations

import search_pmemo_valence_final_output_calibration as base


base.INPUT_DIR = base.ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
base.OUT_DIR = base.ROOT / "outputs" / "valence_hgb_shift_smoothing_final_calibration_20260708"
base.CREATED_AT = "2026-07-08"
base.MODEL_ID = "pmemo_valence_hgb_shift_smoothing_final_calibration_20260708"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


if __name__ == "__main__":
    base.run()
