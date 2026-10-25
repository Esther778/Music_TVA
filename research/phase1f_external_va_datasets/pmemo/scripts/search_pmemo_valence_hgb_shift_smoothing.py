#!/usr/bin/env python3
"""Validation-selected smoothing after HGB song-mean temporal shift."""

from __future__ import annotations

import search_pmemo_valence_current_best_smoothing as mod


mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_hgb_temporal_shift_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_temporal_shift_20260708",
    "rmse": 0.18825628308337505,
    "mae": 0.14662070061648871,
    "pcc": 0.813905998094333,
    "ccc": 0.7996942621388534,
}


if __name__ == "__main__":
    mod.main()
