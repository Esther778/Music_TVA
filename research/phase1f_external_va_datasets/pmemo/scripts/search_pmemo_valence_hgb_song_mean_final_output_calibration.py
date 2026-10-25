#!/usr/bin/env python3
"""Run final output calibration on the HGB song-mean Valence candidate."""

from __future__ import annotations

import search_pmemo_valence_final_output_calibration as base


base.INPUT_DIR = base.ROOT / "outputs" / "valence_extra_wide_hgb_song_mean_20260708"
base.OUT_DIR = base.ROOT / "outputs" / "valence_hgb_song_mean_final_output_calibration_20260708"
base.CREATED_AT = "2026-07-08"
base.MODEL_ID = "pmemo_valence_hgb_song_mean_final_output_calibration_20260708"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_hgb_song_mean_20260708",
    "rmse": 0.1886336187692369,
    "mae": 0.14671297812161363,
    "pcc": 0.8128437820931764,
    "ccc": 0.7988240399841687,
}


if __name__ == "__main__":
    base.run()
