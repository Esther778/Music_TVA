#!/usr/bin/env python3
"""OpenSMILE top-k frame residual on the current HGB+shift+smoothing Valence best."""

from __future__ import annotations

import search_pmemo_valence_best_after_smoothing_opensmile_topk_residual as mod


mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_hgb_smoothing_opensmile_topk_residual_20260709"
mod.MODEL_ID = "pmemo_valence_hgb_smoothing_opensmile_topk_residual_20260709"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


if __name__ == "__main__":
    mod.main()
