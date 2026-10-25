#!/usr/bin/env python3
"""Guarded stacking around the current HGB+shift+smoothing Valence best."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_prediction_stacking_guarded.py"

spec = importlib.util.spec_from_file_location("stacking_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.OUT_DIR = base.OUTPUT_ROOT / "valence_hgb_guarded_stacking_20260709"
base.BASE_MODEL_DIR = base.OUTPUT_ROOT / "valence_hgb_shift_smoothing_20260708"
base.FALLBACK_BASE_MODEL_DIR = base.BASE_MODEL_DIR
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}

base.PREFERRED_MODELS = [
    "valence_hgb_shift_smoothing_20260708",
    "valence_hgb_temporal_shift_20260708",
    "valence_hgb_shift_frame_residual_20260708",
    "valence_hgb_frame_residual_temporal_shift_20260708",
    "valence_extra_wide_rmse_shift_smoothing_20260708",
    "valence_extra_wide_hgb_song_mean_20260708",
    "valence_current_best_opensmile_song_mean_20260708",
    "valence_opensmile_song_mean_affine_calibration_20260708",
    "valence_current_best_smoothing_affine_20260708",
    "valence_deployable_joint_postprocess_fast_20260708",
    "valence_per_split_direct_multilayer_ensemble_20260629",
    "valence_per_split_direct_multilayer_parsimony_20260629",
    "valence_prediction_only_rescale_ccc_preserving_20260628",
    "valence_regularized_stack_ensemble_20260628",
    "valence_multisplit_bilstm_tcn_ensemble_20260627",
    "valence_multisplit_best_20260627",
    "valence_multisplit_tcn_20260627",
    "valence_multisplit_full_librosa_20260628",
    "valence_opensmile_aux_ensemble_20260629",
]


def run() -> None:
    base.run()
    summary_path = base.OUT_DIR / "summary.json"
    text = summary_path.read_text(encoding="utf-8")
    text = text.replace("pmemo_valence_prediction_stacking_guarded_20260707", "pmemo_valence_hgb_guarded_stacking_20260709")
    text = text.replace("2026-07-07", "2026-07-09")
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
