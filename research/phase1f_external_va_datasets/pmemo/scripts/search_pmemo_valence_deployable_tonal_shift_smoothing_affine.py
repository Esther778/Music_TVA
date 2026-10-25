#!/usr/bin/env python3
"""Affine calibration for the deployable tonal shift+smoothing Valence chain."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_affine_calibration.py"

spec = importlib.util.spec_from_file_location("affine_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

affine_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affine_base)

affine_base.INPUT_DIR = (
    affine_base.ROOT / "outputs" / "valence_deployable_tonal_shift_smoothing_20260708"
)
affine_base.OUT_DIR = (
    affine_base.ROOT / "outputs" / "valence_deployable_tonal_shift_smoothing_affine_20260708"
)
affine_base.CURRENT_BEST = {
    "model_id": "pmemo_valence_deployable_tonal_shift_smoothing_20260708",
    "rmse": 0.18890273524857215,
    "mae": 0.14659169034413505,
    "pcc": 0.8117208849284808,
    "ccc": 0.796520332088558,
}


if __name__ == "__main__":
    affine_base.main()
