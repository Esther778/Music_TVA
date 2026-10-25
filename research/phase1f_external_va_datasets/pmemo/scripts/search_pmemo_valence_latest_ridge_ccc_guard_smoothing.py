#!/usr/bin/env python3
"""Smoothing search on the validation CCC-guarded Ridge Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_smoothing.py"

spec = importlib.util.spec_from_file_location("smoothing", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_latest_ridge_ccc_guard_20260709"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_latest_ridge_ccc_guard_smoothing_20260709"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_latest_ridge_ccc_guard_20260709",
    "rmse": 0.18762729443949744,
    "mae": 0.14601122611013231,
    "pcc": 0.8141790484400776,
    "ccc": 0.8002456735609883,
}


if __name__ == "__main__":
    mod.main()
