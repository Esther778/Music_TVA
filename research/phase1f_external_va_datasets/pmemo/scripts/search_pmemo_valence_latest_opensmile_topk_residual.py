#!/usr/bin/env python3
"""OpenSMILE top-k residual search on the latest Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_best_after_smoothing_opensmile_topk_residual.py"

spec = importlib.util.spec_from_file_location("opensmile_topk_residual", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_latest_ridge_ccc_guard_smoothing_20260709"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_latest_opensmile_topk_residual_20260709"
mod.MODEL_ID = "pmemo_valence_latest_opensmile_topk_residual_20260709"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_latest_ridge_ccc_guard_smoothing_20260709",
    "rmse": 0.187623279893854,
    "mae": 0.14601650974911215,
    "pcc": 0.8141844446811181,
    "ccc": 0.8002325331205622,
}


if __name__ == "__main__":
    mod.main()
