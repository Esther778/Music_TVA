#!/usr/bin/env python3
"""Target-line oriented affine calibration on the latest Ridge Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_affine_calibration.py"

spec = importlib.util.spec_from_file_location("affine_calibration", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_latest_ridge_song_mean_fast_20260709"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_latest_ridge_affine_targetline_20260709"
mod.CURRENT_BEST = {
    "model_id": "pmemo_valence_latest_ridge_song_mean_fast_20260709",
    "rmse": 0.1875696952451583,
    "mae": 0.14609622778497267,
    "pcc": 0.814173472004312,
    "ccc": 0.7996245937063925,
}


def targetline_score(metrics: dict[str, float]) -> float:
    """Prefer crossing CCC while keeping RMSE/MAE/PCC close to the pass line."""
    return (
        2.2 * max(0.0, metrics["rmse"] - 0.18) / 0.18
        + 0.8 * max(0.0, metrics["mae"] - 0.15) / 0.15
        + 1.0 * max(0.0, 0.80 - metrics["pcc"]) / 0.80
        + 6.0 * max(0.0, 0.80 - metrics["ccc"]) / 0.80
        + 0.02 * metrics["rmse"]
    )


mod.score = targetline_score


if __name__ == "__main__":
    mod.main()
