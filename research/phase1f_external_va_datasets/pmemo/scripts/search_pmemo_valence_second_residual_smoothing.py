#!/usr/bin/env python3
"""Validation-selected smoothing after the second residual Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_current_best_smoothing.py"

spec = importlib.util.spec_from_file_location("smoothing_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.INPUT_DIR = base.ROOT / "outputs" / "valence_residual_sequence_20260709" / "h6_second_resid_h32_do055_cap002_l2p60"
base.OUT_DIR = base.ROOT / "outputs" / "valence_second_residual_smoothing_20260709"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_second_residual_h6_second_resid_h32_do055_cap002_l2p60",
    "rmse": 0.1878301997979482,
    "mae": 0.14590725302696228,
    "pcc": 0.8140613283748674,
    "ccc": 0.8013470881334128,
}


def run() -> None:
    base.main()
    summary_path = base.OUT_DIR / "summary.json"
    text = summary_path.read_text(encoding="utf-8")
    text = text.replace("pmemo_valence_current_best_smoothing_20260708", "pmemo_valence_second_residual_smoothing_20260709")
    text = text.replace("2026-07-08", "2026-07-09")
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
