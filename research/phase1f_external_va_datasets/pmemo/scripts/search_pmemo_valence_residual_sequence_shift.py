#!/usr/bin/env python3
"""Global temporal shift scan after the residual-sequence Valence artifact."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_temporal_shift.py"

spec = importlib.util.spec_from_file_location("temporal_shift_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.INPUT_DIR = base.ROOT / "outputs" / "valence_residual_sequence_20260709" / "h6_base_resid_h48_do045_cap006"
base.OUT_DIR = base.ROOT / "outputs" / "valence_residual_sequence_shift_20260709"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_residual_sequence_h6_base_resid_h48_do045_cap006",
    "rmse": 0.18797815342744192,
    "mae": 0.146040345231692,
    "pcc": 0.8139891061926631,
    "ccc": 0.8011623335450896,
}


def run() -> None:
    base.main()
    summary_path = base.OUT_DIR / "summary.json"
    text = summary_path.read_text(encoding="utf-8")
    text = text.replace("pmemo_valence_guardtopccc_temporal_shift_20260708", "pmemo_valence_residual_sequence_shift_20260709")
    text = text.replace("2026-07-08", "2026-07-09")
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
