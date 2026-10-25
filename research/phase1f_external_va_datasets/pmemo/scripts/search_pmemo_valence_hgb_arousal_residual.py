#!/usr/bin/env python3
"""Arousal-assisted residual correction on the current HGB Valence best."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_best_after_smoothing_arousal_residual.py"

spec = importlib.util.spec_from_file_location("arousal_residual_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.VALENCE_DIR = base.ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
base.OUT_DIR = base.ROOT / "outputs" / "valence_hgb_arousal_residual_20260709"
base.CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


def run() -> None:
    base.main()
    summary_path = base.OUT_DIR / "summary.json"
    text = summary_path.read_text(encoding="utf-8")
    text = text.replace("pmemo_valence_best_after_smoothing_arousal_residual_20260708", "pmemo_valence_hgb_arousal_residual_20260709")
    text = text.replace("2026-07-08", "2026-07-09")
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
