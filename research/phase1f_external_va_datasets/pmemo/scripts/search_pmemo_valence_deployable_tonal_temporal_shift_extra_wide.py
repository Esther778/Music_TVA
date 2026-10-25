#!/usr/bin/env python3
"""Extra-wide temporal shift scan for the deployable tonal Valence chain."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "search_pmemo_valence_temporal_shift_extra_wide.py"

spec = importlib.util.spec_from_file_location("shift_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")

shift_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shift_base)

shift_base.INPUT_DIR = (
    shift_base.ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
)
shift_base.OUT_DIR = (
    shift_base.ROOT / "outputs" / "valence_deployable_tonal_temporal_shift_extra_wide_20260708"
)


if __name__ == "__main__":
    shift_base.main()
