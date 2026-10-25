#!/usr/bin/env python3
"""Temporal shift after HGB-shift frame residual correction."""

from __future__ import annotations

import search_pmemo_valence_temporal_shift_extra_wide as mod


mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_hgb_shift_frame_residual_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_hgb_frame_residual_temporal_shift_20260708"


if __name__ == "__main__":
    mod.main()
