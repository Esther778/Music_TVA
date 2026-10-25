#!/usr/bin/env python3
"""Extra-wide temporal shift scan for the HGB song-mean Valence candidate."""

from __future__ import annotations

import search_pmemo_valence_temporal_shift_extra_wide as mod


mod.INPUT_DIR = mod.ROOT / "outputs" / "valence_extra_wide_hgb_song_mean_20260708"
mod.OUT_DIR = mod.ROOT / "outputs" / "valence_hgb_temporal_shift_20260708"


if __name__ == "__main__":
    mod.main()
