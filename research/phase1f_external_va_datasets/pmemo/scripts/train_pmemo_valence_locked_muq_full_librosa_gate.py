#!/usr/bin/env python3
"""Run the locked-dev Valence trainer with MuQ hidden-6 + all Librosa groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import train_pmemo_valence_dev_only_trials as trainer
from train_pmemo_valence_selected_fusion import selected_indices as group_indices


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
FEATURES = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
DEFAULT_GROUPS = ["mfcc", "spectral", "harmonic", "energy", "tempo"]
GROUPS = DEFAULT_GROUPS


def all_librosa_indices(_requested: list[str]) -> list[int]:
    return group_indices(GROUPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--suffix", default="full")
    args = parser.parse_args()
    GROUPS = args.groups
    out = ROOT / f"outputs/valence_locked_muq_librosa_{args.suffix}_fold03_20260714"
    # The base trainer resolves this imported symbol when constructing loaders.
    trainer.selected_indices = all_librosa_indices
    sys.argv = [
        sys.argv[0], "--index", str(INDEX), "--mert-dir", str(FEATURES),
        "--out-dir", str(out), "--device", "cpu", "--seed", "20260710",
        "--trial", "v_dev_wider_h96_do035_lr25e4",
    ]
    trainer.main()
