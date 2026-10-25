#!/usr/bin/env python3
"""Refinement trials around the current best hidden_6 MERT + harmonic Valence model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    ROOT,
    TrialConfig,
    build_harmonic_norm,
    run_trial,
    selected_indices,
    resolve_device,
)


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_HIDDEN6_NORM = ROOT / "outputs" / "mert_hidden6_valence_baseline" / "dataset" / "mert_norm.npz"
DEFAULT_OUT = ROOT / "outputs" / "mert_hidden6_harmonic_refine"

TRIALS = [
    TrialConfig("hidden6_harm_h80_proj320_do025_ccc008", 320, 80, 0.25, 4e-4, 2e-4, 24, 0.08),
    TrialConfig("hidden6_harm_h96_proj384_do030_ccc005", 384, 96, 0.30, 4e-4, 2e-4, 22, 0.05),
    TrialConfig("hidden6_harm_h96_proj384_do025_ccc010", 384, 96, 0.25, 3e-4, 2e-4, 24, 0.10),
    TrialConfig("hidden6_harm_h112_proj384_do035_ccc005", 384, 112, 0.35, 3e-4, 2e-4, 22, 0.05),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--mert-norm", type=Path, default=DEFAULT_HIDDEN6_NORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    dataset_summary = build_harmonic_norm(
        index,
        args.librosa_dataset,
        selected_indices(["harmonic"]),
        args.out_dir / "dataset",
        args.force_norm,
    )
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, index, args, device, args.seed) for trial in TRIALS]
    rows = [
        {
            "trial_id": s["trial_id"],
            "best_epoch": s["best_epoch"],
            "test_rmse": s["test_rmse"],
            "test_mae": s["test_mae"],
            "test_pcc": s["test_pcc"],
            "test_ccc": s["test_ccc"],
        }
        for s in summaries
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
