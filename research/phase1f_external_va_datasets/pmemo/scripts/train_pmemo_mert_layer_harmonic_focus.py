#!/usr/bin/env python3
"""Focused MERT-layer + harmonic Valence trials for adjacent MERT layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    DEFAULT_MERT_NORM,
    ROOT,
    TrialConfig,
    build_harmonic_norm,
    run_trial,
    selected_indices,
    resolve_device,
)


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "mert_layer_harmonic_focus"


def make_trials(prefix: str) -> list[TrialConfig]:
    return [
        TrialConfig(f"{prefix}_harm_h48_proj192_do025_mse", 192, 48, 0.25, 5e-4, 2e-4, 28, 0.0),
        TrialConfig(f"{prefix}_harm_h64_proj256_do025_mse", 256, 64, 0.25, 5e-4, 2e-4, 24, 0.0),
        TrialConfig(f"{prefix}_harm_h64_proj256_do025_ccc005", 256, 64, 0.25, 5e-4, 2e-4, 24, 0.05),
        TrialConfig(f"{prefix}_harm_h80_proj320_do030_ccc005", 320, 80, 0.30, 4e-4, 2e-4, 20, 0.05),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-name", default="hiddenX")
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--mert-norm", type=Path, default=DEFAULT_MERT_NORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    dataset_summary = build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), args.out_dir / "dataset", args.force_norm)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, index, args, device, args.seed) for trial in make_trials(args.layer_name)]
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
