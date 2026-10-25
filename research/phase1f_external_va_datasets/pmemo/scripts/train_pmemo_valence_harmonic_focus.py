#!/usr/bin/env python3
"""Focused PMEmo Valence trials around the current harmonic-only best."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_pmemo_valence_selected_fusion import (
    DEFAULT_LIBROSA_DATASET,
    DEFAULT_OPENSMILE_DATASET,
    ROOT,
    TrialConfig,
    resolve_device,
    run_trial,
)


DEFAULT_OUT = ROOT / "outputs" / "valence_opensmile_harmonic_focus"

TRIALS = [
    TrialConfig("v_harm_h80_l1_do020_lr7e4", ["harmonic"], 80, 1, 0.20, 56, 7e-4, 1e-4, 32, 1.0, 0.0, 0.0),
    TrialConfig("v_harm_h112_l1_do020_lr7e4", ["harmonic"], 112, 1, 0.20, 72, 7e-4, 1e-4, 28, 1.0, 0.0, 0.0),
    TrialConfig("v_harm_h96_l1_do025_lr6e4_ccc005", ["harmonic"], 96, 1, 0.25, 64, 6e-4, 2e-4, 32, 1.0, 0.05, 0.0),
    TrialConfig("v_harm_h96_l2_do025_lr6e4_ccc010", ["harmonic"], 96, 2, 0.25, 64, 6e-4, 1e-4, 28, 1.0, 0.10, 0.0),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensmile-dataset", type=Path, default=DEFAULT_OPENSMILE_DATASET)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, args.opensmile_dataset, args.librosa_dataset, args.out_dir, device, args.seed, args.force_norm) for trial in TRIALS]
    rows = []
    for s in summaries:
        rows.append(
            {
                "trial_id": s["trial_id"],
                "feature_groups": "+".join(s["feature_groups"]),
                "selected_librosa_dim": s["selected_librosa_dim"],
                "best_epoch": s["best_epoch"],
                "test_rmse": s["test_rmse"],
                "test_mae": s["test_mae"],
                "test_pcc": s["test_pcc"],
                "test_ccc": s["test_ccc"],
                "test_post_rmse": s["test_post_rmse"],
                "test_post_mae": s["test_post_mae"],
                "test_post_pcc": s["test_post_pcc"],
                "test_post_ccc": s["test_post_ccc"],
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
