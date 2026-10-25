#!/usr/bin/env python3
"""Mean/CCC-focused PMEmo Valence BiLSTM trial.

This is a narrowly scoped follow-up after post-hoc Valence optimization
saturated. It keeps the deployable audio-feature path and fixed song-level
splits, but shifts the loss toward frame CCC and clip/song-level mean
consistency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_harmonic_valence_fusion import DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET
from train_pmemo_mert_valence_baseline import ROOT, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import (
    DEFAULT_MERT_DIR,
    BestConfig,
    aggregate_metrics,
    train_one_split,
)


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_mean_ccc_focus_20260629"

CURRENT_BEST = {
    "model_id": "pmemo_valence_targeted_multilayer_postprocess_20260629",
    "rmse": 0.19270108142080475,
    "mae": 0.15030960055717216,
    "pcc": 0.8029485501074519,
    "ccc": 0.7870809597955878,
}

TRIALS = [
    BestConfig(
        trial_id="h6_harm_h80_mean045_ccc015_std004_do035",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.35,
        learning_rate=3e-4,
        weight_decay=5e-4,
        batch_size=20,
        frame_ccc_weight=0.15,
        clip_mse_weight=0.45,
        extreme_weight=1.10,
        clip_std_weight=0.04,
        patience=10,
        epochs=80,
    ),
    BestConfig(
        trial_id="h6_harm_h64_mean060_ccc020_std006_do045",
        proj_dim=256,
        hidden_dim=64,
        dropout=0.45,
        learning_rate=2.5e-4,
        weight_decay=1e-3,
        batch_size=20,
        frame_ccc_weight=0.20,
        clip_mse_weight=0.60,
        extreme_weight=1.05,
        clip_std_weight=0.06,
        patience=10,
        epochs=80,
    ),
]


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def summarize_trial(rows: list[dict[str, Any]], out_dir: Path, config: BestConfig) -> dict[str, Any]:
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-06-29",
        "model_family": "MERT hidden_6 + Librosa harmonic mean/CCC-focused Valence BiLSTM",
        "fixed_split_source": str(BASE_SPLIT_DIR),
        "config": asdict(config),
        "current_best_reference": CURRENT_BEST,
        "num_splits": len(rows),
        "aggregate": {
            **aggregate_metrics(rows, "val"),
            **aggregate_metrics(rows, "test"),
        },
        "per_split": rows,
    }
    test_mean = {k: aggregate["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["mean_test_gap_rows"] = threshold_rows(test_mean)
    aggregate["mean_test_delta_vs_current_best"] = {
        metric: float(test_mean[metric] - CURRENT_BEST[metric])
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return aggregate


def summarize_all(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for s in summaries:
        cfg = s["config"]
        row = {
            "trial_id": cfg["trial_id"],
            **{f"test_{m}_mean": s["aggregate"][f"test_{m}_mean"] for m in ["rmse", "mae", "pcc", "ccc"]},
            **{f"test_{m}_std": s["aggregate"][f"test_{m}_std"] for m in ["rmse", "mae", "pcc", "ccc"]},
            **{f"delta_vs_current_best_{m}": s["mean_test_delta_vs_current_best"][m] for m in ["rmse", "mae", "pcc", "ccc"]},
        }
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(["test_rmse_mean", "test_ccc_mean"], ascending=[True, False])
    comparison.to_csv(out_dir / "trial_comparison.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-29",
                "current_best_reference": CURRENT_BEST,
                "trial_count": len(summaries),
                "trials": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", choices=[t.trial_id for t in TRIALS] + ["all"], default=TRIALS[0].trial_id)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configs = TRIALS if args.trial_id == "all" else [cfg for cfg in TRIALS if cfg.trial_id == args.trial_id]
    summaries = []
    for config in configs:
        trial_out = args.out_dir / config.trial_id
        trial_out.mkdir(parents=True, exist_ok=True)
        train_args = SimpleNamespace(
            out_dir=trial_out,
            mert_dir=args.mert_dir,
            label_dir=args.label_dir,
            librosa_dataset=args.librosa_dataset,
        )
        rows = []
        for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
            split_id, seed = split_seed(split_dir)
            index = pd.read_csv(split_dir / "sequence_index.csv")
            rows.append(train_one_split(split_id, seed, index, train_args, device, config))
        summaries.append(summarize_trial(rows, trial_out, config))
    summarize_all(summaries, args.out_dir)


if __name__ == "__main__":
    main()
