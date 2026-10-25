#!/usr/bin/env python3
"""Regularized multi-split BiLSTM trials for PMEmo Valence.

The purpose is to attack the remaining Valence overfitting problem without
changing data splits or feature inputs. This script reuses the fixed song-level
splits from valence_multisplit_best_20260627 and trains smaller / more
regularized BiLSTM variants.
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
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_regularized_bilstm_20260628"

CURRENT_MULTI_SPLIT_BEST = {
    "model_id": "pmemo_valence_song_bias_correction_rmse_aware_20260628",
    "rmse": 0.20050966596572084,
    "mae": 0.15405092914501886,
    "pcc": 0.7845220440639044,
    "ccc": 0.7720363870353871,
}


TRIALS = [
    BestConfig(
        trial_id="bilstm_reg_h64_do045_wd001_ccc010_clip010_ext115",
        proj_dim=256,
        hidden_dim=64,
        dropout=0.45,
        learning_rate=3e-4,
        weight_decay=1e-3,
        batch_size=20,
        frame_ccc_weight=0.10,
        clip_mse_weight=0.10,
        extreme_weight=1.15,
        clip_std_weight=0.02,
        patience=10,
        epochs=70,
    ),
    BestConfig(
        trial_id="bilstm_reg_h56_do050_wd0015_ccc012_clip012_ext110",
        proj_dim=224,
        hidden_dim=56,
        dropout=0.50,
        learning_rate=2.5e-4,
        weight_decay=1.5e-3,
        batch_size=20,
        frame_ccc_weight=0.12,
        clip_mse_weight=0.12,
        extreme_weight=1.10,
        clip_std_weight=0.03,
        patience=10,
        epochs=70,
    ),
    BestConfig(
        trial_id="bilstm_songmean_h80_do035_wd0005_ccc008_clip030_std002",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.35,
        learning_rate=3e-4,
        weight_decay=5e-4,
        batch_size=20,
        frame_ccc_weight=0.08,
        clip_mse_weight=0.30,
        extreme_weight=1.25,
        clip_std_weight=0.02,
        patience=10,
        epochs=70,
    ),
]


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def summarize_trial(rows: list[dict[str, Any]], out_dir: Path, config: BestConfig) -> dict[str, Any]:
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-06-28",
        "model_family": "MERT hidden_6 + Librosa harmonic regularized Valence-only BiLSTM",
        "fixed_split_source": str(BASE_SPLIT_DIR),
        "config": asdict(config),
        "current_multi_split_best_reference": CURRENT_MULTI_SPLIT_BEST,
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
        "rmse": float(test_mean["rmse"] - CURRENT_MULTI_SPLIT_BEST["rmse"]),
        "mae": float(test_mean["mae"] - CURRENT_MULTI_SPLIT_BEST["mae"]),
        "pcc": float(test_mean["pcc"] - CURRENT_MULTI_SPLIT_BEST["pcc"]),
        "ccc": float(test_mean["ccc"] - CURRENT_MULTI_SPLIT_BEST["ccc"]),
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
                "created_at": "2026-06-28",
                "current_multi_split_best_reference": CURRENT_MULTI_SPLIT_BEST,
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
    parser.add_argument("--trial-id", choices=[t.trial_id for t in TRIALS], default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configs = [cfg for cfg in TRIALS if args.trial_id in {None, cfg.trial_id}]
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
