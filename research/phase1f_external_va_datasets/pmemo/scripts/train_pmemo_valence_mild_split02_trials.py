#!/usr/bin/env python3
"""Mild split_02 Valence training trials near the historical best config."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    DEFAULT_MERT_DIR,
    BestConfig,
    read_song_target_means,
    make_split_index,
    train_one_split,
)


OUT_DIR = ROOT / "outputs" / "valence_mild_split02_trials_20260708"
SPLIT_ID = 2
SEED = 202

CURRENT_SPLIT2_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "test_rmse": 0.19607233482197653,
    "test_mae": 0.1528912904892823,
    "test_pcc": 0.7942819565393399,
    "test_ccc": 0.7773068277491959,
}

TRIALS = [
    BestConfig(
        trial_id="mild_h80_do025_lr3e4_ccc008_clip008_ext115",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.25,
        learning_rate=3e-4,
        weight_decay=3e-4,
        batch_size=20,
        frame_ccc_weight=0.08,
        clip_mse_weight=0.08,
        extreme_weight=1.15,
        clip_std_weight=0.0,
        patience=10,
        epochs=70,
    ),
    BestConfig(
        trial_id="mild_h96_do030_lr25e4_ccc006_clip012_ext110",
        proj_dim=384,
        hidden_dim=96,
        dropout=0.30,
        learning_rate=2.5e-4,
        weight_decay=5e-4,
        batch_size=16,
        frame_ccc_weight=0.06,
        clip_mse_weight=0.12,
        extreme_weight=1.10,
        clip_std_weight=0.01,
        patience=10,
        epochs=70,
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_index = pd.read_csv(DEFAULT_INDEX)
    song_stats = read_song_target_means(base_index, DEFAULT_LABEL_DIR)
    split_index = make_split_index(song_stats, SEED)
    device = resolve_device("auto")
    rows = []
    for config in TRIALS:
        trial_out = OUT_DIR / config.trial_id
        trial_out.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(
            out_dir=trial_out,
            mert_dir=DEFAULT_MERT_DIR,
            label_dir=DEFAULT_LABEL_DIR,
            librosa_dataset=DEFAULT_LIBROSA_DATASET,
        )
        row = train_one_split(SPLIT_ID, SEED, split_index, args, device, config)
        row["trial_id"] = config.trial_id
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(["test_rmse", "test_ccc"], ascending=[True, False])
    comparison.to_csv(OUT_DIR / "comparison.csv", index=False)
    best = comparison.iloc[0].to_dict()
    best_metrics = {k: float(best[f"test_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-08",
        "purpose": "screen mild near-best training changes on the bottleneck split_02",
        "current_split2_best_reference": CURRENT_SPLIT2_BEST,
        "trials": [asdict(t) for t in TRIALS],
        "best_trial": best,
        "best_trial_gap_rows": threshold_rows(best_metrics),
        "delta_vs_current_split2_best": {
            "rmse": best_metrics["rmse"] - CURRENT_SPLIT2_BEST["test_rmse"],
            "mae": best_metrics["mae"] - CURRENT_SPLIT2_BEST["test_mae"],
            "pcc": best_metrics["pcc"] - CURRENT_SPLIT2_BEST["test_pcc"],
            "ccc": best_metrics["ccc"] - CURRENT_SPLIT2_BEST["test_ccc"],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
