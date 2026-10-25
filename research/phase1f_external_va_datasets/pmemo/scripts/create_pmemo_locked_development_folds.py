#!/usr/bin/env python3
"""Create development folds after locking PMEmo final test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKED_DIR = ROOT / "data" / "splits" / "pmemo_final_locked_20260713"
LOCKED_SPLIT = LOCKED_DIR / "pmemo_final_locked_song_level_split.csv"
OUT_DIR = LOCKED_DIR / "devfolds_3fold_20260713"
SEED = 20260713
N_FOLDS = 3


def summarize(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {"songs": 0, "frames": 0}
    return {
        "songs": int(len(df)),
        "frames": int(df["frames"].sum()),
        "mean_valence_mean": float(df["mean_valence"].mean()),
        "mean_valence_std": float(df["mean_valence"].std(ddof=0)),
        "mean_arousal_mean": float(df["mean_arousal"].mean()),
        "mean_arousal_std": float(df["mean_arousal"].std(ddof=0)),
        "historical_inspected_reference_songs": int(df["historically_inspected_reference"].sum()),
    }


def make_strata(df: pd.DataFrame) -> pd.Series:
    val_bin = pd.qcut(df["mean_valence"], q=5, labels=False, duplicates="drop")
    aro_bin = pd.qcut(df["mean_arousal"], q=5, labels=False, duplicates="drop")
    return val_bin.astype(str) + "_a" + aro_bin.astype(str)


def assign_folds(dev: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = dev.copy()
    out["stratum"] = make_strata(out)
    out["fold_id"] = -1
    for _, group in out.groupby("stratum", sort=True):
        ids = group["musicId"].to_numpy(dtype=int)
        rng.shuffle(ids)
        for i, music_id in enumerate(ids):
            out.loc[out["musicId"].eq(music_id), "fold_id"] = i % N_FOLDS
    return out.drop(columns=["stratum"]).sort_values("musicId")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locked = pd.read_csv(LOCKED_SPLIT)
    dev = locked[locked["locked_split"].eq("development_pool")].copy()
    assigned = assign_folds(dev)
    assigned.to_csv(OUT_DIR / "development_pool_fold_assignments.csv", index=False)

    summaries = {}
    all_final_overlaps = {}
    final_ids = set(locked.loc[locked["locked_split"].eq("final_locked_test"), "musicId"].astype(int))
    for fold_id in range(N_FOLDS):
        fold_name = f"fold_{fold_id + 1:02d}"
        fold_dir = OUT_DIR / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows = assigned.copy()
        rows["split"] = np.where(rows["fold_id"].eq(fold_id), "val", "train")
        index_cols = ["musicId", "split", "frames", "path"]
        rows[index_cols].to_csv(fold_dir / "sequence_index.csv", index=False)
        rows[index_cols + ["historical_split", "historically_inspected_reference", "mean_valence", "mean_arousal", "std_valence", "std_arousal"]].to_csv(
            fold_dir / "sequence_index_with_stats.csv", index=False
        )
        train = rows[rows["split"].eq("train")]
        val = rows[rows["split"].eq("val")]
        overlap = sorted(set(rows["musicId"].astype(int)).intersection(final_ids))
        all_final_overlaps[fold_name] = overlap
        summaries[fold_name] = {
            "train": summarize(train),
            "val": summarize(val),
            "final_locked_test_overlap_count": len(overlap),
            "final_locked_test_overlap_ids": overlap,
        }
    summary = {
        "created_at": "2026-07-13",
        "split_id": "pmemo_final_locked_devfolds_3fold_20260713",
        "source_locked_split": str(LOCKED_SPLIT),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "development_pool_songs": int(len(dev)),
        "final_locked_test_songs_excluded": int(len(final_ids)),
        "folds": summaries,
        "is_final_locked_test_excluded_from_all_folds": all(len(v) == 0 for v in all_final_overlaps.values()),
        "rules": [
            "Use these folds for future development optimization.",
            "Do not use the older pmemo_trainval_devfolds_20260709 for new training because it may contain newly locked final-test songs.",
            "Run scripts/guard_pmemo_final_locked_split.py on any training/validation index before training.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
