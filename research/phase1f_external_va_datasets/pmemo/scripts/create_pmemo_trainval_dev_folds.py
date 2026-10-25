#!/usr/bin/env python3
"""Create reusable song-level development folds from train+val only.

The existing holdout/test rows are excluded. These folds are for model
development and hyperparameter comparison only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_pmemo_mert_valence_baseline import ROOT


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_blind_20260709" / "pmemo_blind_trainvaltest_index.csv"
DEFAULT_SEQUENCE_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_OUT = ROOT / "data" / "splits" / "pmemo_trainval_devfolds_20260709"


def stratify_bins(stats: pd.DataFrame) -> np.ndarray:
    # Joint coarse bins: Valence is the bottleneck, Arousal is included to avoid
    # folds that are balanced for V but skewed for A.
    v_rank = stats["mean_valence"].rank(method="first")
    a_rank = stats["mean_arousal"].rank(method="first")
    v_bins = pd.qcut(v_rank, q=6, labels=False, duplicates="drop").astype(int)
    a_bins = pd.qcut(a_rank, q=4, labels=False, duplicates="drop").astype(int)
    return (v_bins.astype(str) + "_" + a_bins.astype(str)).to_numpy()


def load_stats(index: pd.DataFrame, sequence_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in index.sort_values("musicId").to_dict("records"):
        sid = int(row["musicId"])
        data = np.load(sequence_dir / f"{sid}.npz")
        y = data["Y"].astype(np.float32)
        rows.append(
            {
                "musicId": sid,
                "source_split": row["split"],
                "frames": int(row["frames"]),
                "path": row["path"],
                "mean_valence": float(y[:, 0].mean()),
                "mean_arousal": float(y[:, 1].mean()),
                "std_valence": float(y[:, 0].std()),
                "std_arousal": float(y[:, 1].std()),
            }
        )
    return pd.DataFrame(rows)


def describe(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "songs": int(len(df)),
        "frames": int(df["frames"].sum()),
        "mean_valence_mean": float(df["mean_valence"].mean()),
        "mean_valence_std": float(df["mean_valence"].std(ddof=0)),
        "mean_arousal_mean": float(df["mean_arousal"].mean()),
        "mean_arousal_std": float(df["mean_arousal"].std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_index = pd.read_csv(args.index)
    dev_index = full_index[full_index["split"].isin(["train", "val"])].copy()
    stats = load_stats(dev_index, args.sequence_dir)
    bins = stratify_bins(stats)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_summaries = []
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(stats, bins), start=1):
        fold_dir = args.out_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold = stats.loc[:, ["musicId", "frames", "path"]].copy()
        fold["split"] = "train"
        fold.loc[stats.iloc[val_idx].index, "split"] = "val"
        fold = fold[["musicId", "split", "frames", "path"]].sort_values("musicId").reset_index(drop=True)
        fold.to_csv(fold_dir / "sequence_index.csv", index=False)
        rich = stats.merge(fold[["musicId", "split"]], on="musicId", how="left")
        rich.to_csv(fold_dir / "sequence_index_with_stats.csv", index=False)
        train = rich[rich["split"] == "train"]
        val = rich[rich["split"] == "val"]
        overlap = sorted(set(train["musicId"]) & set(val["musicId"]))
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "index": str(fold_dir / "sequence_index.csv"),
                "train": describe(train),
                "val": describe(val),
                "overlap_count": len(overlap),
            }
        )
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Reusable song-level development folds from train+val only; holdout/test excluded",
        "source_index": str(args.index),
        "folds": args.folds,
        "seed": args.seed,
        "source_rows": full_index["split"].value_counts().to_dict(),
        "dev_songs": int(stats["musicId"].nunique()),
        "dev_frames": int(stats["frames"].sum()),
        "fold_summaries": fold_summaries,
        "boundary": "Use for development. Do not treat as final blind test.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
