#!/usr/bin/env python3
"""Create a permanent PMEmo song-level blind split.

The blind_test subset is intended to stay untouched after creation: it must not
be used for training, validation, model selection, or post-processing selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from train_pmemo_mert_valence_baseline import ROOT


DEFAULT_INDEX = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence_index.csv"
DEFAULT_SEQUENCE_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_OUT_DIR = ROOT / "data" / "splits" / "pmemo_blind_20260709"


def stratify_bins(values: pd.Series, bins: int = 8) -> np.ndarray:
    ranked = values.rank(method="first")
    q = min(bins, max(2, int(len(values) // 20)))
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


def load_song_stats(index_path: Path, sequence_dir: Path) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    rows: list[dict[str, Any]] = []
    for row in index.sort_values("musicId").to_dict("records"):
        sid = int(row["musicId"])
        data = np.load(sequence_dir / f"{sid}.npz")
        y = data["Y"].astype(np.float32)
        rows.append(
            {
                "musicId": sid,
                "frames": int(y.shape[0]),
                "path": str(sequence_dir / f"{sid}.npz"),
                "mean_valence": float(y[:, 0].mean()),
                "mean_arousal": float(y[:, 1].mean()),
                "std_valence": float(y[:, 0].std()),
                "std_arousal": float(y[:, 1].std()),
            }
        )
    return pd.DataFrame(rows)


def create_split(song_stats: pd.DataFrame, seed: int, blind_size: float, val_size: float) -> pd.DataFrame:
    y_bins = stratify_bins(song_stats["mean_valence"])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=blind_size, random_state=seed)
    dev_idx, blind_idx = next(splitter.split(song_stats, y_bins))
    dev = song_stats.iloc[dev_idx].reset_index(drop=True)
    dev_bins = stratify_bins(dev["mean_valence"])
    val_rel_size = val_size / (1.0 - blind_size)
    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=seed + 1000)
    train_rel, val_rel = next(val_splitter.split(dev, dev_bins))

    out = song_stats.copy()
    out["split"] = "train"
    out.loc[out["musicId"].isin(dev.iloc[val_rel]["musicId"]), "split"] = "val"
    out.loc[out["musicId"].isin(song_stats.iloc[blind_idx]["musicId"]), "split"] = "blind_test"
    return out.sort_values("musicId").reset_index(drop=True)


def overlap_report(split_df: pd.DataFrame) -> dict[str, Any]:
    sets = {
        name: set(split_df.loc[split_df["split"] == name, "musicId"].astype(int).tolist())
        for name in ["train", "val", "blind_test"]
    }
    overlaps = {}
    for left, right in [("train", "val"), ("train", "blind_test"), ("val", "blind_test")]:
        ids = sorted(sets[left] & sets[right])
        overlaps[f"{left}_x_{right}"] = {"count": len(ids), "ids": ids[:20]}
    return overlaps


def describe_group(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "songs": int(len(df)),
        "frames": int(df["frames"].sum()),
        "mean_valence_mean": float(df["mean_valence"].mean()),
        "mean_valence_std": float(df["mean_valence"].std(ddof=0)),
        "mean_valence_min": float(df["mean_valence"].min()),
        "mean_valence_max": float(df["mean_valence"].max()),
        "mean_arousal_mean": float(df["mean_arousal"].mean()),
        "mean_arousal_std": float(df["mean_arousal"].std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--blind-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    song_stats = load_song_stats(args.index, args.sequence_dir)
    split_df = create_split(song_stats, args.seed, args.blind_size, args.val_size)
    split_index = split_df[["musicId", "split", "frames", "path"]].copy()
    split_index.to_csv(args.out_dir / "pmemo_blind_song_level_split.csv", index=False)
    split_df.to_csv(args.out_dir / "pmemo_blind_song_level_split_with_stats.csv", index=False)
    split_index.rename(columns={"split": "split_raw"}).assign(
        split=lambda x: x["split_raw"].replace({"blind_test": "test"})
    )[["musicId", "split", "frames", "path"]].to_csv(args.out_dir / "pmemo_blind_trainvaltest_index.csv", index=False)

    overlaps = overlap_report(split_df)
    summary = {
        "created_at": "2026-07-09",
        "split_id": "pmemo_blind_20260709",
        "seed": args.seed,
        "blind_size": args.blind_size,
        "val_size": args.val_size,
        "source_index": str(args.index),
        "sequence_dir": str(args.sequence_dir),
        "total_songs": int(len(split_df)),
        "groups": {name: describe_group(g) for name, g in split_df.groupby("split", sort=True)},
        "overlaps": overlaps,
        "is_song_level_disjoint": all(item["count"] == 0 for item in overlaps.values()),
        "rules": [
            "blind_test songs must not be used for training.",
            "blind_test songs must not be used for validation, model selection, post-processing selection, or hyperparameter tuning.",
            "blind_test metrics should be computed once after the pipeline is frozen.",
        ],
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
