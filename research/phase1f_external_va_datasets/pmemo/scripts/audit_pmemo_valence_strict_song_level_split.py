#!/usr/bin/env python3
"""Audit strict song-level split metrics for a Valence prediction artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "valence_ridge_rmse_ccc_blend_20260709"
DEFAULT_OUT = ROOT / "outputs" / "valence_strict_song_level_audit_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column in {list(df.columns)}")


def load_df(input_dir: Path, split_name: str, data_split: str) -> pd.DataFrame:
    path = input_dir / split_name / f"{data_split}_predictions.csv"
    df = pd.read_csv(path)
    p_col = pred_col(df)
    y_col = target_col(df)
    out = df[["musicId", "time_sec", y_col, p_col]].copy()
    out.columns = ["musicId", "time_sec", "target_valence", "pred_valence"]
    out["musicId"] = out["musicId"].astype(int)
    return out.sort_values(["musicId", "time_sec"]).reset_index(drop=True)


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(
        df["pred_valence"].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def split_song_sets(frames: dict[str, pd.DataFrame]) -> dict[str, set[int]]:
    return {name: set(df["musicId"].astype(int).unique().tolist()) for name, df in frames.items()}


def overlap_report(song_sets: dict[str, set[int]]) -> dict[str, Any]:
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {}
    for left, right in pairs:
        ids = sorted(song_sets[left] & song_sets[right])
        overlaps[f"{left}_x_{right}"] = {"count": len(ids), "ids": ids[:20]}
    return overlaps


def macro(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def pooled(frames: list[pd.DataFrame]) -> dict[str, float]:
    return eval_df(pd.concat(frames, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-id", default="pmemo_valence_ridge_rmse_ccc_blend_20260709")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split: list[dict[str, Any]] = []
    all_test_frames: list[pd.DataFrame] = []
    leak_failures: list[dict[str, Any]] = []

    for split_name in SPLITS:
        frames = {data_split: load_df(input_dir, split_name, data_split) for data_split in DATA_SPLITS}
        song_sets = split_song_sets(frames)
        overlaps = overlap_report(song_sets)
        has_leakage = any(item["count"] > 0 for item in overlaps.values())
        if has_leakage:
            leak_failures.append({"split_name": split_name, "overlaps": overlaps})
        row: dict[str, Any] = {
            "split_name": split_name,
            "has_song_leakage": has_leakage,
            "song_counts": {name: len(ids) for name, ids in song_sets.items()},
            "frame_counts": {name: int(len(df)) for name, df in frames.items()},
            "overlaps": overlaps,
        }
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{k}": v for k, v in eval_df(frames[data_split]).items()})
        per_split.append(row)
        all_test_frames.append(frames["test"])

    macro_test = macro(per_split, "test")
    macro_val = macro(per_split, "val")
    pooled_test = pooled(all_test_frames)
    summary = {
        "created_at": "2026-07-09",
        "model_id": args.model_id,
        "input_dir": str(input_dir.relative_to(ROOT.resolve()) if input_dir.is_relative_to(ROOT.resolve()) else input_dir),
        "purpose": "strict song-level split audit for current Valence artifact",
        "audit_rule": "Within each split, train/val/test musicId sets must be pairwise disjoint.",
        "is_strict_song_level": len(leak_failures) == 0,
        "leak_failures": leak_failures,
        "macro_val": macro_val,
        "macro_test": macro_test,
        "pooled_test": pooled_test,
        "macro_test_threshold_gaps": threshold_rows(macro_test),
        "pooled_test_threshold_gaps": threshold_rows(pooled_test),
        "per_split": per_split,
        "important_note": (
            "This proves the existing three folds are song-level disjoint, but it is not a fresh blind test "
            "because these folds have already been inspected during model development."
        ),
    }
    (out_dir / "strict_song_level_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split_name": row["split_name"],
                "has_song_leakage": row["has_song_leakage"],
                "train_songs": row["song_counts"]["train"],
                "val_songs": row["song_counts"]["val"],
                "test_songs": row["song_counts"]["test"],
                "test_rmse": row["test_rmse"],
                "test_mae": row["test_mae"],
                "test_pcc": row["test_pcc"],
                "test_ccc": row["test_ccc"],
            }
            for row in per_split
        ]
    ).to_csv(out_dir / "strict_song_level_split_metrics.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
