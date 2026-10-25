#!/usr/bin/env python3
"""Create a prospective locked final-test split for PMEmo.

The previous pmemo_blind_20260709 test set has already been inspected during
model development, so it is not eligible as a pristine final test. This script
locks a new prospective final test from the historical train/val pool and marks
the previous blind_test songs as historical reference songs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "splits" / "pmemo_blind_20260709" / "pmemo_blind_song_level_split_with_stats.csv"
OUT_DIR = ROOT / "data" / "splits" / "pmemo_final_locked_20260713"
SEED = 20260713
FINAL_RATIO_OF_TOTAL = 0.15


def summarize(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "songs": 0,
            "frames": 0,
            "mean_valence_mean": None,
            "mean_valence_std": None,
            "mean_arousal_mean": None,
            "mean_arousal_std": None,
        }
    return {
        "songs": int(len(df)),
        "frames": int(df["frames"].sum()),
        "mean_valence_mean": float(df["mean_valence"].mean()),
        "mean_valence_std": float(df["mean_valence"].std(ddof=0)),
        "mean_valence_min": float(df["mean_valence"].min()),
        "mean_valence_max": float(df["mean_valence"].max()),
        "mean_arousal_mean": float(df["mean_arousal"].mean()),
        "mean_arousal_std": float(df["mean_arousal"].std(ddof=0)),
        "mean_arousal_min": float(df["mean_arousal"].min()),
        "mean_arousal_max": float(df["mean_arousal"].max()),
        "frames_mean": float(df["frames"].mean()),
        "frames_min": int(df["frames"].min()),
        "frames_max": int(df["frames"].max()),
    }


def make_strata(df: pd.DataFrame) -> pd.Series:
    val_bin = pd.qcut(df["mean_valence"], q=5, labels=False, duplicates="drop")
    aro_bin = pd.qcut(df["mean_arousal"], q=5, labels=False, duplicates="drop")
    frame_bin = pd.qcut(df["frames"], q=3, labels=False, duplicates="drop")
    return val_bin.astype(str) + "_a" + aro_bin.astype(str) + "_f" + frame_bin.astype(str)


def choose_final_ids(eligible: pd.DataFrame, total_songs: int) -> set[int]:
    rng = np.random.default_rng(SEED)
    target = int(round(total_songs * FINAL_RATIO_OF_TOTAL))
    work = eligible.copy()
    work["stratum"] = make_strata(work)
    chosen: list[int] = []
    for _, group in work.groupby("stratum", sort=True):
        n = int(round(len(group) * target / len(work)))
        if n <= 0 and len(group) >= 8:
            n = 1
        if n <= 0:
            continue
        n = min(n, len(group))
        chosen.extend(rng.choice(group["musicId"].to_numpy(dtype=int), size=n, replace=False).tolist())

    chosen_set = set(int(x) for x in chosen)
    if len(chosen_set) < target:
        remaining = sorted(set(work["musicId"].astype(int)) - chosen_set)
        add = rng.choice(np.array(remaining, dtype=int), size=target - len(chosen_set), replace=False)
        chosen_set.update(int(x) for x in add)
    elif len(chosen_set) > target:
        remove = rng.choice(np.array(sorted(chosen_set), dtype=int), size=len(chosen_set) - target, replace=False)
        chosen_set.difference_update(int(x) for x in remove)
    return chosen_set


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source["historical_split"] = source["split"]
    source["historically_inspected_reference"] = source["historical_split"].eq("blind_test")
    eligible = source[source["historical_split"].isin(["train", "val"])].copy()
    final_ids = choose_final_ids(eligible, total_songs=len(source))

    out = source.copy()
    out["locked_split"] = "development_pool"
    out.loc[out["musicId"].isin(final_ids), "locked_split"] = "final_locked_test"
    out["final_test_eligible_source"] = out["historical_split"].isin(["train", "val"])
    out["final_test_locked"] = out["locked_split"].eq("final_locked_test")
    out["allowed_for_future_training"] = ~out["final_test_locked"]
    out["allowed_for_future_validation"] = ~out["final_test_locked"]
    out["allowed_for_final_evaluation_only"] = out["final_test_locked"]

    cols = [
        "musicId",
        "locked_split",
        "historical_split",
        "historically_inspected_reference",
        "final_test_eligible_source",
        "final_test_locked",
        "allowed_for_future_training",
        "allowed_for_future_validation",
        "allowed_for_final_evaluation_only",
        "frames",
        "path",
        "mean_valence",
        "mean_arousal",
        "std_valence",
        "std_arousal",
    ]
    out = out[cols].sort_values("musicId")
    out.to_csv(OUT_DIR / "pmemo_final_locked_song_level_split.csv", index=False)
    out.to_csv(OUT_DIR / "pmemo_final_locked_song_level_split_with_stats.csv", index=False)
    out[out["locked_split"].eq("development_pool")][["musicId", "locked_split", "frames", "path"]].rename(
        columns={"locked_split": "split"}
    ).to_csv(OUT_DIR / "development_pool_index.csv", index=False)
    out[out["locked_split"].eq("final_locked_test")][["musicId", "locked_split", "frames", "path"]].rename(
        columns={"locked_split": "split"}
    ).to_csv(OUT_DIR / "final_locked_test_index.csv", index=False)

    final = out[out["locked_split"].eq("final_locked_test")]
    dev = out[out["locked_split"].eq("development_pool")]
    historical_ref = out[out["historically_inspected_reference"]]
    trainval_kept = out[out["historical_split"].isin(["train", "val"]) & out["locked_split"].eq("development_pool")]
    overlaps = {
        "final_x_development": sorted(set(final["musicId"]).intersection(set(dev["musicId"]))),
        "final_x_historical_inspected_reference": sorted(set(final["musicId"]).intersection(set(historical_ref["musicId"]))),
    }
    summary = {
        "created_at": "2026-07-13",
        "split_id": "pmemo_final_locked_20260713",
        "seed": SEED,
        "source_split": str(SOURCE),
        "total_songs": int(len(out)),
        "final_ratio_of_total": FINAL_RATIO_OF_TOTAL,
        "important_limitation": (
            "This is a prospective locked final test because all available labelled PMEmo data existed during prior exploration. "
            "The previous pmemo_blind_20260709 blind_test has been inspected and is not used as final_locked_test."
        ),
        "groups": {
            "final_locked_test": summarize(final),
            "development_pool": summarize(dev),
            "historical_inspected_reference_in_development_pool": summarize(historical_ref),
            "historical_trainval_remaining_for_development": summarize(trainval_kept),
        },
        "overlaps": {k: {"count": len(v), "ids": v} for k, v in overlaps.items()},
        "is_song_level_disjoint": all(len(v) == 0 for v in overlaps.values()),
        "rules": [
            "final_locked_test songs must not be used for training.",
            "final_locked_test songs must not be used for validation, model selection, post-processing selection, hyperparameter tuning, normalization fitting, or ensemble-weight selection.",
            "final_locked_test metrics may be computed once only after the full Arousal and Valence teacher pipelines are frozen.",
            "The old pmemo_blind_20260709 blind_test is historical_inspected_reference, not a final blind test.",
            "All future training scripts should read development_pool_index.csv or a fold derived from it, and should run the guard script before training.",
        ],
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
