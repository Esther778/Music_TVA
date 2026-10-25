#!/usr/bin/env python3
"""Extra-wide per-split temporal shift diagnostic for Valence.

This is a diagnostic, not the deployable teacher policy. Each CV split chooses
its own shift by validation RMSE to estimate how much error is attributable to a
single global delay assumption.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_temporal_shift import DATA_SPLITS, INPUT_DIR, SPLITS, apply_shift, eval_frame
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_temporal_shift_extra_wide_per_split_20260708"
CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[f"{data_split}_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split, data_split): pd.read_csv(INPUT_DIR / split / f"{data_split}_predictions.csv")
        for split in SPLITS
        for data_split in DATA_SPLITS
    }

    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_frames: dict[tuple[str, str], pd.DataFrame] = {}

    for split in SPLITS:
        candidates: list[dict[str, Any]] = []
        cache: dict[tuple[int, str], pd.DataFrame] = {}
        for shift_frames in range(-60, 11):
            row: dict[str, Any] = {
                "split_name": split,
                "shift_frames": shift_frames,
                "shift_seconds": shift_frames * 0.5,
            }
            for data_split in DATA_SPLITS:
                shifted = apply_shift(frames[(split, data_split)], shift_frames)
                cache[(shift_frames, data_split)] = shifted
                row.update({f"{data_split}_{k}": v for k, v in eval_frame(shifted).items()})
            candidates.append(row)
            all_rows.append(row)

        selected = sorted(candidates, key=lambda r: (r["val_rmse"], -r["val_ccc"]))[0]
        selected_rows.append(selected)
        for data_split in DATA_SPLITS:
            selected_frames[(split, data_split)] = cache[(int(selected["shift_frames"]), data_split)]

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_per_split_shift_candidates.csv", index=False)

    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        selected = next(row for row in selected_rows if row["split_name"] == split)
        row: dict[str, Any] = {
            "split_name": split,
            "selected_shift_frames": int(selected["shift_frames"]),
            "selected_shift_seconds": float(selected["shift_seconds"]),
        }
        for data_split in DATA_SPLITS:
            df = selected_frames[(split, data_split)]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{k}": v for k, v in eval_frame(df).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_temporal_shift_extra_wide_per_split_20260708",
        "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "selection_rule": "diagnostic only: each split chooses its own temporal shift by validation RMSE; test labels not used for selection",
        "current_best_reference": CURRENT_BEST,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {
            metric: float(selected_test[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
