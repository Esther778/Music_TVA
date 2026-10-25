#!/usr/bin/env python3
"""Extra-wide validation-RMSE temporal shift scan for Valence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_temporal_shift import DATA_SPLITS, INPUT_DIR, ROOT, SPLITS, apply_shift, eval_frame
from train_pmemo_mert_valence_baseline import threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_temporal_shift_extra_wide_rmse_scan_20260708"


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
    cache: dict[int, dict[tuple[str, str], pd.DataFrame]] = {}
    for shift_frames in range(-60, 11):
        per_split = []
        shifted_frames: dict[tuple[str, str], pd.DataFrame] = {}
        for split in SPLITS:
            row: dict[str, Any] = {"split_name": split}
            for data_split in DATA_SPLITS:
                shifted = apply_shift(frames[(split, data_split)], shift_frames)
                shifted_frames[(split, data_split)] = shifted
                row.update({f"{data_split}_{k}": v for k, v in eval_frame(shifted).items()})
            per_split.append(row)
        val = macro(per_split, "val")
        test = macro(per_split, "test")
        all_rows.append(
            {
                "shift_frames": shift_frames,
                "shift_seconds": shift_frames * 0.5,
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
        )
        cache[shift_frames] = shifted_frames
    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_DIR / "all_shift_candidates.csv", index=False)
    selected = result.sort_values(["val_rmse", "val_ccc"], ascending=[True, False]).iloc[0].to_dict()
    selected_shift = int(selected["shift_frames"])
    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split}
        for data_split in DATA_SPLITS:
            df = cache[selected_shift][(split, data_split)]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{k}": v for k, v in eval_frame(df).items()})
        per_split.append(row)
    selected_val = macro(per_split, "val")
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_temporal_shift_extra_wide_rmse_scan_20260708",
        "selection_rule": "extra-wide global temporal shift selected by validation RMSE primary; test labels not used",
        "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "selected_shift_frames": selected_shift,
        "selected_shift_seconds": selected_shift * 0.5,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "per_split": per_split,
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nTop validation RMSE candidates:", flush=True)
    print(result.sort_values(["val_rmse", "val_ccc"], ascending=[True, False]).head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
