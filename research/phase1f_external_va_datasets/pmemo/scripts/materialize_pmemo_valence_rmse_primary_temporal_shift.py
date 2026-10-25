#!/usr/bin/env python3
"""Materialize validation-RMSE-primary temporal shift for Valence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_temporal_shift import (
    CURRENT_BEST,
    DATA_SPLITS,
    INPUT_DIR,
    ROOT,
    SPLITS,
    apply_shift,
    eval_frame,
)
from train_pmemo_mert_valence_baseline import threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_rmseprimary_20260708"
SHIFT_FRAMES = -16


def macro(rows: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[f"{split_name}_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"split_name": split}
        for data_split in DATA_SPLITS:
            df = pd.read_csv(INPUT_DIR / split / f"{data_split}_predictions.csv")
            shifted = apply_shift(df, SHIFT_FRAMES)
            shifted.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{k}": v for k, v in eval_frame(shifted).items()})
        rows.append(row)
    selected_val = macro(rows, "val")
    selected_test = macro(rows, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_guardtopccc_temporal_shift_rmseprimary_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "one global temporal shift selected by validation RMSE primary from the existing validation shift sweep; test labels not used",
        "selected_shift_frames": SHIFT_FRAMES,
        "selected_shift_seconds": SHIFT_FRAMES * 0.5,
        "selected_val": selected_val,
        "per_split": rows,
        "selected_test": selected_test,
        "delta_vs_input": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
