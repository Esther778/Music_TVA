#!/usr/bin/env python3
"""Per-split validation-selected temporal shift diagnostic for Valence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_temporal_shift import DATA_SPLITS, INPUT_DIR, SPLITS, apply_shift, eval_frame, score
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_per_split_20260708"
CURRENT_BEST = {
    "model_id": "pmemo_valence_correlated_song_mean_guardtopccc_20260708",
    "rmse": 0.19051822436650612,
    "mae": 0.1484786057954706,
    "pcc": 0.8086756178359321,
    "ccc": 0.7968998149979719,
}


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split, data): pd.read_csv(INPUT_DIR / split / f"{data}_predictions.csv")
        for split in SPLITS
        for data in DATA_SPLITS
    }
    selected_rows = []
    selected_frames = {}
    all_rows = []
    for split in SPLITS:
        candidates = []
        cache = {}
        for shift_frames in range(-16, 17):
            row = {"split_name": split, "shift_frames": shift_frames, "shift_seconds": shift_frames * 0.5}
            for data in DATA_SPLITS:
                shifted = apply_shift(frames[(split, data)], shift_frames)
                cache[(shift_frames, data)] = shifted
                row.update({f"{data}_{m}": v for m, v in eval_frame(shifted).items()})
            val_m = {m: row[f"val_{m}"] for m in ["rmse", "mae", "pcc", "ccc"]}
            row["val_score"] = score(val_m)
            candidates.append(row)
            all_rows.append(row)
        selected = sorted(candidates, key=lambda r: (r["val_score"], r["val_rmse"], -r["val_ccc"]))[0]
        selected_rows.append(selected)
        for data in DATA_SPLITS:
            selected_frames[(split, data)] = cache[(int(selected["shift_frames"]), data)]
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        for data in DATA_SPLITS:
            selected_frames[(split, data)].to_csv(split_out / f"{data}_predictions.csv", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_temporal_shift_per_split_candidates.csv", index=False)
    per_split = []
    for split in SPLITS:
        row = {"split_name": split}
        selected = next(r for r in selected_rows if r["split_name"] == split)
        row["selected_shift_frames"] = int(selected["shift_frames"])
        row["selected_shift_seconds"] = float(selected["shift_seconds"])
        for data in DATA_SPLITS:
            row.update({f"{data}_{m}": v for m, v in eval_frame(selected_frames[(split, data)]).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_guardtopccc_temporal_shift_per_split_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "diagnostic only: each CV split chooses its own temporal shift by validation score",
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
