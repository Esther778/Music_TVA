#!/usr/bin/env python3
"""Per-split validation-selected temporal shift + smoothing for Valence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_current_best_smoothing import apply_smoothing
from search_pmemo_valence_temporal_shift import apply_shift, eval_frame
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_hgb_song_mean_20260708"
OUT_DIR = ROOT / "outputs" / "valence_per_split_shift_smoothing_20260709"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        4.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 0.9 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.2 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.02 * m["rmse"]
    )


def smooth_candidates() -> list[tuple[str, float, float]]:
    out = [("identity", 0.0, 0.0)]
    out += [("ma", float(w), 1.0) for w in [15, 21, 31]]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv") for s in SPLITS for d in DATA_SPLITS}
    selected_by_split: dict[str, dict[str, Any]] = {}
    selected_frames: dict[tuple[str, str], pd.DataFrame] = {}
    all_rows: list[dict[str, Any]] = []

    for split_name in SPLITS:
        best_row = None
        best_frames = None
        for shift_frames in [-24, -21, -18, -15, -12, -9, -6, -3, 0]:
            shifted = {d: apply_shift(frames[(split_name, d)], shift_frames) for d in DATA_SPLITS}
            for method, param, blend in smooth_candidates():
                smoothed = {d: apply_smoothing(shifted[d], method, param, blend) for d in DATA_SPLITS}
                row = {
                    "split_name": split_name,
                    "shift_frames": int(shift_frames),
                    "shift_seconds": float(shift_frames * 0.5),
                    "smooth_method": method,
                    "smooth_param": float(param),
                    "smooth_blend": float(blend),
                }
                for d in DATA_SPLITS:
                    row.update({f"{d}_{k}": v for k, v in eval_frame(smoothed[d]).items()})
                row["val_score"] = score({k: row[f"val_{k}"] for k in ["rmse", "mae", "pcc", "ccc"]})
                all_rows.append(row)
                key = (row["val_score"], row["val_rmse"], -row["val_ccc"])
                if best_row is None or key < (best_row["val_score"], best_row["val_rmse"], -best_row["val_ccc"]):
                    best_row = row
                    best_frames = smoothed
        if best_row is None or best_frames is None:
            raise RuntimeError(f"No candidate selected for {split_name}")
        selected_by_split[split_name] = best_row
        for d in DATA_SPLITS:
            selected_frames[(split_name, d)] = best_frames[d]

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_per_split_shift_smoothing_candidates.csv", index=False)
    per_split: list[dict[str, Any]] = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name, "selected_config": selected_by_split[split_name]}
        for d in DATA_SPLITS:
            df = selected_frames[(split_name, d)]
            df.to_csv(split_out / f"{d}_predictions.csv", index=False)
            row.update({f"{d}_{k}": v for k, v in eval_frame(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    selected_val = macro(per_split, "val")
    summary = {
        "created_at": "2026-07-09",
        "model_id": "pmemo_valence_per_split_shift_smoothing_20260709",
        "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "selection_rule": "for each train/val/test split, temporal shift and smoothing selected only on that split's validation set; test labels unused",
        "current_best_reference": CURRENT_BEST,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "per_split": per_split,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
