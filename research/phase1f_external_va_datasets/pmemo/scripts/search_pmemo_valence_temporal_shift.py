#!/usr/bin/env python3
"""Validation-selected temporal shift for Valence prediction curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_correlated_song_mean_guardtopccc_20260708"
OUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_wide_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_correlated_song_mean_guardtopccc_20260708",
    "rmse": 0.19051822436650612,
    "mae": 0.1484786057954706,
    "pcc": 0.8086756178359321,
    "ccc": 0.7968998149979719,
}


def shift_one(pred: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0 or len(pred) <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    if shift_frames > 0:
        out[:-shift_frames] = pred[shift_frames:]
        out[-shift_frames:] = pred[-1]
    else:
        k = -shift_frames
        out[:k] = pred[0]
        out[k:] = pred[:-k]
    return out


def apply_shift(df: pd.DataFrame, shift_frames: int) -> pd.DataFrame:
    out = df.copy()
    pred_col = "pred_valence" if "pred_valence" in out.columns else "pred"
    pieces = []
    for _, g in out.groupby("musicId", sort=False):
        pred = g[pred_col].to_numpy(dtype=np.float64)
        gg = g.copy()
        gg["base_pred_valence"] = pred
        gg["pred_valence"] = shift_one(pred, shift_frames)
        gg["temporal_shift_frames"] = int(shift_frames)
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True)


def eval_frame(df: pd.DataFrame) -> dict[str, float]:
    target_col = "target_valence" if "target_valence" in df.columns else "target"
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df[target_col].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - 0.18) / 0.18
        + 1.4 * max(0.0, m["mae"] - 0.15) / 0.15
        + 1.1 * max(0.0, 0.80 - m["pcc"]) / 0.80
        + 2.4 * max(0.0, 0.80 - m["ccc"]) / 0.80
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split, data): pd.read_csv(INPUT_DIR / split / f"{data}_predictions.csv")
        for split in SPLITS
        for data in DATA_SPLITS
    }
    rows = []
    shifted_cache: dict[int, dict[tuple[str, str], pd.DataFrame]] = {}
    for shift_frames in range(-16, 17):
        per_split = []
        shifted = {}
        for split in SPLITS:
            row = {"split_name": split}
            for data in DATA_SPLITS:
                df = apply_shift(frames[(split, data)], shift_frames)
                shifted[(split, data)] = df
                row.update({f"{data}_{m}": v for m, v in eval_frame(df).items()})
            per_split.append(row)
        val_m = macro(per_split, "val")
        test_m = macro(per_split, "test")
        rows.append(
            {
                "shift_frames": int(shift_frames),
                "shift_seconds": float(shift_frames * 0.5),
                **{f"val_{m}": v for m, v in val_m.items()},
                **{f"test_{m}": v for m, v in test_m.items()},
                "val_score": score(val_m),
            }
        )
        shifted_cache[shift_frames] = shifted
    result = pd.DataFrame(rows)
    selected = result.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    shift = int(selected["shift_frames"])
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        for data in DATA_SPLITS:
            shifted_cache[shift][(split, data)].to_csv(split_out / f"{data}_predictions.csv", index=False)
    test = {m: float(selected[f"test_{m}"]) for m in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_guardtopccc_temporal_shift_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "one global temporal shift selected on validation macro score; test labels not used",
        "selected": selected,
        "selected_test": test,
        "delta_vs_current_best": {m: float(test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(test),
    }
    result.to_csv(OUT_DIR / "all_temporal_shift_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
