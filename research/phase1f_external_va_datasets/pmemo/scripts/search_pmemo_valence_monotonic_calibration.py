#!/usr/bin/env python3
"""Validation-selected monotonic calibration for current Valence best."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BASE_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
OUT_DIR = ROOT / "outputs" / "valence_current_best_monotonic_calibration_20260708"
MODEL_ID = "pmemo_valence_current_best_monotonic_calibration_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}


def load(split_name: str, data_split: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / split_name / f"{data_split}_predictions.csv")
    return df.rename(columns={"pred": "pred_valence", "target": "target_valence"})


def fit_iso(train: pd.DataFrame, y_min: float, y_max: float) -> IsotonicRegression:
    return IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip").fit(
        train["pred_valence"].to_numpy(dtype=np.float64),
        train["target_valence"].to_numpy(dtype=np.float64),
    )


def macro(preds: dict[tuple[str, str], np.ndarray], frames: dict[tuple[str, str], pd.DataFrame], data_split: str) -> dict[str, float]:
    rows = []
    for split_name in SPLITS:
        rows.append(metrics(preds[(split_name, data_split)], frames[(split_name, data_split)]["target_valence"].to_numpy(dtype=np.float64)))
    return {k: float(np.mean([row[k] for row in rows])) for k in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - 0.18) / 0.18
        + 1.4 * max(0.0, m["mae"] - 0.15) / 0.15
        + 1.2 * max(0.0, 0.80 - m["pcc"]) / 0.80
        + 2.0 * max(0.0, 0.80 - m["ccc"]) / 0.80
    )


def save_predictions(frames: dict[tuple[str, str], pd.DataFrame], preds: dict[tuple[str, str], np.ndarray]) -> None:
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            frame = frames[(split_name, data_split)].copy()
            frame["base_pred_valence"] = frame["pred_valence"]
            frame["pred_valence"] = preds[(split_name, data_split)]
            frame[["musicId", "time_sec", "target_valence", "pred_valence", "base_pred_valence"]].to_csv(
                split_out / f"{data_split}_predictions.csv", index=False
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(split, data): load(split, data) for split in SPLITS for data in DATA_SPLITS}
    train_targets = np.concatenate([frames[(split, "train")]["target_valence"].to_numpy(dtype=np.float64) for split in SPLITS])
    y_min, y_max = float(np.quantile(train_targets, 0.002)), float(np.quantile(train_targets, 0.998))
    iso_by_split = {split: fit_iso(frames[(split, "train")], y_min, y_max) for split in SPLITS}
    rows: list[dict[str, Any]] = []
    pred_cache: dict[float, dict[tuple[str, str], np.ndarray]] = {}
    for blend in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 1.0]:
        preds = {}
        for split in SPLITS:
            iso = iso_by_split[split]
            for data in DATA_SPLITS:
                base = frames[(split, data)]["pred_valence"].to_numpy(dtype=np.float64)
                calibrated = iso.predict(base)
                preds[(split, data)] = np.clip((1.0 - blend) * base + blend * calibrated, -1.0, 1.0)
        pred_cache[blend] = preds
        val_m = macro(preds, frames, "val")
        test_m = macro(preds, frames, "test")
        rows.append(
            {
                "blend": blend,
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "selection_score": score(val_m),
            }
        )
    table = pd.DataFrame(rows)
    selected = table.sort_values(["selection_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    selected_blend = float(selected["blend"])
    save_predictions(frames, pred_cache[selected_blend])
    test = {k: float(selected[f"test_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-08",
        "model_id": MODEL_ID,
        "base_model": CURRENT_BEST,
        "selection_rule": "isotonic pred-to-target calibrator fitted on train per split; one global blend selected on validation macro score",
        "selected": selected,
        "test_macro": test,
        "delta_vs_current_best": {k: float(test[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
        "test_gap_rows": threshold_rows(test),
    }
    table.to_csv(OUT_DIR / "all_monotonic_calibration_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
