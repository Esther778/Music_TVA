#!/usr/bin/env python3
"""Materialize an RMSE-primary validation-selected Arousal postprocess."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from search_pmemo_arousal_best_postprocess import OUT_DIR as SOURCE_DIR
from search_pmemo_arousal_best_postprocess import INPUT_DIR, apply, eval_df, fit_params
from train_pmemo_opensmile_arousal_trials import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "opensmile_arousal_rmse_primary_postprocess_20260629"
CURRENT_BEST = {
    "model_id": "a_only_h96_l1_do015_mse",
    "rmse": 0.18095970153808594,
    "mae": 0.14009205996990204,
    "pcc": 0.8526338558393907,
    "ccc": 0.8481215815055151,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(SOURCE_DIR / "all_postprocess_candidates.csv")
    allowed = candidates[
        (candidates["val_pcc"] >= PASS["pcc"])
        & (candidates["val_ccc"] >= PASS["ccc"])
    ].copy()
    if allowed.empty:
        allowed = candidates.copy()
    selected = allowed.sort_values(
        ["val_rmse", "val_mae", "val_ccc"],
        ascending=[True, True, False],
    ).iloc[0].to_dict()
    frames = {split: pd.read_csv(INPUT_DIR / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    params = fit_params(frames["val"], str(selected["method"]))
    clip = None if selected["clip"] == "none" else float(str(selected["clip"]).replace("clip", "")) / 100.0
    corrected = {
        split: apply(frames[split], params, float(selected["strength"]), int(selected["smooth_window"]), clip)
        for split in ["train", "val", "test"]
    }
    for split, df in corrected.items():
        df.to_csv(OUT_DIR / f"{split}_predictions.csv", index=False)
    selected_metrics = {split: eval_df(df) for split, df in corrected.items()}
    test_m = selected_metrics["test"]
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_arousal_rmse_primary_postprocess_20260629",
        "purpose": "Arousal postprocess selected with validation RMSE primary after requiring validation PCC/CCC pass",
        "selection_rule": "from validation candidates with val_pcc>=0.80 and val_ccc>=0.80, choose minimum val_rmse; test labels not used for selection",
        "input_model": CURRENT_BEST,
        "selected": selected,
        "selected_metrics": selected_metrics,
        "test_gap_rows": threshold_rows(test_m),
        "delta_vs_input_model": {k: float(test_m[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
