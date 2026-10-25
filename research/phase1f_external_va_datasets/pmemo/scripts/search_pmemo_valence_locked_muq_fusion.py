#!/usr/bin/env python3
"""Fuse the locked-dev Valence reference with MuQ layer-6 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import metrics, threshold_rows
from train_pmemo_valence_dev_only_trials import fit_calibration, moving_average_by_song


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "valence_locked_arousal_assisted_shift_wide_rmse_20260713"
MUQ_DIRS = {
    "fold_01": ROOT / "outputs" / "valence_locked_muq_layer06_fold01_20260713",
    "fold_02": ROOT / "outputs" / "valence_locked_muq_layer06_fold02_20260713",
    "fold_03": ROOT / "outputs" / "valence_locked_muq_layer06_fold03_20260713",
}
TRIAL = "v_dev_wider_h96_do035_lr25e4"
OUT_DIR = ROOT / "outputs" / "valence_locked_muq_reference_fusion_20260713"


def prepare_fold(fold: str) -> pd.DataFrame:
    base = pd.read_csv(BASE_DIR / f"{fold}_val_predictions.csv")
    muq = pd.read_csv(MUQ_DIRS[fold] / TRIAL / "val_predictions.csv")
    merged = base.merge(muq, on=["musicId", "time_sec"], suffixes=("_base", "_muq"), validate="one_to_one")
    if not np.allclose(merged["target_valence_base"], merged["target_valence_muq"]):
        raise ValueError(f"Target mismatch in {fold}")
    target = merged["target_valence_base"].to_numpy(dtype=np.float64)
    muq_raw = merged["pred_valence_muq"].to_numpy(dtype=np.float64)
    muq_smooth = moving_average_by_song(muq_raw, merged["musicId"].to_numpy(), 41)
    muq_post, params = fit_calibration(muq_smooth, target, "linear")
    return pd.DataFrame(
        {
            "fold": fold,
            "musicId": merged["musicId"],
            "time_sec": merged["time_sec"],
            "target_valence": target,
            "base_pred_valence": merged["pred_valence_base"],
            "muq_raw_pred_valence": muq_raw,
            "muq_post_pred_valence": muq_post,
            "muq_calibration_a": float(params["a"]),
            "muq_calibration_b": float(params["b"]),
        }
    )


def evaluate_config(
    data: dict[str, pd.DataFrame],
    weight: float,
    calibration_blend: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    per_fold = {}
    predictions = {}
    for fold, frame in data.items():
        fusion = (1.0 - weight) * frame["base_pred_valence"].to_numpy() + weight * frame["muq_post_pred_valence"].to_numpy()
        calibrated, _ = fit_calibration(fusion, frame["target_valence"].to_numpy(), "mean_std")
        pred = (1.0 - calibration_blend) * fusion + calibration_blend * calibrated
        predictions[fold] = pred
        per_fold[fold] = metrics(pred, frame["target_valence"].to_numpy())
    macro = {
        key: float(np.mean([fold_metrics[key] for fold_metrics in per_fold.values()]))
        for key in ["rmse", "mae", "pcc", "ccc"]
    }
    return macro, per_fold, predictions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {fold: prepare_fold(fold) for fold in MUQ_DIRS}
    candidates = []
    details = {}
    for weight in np.round(np.arange(0.0, 1.0001, 0.025), 3):
        for calibration_blend in np.round(np.arange(0.0, 0.8001, 0.05), 3):
            macro, per_fold, predictions = evaluate_config(data, float(weight), float(calibration_blend))
            candidates.append(
                {
                    "muq_weight": float(weight),
                    "mean_std_blend": float(calibration_blend),
                    **{f"macro_{key}": value for key, value in macro.items()},
                }
            )
            details[(float(weight), float(calibration_blend))] = (per_fold, predictions)
    table = pd.DataFrame(candidates).sort_values(["macro_rmse", "macro_mae"], ascending=True)
    table.to_csv(OUT_DIR / "global_weight_candidates.csv", index=False)
    selected = table.iloc[0]
    weight = float(selected["muq_weight"])
    calibration_blend = float(selected["mean_std_blend"])
    macro = {key: float(selected[f"macro_{key}"]) for key in ["rmse", "mae", "pcc", "ccc"]}
    per_fold, selected_predictions = details[(weight, calibration_blend)]
    for fold, frame in data.items():
        frame = frame.copy()
        frame["fusion_pred_valence"] = (1.0 - weight) * frame["base_pred_valence"] + weight * frame["muq_post_pred_valence"]
        frame["pred_valence"] = selected_predictions[fold]
        frame.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
    ccc_pass = table[
        (table["macro_mae"] <= 0.15)
        & (table["macro_pcc"] >= 0.80)
        & (table["macro_ccc"] >= 0.80)
    ].sort_values(["macro_rmse", "macro_mae"])
    ccc_pass_variant = None if ccc_pass.empty else ccc_pass.iloc[0].to_dict()
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Global-weight fusion of strict locked-dev Valence reference and MuQ layer-6",
        "selection_boundary": "Three locked development validation folds only; final locked test not read.",
        "muq_weight": weight,
        "base_weight": 1.0 - weight,
        "mean_std_blend": calibration_blend,
        "muq_postprocess": "song-wise moving average window 41, then fold-development linear calibration",
        "fusion_calibration": "fold-development mean/std calibration with one global blend",
        "macro": macro,
        "per_fold": per_fold,
        "ccc_pass_variant": ccc_pass_variant,
        "gap_rows": threshold_rows(macro),
        "base_reference_macro": {"rmse": 0.193121, "mae": 0.151375, "pcc": 0.796623, "ccc": 0.776367},
        "delta_vs_base": {
            "rmse": macro["rmse"] - 0.193121,
            "mae": macro["mae"] - 0.151375,
            "pcc": macro["pcc"] - 0.796623,
            "ccc": macro["ccc"] - 0.776367,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
