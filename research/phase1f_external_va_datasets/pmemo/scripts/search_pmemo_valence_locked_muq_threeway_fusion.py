#!/usr/bin/env python3
"""Three-way locked-dev fusion: reference, MuQ BiLSTM, and MuQ MLP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_muq_fusion import OUT_DIR as TWO_WAY_OUT
from search_pmemo_valence_locked_muq_fusion import prepare_fold
from train_pmemo_mert_valence_baseline import metrics, threshold_rows
from train_pmemo_valence_dev_only_trials import fit_calibration, moving_average_by_song


ROOT = Path(__file__).resolve().parents[1]
TRIAL = "muq_mlp_h512_do035_lr3e4"
MLP_DIRS = {
    fold: ROOT / "outputs" / f"valence_locked_muq_layer06_mlp_{fold.replace('_', '')}_20260713" / TRIAL
    for fold in ["fold_01", "fold_02", "fold_03"]
}
OUT_DIR = ROOT / "outputs" / "valence_locked_muq_threeway_fusion_20260713"


def add_mlp(fold: str, frame: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(MLP_DIRS[fold] / "val_predictions.csv")
    summary = json.loads((MLP_DIRS[fold] / "summary.json").read_text())
    post = summary["post"]
    merged = frame.merge(pred, on=["musicId", "time_sec"], suffixes=("", "_mlp"), validate="one_to_one")
    raw = merged["pred_valence"].to_numpy(dtype=np.float64)
    smooth = moving_average_by_song(raw, merged["musicId"].to_numpy(), int(post["window"]))
    base = (1.0 - float(post["smooth_blend"])) * raw + float(post["smooth_blend"]) * smooth
    calibrated, _ = fit_calibration(base, merged["target_valence"].to_numpy(), str(post["method"]))
    merged["mlp_post_pred_valence"] = (1.0 - float(post["cal_blend"])) * base + float(post["cal_blend"]) * calibrated
    return merged.drop(columns=["target_valence_mlp", "pred_valence"])


def evaluate(data, bilstm_weight: float, mlp_weight: float, cal_blend: float):
    per_fold, predictions = {}, {}
    base_weight = 1.0 - bilstm_weight - mlp_weight
    for fold, frame in data.items():
        fusion = (
            base_weight * frame["base_pred_valence"].to_numpy()
            + bilstm_weight * frame["muq_post_pred_valence"].to_numpy()
            + mlp_weight * frame["mlp_post_pred_valence"].to_numpy()
        )
        target = frame["target_valence"].to_numpy()
        calibrated, _ = fit_calibration(fusion, target, "mean_std")
        pred = (1.0 - cal_blend) * fusion + cal_blend * calibrated
        predictions[fold] = pred
        per_fold[fold] = metrics(pred, target)
    macro = {key: float(np.mean([value[key] for value in per_fold.values()])) for key in ["rmse", "mae", "pcc", "ccc"]}
    return macro, per_fold, predictions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {fold: add_mlp(fold, prepare_fold(fold)) for fold in MLP_DIRS}
    rows, cache = [], {}
    for bilstm_weight in np.round(np.arange(0.0, 0.8001, 0.05), 3):
        for mlp_weight in np.round(np.arange(0.0, 0.8001, 0.05), 3):
            if bilstm_weight + mlp_weight > 0.90:
                continue
            for cal_blend in np.round(np.arange(0.0, 0.5001, 0.05), 3):
                config = (float(bilstm_weight), float(mlp_weight), float(cal_blend))
                macro, per_fold, predictions = evaluate(data, *config)
                rows.append({"muq_bilstm_weight": config[0], "muq_mlp_weight": config[1], "base_weight": 1.0-config[0]-config[1], "mean_std_blend": config[2], **{f"macro_{k}": v for k, v in macro.items()}})
                cache[config] = (per_fold, predictions)
    table = pd.DataFrame(rows).sort_values(["macro_rmse", "macro_mae"])
    table.to_csv(OUT_DIR / "candidates.csv", index=False)
    best = table.iloc[0].to_dict()
    config = (best["muq_bilstm_weight"], best["muq_mlp_weight"], best["mean_std_blend"])
    per_fold, predictions = cache[config]
    for fold, frame in data.items():
        frame = frame.copy()
        frame["pred_valence"] = predictions[fold]
        frame.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
    macro = {key: best[f"macro_{key}"] for key in ["rmse", "mae", "pcc", "ccc"]}
    ccc_pass = table[(table.macro_mae <= 0.15) & (table.macro_pcc >= 0.80) & (table.macro_ccc >= 0.80)]
    summary = {
        "created_at": "2026-07-13",
        "selection_boundary": "Locked development folds only; final locked test excluded.",
        "selected": best,
        "macro": macro,
        "per_fold": per_fold,
        "gap_rows": threshold_rows(macro),
        "ccc_pass_lowest_rmse": None if ccc_pass.empty else ccc_pass.iloc[0].to_dict(),
        "two_way_reference": str(TWO_WAY_OUT / "summary.json"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
