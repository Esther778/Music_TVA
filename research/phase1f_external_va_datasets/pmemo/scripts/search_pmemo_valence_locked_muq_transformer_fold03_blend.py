#!/usr/bin/env python3
"""Gate MuQ Transformer complementarity against the current fold03 fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import metrics
from train_pmemo_valence_dev_only_trials import moving_average_by_song


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713/fold_03_val_predictions.csv"
TF_DIR = ROOT / "outputs/valence_locked_muq_transformer_smile_fold03_20260714"
OUT = ROOT / "outputs/valence_locked_muq_transformer_fold03_blend_20260714"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)[["musicId", "time_sec", "target_valence", "pred_valence"]].rename(columns={"pred_valence": "base_pred"})
    tf = pd.read_csv(TF_DIR / "val_predictions.csv")
    summary = json.loads((TF_DIR / "summary.json").read_text())
    merged = base.merge(tf[["musicId", "time_sec", "pred_valence"]], on=["musicId", "time_sec"], validate="one_to_one")
    raw = merged["pred_valence"].to_numpy(dtype=np.float64)
    post = summary["best_val_postprocess"]
    smooth = moving_average_by_song(raw, merged["musicId"].to_numpy(), int(post["window"]))
    smooth_base = (1.0 - float(post["smooth_blend"])) * raw + float(post["smooth_blend"]) * smooth
    params = json.loads(post["params"])
    calibrated = float(params["a"]) * smooth_base + float(params["b"])
    tf_post = (1.0 - float(post["cal_blend"])) * smooth_base + float(post["cal_blend"]) * calibrated
    target = merged["target_valence"].to_numpy(dtype=np.float64)
    base_pred = merged["base_pred"].to_numpy(dtype=np.float64)
    rows = []
    for weight in np.arange(0.0, 0.301, 0.01):
        pred = (1.0 - weight) * base_pred + weight * tf_post
        rows.append({"transformer_weight": float(weight), **metrics(pred, target)})
    table = pd.DataFrame(rows).sort_values(["rmse", "mae"])
    table.to_csv(OUT / "candidates.csv", index=False)
    selected = table.iloc[0].to_dict()
    result = {
        "created_at": "2026-07-14",
        "selection_boundary": "fold03 development validation only; final locked test excluded",
        "base": metrics(base_pred, target),
        "selected": selected,
        "decision": "expand_transformer_all_folds" if selected["rmse"] <= metrics(base_pred, target)["rmse"] - 0.0003 else "reject_transformer_expansion",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
