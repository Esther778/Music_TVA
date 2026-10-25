#!/usr/bin/env python3
"""Blend a postprocessed fold03 candidate into the current three-way base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import metrics
from train_pmemo_valence_dev_only_trials import moving_average_by_song


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713/fold_03_val_predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)[["musicId", "time_sec", "target_valence", "pred_valence"]].rename(columns={"pred_valence": "base_pred"})
    candidate = pd.read_csv(args.candidate_dir / "val_predictions.csv")
    summary = json.loads((args.candidate_dir / "summary.json").read_text())
    post = summary["best_val_postprocess"]
    merged = base.merge(candidate[["musicId", "time_sec", "pred_valence"]], on=["musicId", "time_sec"], validate="one_to_one")
    raw = merged["pred_valence"].to_numpy(dtype=np.float64)
    smooth = moving_average_by_song(raw, merged["musicId"].to_numpy(), int(post["window"]))
    interim = (1.0 - float(post["smooth_blend"])) * raw + float(post["smooth_blend"]) * smooth
    params = json.loads(post["params"])
    if post["method"] == "bias":
        calibrated = interim + float(params["b"])
    else:
        calibrated = float(params.get("a", 1.0)) * interim + float(params.get("b", 0.0))
    candidate_post = (1.0 - float(post["cal_blend"])) * interim + float(post["cal_blend"]) * calibrated
    target = merged["target_valence"].to_numpy(dtype=np.float64)
    base_pred = merged["base_pred"].to_numpy(dtype=np.float64)
    rows = []
    for weight in np.arange(0.0, 0.301, 0.01):
        pred = (1.0 - weight) * base_pred + weight * candidate_post
        rows.append({"candidate_weight": float(weight), **metrics(pred, target)})
    table = pd.DataFrame(rows).sort_values(["rmse", "mae"])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    base_metrics = metrics(base_pred, target)
    selected = table.iloc[0].to_dict()
    result = {
        "created_at": "2026-07-14", "candidate": str(args.candidate_dir),
        "selection_boundary": "fold03 development validation only; final locked test excluded",
        "base": base_metrics, "selected": selected,
        "decision": "promote_candidate" if selected["rmse"] <= base_metrics["rmse"] - 0.0003 else "reject_candidate",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
