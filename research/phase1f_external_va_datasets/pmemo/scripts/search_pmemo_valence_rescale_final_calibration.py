#!/usr/bin/env python3
"""Final output calibration after the promoted prediction-only Valence rescale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_final_output_calibration import (
    apply_calibrated,
    eval_df,
    fit_calibrators,
    selection_score,
)
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
OUT_DIR = ROOT / "outputs" / "valence_rescale_final_calibration_20260628"
CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.1953163033728265,
    "mae": 0.15179290177231072,
    "pcc": 0.7977655929469954,
    "ccc": 0.7848342582143512,
}
CCC_EPS = 0.001


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def evaluate_split(split_dir: Path) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    pcol = pred_col(frames["train"])
    tcol = target_col(frames["train"])
    calibrators = fit_calibrators(
        frames["train"][pcol].to_numpy(dtype=np.float64),
        frames["train"][tcol].to_numpy(dtype=np.float64),
    )
    baseline = {split: eval_df(frames[split]) for split in ["train", "val", "test"]}
    rows = []
    cache = {}
    for calibrator in calibrators:
        for blend in np.linspace(0.0, 1.0, 21):
            for clip_name, clip in [("none", None), ("clip095", 0.95), ("clip090", 0.90), ("clip085", 0.85)]:
                split_metrics = {}
                for split in ["train", "val", "test"]:
                    corrected = apply_calibrated(frames[split], calibrator, float(blend), clip)
                    split_metrics[split] = eval_df(corrected)
                    cache[(split, calibrator.name, float(blend), clip_name)] = corrected
                rows.append(
                    {
                        "split_name": split_dir.name,
                        "calibrator": calibrator.name,
                        "blend": float(blend),
                        "clip": clip_name,
                        **{f"{split}_{k}": v for split, m in split_metrics.items() for k, v in m.items()},
                        "val_selection_score": selection_score(split_metrics["val"]),
                    }
                )
    df = pd.DataFrame(rows)
    allowed = df[df["val_ccc"] >= baseline["val"]["ccc"] - CCC_EPS].copy()
    if allowed.empty:
        selected = {
            "split_name": split_dir.name,
            "source": "baseline",
            "calibrator": "identity",
            "blend": 0.0,
            "clip": "none",
            **{f"{split}_{k}": v for split, m in baseline.items() for k, v in m.items()},
            "val_selection_score": None,
        }
    else:
        selected = allowed.sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False]).iloc[0].to_dict()
        selected["source"] = "calibration"
    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        if selected["source"] == "baseline":
            frames[split].to_csv(out_split / f"{split}_predictions.csv", index=False)
        else:
            cache[(split, str(selected["calibrator"]), float(selected["blend"]), str(selected["clip"]))].to_csv(
                out_split / f"{split}_predictions.csv", index=False
            )
    summary = {
        "split_name": split_dir.name,
        "ccc_eps": CCC_EPS,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_rescale_final_calibration_20260628",
        "input_model": CURRENT_BEST,
        "selection_rule": f"final calibrator selected only if val_ccc >= baseline_val_ccc - {CCC_EPS}",
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            vals = np.asarray(
                [
                    r["selected"][f"test_{metric}"] if variant == "selected" else r["baseline"]["test"][metric]
                    for r in rows
                ],
                dtype=np.float64,
            )
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: out[f"selected_test_{metric}_mean"] for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    candidates = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        summary, df = evaluate_split(split_dir)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_calibration_results.csv", index=False)
    pd.DataFrame([r["selected"] for r in rows]).to_csv(OUT_DIR / "selected_by_split.csv", index=False)
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
