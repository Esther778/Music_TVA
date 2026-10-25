#!/usr/bin/env python3
"""Try a tiny validation-selected blend of current Valence best and OpenSMILE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


BASE_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
CANDIDATE_DIR = ROOT / "outputs" / "opensmile_valence_multisplit_trials_20260708" / "v_smile_h96_l1_do010_rmse"
OUT_DIR = ROOT / "outputs" / "valence_current_best_opensmile_weak_blend_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def load_pair(split_name: str, data_split: str) -> pd.DataFrame:
    base = pd.read_csv(BASE_DIR / split_name / f"{data_split}_predictions.csv")
    cand = pd.read_csv(CANDIDATE_DIR / split_name / f"{data_split}_predictions.csv")
    out = base[["musicId", "time_sec", target_col(base), pred_col(base)]].rename(
        columns={target_col(base): "target_valence", pred_col(base): "base_pred"}
    )
    mini = cand[["musicId", "time_sec", pred_col(cand)]].rename(columns={pred_col(cand): "candidate_pred"})
    out = out.merge(mini, on=["musicId", "time_sec"], how="inner")
    if len(out) != len(base):
        raise ValueError(f"Candidate/base mismatch for {split_name} {data_split}")
    return out


def eval_frame(frame: pd.DataFrame, weight: float) -> dict[str, float]:
    pred = (1.0 - weight) * frame["base_pred"].to_numpy(dtype=np.float64) + weight * frame["candidate_pred"].to_numpy(dtype=np.float64)
    return metrics(pred, frame["target_valence"].to_numpy(dtype=np.float64))


def macro(frames: dict[tuple[str, str], pd.DataFrame], data_split: str, weight: float) -> dict[str, float]:
    rows = [eval_frame(frames[(split, data_split)], weight) for split in SPLITS]
    return {k: float(np.mean([row[k] for row in rows])) for k in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - 0.18) / 0.18
        + 1.4 * max(0.0, m["mae"] - 0.15) / 0.15
        + 1.2 * max(0.0, 0.80 - m["pcc"]) / 0.80
        + 2.0 * max(0.0, 0.80 - m["ccc"]) / 0.80
    )


def save_predictions(frames: dict[tuple[str, str], pd.DataFrame], weight: float) -> None:
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            frame = frames[(split_name, data_split)].copy()
            frame["pred_valence"] = (1.0 - weight) * frame["base_pred"] + weight * frame["candidate_pred"]
            frame[["musicId", "time_sec", "target_valence", "pred_valence", "base_pred", "candidate_pred"]].to_csv(
                split_out / f"{data_split}_predictions.csv", index=False
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(split, data): load_pair(split, data) for split in SPLITS for data in DATA_SPLITS}
    rows: list[dict[str, Any]] = []
    for weight in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        val_m = macro(frames, "val", weight)
        test_m = macro(frames, "test", weight)
        rows.append(
            {
                "weight_opensmile": weight,
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "selection_score": score(val_m),
            }
        )
    table = pd.DataFrame(rows)
    selected = table.sort_values(["selection_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
    save_predictions(frames, float(selected["weight_opensmile"]))
    test = {k: float(selected[f"test_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_current_best_opensmile_weak_blend_20260708",
        "base_model": CURRENT_BEST,
        "candidate_model": str(CANDIDATE_DIR),
        "selection_rule": "one global OpenSMILE blend weight selected on validation macro score; test reported after selection",
        "selected": selected,
        "test_macro": test,
        "delta_vs_current_best": {k: float(test[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
        "test_gap_rows": threshold_rows(test),
    }
    table.to_csv(OUT_DIR / "all_weak_blend_candidates.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
