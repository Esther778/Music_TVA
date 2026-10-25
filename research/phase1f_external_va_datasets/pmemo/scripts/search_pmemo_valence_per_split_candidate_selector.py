#!/usr/bin/env python3
"""Validation-only per-split selector over existing Valence prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUTPUT_ROOT = ROOT / "outputs"
OUT_DIR = OUTPUT_ROOT / "valence_per_split_candidate_selector_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST_DIR = OUTPUT_ROOT / "valence_extra_wide_rmse_shift_smoothing_20260708"
CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_valence", "pred"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {df.columns.tolist()}")


def target_col(df: pd.DataFrame) -> str:
    for col in ["target_valence", "target"]:
        if col in df.columns:
            return col
    raise ValueError(f"No target column in {df.columns.tolist()}")


def eval_path(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return metrics(df[pred_col(df)].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def compatible_candidate_dirs() -> list[Path]:
    out = []
    for root in OUTPUT_ROOT.iterdir():
        if not root.is_dir():
            continue
        ok = True
        for split in SPLITS:
            for data in DATA_SPLITS:
                if not (root / split / f"{data}_predictions.csv").exists():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            try:
                sample = pd.read_csv(root / SPLITS[0] / "val_predictions.csv", nrows=5)
                pred_col(sample)
                target_col(sample)
            except Exception:
                ok = False
        if ok:
            out.append(root)
    return sorted(out)


def macro(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{prefix}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def normalize_prediction_file(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)
    pc = pred_col(df)
    tc = target_col(df)
    out = df.copy()
    if pc != "pred_valence":
        out["pred_valence"] = out[pc]
    if tc != "target_valence":
        out["target_valence"] = out[tc]
    out.to_csv(dst, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = compatible_candidate_dirs()
    rows = []
    for cand in candidates:
        for split in SPLITS:
            try:
                val_m = eval_path(cand / split / "val_predictions.csv")
                test_m = eval_path(cand / split / "test_predictions.csv")
            except Exception as exc:
                rows.append({"candidate": cand.name, "split_name": split, "error": repr(exc)})
                continue
            rows.append(
                {
                    "candidate": cand.name,
                    "candidate_path": str(cand),
                    "split_name": split,
                    **{f"val_{m}": v for m, v in val_m.items()},
                    **{f"test_{m}": v for m, v in test_m.items()},
                    "val_score": score(val_m),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "all_per_split_candidate_scores.csv", index=False)
    usable = table[~table.get("error", pd.Series(index=table.index, dtype=object)).notna()].copy()

    selected_rows = []
    for split in SPLITS:
        split_table = usable[usable["split_name"] == split].copy()
        # Validation-only selection; test columns are ignored here.
        selected = split_table.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected_rows.append(selected)
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        src_root = Path(str(selected["candidate_path"]))
        for data in DATA_SPLITS:
            normalize_prediction_file(src_root / split / f"{data}_predictions.csv", split_out / f"{data}_predictions.csv")

    selected_test = macro(selected_rows, "test")
    selected_val = macro(selected_rows, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_per_split_candidate_selector_20260708",
        "purpose": "diagnostic validation-only selector over existing Valence prediction artifacts",
        "selection_rule": "for each fixed song-level split, select candidate by validation pass-gap score only; test ignored until final report",
        "num_candidates": int(len(candidates)),
        "current_best_reference": CURRENT_BEST,
        "selected_per_split": selected_rows,
        "selected_val": selected_val,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
        "note": "If this improves, inspect selected families and convert them into a single deployable training policy before final teacher use.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
