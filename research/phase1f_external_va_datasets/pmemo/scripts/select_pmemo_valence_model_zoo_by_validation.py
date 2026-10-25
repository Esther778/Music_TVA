#!/usr/bin/env python3
"""Validation-selected model-zoo diagnostic for PMEmo Valence.

For each split, select one existing prediction artifact by validation metrics,
then report the held-out test result. This is a diagnostic for whether the
remaining gap is due to global model selection hiding split-specific strengths.
It does not use test labels for selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUTPUT_ROOT = ROOT / "outputs"
OUT_DIR = OUTPUT_ROOT / "valence_model_zoo_validation_selector_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def pred_col(df: pd.DataFrame) -> str | None:
    for c in ["pred_valence", "pred"]:
        if c in df.columns:
            return c
    return None


def target_col(df: pd.DataFrame) -> str | None:
    for c in ["target_valence", "target"]:
        if c in df.columns:
            return c
    return None


def eval_file(path: Path) -> dict[str, float] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    pc = pred_col(df)
    tc = target_col(df)
    if pc is None or tc is None:
        return None
    return metrics(df[pc].to_numpy(dtype=np.float64), df[tc].to_numpy(dtype=np.float64))


def score(m: dict[str, float]) -> float:
    return (
        3.3 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.2 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.8 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def candidate_dirs() -> list[Path]:
    out = []
    for d in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("valence")):
        if d == OUT_DIR:
            continue
        ok = True
        for split in SPLITS:
            if not (d / split / "val_predictions.csv").exists() or not (d / split / "test_predictions.csv").exists():
                ok = False
                break
        if ok:
            out.append(d)
    return out


def load_standard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pc = pred_col(df)
    tc = target_col(df)
    if pc is None or tc is None:
        raise ValueError(path)
    cols = ["musicId", "time_sec", tc, pc]
    return df[cols].rename(columns={tc: "target_valence", pc: "pred_valence"})


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = candidate_dirs()
    all_rows = []
    selected_rows = []
    for split in SPLITS:
        split_rows = []
        for d in dirs:
            val_m = eval_file(d / split / "val_predictions.csv")
            test_m = eval_file(d / split / "test_predictions.csv")
            train_m = eval_file(d / split / "train_predictions.csv") if (d / split / "train_predictions.csv").exists() else None
            if val_m is None or test_m is None:
                continue
            row = {
                "split_name": split,
                "candidate_dir": d.name,
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "val_score": score(val_m),
            }
            if train_m is not None:
                row.update({f"train_{k}": v for k, v in train_m.items()})
            split_rows.append(row)
        split_df = pd.DataFrame(split_rows)
        split_df.to_csv(OUT_DIR / f"{split}_candidate_scores.csv", index=False)
        selected = split_df.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "best validation pass-gap score for this split"
        selected_rows.append(selected)
        all_rows.extend(split_rows)
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        selected_dir = OUTPUT_ROOT / str(selected["candidate_dir"])
        for data_split in DATA_SPLITS:
            src = selected_dir / split / f"{data_split}_predictions.csv"
            if src.exists():
                load_standard(src).to_csv(split_out / f"{data_split}_predictions.csv", index=False)

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "all_model_zoo_candidate_scores.csv", index=False)
    selected_test = macro(selected_rows, "test")
    selected_val = macro(selected_rows, "val")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_model_zoo_validation_selector_20260708",
        "purpose": "diagnose whether validation-selected split-specific candidate choice can close the Valence gap",
        "selection_rule": "per split, candidate selected by validation metrics only; test labels unused",
        "candidate_count": len(dirs),
        "selected_val": selected_val,
        "selected_test": selected_test,
        "per_split_selected": selected_rows,
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
