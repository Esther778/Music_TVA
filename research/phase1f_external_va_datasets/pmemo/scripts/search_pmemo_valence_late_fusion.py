#!/usr/bin/env python3
"""Validation-selected late fusion for PMEmo Valence candidate predictions."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "valence_late_fusion_20260625"

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}


CANDIDATES = [
    {
        "id": "extreme_ext125_best",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext125_clip010_ccc005",
    },
    {
        "id": "extreme_ext075",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext075_clip010_ccc005",
    },
    {
        "id": "extreme_std005",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext100_clip010_std005",
    },
    {
        "id": "clip_aux_best",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_ccc005",
    },
    {
        "id": "clip_aux_light",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse005_ccc005",
    },
    {
        "id": "clip_aux_clipccc",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_clipccc005",
    },
    {
        "id": "hidden6_focus_best",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_focus/hidden6_harm_h80_proj320_do030_ccc005",
    },
    {
        "id": "hidden6_focus_ccc005_small",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_focus/hidden6_harm_h64_proj256_do025_ccc005",
    },
    {
        "id": "hidden6_refine_h80",
        "dir": ROOT
        / "outputs/mert_hidden6_harmonic_refine/hidden6_harm_h80_proj320_do025_ccc008",
    },
    {
        "id": "hidden8_harm_mse",
        "dir": ROOT
        / "outputs/mert_hidden8_harmonic_valence_fusion/mert_harm_h64_proj256_do030_mse",
    },
    {
        "id": "hidden4_focus_h80",
        "dir": ROOT
        / "outputs/mert_hidden4_harmonic_focus/hidden4_harm_h80_proj320_do030_ccc005",
    },
    {
        "id": "opensmile_harmonic_best",
        "dir": ROOT
        / "outputs/valence_opensmile_harmonic_focus/v_harm_h96_l1_do025_lr6e4_ccc005",
    },
    {
        "id": "mert_opensmile_fixed",
        "dir": ROOT
        / "outputs/mert_hidden6_opensmile_valence_fusion_fixed/h6_smile_h80_proj320_do030_ccc005",
    },
]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pcc = 0.0
    else:
        pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true))
    var_pred = float(np.var(y_pred))
    cov = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    ccc = float((2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12))
    return {"rmse": rmse, "mae": mae, "pcc": pcc, "ccc": ccc}


def score_for_selection(m: dict[str, float]) -> float:
    """Lower is better; validation-only score weighted toward pass thresholds."""
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.15 * m["rmse"]
    )


def read_candidate(split: str, candidate: dict[str, Path | str]) -> pd.DataFrame:
    path = Path(candidate["dir"]) / f"{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    pred_col = "pred_valence"
    if pred_col not in df.columns and "pred_valence_raw" in df.columns:
        pred_col = "pred_valence_raw"
    required = {"musicId", "time_sec", "target_valence", pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    out = df.loc[:, ["musicId", "time_sec", "target_valence", pred_col]].copy()
    out["musicId"] = out["musicId"].astype(str)
    out["time_sec"] = out["time_sec"].astype(float).round(3)
    out = out.rename(columns={pred_col: str(candidate["id"])})
    return out


def build_matrix(split: str, candidates: list[dict[str, Path | str]]) -> pd.DataFrame:
    base = None
    for cand in candidates:
        df = read_candidate(split, cand)
        if base is None:
            base = df
            continue
        base = base.merge(
            df.drop(columns=["target_valence"]),
            on=["musicId", "time_sec"],
            how="inner",
            validate="one_to_one",
        )
    assert base is not None
    if base.empty:
        raise ValueError(f"No aligned rows for split={split}")
    return base.sort_values(["musicId", "time_sec"]).reset_index(drop=True)


def weight_grid(n: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    weights = []
    for parts in itertools.product(range(units + 1), repeat=n):
        if sum(parts) == units:
            weights.append(np.array(parts, dtype=np.float64) / units)
    return weights


def evaluate_combo(df: pd.DataFrame, ids: list[str], weights: np.ndarray) -> dict[str, float]:
    preds = df[ids].to_numpy(dtype=np.float64) @ weights
    return metrics(df["target_valence"].to_numpy(dtype=np.float64), preds)


def run_search() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [c for c in CANDIDATES if (Path(c["dir"]) / "val_predictions.csv").exists()]
    val = build_matrix("val", candidates)
    test = build_matrix("test", candidates)
    ids = [str(c["id"]) for c in candidates]

    single_rows = []
    for cid in ids:
        val_m = metrics(val["target_valence"], val[cid])
        test_m = metrics(test["target_valence"], test[cid])
        single_rows.append(
            {
                "combo": cid,
                "weights": json.dumps({cid: 1.0}, ensure_ascii=True),
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "selection_score": score_for_selection(val_m),
            }
        )

    search_rows = []
    for size, step in [(2, 0.01), (3, 0.05), (4, 0.10)]:
        for combo in itertools.combinations(ids, size):
            for weights in weight_grid(size, step):
                if np.count_nonzero(weights) != size:
                    continue
                val_m = evaluate_combo(val, list(combo), weights)
                search_rows.append(
                    {
                        "combo": "+".join(combo),
                        "weights": json.dumps(
                            {cid: float(w) for cid, w in zip(combo, weights)},
                            ensure_ascii=True,
                        ),
                        **{f"val_{k}": v for k, v in val_m.items()},
                        "selection_score": score_for_selection(val_m),
                    }
                )

    search_rows.extend(single_rows)
    ranked = sorted(search_rows, key=lambda r: (r["selection_score"], r["val_rmse"], -r["val_ccc"]))
    top_rows = []
    for row in ranked[:200]:
        weights = json.loads(row["weights"])
        combo = list(weights.keys())
        w = np.array([weights[cid] for cid in combo], dtype=np.float64)
        test_m = evaluate_combo(test, combo, w)
        top_rows.append({**row, **{f"test_{k}": v for k, v in test_m.items()}})

    selected = top_rows[0]
    baseline_id = "extreme_ext125_best" if any(r["combo"] == "extreme_ext125_best" for r in single_rows) else "clip_aux_best"
    baseline = next(r for r in single_rows if r["combo"] == baseline_id)

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        keys = list(rows[0].keys()) if rows else []
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUTPUT_DIR / "single_model_metrics.csv", single_rows)
    write_csv(OUTPUT_DIR / "top_late_fusion_results.csv", top_rows)

    selected_weights = json.loads(selected["weights"])
    selected_ids = list(selected_weights.keys())
    selected_w = np.array([selected_weights[cid] for cid in selected_ids], dtype=np.float64)
    for split_name, df in [("val", val), ("test", test)]:
        pred = df[selected_ids].to_numpy(dtype=np.float64) @ selected_w
        out = df.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
        out["pred_valence"] = pred
        out.to_csv(OUTPUT_DIR / f"selected_{split_name}_predictions.csv", index=False)

    summary = {
        "created_at": "2026-06-25",
        "selection_rule": "lowest validation pass-gap composite score; test not used for weight selection",
        "candidate_count": len(candidates),
        "aligned_val_rows": int(len(val)),
        "aligned_test_rows": int(len(test)),
        "selected": selected,
        "current_best_baseline": baseline,
        "current_best_baseline_id": baseline_id,
        "test_delta_vs_current_best": {
            "rmse": selected["test_rmse"] - baseline["test_rmse"],
            "mae": selected["test_mae"] - baseline["test_mae"],
            "pcc": selected["test_pcc"] - baseline["test_pcc"],
            "ccc": selected["test_ccc"] - baseline["test_ccc"],
        },
        "threshold_gap_selected_test": {
            "rmse_to_pass": selected["test_rmse"] - PASS["rmse"],
            "mae_to_pass": selected["test_mae"] - PASS["mae"],
            "pcc_to_pass": PASS["pcc"] - selected["test_pcc"],
            "ccc_to_pass": PASS["ccc"] - selected["test_ccc"],
            "rmse_to_good": selected["test_rmse"] - GOOD["rmse"],
            "mae_to_good": selected["test_mae"] - GOOD["mae"],
            "pcc_to_good": GOOD["pcc"] - selected["test_pcc"],
            "ccc_to_good": GOOD["ccc"] - selected["test_ccc"],
            "rmse_to_excellent": selected["test_rmse"] - EXCELLENT["rmse"],
            "mae_to_excellent": selected["test_mae"] - EXCELLENT["mae"],
            "pcc_to_excellent": EXCELLENT["pcc"] - selected["test_pcc"],
            "ccc_to_excellent": EXCELLENT["ccc"] - selected["test_ccc"],
        },
        "candidate_ids": ids,
    }
    (OUTPUT_DIR / "late_fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_search()
