#!/usr/bin/env python3
"""Train/validation/test Ridge stacking for PMEmo Valence predictions.

The stacker is fit only on train predictions. Validation is used to choose the
regularization strength and candidate subset. Test is evaluated once for the
selected stacker.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "valence_ridge_stacking_20260625"
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}

CANDIDATES = [
    ("extreme_ext125_best", "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext125_clip010_ccc005"),
    ("extreme_ext075", "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext075_clip010_ccc005"),
    ("extreme_std005", "outputs/mert_hidden6_harmonic_extreme_weighted/h6_harm_h80_ext100_clip010_std005"),
    ("clip_aux_best", "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_ccc005"),
    ("clip_aux_light", "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse005_ccc005"),
    ("clip_aux_clipccc", "outputs/mert_hidden6_harmonic_clip_aux/h6_harm_h80_clipmse010_clipccc005"),
    ("hidden6_focus_best", "outputs/mert_hidden6_harmonic_focus/hidden6_harm_h80_proj320_do030_ccc005"),
    ("hidden6_focus_small", "outputs/mert_hidden6_harmonic_focus/hidden6_harm_h64_proj256_do025_ccc005"),
    ("hidden6_refine_h80", "outputs/mert_hidden6_harmonic_refine/hidden6_harm_h80_proj320_do025_ccc008"),
    ("hidden8_harm_mse", "outputs/mert_hidden8_harmonic_valence_fusion/mert_harm_h64_proj256_do030_mse"),
    ("hidden4_focus_h80", "outputs/mert_hidden4_harmonic_focus/hidden4_harm_h80_proj320_do030_ccc005"),
    ("opensmile_harmonic_best", "outputs/valence_opensmile_harmonic_focus/v_harm_h96_l1_do025_lr6e4_ccc005"),
    ("mert_opensmile_fixed", "outputs/mert_hidden6_opensmile_valence_fusion_fixed/h6_smile_h80_proj320_do030_ccc005"),
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


def selection_score(m: dict[str, float]) -> float:
    return (
        max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.15 * m["rmse"]
    )


def read_prediction(split: str, cid: str, rel_dir: str) -> pd.DataFrame:
    path = ROOT / rel_dir / f"{split}_predictions.csv"
    df = pd.read_csv(path)
    pred_col = "pred_valence"
    if pred_col not in df.columns and "pred_valence_raw" in df.columns:
        pred_col = "pred_valence_raw"
    out = df.loc[:, ["musicId", "time_sec", "target_valence", pred_col]].copy()
    out["musicId"] = out["musicId"].astype(str)
    out["time_sec"] = out["time_sec"].astype(float).round(3)
    return out.rename(columns={pred_col: cid})


def build(split: str) -> pd.DataFrame:
    base = None
    for cid, rel_dir in CANDIDATES:
        if not (ROOT / rel_dir / f"{split}_predictions.csv").exists():
            continue
        df = read_prediction(split, cid, rel_dir)
        if base is None:
            base = df
        else:
            base = base.merge(
                df.drop(columns=["target_valence"]),
                on=["musicId", "time_sec"],
                how="inner",
                validate="one_to_one",
            )
    if base is None or base.empty:
        raise ValueError(f"No rows for split={split}")
    return base.sort_values(["musicId", "time_sec"]).reset_index(drop=True)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-8
    xs = (X - x_mean) / x_std
    design = np.concatenate([xs, np.ones((len(xs), 1))], axis=1)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    coef_scaled = beta[:-1]
    intercept_scaled = float(beta[-1])
    coef = coef_scaled / x_std.ravel()
    intercept = intercept_scaled - float(np.sum(coef_scaled * x_mean.ravel() / x_std.ravel()))
    return coef.astype(np.float64), intercept


def predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coef + intercept


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build("train")
    val = build("val")
    test = build("test")
    ids = [c for c in train.columns if c not in {"musicId", "time_sec", "target_valence"}]
    y_train = train["target_valence"].to_numpy(dtype=np.float64)
    y_val = val["target_valence"].to_numpy(dtype=np.float64)
    y_test = test["target_valence"].to_numpy(dtype=np.float64)

    alpha_grid = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    subset_pool = []
    for size in [1, 2, 3, 4, 5, len(ids)]:
        for combo in itertools.combinations(ids, size):
            subset_pool.append(combo)
    rows = []
    for combo in subset_pool:
        cols = list(combo)
        x_train = train[cols].to_numpy(dtype=np.float64)
        x_val = val[cols].to_numpy(dtype=np.float64)
        for alpha in alpha_grid:
            coef, intercept = fit_ridge(x_train, y_train, alpha)
            val_pred = predict(x_val, coef, intercept)
            val_m = metrics(y_val, val_pred)
            rows.append(
                {
                    "combo": "+".join(cols),
                    "alpha": alpha,
                    "coef": json.dumps({c: float(v) for c, v in zip(cols, coef)}, ensure_ascii=True),
                    "intercept": intercept,
                    **{f"val_{k}": v for k, v in val_m.items()},
                    "selection_score": selection_score(val_m),
                }
            )

    ranked = sorted(rows, key=lambda r: (r["selection_score"], r["val_rmse"], -r["val_ccc"]))
    top_rows = []
    for row in ranked[:200]:
        cols = row["combo"].split("+")
        coef = np.array(list(json.loads(row["coef"]).values()), dtype=np.float64)
        test_pred = predict(test[cols].to_numpy(dtype=np.float64), coef, float(row["intercept"]))
        test_m = metrics(y_test, test_pred)
        top_rows.append({**row, **{f"test_{k}": v for k, v in test_m.items()}})

    selected = top_rows[0]
    baseline_id = "extreme_ext125_best" if "extreme_ext125_best" in test.columns else "clip_aux_best"
    baseline = metrics(y_test, test[baseline_id].to_numpy(dtype=np.float64))
    selected_cols = selected["combo"].split("+")
    selected_coef = np.array(list(json.loads(selected["coef"]).values()), dtype=np.float64)
    for split_name, df in [("val", val), ("test", test)]:
        pred = predict(df[selected_cols].to_numpy(dtype=np.float64), selected_coef, float(selected["intercept"]))
        out = df.loc[:, ["musicId", "time_sec", "target_valence"]].copy()
        out["pred_valence"] = pred
        out.to_csv(OUTPUT_DIR / f"selected_{split_name}_predictions.csv", index=False)

    pd.DataFrame(top_rows).to_csv(OUTPUT_DIR / "top_ridge_stacking_results.csv", index=False)
    summary = {
        "created_at": "2026-06-25",
        "selection_rule": "fit ridge stacker on train predictions; choose subset and alpha by validation pass-gap composite; test not used for selection",
        "candidate_count": len(ids),
        "aligned_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "selected": selected,
        "current_best_baseline_test": baseline,
        "current_best_baseline_id": baseline_id,
        "test_delta_vs_current_best": {
            "rmse": selected["test_rmse"] - baseline["rmse"],
            "mae": selected["test_mae"] - baseline["mae"],
            "pcc": selected["test_pcc"] - baseline["pcc"],
            "ccc": selected["test_ccc"] - baseline["ccc"],
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
    (OUTPUT_DIR / "ridge_stacking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
