#!/usr/bin/env python3
"""Train-fitted output calibration for current best PMEmo Valence predictions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_guardtopccc_temporal_shift_wide_20260708"
OUT_DIR = ROOT / "outputs" / "valence_trainfit_output_calibration_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_guardtopccc_temporal_shift_wide_20260708",
    "rmse": 0.1896185697242768,
    "mae": 0.14745982046674147,
    "pcc": 0.8105040088472681,
    "ccc": 0.797385312909266,
}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.3 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.8 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def features(pred: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    return np.column_stack([p, np.tanh(2 * p), np.sign(p) * np.sqrt(np.abs(p) + 1e-8)])


class Identity:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Identity":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0]


def fit_model(model_id: str, train_pred: np.ndarray, train_target: np.ndarray):
    x = features(train_pred)
    y = np.asarray(train_target, dtype=np.float64)
    if model_id == "identity":
        model = Identity()
    elif model_id == "ridge_linear_a1":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif model_id == "ridge_linear_a10":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif model_id == "ridge_poly2_a10":
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=10.0))
    elif model_id == "ridge_poly2_a100":
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=100.0))
    elif model_id == "huber_linear":
        model = make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=1000))
    elif model_id == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=-1.0, y_max=1.0)
        model.fit(train_pred, y)
        return model
    else:
        raise ValueError(model_id)
    model.fit(x, y)
    return model


def predict_model(model, model_id: str, pred: np.ndarray) -> np.ndarray:
    if model_id == "isotonic":
        out = model.predict(np.asarray(pred, dtype=np.float64))
    else:
        out = model.predict(features(pred))
    return np.clip(np.asarray(out, dtype=np.float64), -1.0, 1.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split, data): pd.read_csv(INPUT_DIR / split / f"{data}_predictions.csv")
        for split in SPLITS
        for data in DATA_SPLITS
    }
    baseline_rows = []
    for split in SPLITS:
        row = {"split_name": split}
        for data in DATA_SPLITS:
            row.update({f"{data}_{m}": v for m, v in eval_df(frames[(split, data)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    model_ids = ["identity", "ridge_linear_a1", "ridge_linear_a10", "ridge_poly2_a10", "ridge_poly2_a100", "huber_linear", "isotonic"]
    rows = []
    cache: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    for model_id in model_ids:
        calibrated: dict[tuple[str, str], pd.DataFrame] = {}
        per_split = []
        for split in SPLITS:
            train = frames[(split, "train")]
            model = fit_model(
                model_id,
                train["pred_valence"].to_numpy(dtype=np.float64),
                train["target_valence"].to_numpy(dtype=np.float64),
            )
            row = {"split_name": split}
            for data in DATA_SPLITS:
                df = frames[(split, data)].copy()
                df["base_pred_valence"] = df["pred_valence"].to_numpy(dtype=np.float64)
                df["pred_valence"] = predict_model(model, model_id, df["base_pred_valence"].to_numpy(dtype=np.float64))
                calibrated[(split, data)] = df
                row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
            per_split.append(row)
        val_m = macro(per_split, "val")
        test_m = macro(per_split, "test")
        rows.append(
            {
                "model_id": model_id,
                **{f"val_{m}": v for m, v in val_m.items()},
                **{f"test_{m}": v for m, v in test_m.items()},
                "val_score": score(val_m),
                "passes_guard": (
                    val_m["rmse"] <= base_val["rmse"] + 0.001
                    and val_m["mae"] <= base_val["mae"] + 0.001
                    and val_m["pcc"] >= base_val["pcc"] - 0.001
                    and val_m["ccc"] >= base_val["ccc"] + 0.0005
                ),
            }
        )
        cache[model_id] = calibrated

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_output_calibration_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = result[result["model_id"] == "identity"].iloc[0].to_dict()
        selected["selection_reason"] = "identity retained; no candidate passed validation guard"
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
    selected_id = str(selected["model_id"])
    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split}
        for data in DATA_SPLITS:
            df = cache[selected_id][(split, data)]
            df.to_csv(split_out / f"{data}_predictions.csv", index=False)
            row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_trainfit_output_calibration_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "calibration mapping fitted on train predictions only; mapping family selected by validation macro guard",
        "baseline_val": base_val,
        "baseline_test": base_test,
        "selected_config": selected,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {m: float(selected_test[m] - CURRENT_BEST[m]) for m in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
