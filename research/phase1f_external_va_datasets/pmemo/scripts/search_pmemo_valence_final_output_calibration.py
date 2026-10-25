#!/usr/bin/env python3
"""Final validation-selected output calibration for the current Valence best.

This script does not retrain the audio model. It fits lightweight output
calibrators on train predictions, selects the calibrator/blend on validation
predictions, and reports held-out test metrics. The goal is to test whether the
near-pass Valence model can be improved by a deployable final calibration step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_frame_calibrator_song_bias_20260628"
OUT_DIR = ROOT / "outputs" / "valence_final_output_calibration_20260628"
CREATED_AT = "2026-06-28"
MODEL_ID = "pmemo_valence_final_output_calibration_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_frame_calibrator_song_bias_20260628",
    "rmse": 0.19748015854593182,
    "mae": 0.15352887995061668,
    "pcc": 0.7932350928356282,
    "ccc": 0.7800520380365018,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


@dataclass
class Calibrator:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column found in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column found in {list(df.columns)}")


def selection_score(m: dict[str, float]) -> float:
    return (
        2.20 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 2.20 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 1.70 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64))


def fit_quantile_mapping(x_train: np.ndarray, y_train: np.ndarray) -> Calibrator:
    order_x = np.argsort(x_train)
    order_y = np.argsort(y_train)
    xp = x_train[order_x]
    fp = y_train[order_y]
    # np.interp requires increasing xp. Collapse duplicate prediction values.
    unique_x, inverse = np.unique(xp, return_inverse=True)
    unique_y = np.zeros_like(unique_x, dtype=np.float64)
    for idx in range(len(unique_x)):
        unique_y[idx] = float(np.mean(fp[inverse == idx]))

    def predict(x: np.ndarray) -> np.ndarray:
        return np.interp(x, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])

    return Calibrator("quantile_mapping", predict)


def fit_calibrators(x_train: np.ndarray, y_train: np.ndarray) -> list[Calibrator]:
    x2 = x_train.reshape(-1, 1)
    calibrators: list[Calibrator] = [Calibrator("identity", lambda x: x)]

    linear = LinearRegression().fit(x2, y_train)
    calibrators.append(Calibrator("linear", lambda x, model=linear: model.predict(x.reshape(-1, 1))))

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha).fit(x2, y_train)
        calibrators.append(Calibrator(f"ridge_a{alpha:g}", lambda x, model=ridge: model.predict(x.reshape(-1, 1))))

    for degree in [2, 3]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            poly = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha))
            poly.fit(x2, y_train)
            calibrators.append(Calibrator(f"poly{degree}_ridge_a{alpha:g}", lambda x, model=poly: model.predict(x.reshape(-1, 1))))

    huber = HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=1000).fit(x2, y_train)
    calibrators.append(Calibrator("huber", lambda x, model=huber: model.predict(x.reshape(-1, 1))))

    iso = IsotonicRegression(out_of_bounds="clip").fit(x_train, y_train)
    calibrators.append(Calibrator("isotonic", lambda x, model=iso: model.predict(x)))

    calibrators.append(fit_quantile_mapping(x_train, y_train))
    return calibrators


def apply_calibrated(df: pd.DataFrame, calibrator: Calibrator, blend: float, clip: float | None) -> pd.DataFrame:
    out = df.copy()
    pcol = pred_col(out)
    raw = out[pcol].to_numpy(dtype=np.float64)
    calibrated = calibrator.predict(raw)
    pred = (1.0 - blend) * raw + blend * calibrated
    if clip is not None:
        pred = np.clip(pred, -float(clip), float(clip))
    out[pcol] = pred
    out["final_calibrator"] = calibrator.name
    out["final_calibration_blend"] = blend
    out["final_calibration_clip"] = "none" if clip is None else clip
    return out


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return eval_arrays(df[pred_col(df)].to_numpy(), df[target_col(df)].to_numpy())


def evaluate_split(split_dir: Path) -> tuple[dict, pd.DataFrame]:
    frames = {split: pd.read_csv(split_dir / f"{split}_predictions.csv") for split in ["train", "val", "test"]}
    pcol = pred_col(frames["train"])
    tcol = target_col(frames["train"])
    calibrators = fit_calibrators(frames["train"][pcol].to_numpy(dtype=np.float64), frames["train"][tcol].to_numpy(dtype=np.float64))
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

    df = pd.DataFrame(rows).sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    selected = df.iloc[0].to_dict()
    selected_key = (str(selected["calibrator"]), float(selected["blend"]), str(selected["clip"]))
    out_split = OUT_DIR / split_dir.name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        cache[(split, *selected_key)].to_csv(out_split / f"{split}_predictions.csv", index=False)

    summary = {
        "split_name": split_dir.name,
        "baseline": baseline,
        "selected": selected,
        "delta_vs_input_test": {metric: float(selected[f"test_{metric}"] - baseline["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
    }
    return summary, df


def aggregate(rows: list[dict]) -> dict:
    out = {
        "created_at": CREATED_AT,
        "model_id": MODEL_ID,
        "input_model": CURRENT_BEST,
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([r["selected"][f"test_{metric}"] for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([r["baseline"]["test"][metric] for r in rows], dtype=np.float64)
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
    summary = aggregate(rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
