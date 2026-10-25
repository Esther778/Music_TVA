#!/usr/bin/env python3
"""Joint validation-selected postprocess search for deployable Valence.

The earlier deployable chain selected temporal shift, smoothing, and affine
calibration sequentially. This script searches the three decisions together on
the same raw-audio-deployable prediction base. Test labels are reported only
after the validation-selected policy is fixed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_temporal_shift import apply_shift
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
OUT_DIR = ROOT / "outputs" / "valence_deployable_joint_postprocess_fast_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_DEPLOYABLE = {
    "model_id": "pmemo_valence_deployable_tonal_shift_smoothing_affine_20260708",
    "rmse": 0.1889116153150956,
    "mae": 0.1465878035891668,
    "pcc": 0.8117208849284806,
    "ccc": 0.7967490459172722,
}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 2:
        return x.copy()
    pad = window // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) <= 1:
        return x.copy()
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = float(alpha) * x[i] + (1.0 - float(alpha)) * y[i - 1]
    return y


def apply_smoothing(df: pd.DataFrame, method: str, param: float, blend: float) -> pd.DataFrame:
    pieces = []
    for _, g in df.groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        if method == "ma":
            smoothed = moving_average(pred, int(param))
        elif method == "exp":
            smoothed = exp_smooth(pred, float(param))
        elif method == "identity":
            smoothed = pred.copy()
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        out = g.copy()
        out["pre_smooth_pred_valence"] = pred
        out["pred_valence"] = np.clip((1.0 - float(blend)) * pred + float(blend) * smoothed, -1.0, 1.0)
        out["smooth_method"] = method
        out["smooth_param"] = float(param)
        out["smooth_blend"] = float(blend)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def fit_affine(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    var = float(np.var(x))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    b = float(y.mean() - a * x.mean())
    return a, b


def fit_mean_std(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    sx = float(np.std(x))
    if sx < 1e-12:
        return 1.0, float(y.mean() - x.mean())
    a = float(np.std(y) / sx)
    b = float(y.mean() - a * x.mean())
    return a, b


def apply_affine(df: pd.DataFrame, fit_mode: str, a: float, b: float, blend: float) -> pd.DataFrame:
    out = df.copy()
    pred = out["pred_valence"].to_numpy(dtype=np.float64)
    if fit_mode == "identity" or blend <= 0.0:
        calibrated = pred
        a = 1.0
        b = 0.0
        blend = 0.0
    else:
        transformed = float(a) * pred + float(b)
        calibrated = (1.0 - float(blend)) * pred + float(blend) * transformed
    out["pre_affine_pred_valence"] = pred
    out["pred_valence"] = np.clip(calibrated, -1.0, 1.0)
    out["affine_mode"] = fit_mode
    out["affine_a"] = float(a)
    out["affine_b"] = float(b)
    out["affine_blend"] = float(blend)
    return out


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(
        df["pred_valence"].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[f"{data_split}_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }


def pass_gap_score(m: dict[str, float]) -> float:
    return (
        3.4 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.7 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split, data_split): pd.read_csv(INPUT_DIR / split / f"{data_split}_predictions.csv")
        for split in SPLITS
        for data_split in DATA_SPLITS
    }

    base_rows = []
    for split in SPLITS:
        row: dict[str, Any] = {"split_name": split}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split, data_split)]).items()})
        base_rows.append(row)
    base_val = macro(base_rows, "val")
    base_test = macro(base_rows, "test")

    smoothing_candidates: list[tuple[str, float, float]] = [("identity", 0.0, 0.0)]
    smoothing_candidates += [("ma", float(w), b) for w in [17, 19, 21, 23, 25] for b in [0.85, 1.0]]
    smoothing_candidates += [("exp", a, b) for a in [0.50, 0.65] for b in [0.70, 1.0]]

    affine_modes = ["identity", "affine", "mean_std"]
    affine_blends = [0.0, 0.02, 0.03, 0.05]
    rows: list[dict[str, Any]] = []

    for shift_frames in range(-26, -17):
        shifted = {
            (split, data_split): apply_shift(frames[(split, data_split)], shift_frames)
            for split in SPLITS
            for data_split in DATA_SPLITS
        }
        for smooth_method, smooth_param, smooth_blend in smoothing_candidates:
            smoothed = {
                key: apply_smoothing(df, smooth_method, smooth_param, smooth_blend)
                for key, df in shifted.items()
            }
            train_all = pd.concat([smoothed[(split, "train")] for split in SPLITS], ignore_index=True)
            train_pred = train_all["pred_valence"].to_numpy(dtype=np.float64)
            train_target = train_all["target_valence"].to_numpy(dtype=np.float64)
            params = {
                "identity": (1.0, 0.0),
                "affine": fit_affine(train_pred, train_target),
                "mean_std": fit_mean_std(train_pred, train_target),
            }
            for affine_mode in affine_modes:
                for affine_blend in affine_blends:
                    if affine_mode == "identity" and affine_blend != 0.0:
                        continue
                    a, b = params[affine_mode]
                    calibrated = {
                        key: apply_affine(df, affine_mode, a, b, affine_blend)
                        for key, df in smoothed.items()
                    }
                    per_split = []
                    for split in SPLITS:
                        row: dict[str, Any] = {"split_name": split}
                        for data_split in DATA_SPLITS:
                            row.update({f"{data_split}_{m}": v for m, v in eval_df(calibrated[(split, data_split)]).items()})
                        per_split.append(row)
                    val_m = macro(per_split, "val")
                    test_m = macro(per_split, "test")
                    key = (shift_frames, smooth_method, float(smooth_param), float(smooth_blend), affine_mode, float(affine_blend))
                    rows.append(
                        {
                            "shift_frames": int(shift_frames),
                            "shift_seconds": float(shift_frames * 0.5),
                            "smooth_method": smooth_method,
                            "smooth_param": float(smooth_param),
                            "smooth_blend": float(smooth_blend),
                            "affine_mode": affine_mode,
                            "affine_blend": float(affine_blend),
                            "affine_a": float(a),
                            "affine_b": float(b),
                            **{f"val_{m}": v for m, v in val_m.items()},
                            **{f"test_{m}": v for m, v in test_m.items()},
                            "val_score": pass_gap_score(val_m),
                        }
                    )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_joint_postprocess_candidates.csv", index=False)
    selected = result.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()

    selected_shift = int(selected["shift_frames"])
    selected_smooth_method = str(selected["smooth_method"])
    selected_smooth_param = float(selected["smooth_param"])
    selected_smooth_blend = float(selected["smooth_blend"])
    selected_affine_mode = str(selected["affine_mode"])
    selected_affine_blend = float(selected["affine_blend"])

    shifted = {
        (split, data_split): apply_shift(frames[(split, data_split)], selected_shift)
        for split in SPLITS
        for data_split in DATA_SPLITS
    }
    smoothed = {
        key: apply_smoothing(df, selected_smooth_method, selected_smooth_param, selected_smooth_blend)
        for key, df in shifted.items()
    }
    train_all = pd.concat([smoothed[(split, "train")] for split in SPLITS], ignore_index=True)
    train_pred = train_all["pred_valence"].to_numpy(dtype=np.float64)
    train_target = train_all["target_valence"].to_numpy(dtype=np.float64)
    if selected_affine_mode == "affine":
        selected_a, selected_b = fit_affine(train_pred, train_target)
    elif selected_affine_mode == "mean_std":
        selected_a, selected_b = fit_mean_std(train_pred, train_target)
    else:
        selected_a, selected_b = 1.0, 0.0
    selected_frames = {
        key: apply_affine(df, selected_affine_mode, selected_a, selected_b, selected_affine_blend)
        for key, df in smoothed.items()
    }
    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split}
        for data_split in DATA_SPLITS:
            df = selected_frames[(split, data_split)]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_deployable_joint_postprocess_fast_20260708",
        "input_dir": str(INPUT_DIR.relative_to(ROOT)),
        "deployability": "global shift, smoothing, and affine parameters selected on validation; can be applied to arbitrary raw-MP3 predictions",
        "selection_rule": "joint postprocess selected by validation pass-gap score; test labels not used for selection",
        "base_val": base_val,
        "base_test": base_test,
        "current_deployable_reference": CURRENT_DEPLOYABLE,
        "selected_config": selected,
        "selected_test": selected_test,
        "per_split": per_split,
        "delta_vs_current_deployable": {
            m: float(selected_test[m] - CURRENT_DEPLOYABLE[m])
            for m in ["rmse", "mae", "pcc", "ccc"]
        },
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nTop validation-selected candidates:", flush=True)
    print(result.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
