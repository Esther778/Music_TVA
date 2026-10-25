#!/usr/bin/env python3
"""Low-capacity affine calibration for the current Valence best.

This experiment is intentionally simpler than prediction stacking. It fits
global affine/rescale parameters on train predictions only and selects a small
blend/smoothing policy on validation. Test labels are only reported after
selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
OUT_DIR = ROOT / "outputs" / "valence_current_best_affine_calibration_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


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
    sy = float(np.std(y))
    if sx < 1e-12:
        return 1.0, float(y.mean() - x.mean())
    a = sy / sx
    b = float(y.mean() - a * x.mean())
    return a, b


def moving_average_by_song(pred: np.ndarray, song_ids: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    out = pred.copy()
    for sid in np.unique(song_ids):
        idx = np.where(song_ids == sid)[0]
        local = min(int(window), len(idx))
        if local <= 1:
            continue
        kernel = np.ones(local, dtype=np.float64) / float(local)
        out[idx] = np.convolve(pred[idx], kernel, mode="same")
    return out


def apply_policy(df: pd.DataFrame, a: float, b: float, blend: float, smooth_window: int, clip: str) -> pd.DataFrame:
    out = df.copy()
    base = out["pred_valence"].to_numpy(dtype=np.float64)
    transformed = float(a) * base + float(b)
    pred = (1.0 - float(blend)) * base + float(blend) * transformed
    pred = moving_average_by_song(pred, out["musicId"].to_numpy(), int(smooth_window))
    if clip == "hard":
        pred = np.clip(pred, -1.0, 1.0)
    elif clip == "soft095":
        pred = np.clip(pred, -0.95, 0.95)
    elif clip != "none":
        raise ValueError(f"Unknown clip mode: {clip}")
    out["affine_base_pred_valence"] = base
    out["pred_valence"] = pred
    out["affine_a"] = float(a)
    out["affine_b"] = float(b)
    out["affine_blend"] = float(blend)
    out["affine_smooth_window"] = int(smooth_window)
    out["affine_clip"] = clip
    return out


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.4 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.2 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.2 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split_name, data_split): pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv")
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{k}": v for k, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    cache: dict[tuple[str, str, float, int, str], dict[tuple[str, str], pd.DataFrame]] = {}
    for fit_mode in ["affine", "mean_std"]:
        for scope in ["per_split", "global"]:
            params: dict[str, tuple[float, float]] = {}
            if scope == "global":
                train_all = pd.concat([frames[(s, "train")] for s in SPLITS], ignore_index=True)
                fn = fit_affine if fit_mode == "affine" else fit_mean_std
                params["global"] = fn(
                    train_all["pred_valence"].to_numpy(dtype=np.float64),
                    train_all["target_valence"].to_numpy(dtype=np.float64),
                )
            else:
                fn = fit_affine if fit_mode == "affine" else fit_mean_std
                for split_name in SPLITS:
                    train = frames[(split_name, "train")]
                    params[split_name] = fn(
                        train["pred_valence"].to_numpy(dtype=np.float64),
                        train["target_valence"].to_numpy(dtype=np.float64),
                    )
            for blend in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 1.0]:
                for smooth_window in [1, 3, 5, 7, 9]:
                    for clip in ["none", "hard", "soft095"]:
                        key = (fit_mode, scope, float(blend), int(smooth_window), clip)
                        selected_frames = {}
                        rows = []
                        for split_name in SPLITS:
                            a, b = params["global"] if scope == "global" else params[split_name]
                            row = {"split_name": split_name}
                            for data_split in DATA_SPLITS:
                                corrected = apply_policy(frames[(split_name, data_split)], a, b, float(blend), int(smooth_window), clip)
                                selected_frames[(split_name, data_split)] = corrected
                                row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                            rows.append(row)
                        val_m = macro(rows, "val")
                        test_m = macro(rows, "test")
                        candidates.append(
                            {
                                "fit_mode": fit_mode,
                                "scope": scope,
                                "blend": float(blend),
                                "smooth_window": int(smooth_window),
                                "clip": clip,
                                **{f"val_{k}": v for k, v in val_m.items()},
                                **{f"test_{k}": v for k, v in test_m.items()},
                                "val_score": score(val_m),
                                "passes_guard": (
                                    val_m["rmse"] <= base_val["rmse"] + 0.0005
                                    and val_m["mae"] <= base_val["mae"] + 0.0005
                                    and val_m["pcc"] >= base_val["pcc"] - 0.0005
                                    and val_m["ccc"] >= base_val["ccc"] + 0.0002
                                ),
                            }
                        )
                        cache[key] = selected_frames

    result_df = pd.DataFrame(candidates)
    result_df.to_csv(OUT_DIR / "all_affine_calibration_candidates.csv", index=False)
    guarded = result_df[result_df["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "fit_mode": "identity",
            "scope": "identity",
            "blend": 0.0,
            "smooth_window": 1,
            "clip": "none",
            **{f"val_{k}": v for k, v in base_val.items()},
            **{f"test_{k}": v for k, v in base_test.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_frames = frames
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_frames = cache[(str(selected["fit_mode"]), str(selected["scope"]), float(selected["blend"]), int(selected["smooth_window"]), str(selected["clip"]))]

    per_split = []
    for split_name in SPLITS:
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = selected_frames[(split_name, data_split)]
            df.to_csv(out_dir / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_current_best_affine_calibration_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "affine/mean-std params fitted on train; blend/smoothing selected on validation macro guard; test labels not used",
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
