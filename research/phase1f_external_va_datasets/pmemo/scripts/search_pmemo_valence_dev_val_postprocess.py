#!/usr/bin/env python3
"""Validation-only Valence postprocess search for the sealed-split dev model.

This script intentionally reads only validation predictions. It must not be
used to claim blind/test performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "valence_blind_holdout_base_20260709"
    / "split_01_seed_20260709"
)
DEFAULT_OUT = ROOT / "outputs" / "valence_dev_val_postprocess_20260709"


def moving_average_by_song(values: np.ndarray, song_ids: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    half = window // 2
    for sid in np.unique(song_ids):
        idx = np.where(song_ids == sid)[0]
        seq = values[idx]
        padded = np.pad(seq, (half, half), mode="edge")
        out[idx] = np.convolve(padded, kernel, mode="valid")[: len(seq)]
    return out


def fit_calibration(pred: np.ndarray, target: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float | str]]:
    if method == "identity":
        return pred.copy(), {"method": method}
    if method == "bias":
        b = float(np.mean(target) - np.mean(pred))
        return pred + b, {"method": method, "b": b}
    if method == "linear":
        a, b = np.polyfit(pred, target, deg=1)
        return a * pred + b, {"method": method, "a": float(a), "b": float(b)}
    if method == "mean_std":
        pred_std = float(np.std(pred))
        target_std = float(np.std(target))
        a = target_std / max(pred_std, 1e-8)
        b = float(np.mean(target) - a * np.mean(pred))
        return a * pred + b, {"method": method, "a": float(a), "b": b}
    raise ValueError(f"Unknown calibration method: {method}")


def score(row: dict[str, float]) -> float:
    return (
        row["rmse"]
        + 0.015 * max(0.0, row["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - row["ccc"])
        + 0.02 * max(0.0, 0.80 - row["pcc"])
    )


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(DEFAULT_INPUT / "val_predictions.csv")
    pred = pred_df["pred_valence"].to_numpy(dtype=np.float64)
    target = pred_df["target_valence"].to_numpy(dtype=np.float64)
    song = pred_df["musicId"].to_numpy()
    baseline = metrics(pred, target)

    rows: list[dict[str, float | int | str]] = []
    best: dict[str, float | int | str] | None = None
    for window in [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 61]:
        smoothed = moving_average_by_song(pred, song, window)
        for smooth_blend in [0.0, 0.25, 0.50, 0.75, 1.0]:
            smoothed_blend = (1.0 - smooth_blend) * pred + smooth_blend * smoothed
            for method in ["identity", "bias", "linear", "mean_std"]:
                calibrated, params = fit_calibration(smoothed_blend, target, method)
                for cal_blend in [0.25, 0.50, 0.75, 1.0]:
                    raw_pp = (1.0 - cal_blend) * smoothed_blend + cal_blend * calibrated
                    for clip in [False, True]:
                        pp = np.clip(raw_pp, -1.0, 1.0) if clip else raw_pp
                        m = metrics(pp, target)
                        row = {
                            "window": window,
                            "smooth_blend": smooth_blend,
                            "method": method,
                            "cal_blend": cal_blend,
                            "clip": clip,
                            "params": json.dumps(params, ensure_ascii=False),
                            **m,
                        }
                        row["score"] = score(row)  # type: ignore[arg-type]
                        rows.append(row)
                        if best is None or float(row["score"]) < float(best["score"]):
                            best = row

    candidates = pd.DataFrame(rows).sort_values(["score", "rmse", "ccc"], ascending=[True, True, False])
    candidates.to_csv(DEFAULT_OUT / "all_val_postprocess_candidates.csv", index=False)
    assert best is not None
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Validation-only Valence postprocess search; blind_test not read",
        "input_dir": str(DEFAULT_INPUT),
        "baseline_val": baseline,
        "selected_config": best,
        "val_gap_rows": threshold_rows(
            {
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "pcc": float(best["pcc"]),
                "ccc": float(best["ccc"]),
            }
        ),
        "selection_boundary": "Development only. Do not claim blind/test performance from this file.",
    }
    (DEFAULT_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_df = pred_df.copy()
    best_window = int(best["window"])
    best_smooth_blend = float(best["smooth_blend"])
    base = (1.0 - best_smooth_blend) * pred + best_smooth_blend * moving_average_by_song(pred, song, best_window)
    calibrated, _ = fit_calibration(base, target, str(best["method"]))
    post = (1.0 - float(best["cal_blend"])) * base + float(best["cal_blend"]) * calibrated
    if bool(best.get("clip", False)):
        post = np.clip(post, -1.0, 1.0)
    out_df["pred_valence_post"] = post
    out_df.to_csv(DEFAULT_OUT / "val_predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
