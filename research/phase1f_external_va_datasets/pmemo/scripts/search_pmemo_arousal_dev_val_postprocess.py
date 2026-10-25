#!/usr/bin/env python3
"""Validation-only postprocess search for Arousal dev predictions.

This script intentionally reads only validation predictions. It must not be
used to report blind/test performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_opensmile_arousal_trials import ROOT, metrics, threshold_rows


DEFAULT_INPUT = ROOT / "outputs" / "opensmile_arousal_dev_only_seed20260709_20260709" / "a_dev_h96_l1_do015_lr8e4_mse"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_arousal_dev_val_postprocess_20260709"


def fit_linear(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    var = float(np.var(pred))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((pred - pred.mean()) * (target - target.mean())) / var)
    b = float(target.mean() - a * pred.mean())
    return a, b


def smooth_by_song(df: pd.DataFrame, values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    tmp = df[["musicId", "time_sec"]].copy()
    tmp["value"] = values
    tmp["_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["musicId", "time_sec"])
    sm = (
        tmp.groupby("musicId", sort=False)["value"]
        .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
        .to_numpy(dtype=np.float64)
    )
    out = np.empty_like(sm)
    out[tmp["_order"].to_numpy(dtype=np.int64)] = sm
    return out


def transform(pred: np.ndarray, target: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "identity":
        return pred.copy(), {"method": method}
    if method == "bias":
        b = float(np.mean(target - pred))
        return pred + b, {"method": method, "b": b}
    if method == "linear":
        a, b = fit_linear(pred, target)
        return a * pred + b, {"method": method, "a": a, "b": b}
    if method == "mean_std":
        pm, ps = float(pred.mean()), float(pred.std() + 1e-12)
        tm, ts = float(target.mean()), float(target.std() + 1e-12)
        return (pred - pm) / ps * ts + tm, {"method": method, "pred_mean": pm, "pred_std": ps, "target_mean": tm, "target_std": ts}
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(args.input_dir / "val_predictions.csv")
    pred = val["pred_arousal"].to_numpy(dtype=np.float64)
    target = val["target_arousal"].to_numpy(dtype=np.float64)
    rows = []
    best: dict[str, Any] | None = None
    for window in [1, 3, 5, 7, 9, 11, 15, 21]:
        sm = smooth_by_song(val, pred, window)
        for smooth_blend in [0.0, 0.25, 0.5, 0.75, 1.0]:
            base = (1.0 - smooth_blend) * pred + smooth_blend * sm
            for method in ["identity", "bias", "linear", "mean_std"]:
                cal, params = transform(base, target, method)
                for cal_blend in [0.25, 0.5, 0.75, 1.0] if method != "identity" else [0.0]:
                    final = base if method == "identity" else (1.0 - cal_blend) * base + cal_blend * cal
                    final = np.clip(final, -1.0, 1.0)
                    m = metrics(final, target)
                    score = m["rmse"] + 0.01 * max(0.0, m["mae"] - 0.15) + 0.005 * max(0.0, 0.80 - m["ccc"])
                    row = {
                        "window": window,
                        "smooth_blend": smooth_blend,
                        "method": method,
                        "cal_blend": cal_blend,
                        "params": json.dumps(params, ensure_ascii=True),
                        **m,
                        "score": score,
                    }
                    rows.append(row)
                    if best is None or score < best["score"]:
                        best = row
                        best["pred"] = final
    assert best is not None
    result = pd.DataFrame([{k: v for k, v in row.items() if k != "pred"} for row in rows])
    result.to_csv(args.out_dir / "all_val_postprocess_candidates.csv", index=False)
    out = val.copy()
    out["base_pred_arousal"] = pred
    out["pred_arousal"] = best["pred"]
    out.to_csv(args.out_dir / "val_predictions.csv", index=False)
    best_public = {k: v for k, v in best.items() if k != "pred"}
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Validation-only Arousal postprocess search; blind_test not read",
        "input_dir": str(args.input_dir),
        "baseline_val": metrics(pred, target),
        "selected_config": best_public,
        "val_gap_rows": threshold_rows({k: best_public[k] for k in ["rmse", "mae", "pcc", "ccc"]}),
        "selection_boundary": "Development only. Do not claim blind/test performance from this file.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
