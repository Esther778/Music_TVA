#!/usr/bin/env python3
"""Validation-selected postprocessing search for PMEmo Valence predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import metrics, threshold_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "mert_hidden6_harmonic_focus" / "hidden6_harm_h80_proj320_do030_ccc005"
DEFAULT_OUT = ROOT / "outputs" / "valence_hidden6_harmonic_postprocess"

HISTORICAL_BEST = {
    "model_id": "pmemo_valence_mert_hidden6_harmonic_focus_v1",
    "trial_id": "hidden6_harm_h80_proj320_do030_ccc005",
    "rmse": 0.20619146525859833,
    "mae": 0.16583308577537537,
    "pcc": 0.7784220421543382,
    "ccc": 0.7577601526573324,
}


def load_split(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {
        "song_id": df["musicId"].to_numpy(),
        "time_sec": df["time_sec"].to_numpy(dtype=np.float32),
        "target": df["target_valence"].to_numpy(dtype=np.float32),
        "pred": df["pred_valence"].to_numpy(dtype=np.float32),
    }


def moving_average_by_song(pred: np.ndarray, song_ids: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.astype(np.float64).copy()
    out = pred.astype(np.float64).copy()
    for sid in np.unique(song_ids):
        idx = np.where(song_ids == sid)[0]
        local_window = min(window, len(idx))
        if local_window <= 1:
            continue
        kernel = np.ones(local_window, dtype=np.float64) / float(local_window)
        padded = np.pad(out[idx], (local_window // 2, local_window - 1 - local_window // 2), mode="edge")
        out[idx] = np.convolve(padded, kernel, mode="valid")
    return out


def exp_smooth_by_song(pred: np.ndarray, song_ids: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return pred.astype(np.float64).copy()
    out = pred.astype(np.float64).copy()
    for sid in np.unique(song_ids):
        idx = np.where(song_ids == sid)[0]
        values = out[idx].copy()
        if len(values) <= 1:
            continue
        forward = values.copy()
        for i in range(1, len(forward)):
            forward[i] = alpha * forward[i] + (1.0 - alpha) * forward[i - 1]
        backward = values.copy()
        for i in range(len(backward) - 2, -1, -1):
            backward[i] = alpha * backward[i] + (1.0 - alpha) * backward[i + 1]
        out[idx] = 0.5 * (forward + backward)
    return out


def fit_linear(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    var = float(np.var(x))
    if var < 1e-12:
        return {"kind": "linear", "a": 1.0, "b": 0.0}
    a = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    b = float(y.mean() - a * x.mean())
    return {"kind": "linear", "a": a, "b": b}


def fit_mean_std(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12:
        return {"kind": "mean_std", "a": 1.0, "b": 0.0}
    a = sy / sx
    b = float(y.mean() - a * x.mean())
    return {"kind": "mean_std", "a": float(a), "b": b}


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    if calibration["kind"] == "none":
        return pred.astype(np.float64).copy()
    return float(calibration["a"]) * pred.astype(np.float64) + float(calibration["b"])


def apply_config(data: dict[str, np.ndarray], config: dict[str, Any]) -> np.ndarray:
    pred = data["pred"].astype(np.float64)
    if config["smooth_kind"] == "moving_average":
        pred = moving_average_by_song(pred, data["song_id"], int(config["window"]))
    elif config["smooth_kind"] == "exp":
        pred = exp_smooth_by_song(pred, data["song_id"], float(config["alpha"]))
    elif config["smooth_kind"] != "none":
        raise ValueError(config["smooth_kind"])
    pred = apply_calibration(pred, config["calibration"])
    if config["clip"]:
        pred = np.clip(pred, -1.0, 1.0)
    return pred.astype(np.float32)


def score_metrics(m: dict[str, float], score_name: str) -> float:
    if score_name == "rmse":
        return float(m["rmse"])
    if score_name == "rmse_ccc":
        return float(m["rmse"] - 0.01 * m["ccc"])
    if score_name == "pass_composite":
        return float(m["rmse"] + m["mae"] + max(0.0, 0.8 - m["pcc"]) + max(0.0, 0.8 - m["ccc"]))
    raise ValueError(score_name)


def candidate_configs(val: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    smooths: list[dict[str, Any]] = [{"smooth_kind": "none", "window": 1, "alpha": 0.0}]
    smooths.extend({"smooth_kind": "moving_average", "window": w, "alpha": 0.0} for w in [3, 5, 7, 9, 11, 15])
    smooths.extend({"smooth_kind": "exp", "window": 1, "alpha": a} for a in [0.3, 0.5, 0.7, 0.85])
    configs = []
    for smooth in smooths:
        temp_cfg = {**smooth, "calibration": {"kind": "none"}, "clip": False}
        smoothed_val = apply_config(val, temp_cfg)
        calibrations = [
            {"kind": "none"},
            fit_linear(smoothed_val, val["target"]),
            fit_mean_std(smoothed_val, val["target"]),
        ]
        for calibration in calibrations:
            for clip in [False, True]:
                configs.append({**smooth, "calibration": calibration, "clip": clip})
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--score", default="rmse", choices=["rmse", "rmse_ccc", "pass_composite"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val = load_split(args.model_dir / "val_predictions.csv")
    test = load_split(args.model_dir / "test_predictions.csv")
    train = load_split(args.model_dir / "train_predictions.csv")

    rows = []
    best: dict[str, Any] | None = None
    for cfg in candidate_configs(val):
        val_pred = apply_config(val, cfg)
        test_pred = apply_config(test, cfg)
        val_m = metrics(val_pred, val["target"])
        test_m = metrics(test_pred, test["target"])
        row = {
            "config": cfg,
            "score_name": args.score,
            "score": score_metrics(val_m, args.score),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        rows.append(row)
        if best is None or row["score"] < best["score"]:
            best = row
    assert best is not None
    best_cfg = best["config"]
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        pred = apply_config(data, best_cfg)
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence_raw": data["pred"],
                "pred_valence_postprocessed": pred,
            }
        ).to_csv(args.out_dir / f"{split_name}_postprocessed_predictions.csv", index=False)

    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "config"}
        flat.update(
            {
                "smooth_kind": row["config"]["smooth_kind"],
                "window": row["config"]["window"],
                "alpha": row["config"]["alpha"],
                "calibration_kind": row["config"]["calibration"]["kind"],
                "calibration_a": row["config"]["calibration"].get("a", ""),
                "calibration_b": row["config"]["calibration"].get("b", ""),
                "clip": row["config"]["clip"],
            }
        )
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).sort_values("score").to_csv(args.out_dir / "postprocess_search.csv", index=False)
    result = {
        "selection": "validation_only",
        "score": args.score,
        "historical_best": HISTORICAL_BEST,
        "raw_val_metrics": metrics(val["pred"], val["target"]),
        "raw_test_metrics": metrics(test["pred"], test["target"]),
        "best_config": best_cfg,
        "best_val_metrics": {k.replace("val_", ""): best[f"val_{k.replace('val_', '')}"] for k in ["val_rmse", "val_mae", "val_pcc", "val_ccc"]},
        "best_test_metrics": {k.replace("test_", ""): best[f"test_{k.replace('test_', '')}"] for k in ["test_rmse", "test_mae", "test_pcc", "test_ccc"]},
        "delta_vs_historical_best": {
            "rmse": float(best["test_rmse"] - HISTORICAL_BEST["rmse"]),
            "mae": float(best["test_mae"] - HISTORICAL_BEST["mae"]),
            "pcc": float(best["test_pcc"] - HISTORICAL_BEST["pcc"]),
            "ccc": float(best["test_ccc"] - HISTORICAL_BEST["ccc"]),
        },
        "test_gap_rows": threshold_rows({k.replace("test_", ""): best[f"test_{k.replace('test_', '')}"] for k in ["test_rmse", "test_mae", "test_pcc", "test_ccc"]}),
    }
    (args.out_dir / "postprocess_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
