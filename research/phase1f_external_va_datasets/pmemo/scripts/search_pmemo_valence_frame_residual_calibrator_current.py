#!/usr/bin/env python3
"""Frame-level residual calibrator for current best Valence predictions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_frame_residual_calibrator_current_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_smoothing_20260708",
    "rmse": 0.18955274335438643,
    "mae": 0.1474380568179414,
    "pcc": 0.8106359569208683,
    "ccc": 0.7974758948321773,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.groupby("musicId", sort=False):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        time = g["time_sec"].to_numpy(dtype=np.float64)
        duration = max(float(time.max() - time.min() + 0.5), 0.5)
        pos = (time - time.min()) / duration
        out = g.copy()
        out["f_pred"] = pred
        out["f_pred2"] = pred**2
        out["f_pred3"] = pred**3
        out["f_abs_pred"] = np.abs(pred)
        out["f_pos"] = pos
        out["f_pos2"] = pos**2
        out["f_sin_pos"] = np.sin(2.0 * np.pi * pos)
        out["f_cos_pos"] = np.cos(2.0 * np.pi * pos)
        out["f_song_pred_mean"] = float(np.mean(pred))
        out["f_song_pred_std"] = float(np.std(pred))
        out["f_song_pred_min"] = float(np.min(pred))
        out["f_song_pred_max"] = float(np.max(pred))
        out["f_song_pred_range"] = float(np.max(pred) - np.min(pred))
        out["f_song_pred_q10"] = float(np.quantile(pred, 0.10))
        out["f_song_pred_q90"] = float(np.quantile(pred, 0.90))
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


FEATURE_COLS = [
    "f_pred",
    "f_pred2",
    "f_pred3",
    "f_abs_pred",
    "f_pos",
    "f_pos2",
    "f_sin_pos",
    "f_cos_pos",
    "f_song_pred_mean",
    "f_song_pred_std",
    "f_song_pred_min",
    "f_song_pred_max",
    "f_song_pred_range",
    "f_song_pred_q10",
    "f_song_pred_q90",
]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("elastic_a003", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=20000))),
        ("elastic_a010", make_pipeline(StandardScaler(), ElasticNet(alpha=0.10, l1_ratio=0.2, max_iter=20000))),
    ]


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def apply_residual(df: pd.DataFrame, model, strength: float, cap: float) -> pd.DataFrame:
    out = df.copy()
    base = out["pred_valence"].to_numpy(dtype=np.float64)
    residual = model.predict(out[FEATURE_COLS].to_numpy(dtype=np.float64))
    residual = np.clip(residual, -cap, cap)
    out["base_pred_valence"] = base
    out["residual_correction"] = strength * residual
    out["pred_valence"] = np.clip(base + out["residual_correction"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): add_features(pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv")) for s in SPLITS for d in DATA_SPLITS}
    baseline_rows = []
    for split in SPLITS:
        row = {"split_name": split}
        for data in DATA_SPLITS:
            row.update({f"{data}_{m}": v for m, v in eval_df(frames[(split, data)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")

    rows = []
    cache: dict[tuple[str, float, float], dict[tuple[str, str], pd.DataFrame]] = {}
    for model_id, _ in model_candidates():
        for strength in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
            for cap in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]:
                per_split = []
                corrected = {}
                for split in SPLITS:
                    model = dict(model_candidates())[model_id]
                    train = frames[(split, "train")]
                    y_res = train["target_valence"].to_numpy(dtype=np.float64) - train["pred_valence"].to_numpy(dtype=np.float64)
                    model.fit(train[FEATURE_COLS].to_numpy(dtype=np.float64), y_res)
                    row = {"split_name": split}
                    for data in DATA_SPLITS:
                        df = apply_residual(frames[(split, data)], model, float(strength), float(cap))
                        corrected[(split, data)] = df
                        row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
                    per_split.append(row)
                val_m = macro(per_split, "val")
                test_m = macro(per_split, "test")
                rows.append(
                    {
                        "model_id": model_id,
                        "strength": float(strength),
                        "cap": float(cap),
                        **{f"val_{m}": v for m, v in val_m.items()},
                        **{f"test_{m}": v for m, v in test_m.items()},
                        "val_score": score(val_m),
                        "passes_guard": (
                            val_m["rmse"] <= base_val["rmse"] + 0.001
                            and val_m["mae"] <= base_val["mae"] + 0.001
                            and val_m["pcc"] >= base_val["pcc"] - 0.001
                            and val_m["ccc"] >= base_val["ccc"] + 0.0003
                        ),
                    }
                )
                cache[(model_id, float(strength), float(cap))] = corrected

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_frame_residual_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "model_id": "identity",
            "strength": 0.0,
            "cap": 0.0,
            **{f"val_{m}": v for m, v in base_val.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_key = None
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_key = (str(selected["model_id"]), float(selected["strength"]), float(selected["cap"]))

    per_split = []
    for split in SPLITS:
        split_out = OUT_DIR / split
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split}
        for data in DATA_SPLITS:
            df = frames[(split, data)] if selected_key is None else cache[selected_key][(split, data)]
            keep_cols = [c for c in df.columns if not c.startswith("f_")]
            df[keep_cols].to_csv(split_out / f"{data}_predictions.csv", index=False)
            row.update({f"{data}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_frame_residual_calibrator_current_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "frame residual calibrator fitted on train only; global config selected on validation macro guard",
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
