#!/usr/bin/env python3
"""Frame-level HGB residual correction on the best HGB+shift+smoothing Valence curve."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_hgb_shift_smoothing_20260708"
OUT_DIR = ROOT / "outputs" / "valence_hgb_shift_frame_residual_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_hgb_shift_smoothing_20260708",
    "rmse": 0.1882446005417573,
    "mae": 0.14662528843869826,
    "pcc": 0.8139312132446679,
    "ccc": 0.799716001809544,
}


def pred_col(df: pd.DataFrame) -> str:
    return "pred_valence" if "pred_valence" in df.columns else "pred"


def target_col(df: pd.DataFrame) -> str:
    return "target_valence" if "target_valence" in df.columns else "target"


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df[pred_col(df)].to_numpy(dtype=np.float64), df[target_col(df)].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        4.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 0.9 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.4 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.03 * m["rmse"]
    )


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["musicId", "time_sec"]).copy()
    p = pred_col(out)
    g = out.groupby("musicId", sort=False)[p]
    out["pred_song_mean"] = g.transform("mean")
    out["pred_song_std"] = g.transform("std").fillna(0.0)
    out["pred_centered"] = out[p] - out["pred_song_mean"]
    out["pred_delta"] = g.diff().fillna(0.0)
    out["pred_abs_delta"] = out["pred_delta"].abs()
    out["time_norm"] = out.groupby("musicId", sort=False)["time_sec"].transform(
        lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
    )
    out["time_norm_sq"] = out["time_norm"] ** 2
    return out


def feature_cols() -> list[str]:
    return [
        "pred_valence",
        "pred_song_mean",
        "pred_song_std",
        "pred_centered",
        "pred_delta",
        "pred_abs_delta",
        "time_norm",
        "time_norm_sq",
    ]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        (
            "hgb_l2_010_leaf7",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=140,
                max_leaf_nodes=7,
                min_samples_leaf=50,
                l2_regularization=0.10,
                random_state=31,
            ),
        ),
        (
            "hgb_l2_100_leaf5",
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.02,
                max_iter=120,
                max_leaf_nodes=5,
                min_samples_leaf=80,
                l2_regularization=1.00,
                random_state=31,
            ),
        ),
    ]


def corrected(df: pd.DataFrame, residual: np.ndarray, strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    p = pred_col(out)
    corr = residual.astype(np.float64)
    if cap is not None:
        corr = np.clip(corr, -cap, cap)
    out["base_pred_valence"] = out[p].to_numpy(dtype=np.float64)
    out[p] = np.clip(out["base_pred_valence"].to_numpy(dtype=np.float64) + strength * corr, -1.0, 1.0)
    return out


def cap_value(name: str) -> float | None:
    return None if name == "none" else float(name.replace("cap", "")) / 1000.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): enrich(pd.read_csv(INPUT_DIR / s / f"{d}_predictions.csv")) for s in SPLITS for d in DATA_SPLITS}
    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    best_row = None
    best_payload = None
    for model_id, model_template in model_candidates():
        residuals_by_split = {}
        for split_name in SPLITS:
            train = frames[(split_name, "train")]
            y = train[target_col(train)].to_numpy(dtype=np.float64) - train[pred_col(train)].to_numpy(dtype=np.float64)
            model = model_template
            model.fit(train[feature_cols()].to_numpy(dtype=np.float64), y)
            residuals_by_split[split_name] = {
                data_split: model.predict(frames[(split_name, data_split)][feature_cols()].to_numpy(dtype=np.float64))
                for data_split in DATA_SPLITS
            }
        for strength in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]:
            for cap_name in ["cap002", "cap003", "cap005", "cap008", "cap010", "cap015", "none"]:
                rows = []
                for split_name in SPLITS:
                    row = {"split_name": split_name}
                    for data_split in DATA_SPLITS:
                        df = corrected(frames[(split_name, data_split)], residuals_by_split[split_name][data_split], float(strength), cap_value(cap_name))
                        row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
                    rows.append(row)
                val_m = macro(rows, "val")
                test_m = macro(rows, "test")
                row = {
                    "model_id": model_id,
                    "strength": float(strength),
                    "cap": cap_name,
                    **{f"val_{m}": v for m, v in val_m.items()},
                    **{f"test_{m}": v for m, v in test_m.items()},
                    "val_score": score(val_m),
                    "passes_guard": (
                        val_m["rmse"] <= base_val["rmse"] + 0.0008
                        and val_m["mae"] <= base_val["mae"] + 0.0008
                        and val_m["pcc"] >= base_val["pcc"] - 0.001
                        and val_m["ccc"] >= base_val["ccc"] - 0.001
                    ),
                }
                candidates.append(row)
                if row["passes_guard"] and (
                    best_row is None
                    or (row["val_score"], row["val_rmse"], -row["val_ccc"]) < (best_row["val_score"], best_row["val_rmse"], -best_row["val_ccc"])
                ):
                    best_row = row
                    best_payload = (residuals_by_split, float(strength), cap_name)
    pd.DataFrame(candidates).to_csv(OUT_DIR / "all_frame_residual_candidates.csv", index=False)
    if best_row is None or best_payload is None:
        selected = {
            "model_id": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            **{f"test_{m}": v for m, v in base_test.items()},
            "selection_reason": "identity retained; no frame-residual candidate passed validation guard",
        }
        per_split = baseline_rows
        selected_frames = frames
    else:
        selected = dict(best_row)
        selected["selection_reason"] = "selected by validation macro guard"
        residuals_by_split, strength, cap_name = best_payload
        per_split = []
        selected_frames = {}
        for split_name in SPLITS:
            row = {"split_name": split_name}
            for data_split in DATA_SPLITS:
                df = corrected(frames[(split_name, data_split)], residuals_by_split[split_name][data_split], strength, cap_value(cap_name))
                selected_frames[(split_name, data_split)] = df
                row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
            per_split.append(row)
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for data_split in DATA_SPLITS:
            selected_frames[(split_name, data_split)].to_csv(split_out / f"{data_split}_predictions.csv", index=False)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_hgb_shift_frame_residual_20260708",
        "input_model": CURRENT_BEST,
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
