#!/usr/bin/env python3
"""Arousal-assisted Valence residual search after current closest artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


VALENCE_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
AROUSAL_BASE = ROOT / "outputs" / "valence_arousal_bilstm_assisted_residual_20260629"
OUT_DIR = ROOT / "outputs" / "valence_best_after_smoothing_arousal_residual_20260708"
SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
CURRENT_BEST = {
    "model_id": "pmemo_valence_extra_wide_rmse_shift_smoothing_20260708",
    "rmse": 0.18879084858935094,
    "mae": 0.14645707170893926,
    "pcc": 0.8121939345532717,
    "ccc": 0.7972789406085162,
}


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.0 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.6 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def load_merged(split_name: str, data_split: str) -> pd.DataFrame:
    val = pd.read_csv(VALENCE_DIR / split_name / f"{data_split}_predictions.csv")
    ar = pd.read_csv(AROUSAL_BASE / split_name / "arousal_postprocess" / f"{data_split}_predictions.csv")
    val["time_key"] = val["time_sec"].round(6)
    ar["time_key"] = ar["time_sec"].round(6)
    merged = val.merge(ar[["musicId", "time_key", "pred_arousal"]], on=["musicId", "time_key"], how="inner")
    if len(merged) != len(val):
        raise ValueError(f"{split_name}/{data_split} alignment mismatch: valence={len(val)} merged={len(merged)}")
    merged = merged.sort_values(["musicId", "time_sec"]).reset_index(drop=True)
    for col in ["pred_valence", "pred_arousal"]:
        grouped = merged.groupby("musicId", sort=False)[col]
        merged[f"{col}_song_mean"] = grouped.transform("mean")
        merged[f"{col}_song_std"] = grouped.transform("std").fillna(0.0)
        merged[f"{col}_centered"] = merged[col] - merged[f"{col}_song_mean"]
        merged[f"{col}_delta"] = grouped.diff().fillna(0.0)
    merged["time_norm"] = merged.groupby("musicId", sort=False)["time_sec"].transform(
        lambda s: (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
    )
    return merged


def feature_cols() -> list[str]:
    return [
        "pred_valence",
        "pred_valence_song_mean",
        "pred_valence_song_std",
        "pred_valence_centered",
        "pred_valence_delta",
        "pred_arousal",
        "pred_arousal_song_mean",
        "pred_arousal_song_std",
        "pred_arousal_centered",
        "pred_arousal_delta",
        "time_sec",
        "time_norm",
    ]


def residual_models() -> list[tuple[str, Any]]:
    return [
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("huber", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500))),
    ]


def corrected(df: pd.DataFrame, residual: np.ndarray, strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    corr = residual.astype(np.float64)
    if cap is not None:
        corr = np.clip(corr, -cap, cap)
    out["base_pred_valence"] = out["pred_valence"].to_numpy(dtype=np.float64)
    out["pred_valence"] = np.clip(out["base_pred_valence"].to_numpy(dtype=np.float64) + float(strength) * corr, -1.0, 1.0)
    return out


def cap_value(name: str) -> float | None:
    return None if name == "none" else float(name.replace("cap", "")) / 1000.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {(s, d): load_merged(s, d) for s in SPLITS for d in DATA_SPLITS}
    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    cache: dict[tuple[str, str, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
    for scope in ["per_split", "global"]:
        fitted: dict[str, Any] = {}
        if scope == "global":
            train_all = pd.concat([frames[(s, "train")] for s in SPLITS], ignore_index=True)
            y = train_all["target_valence"].to_numpy(dtype=np.float64) - train_all["pred_valence"].to_numpy(dtype=np.float64)
            for model_id, model in residual_models():
                model.fit(train_all[feature_cols()].to_numpy(dtype=np.float64), y)
                fitted[model_id] = model
        else:
            for split_name in SPLITS:
                train = frames[(split_name, "train")]
                y = train["target_valence"].to_numpy(dtype=np.float64) - train["pred_valence"].to_numpy(dtype=np.float64)
                fitted[split_name] = {}
                for model_id, model in residual_models():
                    model.fit(train[feature_cols()].to_numpy(dtype=np.float64), y)
                    fitted[split_name][model_id] = model
        for model_id, _ in residual_models():
            for strength in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
                for cap_name in ["cap002", "cap005", "cap010", "cap015", "cap020", "cap030", "none"]:
                    eval_rows = []
                    dfs = {}
                    for split_name in SPLITS:
                        row = {"split_name": split_name}
                        model = fitted[model_id] if scope == "global" else fitted[split_name][model_id]
                        for data_split in DATA_SPLITS:
                            base = frames[(split_name, data_split)]
                            resid = model.predict(base[feature_cols()].to_numpy(dtype=np.float64))
                            df = corrected(base, resid, strength, cap_value(cap_name))
                            dfs[(split_name, data_split)] = df
                            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
                        eval_rows.append(row)
                    val_m = macro(eval_rows, "val")
                    test_m = macro(eval_rows, "test")
                    key = (scope, model_id, float(strength), cap_name)
                    cache[key] = dfs
                    candidates.append(
                        {
                            "scope": scope,
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
                    )

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_arousal_residual_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "scope": "identity",
            "model_id": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            **{f"test_{m}": v for m, v in base_test.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_dfs = frames
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_dfs = cache[(str(selected["scope"]), str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))]

    per_split = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = selected_dfs[(split_name, data_split)]
            df.to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_best_after_smoothing_arousal_residual_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "cached predicted Arousal plus current Valence features; residual model selected by validation macro guard",
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
