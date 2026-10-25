#!/usr/bin/env python3
"""OpenSMILE dynamic-feature residual calibration for current best Valence.

The base prediction is the current best Valence output.  For each song-level
split, this script fits a small residual model on train frames only:

    residual = true_valence - base_pred_valence

Validation selects one global model/strength/cap configuration. Test metrics
are reported once after selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_extra_wide_rmse_shift_smoothing_20260708"
OPENSMILE_DATASET = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
OUT_DIR = ROOT / "outputs" / "valence_best_after_smoothing_opensmile_frame_residual_20260708"
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
    return metrics(
        df["pred_valence"].to_numpy(dtype=np.float64),
        df["target_valence"].to_numpy(dtype=np.float64),
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {
        m: float(np.mean([row[f"{data_split}_{m}"] for row in rows]))
        for m in ["rmse", "mae", "pcc", "ccc"]
    }


def pass_gap_score(m: dict[str, float]) -> float:
    return (
        3.5 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.1 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.5 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def load_opensmile_features(music_id: int) -> pd.DataFrame:
    data = np.load(OPENSMILE_DATASET / f"{int(music_id)}.npz")
    x = data["X"].astype(np.float64)
    time_sec = data["time_sec"].astype(np.float64)
    out = pd.DataFrame(x, columns=[f"smile_{i:03d}" for i in range(x.shape[1])])
    out.insert(0, "time_key", np.round(time_sec, 3))
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    cache: dict[int, pd.DataFrame] = {}
    for music_id, g in df.groupby("musicId", sort=False):
        music_id_int = int(music_id)
        if music_id_int not in cache:
            cache[music_id_int] = load_opensmile_features(music_id_int)
        base = g.copy()
        base["time_key"] = np.round(base["time_sec"].astype(float), 3)
        merged = base.merge(cache[music_id_int], on="time_key", how="left", validate="many_to_one")
        smile_cols = [c for c in merged.columns if c.startswith("smile_")]
        if merged[smile_cols].isna().any().any():
            raise ValueError(f"OpenSMILE alignment produced NaN features for musicId={music_id_int}")

        pred = merged["pred_valence"].to_numpy(dtype=np.float64)
        time = merged["time_sec"].to_numpy(dtype=np.float64)
        duration = max(float(time.max() - time.min() + 0.5), 0.5)
        pos = (time - time.min()) / duration
        merged["f_pred"] = pred
        merged["f_pred2"] = pred**2
        merged["f_abs_pred"] = np.abs(pred)
        merged["f_pos"] = pos
        merged["f_pos2"] = pos**2
        merged["f_sin_pos"] = np.sin(2.0 * np.pi * pos)
        merged["f_cos_pos"] = np.cos(2.0 * np.pi * pos)
        merged["f_song_pred_mean"] = float(np.mean(pred))
        merged["f_song_pred_std"] = float(np.std(pred))
        merged["f_song_pred_range"] = float(np.max(pred) - np.min(pred))
        chunks.append(merged)
    return pd.concat(chunks, ignore_index=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("smile_") or c.startswith("f_")]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("ridge_a1000", make_pipeline(StandardScaler(), Ridge(alpha=1000.0))),
        ("ridge_a3000", make_pipeline(StandardScaler(), Ridge(alpha=3000.0))),
        ("elastic_a002", make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.15, max_iter=20000))),
        ("elastic_a005", make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.15, max_iter=20000))),
        ("elastic_a010", make_pipeline(StandardScaler(), ElasticNet(alpha=0.10, l1_ratio=0.15, max_iter=20000))),
    ]


def apply_residual(df: pd.DataFrame, model: object, cols: list[str], strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    base = out["pred_valence"].to_numpy(dtype=np.float64)
    residual = np.asarray(model.predict(out[cols].to_numpy(dtype=np.float64)), dtype=np.float64)
    if cap is not None:
        residual = np.clip(residual, -cap, cap)
    out["base_pred_valence"] = base
    out["opensmile_residual_correction"] = strength * residual
    out["pred_valence"] = np.clip(base + out["opensmile_residual_correction"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def apply_residual_array(df: pd.DataFrame, residual: np.ndarray, strength: float, cap: float | None) -> pd.DataFrame:
    out = df.copy()
    base = out["pred_valence"].to_numpy(dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if cap is not None:
        correction = np.clip(correction, -cap, cap)
    out["base_pred_valence"] = base
    out["opensmile_residual_correction"] = strength * correction
    out["pred_valence"] = np.clip(base + out["opensmile_residual_correction"].to_numpy(dtype=np.float64), -1.0, 1.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split_name, data_split): add_features(pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv"))
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    cols = feature_cols(next(iter(frames.values())))

    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")

    rows = []
    cache: dict[tuple[str, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
    residual_cache: dict[tuple[str, str, str], np.ndarray] = {}
    model_lookup = dict(model_candidates())
    for model_id in model_lookup:
        for split_name in SPLITS:
            model = model_lookup[model_id]
            train = frames[(split_name, "train")]
            y_res = train["target_valence"].to_numpy(dtype=np.float64) - train["pred_valence"].to_numpy(dtype=np.float64)
            model.fit(train[cols].to_numpy(dtype=np.float64), y_res)
            for data_split in DATA_SPLITS:
                residual_cache[(model_id, split_name, data_split)] = np.asarray(
                    model.predict(frames[(split_name, data_split)][cols].to_numpy(dtype=np.float64)),
                    dtype=np.float64,
                )

    caps: list[float | None] = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, None]
    strengths = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    for model_id in model_lookup:
        for strength in strengths:
            for cap in caps:
                corrected: dict[tuple[str, str], pd.DataFrame] = {}
                per_split = []
                for split_name in SPLITS:
                    row = {"split_name": split_name}
                    for data_split in DATA_SPLITS:
                        df = apply_residual_array(
                            frames[(split_name, data_split)],
                            residual_cache[(model_id, split_name, data_split)],
                            float(strength),
                            cap,
                        )
                        corrected[(split_name, data_split)] = df
                        row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
                    per_split.append(row)
                val_m = macro(per_split, "val")
                test_m = macro(per_split, "test")
                cap_name = "none" if cap is None else f"{cap:.3f}"
                val_score = pass_gap_score(val_m)
                rows.append(
                    {
                        "model_id": model_id,
                        "strength": float(strength),
                        "cap": cap_name,
                        **{f"val_{m}": v for m, v in val_m.items()},
                        **{f"test_{m}": v for m, v in test_m.items()},
                        "val_score": val_score,
                        "passes_guard": (
                            val_score <= pass_gap_score(base_val) - 0.00005
                            and val_m["rmse"] <= base_val["rmse"] + 0.0015
                            and val_m["mae"] <= base_val["mae"] + 0.0015
                            and val_m["pcc"] >= base_val["pcc"] - 0.0025
                            and val_m["ccc"] >= base_val["ccc"] - 0.0015
                        ),
                    }
                )
                cache[(model_id, float(strength), cap_name)] = corrected

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_opensmile_frame_residual_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "model_id": "identity",
            "strength": 0.0,
            "cap": "none",
            **{f"val_{m}": v for m, v in base_val.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_key = None
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_key = (str(selected["model_id"]), float(selected["strength"]), str(selected["cap"]))

    per_split = []
    for split_name in SPLITS:
        split_out = OUT_DIR / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = frames[(split_name, data_split)] if selected_key is None else cache[selected_key][(split_name, data_split)]
            keep_cols = [c for c in df.columns if not (c.startswith("smile_") or c.startswith("f_") or c == "time_key")]
            df[keep_cols].to_csv(split_out / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)

    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_best_after_smoothing_opensmile_frame_residual_20260708",
        "input_model": CURRENT_BEST,
        "feature_source": "PMEmo official dynamic OpenSMILE sequence features plus current prediction/time features",
        "selection_rule": "fit residual on train only per split; one global config selected by validation pass-gap guard; test not used for selection",
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
