#!/usr/bin/env python3
"""OpenSMILE song-level mean correction for the current Valence best.

Uses PMEmo released dynamic/static OpenSMILE features as audio-side
song-level predictors of Valence mean. Selection is global across splits and
validation-only; test labels are reported after selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
FEATURE_ROOT = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019" / "features"
OUT_DIR = ROOT / "outputs" / "valence_current_best_opensmile_song_mean_20260708"
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


def load_dynamic_table() -> pd.DataFrame:
    dyn = pd.read_csv(FEATURE_ROOT / "dynamic_features.csv")
    value_cols = [c for c in dyn.columns if c not in {"musicId", "frameTime"}]
    grouped = dyn.groupby("musicId", sort=False)[value_cols]
    mean = grouped.mean().add_prefix("dyn_mean_")
    std = grouped.std(ddof=0).fillna(0.0).add_prefix("dyn_std_")
    q10 = grouped.quantile(0.10).add_prefix("dyn_q10_")
    q90 = grouped.quantile(0.90).add_prefix("dyn_q90_")
    out = pd.concat([mean, std, q10, q90], axis=1).reset_index()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_static_table() -> pd.DataFrame:
    sta = pd.read_csv(FEATURE_ROOT / "static_features.csv")
    sta = sta.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return sta


def model_candidates(train_rows: int, dim: int) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = [
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
    ]
    for n in [16, 32, 64]:
        k = min(n, max(2, train_rows - 2), max(2, dim - 1))
        out.append((f"ridge_pca{k}_a30", make_pipeline(StandardScaler(), PCA(n_components=k, random_state=42), Ridge(alpha=30.0))))
        out.append((f"ridge_pca{k}_a100", make_pipeline(StandardScaler(), PCA(n_components=k, random_state=42), Ridge(alpha=100.0))))
    return out


def cap_value(name: str) -> float | None:
    if name == "none":
        return None
    return float(name.replace("cap", "")) / 1000.0


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.5 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.1 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def make_feature_tables(all_ids: list[int]) -> dict[str, pd.DataFrame]:
    dyn = load_dynamic_table()
    ids = pd.DataFrame({"musicId": all_ids})
    pred_only = ids.copy()
    dyn_t = ids.merge(dyn, on="musicId", how="left").fillna(0.0)
    return {
        "pred_only": pred_only,
        "opensmile_dynamic": dyn_t,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {
        (split_name, data_split): pd.read_csv(INPUT_DIR / split_name / f"{data_split}_predictions.csv")
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    song_tables = {
        (split_name, data_split): prediction_song_features(frames[(split_name, data_split)])
        for split_name in SPLITS
        for data_split in DATA_SPLITS
    }
    for table in song_tables.values():
        table["base_residual_mean"] = table["target_mean"].to_numpy(dtype=np.float64) - table["pred_mean"].to_numpy(dtype=np.float64)
    all_ids = sorted({int(sid) for table in song_tables.values() for sid in table["musicId"].tolist()})
    audio_tables = make_feature_tables(all_ids)
    pred_cols = [
        "frames",
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q10",
        "pred_q25",
        "pred_q75",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
        "pred_slope_proxy",
        "pred_abs_delta_mean",
    ]

    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates = []
    materialized: dict[tuple[str, str, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
    for feature_set, audio in audio_tables.items():
        merged = {}
        audio_cols = [c for c in audio.columns if c != "musicId"]
        for split_name in SPLITS:
            for data_split in DATA_SPLITS:
                merged[(split_name, data_split)] = song_tables[(split_name, data_split)].merge(audio, on="musicId", how="left").fillna(0.0)
        feature_cols_by_mode = {
            "target_mean": pred_cols + audio_cols if feature_set != "pred_only" else pred_cols,
            "residual_mean": pred_cols + audio_cols if feature_set != "pred_only" else pred_cols,
        }
        for target_mode, feature_cols in feature_cols_by_mode.items():
            train_rows = len(merged[(SPLITS[0], "train")])
            for model_id, _ in model_candidates(train_rows, len(feature_cols)):
                corrected_by_split: dict[tuple[str, str, float, str], dict[tuple[str, str], pd.DataFrame]] = {}
                pred_targets_by_split: dict[str, dict[str, dict[int, float]]] = {}
                for split_name in SPLITS:
                    train = merged[(split_name, "train")]
                    model = dict(model_candidates(len(train), len(feature_cols)))[model_id]
                    y_col = "target_mean" if target_mode == "target_mean" else "base_residual_mean"
                    model.fit(train[feature_cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                    pred_targets_by_split[split_name] = {}
                    for data_split in DATA_SPLITS:
                        table = merged[(split_name, data_split)]
                        pred = model.predict(table[feature_cols].to_numpy(dtype=np.float64))
                        if target_mode == "residual_mean":
                            pred = table["pred_mean"].to_numpy(dtype=np.float64) + pred
                        pred_targets_by_split[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred.tolist()))
                for strength in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70]:
                    for cap_name in ["cap005", "cap010", "cap015", "cap020", "cap030", "cap040", "cap060", "none"]:
                        key = (feature_set + "__" + target_mode, model_id, float(strength), cap_name)
                        rows = []
                        dfs: dict[tuple[str, str], pd.DataFrame] = {}
                        for split_name in SPLITS:
                            row = {"split_name": split_name}
                            for data_split in DATA_SPLITS:
                                corrected = apply_song_mean_shift(
                                    frames[(split_name, data_split)],
                                    pred_targets_by_split[split_name][data_split],
                                    float(strength),
                                    cap_value(cap_name),
                                )
                                dfs[(split_name, data_split)] = corrected
                                row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                            rows.append(row)
                        val_m = macro(rows, "val")
                        test_m = macro(rows, "test")
                        candidates.append(
                            {
                                "feature_set": feature_set,
                                "target_mode": target_mode,
                                "model_id": model_id,
                                "strength": float(strength),
                                "cap": cap_name,
                                **{f"val_{m}": v for m, v in val_m.items()},
                                **{f"test_{m}": v for m, v in test_m.items()},
                                "val_score": score(val_m),
                                "passes_guard": (
                                    val_m["rmse"] <= base_val["rmse"] + 0.001
                                    and val_m["mae"] <= base_val["mae"] + 0.001
                                    and val_m["pcc"] >= base_val["pcc"] - 0.001
                                    and val_m["ccc"] >= base_val["ccc"] + 0.0005
                                ),
                            }
                        )
                        materialized[key] = dfs

    cand = pd.DataFrame(candidates)
    cand.to_csv(OUT_DIR / "all_opensmile_song_mean_candidates.csv", index=False)
    guarded = cand[cand["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "feature_set": "identity",
            "target_mode": "identity",
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
        selected_dfs = materialized[
            (
                str(selected["feature_set"]) + "__" + str(selected["target_mode"]),
                str(selected["model_id"]),
                float(selected["strength"]),
                str(selected["cap"]),
            )
        ]

    per_split = []
    for split_name in SPLITS:
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            df = selected_dfs[(split_name, data_split)]
            df.to_csv(out_dir / f"{data_split}_predictions.csv", index=False)
            row.update({f"{data_split}_{m}": v for m, v in eval_df(df).items()})
        per_split.append(row)
    selected_test = macro(per_split, "test")
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_current_best_opensmile_song_mean_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "song-level target/residual model fitted on train using pred stats plus OpenSMILE features; global config selected on validation macro guard; test labels not used",
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
