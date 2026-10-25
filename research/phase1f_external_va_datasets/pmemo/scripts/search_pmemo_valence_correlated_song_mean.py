#!/usr/bin/env python3
"""Correlated OpenSMILE song-level correction for Valence.

This experiment targets the remaining Valence song mean/polarity bias. It
selects compact song-level audio features by train-only correlation with the
song-level target/residual, then applies a capped correction to the current
frame-level Valence curve. One global configuration is selected on validation.
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

from search_pmemo_valence_current_best_tonal_correction import apply_song_mean_shift, eval_df, prediction_song_features
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


INPUT_DIR = ROOT / "outputs" / "valence_current_best_tonal_correction_20260707"
FEATURE_ROOT = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019" / "features"
OUT_DIR = ROOT / "outputs" / "valence_correlated_song_mean_20260708"
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

PRED_COLS = [
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


def cap_value(name: str) -> float | None:
    if name == "none":
        return None
    return float(name.replace("cap", "")) / 1000.0


def load_dynamic_song_table() -> pd.DataFrame:
    dyn = pd.read_csv(FEATURE_ROOT / "dynamic_features.csv")
    value_cols = [c for c in dyn.columns if c not in {"musicId", "frameTime"}]
    grouped = dyn.groupby("musicId", sort=False)[value_cols]
    blocks = [
        grouped.mean().add_prefix("dyn_mean_"),
        grouped.std(ddof=0).fillna(0.0).add_prefix("dyn_std_"),
        grouped.quantile(0.10).add_prefix("dyn_q10_"),
        grouped.quantile(0.90).add_prefix("dyn_q90_"),
    ]
    return pd.concat(blocks, axis=1).reset_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_static_song_table() -> pd.DataFrame:
    sta = pd.read_csv(FEATURE_ROOT / "static_features.csv")
    return sta.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def audio_tables(all_ids: list[int]) -> dict[str, pd.DataFrame]:
    ids = pd.DataFrame({"musicId": all_ids})
    dyn = ids.merge(load_dynamic_song_table(), on="musicId", how="left").fillna(0.0)
    sta = ids.merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)
    both = dyn.merge(sta, on="musicId", how="left", suffixes=("", "_static")).fillna(0.0)
    return {"dynamic": dyn, "static": sta, "dynamic_static": both}


def score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.4 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.1 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.4 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def macro(rows: list[dict[str, Any]], data_split: str) -> dict[str, float]:
    return {m: float(np.mean([row[f"{data_split}_{m}"] for row in rows])) for m in ["rmse", "mae", "pcc", "ccc"]}


def select_audio_cols(train: pd.DataFrame, audio_cols: list[str], y_col: str, k: int) -> list[str]:
    if k <= 0 or not audio_cols:
        return []
    y = train[y_col].to_numpy(dtype=np.float64)
    y = y - y.mean()
    y_std = float(np.sqrt(np.mean(y**2)))
    if y_std < 1e-12:
        return []
    scores: list[tuple[float, str]] = []
    for col in audio_cols:
        x = train[col].to_numpy(dtype=np.float64)
        x = x - x.mean()
        x_std = float(np.sqrt(np.mean(x**2)))
        if x_std < 1e-12:
            continue
        corr = abs(float(np.mean(x * y) / (x_std * y_std)))
        if np.isfinite(corr):
            scores.append((corr, col))
    scores.sort(reverse=True)
    return [col for _, col in scores[: min(k, len(scores))]]


def model_candidates() -> list[tuple[str, object]]:
    return [
        ("ridge_a30", make_pipeline(StandardScaler(), Ridge(alpha=30.0))),
        ("ridge_a100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("ridge_a300", make_pipeline(StandardScaler(), Ridge(alpha=300.0))),
        ("elastic_a003_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=10000))),
    ]


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
    # Keep this first pass compact. Loading all dynamic+static features was too
    # memory hungry on the local Mac, so this run only screens static features.
    ids = pd.DataFrame({"musicId": all_ids})
    audio_by_source = {"static": ids.merge(load_static_song_table(), on="musicId", how="left").fillna(0.0)}

    baseline_rows = []
    for split_name in SPLITS:
        row = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            row.update({f"{data_split}_{m}": v for m, v in eval_df(frames[(split_name, data_split)]).items()})
        baseline_rows.append(row)
    base_val = macro(baseline_rows, "val")
    base_test = macro(baseline_rows, "test")

    candidates: list[dict[str, Any]] = []
    model_cache: dict[tuple[str, str, int, str], dict[str, dict[str, dict[int, float]]]] = {}
    for source_name, audio in audio_by_source.items():
        audio_cols = [c for c in audio.columns if c != "musicId"]
        merged = {
            (split_name, data_split): song_tables[(split_name, data_split)].merge(audio, on="musicId", how="left").fillna(0.0)
            for split_name in SPLITS
            for data_split in DATA_SPLITS
        }
        for target_mode, y_col in [("target_mean", "target_mean"), ("residual_mean", "base_residual_mean")]:
            for top_k in [8, 16, 32, 64]:
                selected_cols_by_split = {
                    split_name: select_audio_cols(merged[(split_name, "train")], audio_cols, y_col, top_k)
                    for split_name in SPLITS
                }
                for model_id, _ in model_candidates():
                    pred_targets: dict[str, dict[str, dict[int, float]]] = {}
                    for split_name in SPLITS:
                        cols = PRED_COLS + selected_cols_by_split[split_name]
                        train = merged[(split_name, "train")]
                        model = dict(model_candidates())[model_id]
                        model.fit(train[cols].to_numpy(dtype=np.float64), train[y_col].to_numpy(dtype=np.float64))
                        pred_targets[split_name] = {}
                        for data_split in DATA_SPLITS:
                            table = merged[(split_name, data_split)]
                            pred = model.predict(table[cols].to_numpy(dtype=np.float64))
                            if target_mode == "residual_mean":
                                pred = table["pred_mean"].to_numpy(dtype=np.float64) + pred
                            pred_targets[split_name][data_split] = dict(zip(table["musicId"].astype(int).tolist(), pred.tolist()))
                    model_cache[(source_name, target_mode, int(top_k), model_id)] = pred_targets
                    for strength in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55]:
                        for cap_name in ["cap005", "cap010", "cap015", "cap020", "cap030", "cap040", "cap060", "none"]:
                            rows = []
                            for split_name in SPLITS:
                                row = {"split_name": split_name}
                                for data_split in DATA_SPLITS:
                                    corrected = apply_song_mean_shift(
                                        frames[(split_name, data_split)],
                                        pred_targets[split_name][data_split],
                                        float(strength),
                                        cap_value(cap_name),
                                    )
                                    row.update({f"{data_split}_{m}": v for m, v in eval_df(corrected).items()})
                                rows.append(row)
                            val_m = macro(rows, "val")
                            test_m = macro(rows, "test")
                            candidates.append(
                                {
                                    "source": source_name,
                                    "target_mode": target_mode,
                                    "top_k": int(top_k),
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
                                        and val_m["ccc"] >= base_val["ccc"] + 0.0003
                                    ),
                                }
                            )

    result = pd.DataFrame(candidates)
    result.to_csv(OUT_DIR / "all_correlated_song_mean_candidates.csv", index=False)
    guarded = result[result["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "source": "identity",
            "target_mode": "identity",
            "top_k": 0,
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
        pred_targets = model_cache[
            (str(selected["source"]), str(selected["target_mode"]), int(selected["top_k"]), str(selected["model_id"]))
        ]
        selected_dfs = {}
        for split_name in SPLITS:
            for data_split in DATA_SPLITS:
                selected_dfs[(split_name, data_split)] = apply_song_mean_shift(
                    frames[(split_name, data_split)],
                    pred_targets[split_name][data_split],
                    float(selected["strength"]),
                    cap_value(str(selected["cap"])),
                )

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
        "model_id": "pmemo_valence_correlated_song_mean_20260708",
        "input_model": CURRENT_BEST,
        "selection_rule": "train-only correlation feature selection plus song-level mean correction; one global config selected on validation macro guard",
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
