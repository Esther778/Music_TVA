#!/usr/bin/env python3
"""Refine pred-only song-mean correction for the current Valence best."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_current_best_tonal_correction import (
    CURRENT_BEST,
    INPUT_DIR,
    OUT_DIR as TONAL_OUT_DIR,
    apply_song_mean_shift,
    eval_df,
    prediction_song_features,
)
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


OUT_DIR = ROOT / "outputs" / "valence_current_best_predonly_refine_20260707"


FEATURE_SETS = {
    "basic": [
        "frames",
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_min",
        "pred_max",
        "pred_range",
        "pred_q10",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
    ],
    "expanded": [
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
    ],
    "compact": [
        "pred_mean",
        "pred_std",
        "pred_range",
        "pred_q10",
        "pred_q90",
        "pred_head_mean",
        "pred_tail_mean",
        "pred_slope_proxy",
    ],
    "mean_shape": [
        "duration_proxy",
        "pred_mean",
        "pred_std",
        "pred_range",
        "pred_head_mean",
        "pred_tail_mean",
        "pred_abs_delta_mean",
    ],
}


def model_candidates() -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        out.append((f"ridge_a{str(alpha).replace('.', 'p')}", make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
    out.extend(
        [
            ("huber_a0001", make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=2000))),
            ("elastic_a0005_l101", make_pipeline(StandardScaler(), ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=12000))),
            ("elastic_a001_l102", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=12000))),
        ]
    )
    return out


def cap_value(name: str) -> float | None:
    if name == "none":
        return None
    return float(name.replace("cap", "")) / 1000.0


def build_candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        frames = {part: pd.read_csv(split_dir / f"{part}_predictions.csv") for part in ["train", "val", "test"]}
        song_tables = {part: prediction_song_features(frames[part]) for part in ["train", "val", "test"]}
        baseline = {part: eval_df(frames[part]) for part in ["train", "val", "test"]}
        rows.append(
            {
                "split_name": split_dir.name,
                "feature_set": "baseline",
                "model_id": "no_change",
                "strength": 0.0,
                "cap": "none",
                **{f"{part}_{k}": v for part, m in baseline.items() for k, v in m.items()},
            }
        )
        for feature_set, cols in FEATURE_SETS.items():
            x_train = song_tables["train"][cols].to_numpy(dtype=np.float64)
            y_train = song_tables["train"]["target_mean"].to_numpy(dtype=np.float64)
            for model_id, model in model_candidates():
                model.fit(x_train, y_train)
                pred_means = {
                    part: dict(
                        zip(
                            song_tables[part]["musicId"].astype(int).tolist(),
                            model.predict(song_tables[part][cols].to_numpy(dtype=np.float64)),
                        )
                    )
                    for part in ["train", "val", "test"]
                }
                for strength in np.round(np.arange(0.10, 0.96, 0.05), 2).tolist():
                    for cap_name in ["cap005", "cap010", "cap015", "cap020", "cap030", "cap040", "cap060", "cap080", "cap100", "none"]:
                        key = (split_dir.name, feature_set, model_id, float(strength), cap_name)
                        split_metrics = {}
                        for part in ["train", "val", "test"]:
                            corrected = apply_song_mean_shift(frames[part], pred_means[part], float(strength), cap_value(cap_name))
                            split_metrics[part] = eval_df(corrected)
                        rows.append(
                            {
                                "split_name": split_dir.name,
                                "feature_set": feature_set,
                                "model_id": model_id,
                                "strength": float(strength),
                                "cap": cap_name,
                                **{f"{part}_{k}": v for part, m in split_metrics.items() for k, v in m.items()},
                            }
                        )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["feature_set", "model_id", "strength", "cap"]
    cols = [f"{part}_{metric}" for part in ["val", "test"] for metric in ["rmse", "mae", "pcc", "ccc"]]
    return df.groupby(keys, dropna=False)[cols].mean().reset_index()


def select_rows(agg_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    base = agg_df[agg_df["model_id"].eq("no_change")].iloc[0]
    guarded = agg_df[
        (agg_df["val_rmse"] <= base["val_rmse"] + 0.0015)
        & (agg_df["val_mae"] <= base["val_mae"] + 0.0015)
        & (agg_df["val_pcc"] >= base["val_pcc"] - 0.0015)
        & (agg_df["val_ccc"] >= base["val_ccc"] + 0.0005)
    ].copy()
    guarded["balanced_score"] = (
        3.0 * np.maximum(0.0, guarded["val_rmse"] - 0.18) / 0.18
        + 1.6 * np.maximum(0.0, guarded["val_mae"] - 0.15) / 0.15
        + 1.8 * np.maximum(0.0, 0.8 - guarded["val_ccc"]) / 0.8
        + 1.2 * np.maximum(0.0, 0.8 - guarded["val_pcc"]) / 0.8
    )
    out = {
        "balanced": guarded.sort_values(["balanced_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict(),
        "ccc_priority": guarded.sort_values(["val_ccc", "val_rmse", "val_mae"], ascending=[False, True, True]).iloc[0].to_dict(),
        "rmse_priority": guarded.sort_values(["val_rmse", "val_ccc", "val_mae"], ascending=[True, False, True]).iloc[0].to_dict(),
    }
    for row in out.values():
        row["baseline_val_rmse"] = float(base["val_rmse"])
        row["baseline_val_mae"] = float(base["val_mae"])
        row["baseline_val_pcc"] = float(base["val_pcc"])
        row["baseline_val_ccc"] = float(base["val_ccc"])
    return out


def selected_model(model_id: str):
    for name, model in model_candidates():
        if name == model_id:
            return model
    raise ValueError(f"Unknown model_id: {model_id}")


def materialize_selection(name: str, selected: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_dir in sorted(p for p in INPUT_DIR.glob("split_*") if p.is_dir()):
        frames = {part: pd.read_csv(split_dir / f"{part}_predictions.csv") for part in ["train", "val", "test"]}
        if str(selected["model_id"]) == "no_change":
            corrected = frames
        else:
            feature_set = str(selected["feature_set"])
            cols = FEATURE_SETS[feature_set]
            song_tables = {part: prediction_song_features(frames[part]) for part in ["train", "val", "test"]}
            model = selected_model(str(selected["model_id"]))
            model.fit(
                song_tables["train"][cols].to_numpy(dtype=np.float64),
                song_tables["train"]["target_mean"].to_numpy(dtype=np.float64),
            )
            pred_means = {
                part: dict(
                    zip(
                        song_tables[part]["musicId"].astype(int).tolist(),
                        model.predict(song_tables[part][cols].to_numpy(dtype=np.float64)),
                    )
                )
                for part in ["train", "val", "test"]
            }
            corrected = {
                part: apply_song_mean_shift(
                    frames[part],
                    pred_means[part],
                    float(selected["strength"]),
                    cap_value(str(selected["cap"])),
                )
                for part in ["train", "val", "test"]
            }
        split_out = OUT_DIR / name / split_dir.name
        split_out.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"split_name": split_dir.name}
        for part in ["train", "val", "test"]:
            df = corrected[part]
            df.to_csv(split_out / f"{part}_predictions.csv", index=False)
            row.update({f"{part}_{metric}": value for metric, value in eval_df(df).items()})
        rows.append(row)
    return rows


def summarize_selection(name: str, selected: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    test = {
        metric: float(np.mean([row[f"test_{metric}"] for row in rows]))
        for metric in ["rmse", "mae", "pcc", "ccc"]
    }
    return {
        "selection_name": name,
        "selected_config": selected,
        "test_metrics": test,
        "delta_vs_current_best": {metric: float(test[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(test),
        "per_split": rows,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    candidates.to_csv(OUT_DIR / "all_predonly_refine_candidates_by_split.csv", index=False)
    agg_df = aggregate(candidates)
    agg_df.to_csv(OUT_DIR / "all_predonly_refine_candidates_global.csv", index=False)
    selections = select_rows(agg_df)
    summaries = []
    for name, selected in selections.items():
        rows = materialize_selection(name, selected)
        summaries.append(summarize_selection(name, selected, rows))
    result = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_current_best_predonly_refine_20260707",
        "input_model": CURRENT_BEST,
        "source_tonal_experiment": str(TONAL_OUT_DIR.relative_to(ROOT)),
        "selection_rule": "global shared pred-only song-mean correction; validation guard; no test-label tuning",
        "summaries": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
