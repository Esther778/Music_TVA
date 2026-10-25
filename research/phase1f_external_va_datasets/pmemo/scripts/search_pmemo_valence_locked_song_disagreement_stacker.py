#!/usr/bin/env python3
"""Song-level audio-predictor disagreement stacker for locked Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_arousal_assisted_crossfold import AROUSAL_DIRS
from search_pmemo_valence_locked_audio_ensemble_postprocess import pred_col
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_monotonic_song_calibration_20260714"
THREEWAY = ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713"
MOOD = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_mood_semantics_val207_20260714/song_features.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_song_disagreement_stacker_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def summarize(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    grouped = frame.groupby("musicId", sort=False)
    result = pd.DataFrame({"musicId": grouped.size().index})
    for col in columns:
        name = f"{prefix}_{col}"
        stats = grouped[col].agg(["mean", "std", "min", "max"]).reset_index()
        stats["std"] = stats["std"].fillna(0.0)
        stats[f"{name}_range"] = stats["max"] - stats["min"]
        stats = stats.rename(columns={s: f"{name}_{s}" for s in ["mean", "std", "min", "max"]})
        result = result.merge(stats, on="musicId", validate="one_to_one")
    return result


def load_fold(base_dir: Path, fold: str, mood: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    frame = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
    song = summarize(frame, ["pred_valence"], "current")
    target = frame.groupby("musicId", sort=False)["target_valence"].mean().rename("target_song").reset_index()
    song = song.merge(target, on="musicId", validate="one_to_one")

    threeway = pd.read_csv(THREEWAY / f"{fold}_val_predictions.csv")
    val_cols = [c for c in ["base_pred_valence", "muq_raw_pred_valence", "muq_post_pred_valence", "mlp_post_pred_valence", "pred_valence"] if c in threeway]
    val_stats = summarize(threeway, val_cols, "v")
    song = song.merge(val_stats, on="musicId", validate="one_to_one")

    arousal_features: list[str] = []
    for name, directory in AROUSAL_DIRS.items():
        arousal = pd.read_csv(directory / f"{fold}_val_predictions.csv")
        pc = "pred_arousal" if "pred_arousal" in arousal else pred_col(arousal)
        stats = summarize(arousal, [pc], name)
        arousal_features.extend([c for c in stats if c != "musicId"])
        song = song.merge(stats, on="musicId", validate="one_to_one")

    mood_cols = [c for c in mood if c.startswith("happy_") or c.startswith("sad_")]
    song = song.merge(mood[["musicId", *mood_cols]], on="musicId", validate="one_to_one")
    base_features = [c for c in song if c.startswith("current_")]
    valence_features = base_features + [c for c in song if c.startswith("v_")]
    sets = {
        "valence": valence_features,
        "valence_arousal": valence_features + arousal_features,
        "all": valence_features + arousal_features + mood_cols,
    }
    return frame, song, sets


def model_specs() -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    for alpha in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 10.0, 100.0, 1000.0]:
        specs.append((f"ridge_{alpha:g}", make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
    specs.extend([
        ("elastic_0003_l01", make_pipeline(StandardScaler(), ElasticNet(alpha=0.0003, l1_ratio=0.1, max_iter=20000))),
        ("elastic_0003_l05", make_pipeline(StandardScaler(), ElasticNet(alpha=0.0003, l1_ratio=0.5, max_iter=20000))),
        ("elastic_001_l01", make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000))),
        ("elastic_001_l05", make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000))),
        ("huber", make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=1000))),
        ("bayesian", make_pipeline(StandardScaler(), BayesianRidge())),
        ("pls_2", make_pipeline(StandardScaler(), PLSRegression(n_components=2, scale=False, max_iter=1000))),
        ("pls_4", make_pipeline(StandardScaler(), PLSRegression(n_components=4, scale=False, max_iter=1000))),
        ("pls_8", make_pipeline(StandardScaler(), PLSRegression(n_components=8, scale=False, max_iter=1000))),
        ("hgb_leaf4", HistGradientBoostingRegressor(learning_rate=0.025, max_leaf_nodes=4, min_samples_leaf=30, l2_regularization=3.0, max_iter=200, random_state=20260714)),
        ("hgb_leaf6", HistGradientBoostingRegressor(learning_rate=0.025, max_leaf_nodes=6, min_samples_leaf=30, l2_regularization=3.0, max_iter=200, random_state=20260714)),
        ("extra_leaf5", ExtraTreesRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.7, random_state=20260714, n_jobs=-1)),
        ("extra_leaf10", ExtraTreesRegressor(n_estimators=150, min_samples_leaf=10, max_features=0.7, random_state=20260714, n_jobs=-1)),
    ])
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mood-csv", type=Path, default=MOOD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-sets", nargs="+", default=None)
    parser.add_argument("--model-names", nargs="+", default=None)
    parser.add_argument("--strengths", nargs="+", type=float, default=None)
    parser.add_argument("--caps", nargs="+", type=float, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mood = pd.read_csv(args.mood_csv)
    frames, songs, feature_sets = {}, {}, {}
    for fold in FOLDS:
        frames[fold], songs[fold], feature_sets = load_fold(args.base_dir, fold, mood)
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    candidates: list[dict[str, Any]] = [{"feature_set": "identity", "model": "identity", "target_mode": "identity", "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True}]
    raw_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    selected_feature_sets = feature_sets if args.feature_sets is None else {name: feature_sets[name] for name in args.feature_sets}
    selected_models = model_specs()
    if args.model_names is not None:
        selected_models = [(name, model) for name, model in selected_models if name in set(args.model_names)]
    strengths = args.strengths or [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]
    caps = args.caps or [0.003, 0.005, 0.0075, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040, 0.060, 0.080]
    for feature_name, columns in selected_feature_sets.items():
        for model_name, model in selected_models:
            for target_mode in ["direct", "residual"]:
                shifts: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                    test = songs[held]
                    target = train["target_song"].to_numpy(dtype=np.float64)
                    base_name = "current_pred_valence_mean"
                    if target_mode == "residual":
                        target = target - train[base_name].to_numpy(dtype=np.float64)
                    model.fit(train[columns].to_numpy(dtype=np.float64), target)
                    estimate = np.asarray(model.predict(test[columns].to_numpy(dtype=np.float64))).reshape(-1)
                    if target_mode == "direct":
                        estimate = estimate - test[base_name].to_numpy(dtype=np.float64)
                    shifts[held] = estimate
                raw_cache[(feature_name, model_name, target_mode)] = shifts
                for strength in strengths:
                    for cap in caps:
                        per_fold = []
                        for fold in FOLDS:
                            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                            shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                            per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                        result = macro(per_fold)
                        guarded = result["mae"] <= base_macro["mae"] + 0.0005 and result["pcc"] >= base_macro["pcc"] - 0.0005 and result["ccc"] >= 0.80
                        candidates.append({"feature_set": feature_name, "model": model_name, "target_mode": target_mode, "strength": strength, "cap": cap, **result, "guarded": guarded})

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["feature_set"] == "identity":
        selected_preds = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        shifts = raw_cache[(str(selected["feature_set"]), str(selected["model"]), str(selected["target_mode"]))]
        selected_preds = {}
        for fold in FOLDS:
            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
            shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
            selected_preds[fold] = np.clip(base + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0)
    per_fold_rows = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_disagreement_pred_valence"})
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Song stacker fit on other two development folds using deployable audio predictions only; final locked test excluded",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
