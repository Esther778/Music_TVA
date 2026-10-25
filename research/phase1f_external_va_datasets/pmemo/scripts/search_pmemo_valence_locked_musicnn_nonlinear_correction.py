#!/usr/bin/env python3
"""Screen compact nonlinear song-polarity models on locked PMEmo dev folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from search_pmemo_valence_locked_musicnn_embedding_correction import (
    DEFAULT_FEATURES, FOLDS, feature_matrix, load_song_table, macro,
)
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs" / "valence_locked_musicnn_mood_correction_20260714"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_musicnn_nonlinear_correction_20260714"


def model_specs() -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    for components in [2, 4, 8, 16]:
        specs.append((f"pls_{components}", make_pipeline(StandardScaler(), PLSRegression(n_components=components, scale=False, max_iter=1000))))
    for kernel in ["linear", "rbf"]:
        for c in [0.3, 1.0, 3.0, 10.0]:
            for epsilon in [0.03, 0.06]:
                specs.append((f"svr_{kernel}_c{c:g}_e{epsilon:g}", make_pipeline(StandardScaler(), SVR(kernel=kernel, C=c, epsilon=epsilon, gamma="scale"))))
    for leaf in [3, 5, 10]:
        for max_features in [0.3, 0.7]:
            specs.append((f"extra_leaf{leaf}_mf{max_features:g}", ExtraTreesRegressor(
                n_estimators=100, min_samples_leaf=leaf, max_features=max_features,
                random_state=20260714, n_jobs=-1,
            )))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-sets", nargs="+", default=["mean_base", "mean_std_base"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    songs = {fold: load_song_table(frames[fold], fold, args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    candidates: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for feature_set in args.feature_sets:
        for model_id, model in model_specs():
            shifts: dict[str, np.ndarray] = {}
            for held in FOLDS:
                train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                test = songs[held]
                model.fit(feature_matrix(train, feature_set), train["target_song"].to_numpy(dtype=np.float64))
                estimate = np.asarray(model.predict(feature_matrix(test, feature_set))).reshape(-1)
                shifts[held] = estimate - test["base_mean"].to_numpy(dtype=np.float64)
            raw_cache[(feature_set, model_id)] = shifts
            for strength in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.25, 1.50]:
                for cap in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080]:
                    per_fold = []
                    for fold in FOLDS:
                        song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                        shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
                        base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                        pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                        per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                    result = macro(per_fold)
                    guarded = (
                        result["mae"] <= base_macro["mae"] + 0.001
                        and result["pcc"] >= base_macro["pcc"] - 0.001
                        and result["ccc"] >= base_macro["ccc"] - 0.001
                    )
                    candidates.append({"feature_set": feature_set, "model": model_id, "strength": strength, "cap": cap, **result, "guarded": guarded})

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    shifts = raw_cache[(str(selected["feature_set"]), str(selected["model"]))]
    per_fold_rows = []
    for fold in FOLDS:
        song_shift = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
        shift = frames[fold]["musicId"].astype(int).map(song_shift).to_numpy(dtype=np.float64)
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_nonlinear_pred_valence"})
        out["musicnn_nonlinear_song_shift"] = shift
        out["pred_valence"] = np.clip(
            out["pre_nonlinear_pred_valence"].to_numpy(dtype=np.float64)
            + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0,
        )
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Cross-devfold song-level fit; final locked test excluded",
        "base_macro": base_macro, "selected": selected, "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
