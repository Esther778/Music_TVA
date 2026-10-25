#!/usr/bin/env python3
"""Train-fold-only MuQ song-mean correction for the locked-dev Valence best."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import metrics, threshold_rows


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713"
FEATURES = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
LABELS = ROOT / "outputs/opensmile_baseline/dataset/sequence"
BASE = ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713"
OUT = ROOT / "outputs/valence_locked_muq_song_mean_correction_20260713"
FOLDS = ["fold_01", "fold_02", "fold_03"]


@lru_cache(maxsize=None)
def song_feature(sid: int, feature_set: str) -> np.ndarray:
    embedding = np.load(FEATURES / f"{sid}.npz")["embedding"].astype(np.float64)
    mean = embedding.mean(axis=0)
    if feature_set == "mean":
        return mean
    return np.concatenate([mean, embedding.std(axis=0)])


@lru_cache(maxsize=None)
def target_mean(sid: int) -> float:
    return float(np.load(LABELS / f"{sid}.npz")["Y"][:, 0].mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fold_data = {}
    predictions = {}
    for feature_set in ["mean", "mean_std"]:
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            key = (feature_set, alpha)
            predictions[key] = {}
            for fold in FOLDS:
                index = pd.read_csv(SPLITS / fold / "sequence_index.csv")
                train_ids = index.loc[index.split == "train", "musicId"].astype(int).tolist()
                val_ids = index.loc[index.split == "val", "musicId"].astype(int).tolist()
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(np.stack([song_feature(sid, feature_set) for sid in train_ids]), np.array([target_mean(sid) for sid in train_ids]))
                predictions[key][fold] = dict(zip(val_ids, model.predict(np.stack([song_feature(sid, feature_set) for sid in val_ids]))))
                if fold not in fold_data:
                    fold_data[fold] = pd.read_csv(BASE / f"{fold}_val_predictions.csv")

    rows, cache = [], {}
    for key, fold_predictions in predictions.items():
        feature_set, alpha = key
        for strength in [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]:
            for cap in [0.02, 0.04, 0.06, 0.08, 0.12, 0.20, None]:
                per_fold, corrected = {}, {}
                for fold, frame in fold_data.items():
                    frame = frame.copy()
                    current_mean = frame.groupby("musicId")["pred_valence"].transform("mean").to_numpy()
                    predicted_mean = frame["musicId"].map(fold_predictions[fold]).to_numpy(dtype=np.float64)
                    delta = float(strength) * (predicted_mean - current_mean)
                    if cap is not None:
                        delta = np.clip(delta, -float(cap), float(cap))
                    pred = frame["pred_valence"].to_numpy() + delta
                    corrected[fold] = pred
                    per_fold[fold] = metrics(pred, frame["target_valence"].to_numpy())
                macro = {metric: float(np.mean([value[metric] for value in per_fold.values()])) for metric in ["rmse", "mae", "pcc", "ccc"]}
                config = (feature_set, alpha, strength, cap)
                rows.append({"feature_set": feature_set, "alpha": alpha, "strength": strength, "cap": "none" if cap is None else cap, **{f"macro_{k}": v for k, v in macro.items()}})
                cache[config] = (per_fold, corrected)
    table = pd.DataFrame(rows).sort_values(["macro_rmse", "macro_mae"])
    table.to_csv(OUT / "candidates.csv", index=False)
    best = table.iloc[0].to_dict()
    cap = None if best["cap"] == "none" else float(best["cap"])
    config = (str(best["feature_set"]), float(best["alpha"]), float(best["strength"]), cap)
    per_fold, corrected = cache[config]
    for fold, frame in fold_data.items():
        frame = frame.copy()
        frame["pred_valence_before_song_mean"] = frame["pred_valence"]
        frame["pred_valence"] = corrected[fold]
        frame.to_csv(OUT / f"{fold}_val_predictions.csv", index=False)
    macro = {metric: float(best[f"macro_{metric}"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-13",
        "selection_boundary": "Correction models fit on each fold's train songs and evaluated on val songs; final test excluded.",
        "selected": best,
        "macro": macro,
        "per_fold": per_fold,
        "gap_rows": threshold_rows(macro),
        "base_macro": {"rmse": 0.18656091429128618, "mae": 0.1462227531380141, "pcc": 0.8122931092821416, "ccc": 0.7948559621324526},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
