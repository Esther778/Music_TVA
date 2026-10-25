#!/usr/bin/env python3
"""Cross-fold dynamic MusiCNN Ridge branch for locked-development Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_monotonic_song_calibration_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_musicnn_sequence_embeddings_20260714/per_song"
DEFAULT_OUT = ROOT / "outputs/valence_locked_musicnn_dynamic_ridge_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def interpolate_embedding(embedding: np.ndarray, frames: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, embedding.shape[0])
    target = np.linspace(0.0, 1.0, frames)
    return np.stack([np.interp(target, source, embedding[:, dim]) for dim in range(embedding.shape[1])], axis=1)


def build_fold(frame: pd.DataFrame, feature_dir: Path) -> dict[str, np.ndarray]:
    embeddings, base_features, targets, weights, ids = [], [], [], [], []
    for music_id, song in frame.groupby("musicId", sort=False):
        embedding = np.load(feature_dir / f"{int(music_id)}.npz")["embedding"].astype(np.float64)
        aligned = interpolate_embedding(embedding, len(song))
        base = song["pred_valence"].to_numpy(dtype=np.float64)
        song_mean = float(base.mean())
        position = np.linspace(0.0, 1.0, len(song))
        embeddings.append(aligned)
        base_features.append(np.column_stack([base, base - song_mean, np.full(len(song), song_mean), position]))
        targets.append(song["target_valence"].to_numpy(dtype=np.float64))
        weights.append(np.full(len(song), 1.0 / len(song), dtype=np.float64))
        ids.append(np.full(len(song), int(music_id), dtype=np.int64))
    return {
        "embedding": np.concatenate(embeddings), "base": np.concatenate(base_features),
        "target": np.concatenate(targets), "weight": np.concatenate(weights),
        "musicId": np.concatenate(ids),
    }


def matrix(data: dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    if feature_set == "embedding":
        return data["embedding"]
    if feature_set == "embedding_base":
        return np.concatenate([data["embedding"], data["base"]], axis=1)
    raise ValueError(feature_set)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    data = {fold: build_fold(frames[fold], args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    raw_cache: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    candidates: list[dict[str, Any]] = [{"feature_set": "identity", "target_mode": "identity", "alpha": 0.0, "blend": 0.0, "cap": 0.0, **base_macro, "guarded": True}]
    for feature_set in ["embedding", "embedding_base"]:
        for target_mode in ["direct", "residual"]:
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
                raw: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = {key: np.concatenate([data[f][key] for f in FOLDS if f != held]) for key in data[held]}
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    target = train["target"]
                    if target_mode == "residual":
                        target = target - train["base"][:, 0]
                    model.fit(matrix(train, feature_set), target, ridge__sample_weight=train["weight"])
                    estimate = model.predict(matrix(data[held], feature_set))
                    if target_mode == "residual":
                        estimate = estimate + data[held]["base"][:, 0]
                    raw[held] = estimate
                raw_cache[(feature_set, target_mode, alpha)] = raw
                for blend in [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
                    for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.120]:
                        per_fold = []
                        for fold in FOLDS:
                            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            pred = np.clip(base + np.clip(blend * (raw[fold] - base), -cap, cap), -1.0, 1.0)
                            per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                        result = macro(per_fold)
                        guarded = (
                            result["mae"] <= base_macro["mae"] + 0.0005
                            and result["pcc"] >= base_macro["pcc"] - 0.0005
                            and result["ccc"] >= 0.80
                        )
                        candidates.append({"feature_set": feature_set, "target_mode": target_mode, "alpha": alpha, "blend": blend, "cap": cap, **result, "guarded": guarded})

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["feature_set"] == "identity":
        selected_preds = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        raw = raw_cache[(str(selected["feature_set"]), str(selected["target_mode"]), float(selected["alpha"]))]
        selected_preds = {}
        for fold in FOLDS:
            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
            selected_preds[fold] = np.clip(base + np.clip(float(selected["blend"]) * (raw[fold] - base), -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0)
    per_fold_rows = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_musicnn_dynamic_pred_valence"})
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Dynamic Ridge fit on the other two development folds with song-balanced weights; final locked test excluded",
        "alignment": "Per-song normalized-time linear interpolation from MusiCNN patches to label frames",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
