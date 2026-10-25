from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cross_validate_mert_layer_mix import (
    INDEX,
    ROOT,
    macro_metrics,
    window_metrics,
)
from train_valence_harmonic_context import scalar_metrics


def load_target_and_ids():
    manifest = pd.read_csv(ROOT / "data/five_fold_manifest.csv").sort_values("song_id")
    index = pd.read_csv(INDEX).set_index("song_id")
    targets, song_ids, fold_ids = [], [], []
    for row in manifest.itertuples():
        with np.load(index.loc[int(row.song_id), "sequence_path"]) as data:
            target = data["Y"][:, 0].astype(np.float32)
        targets.append(target)
        song_ids.append(np.full(len(target), int(row.song_id), dtype=np.int32))
        fold_ids.append(np.full(len(target), int(row.fold), dtype=np.int8))
    return np.concatenate(targets), np.concatenate(song_ids), np.concatenate(fold_ids)


def load_deep_oof(song_ids: np.ndarray, fold_ids: np.ndarray):
    prediction = np.empty(len(song_ids), dtype=np.float32)
    for fold in range(5):
        fold_prediction = np.load(
            ROOT / f"outputs/global_local_temporal_fold{fold}/held_pred.npy"
        )
        mask = fold_ids == fold
        held_song_ids = np.unique(song_ids[mask])
        parts = []
        for song_id in held_song_ids:
            parts.append(np.full((song_ids == song_id).sum(), song_id, dtype=np.int32))
        expected_ids = np.concatenate(parts)
        actual_ids = song_ids[mask]
        if not np.array_equal(expected_ids, actual_ids):
            raise ValueError(f"Fold {fold}: held prediction song order mismatch.")
        if len(fold_prediction) != mask.sum():
            raise ValueError(f"Fold {fold}: prediction length mismatch.")
        prediction[mask] = fold_prediction
    return prediction


def optimize_weights(predictions: np.ndarray, target: np.ndarray):
    def objective(weights):
        values = np.tensordot(weights, predictions, axes=(0, 0))
        return float(np.mean((values - target) ** 2))

    result = minimize(
        objective,
        np.full(len(predictions), 1.0 / len(predictions)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(predictions),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def main() -> None:
    target, song_ids, fold_ids = load_target_and_ids()
    names = ["layer6_ridge", "global_local_ridge", "global_local_temporal"]
    predictions = np.stack(
        [
            np.load(ROOT / "outputs/oof_layer6_valence.npy"),
            np.load(ROOT / "outputs/oof_global_local_valence.npy"),
            load_deep_oof(song_ids, fold_ids),
        ]
    )
    stacked = np.empty_like(target)
    fold_rows = []
    for fold in range(5):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        weights = optimize_weights(predictions[:, train_mask], target[train_mask])
        held_prediction = np.tensordot(
            weights, predictions[:, held_mask], axes=(0, 0)
        )
        stacked[held_mask] = held_prediction
        metrics = scalar_metrics(held_prediction, target[held_mask])
        fold_rows.append(
            {
                "fold": fold,
                "weights": dict(zip(names, weights.tolist())),
                **metrics,
            }
        )
        print(
            f"fold={fold} weights={weights.round(4).tolist()} "
            f"rmse={metrics['rmse']:.6f} pcc={metrics['pcc']:.6f} "
            f"ccc={metrics['ccc']:.6f}"
        )
    micro = scalar_metrics(stacked, target)
    macro = macro_metrics(stacked, target, song_ids)
    window5 = window_metrics(stacked, target, song_ids, 10)
    window10 = window_metrics(stacked, target, song_ids, 20)
    centered_prediction, centered_target = [], []
    song_mean_prediction, song_mean_target = [], []
    for song_id in np.unique(song_ids):
        mask = song_ids == song_id
        pred_song = stacked[mask]
        target_song = target[mask]
        centered_prediction.append(pred_song - pred_song.mean())
        centered_target.append(target_song - target_song.mean())
        song_mean_prediction.append(float(pred_song.mean()))
        song_mean_target.append(float(target_song.mean()))
    within = scalar_metrics(
        np.concatenate(centered_prediction), np.concatenate(centered_target)
    )
    means = scalar_metrics(
        np.asarray(song_mean_prediction), np.asarray(song_mean_target)
    )
    result = {
        "dataset": "DEAM original train split only",
        "method": "cross-fitted nonnegative stacking, weights sum to one",
        "meta_training": "four other OOF folds only",
        "test_used": False,
        "legacy_validation_used": False,
        "models": names,
        "folds": fold_rows,
        "metrics": {
            **{f"micro_{key}": value for key, value in micro.items()},
            **{f"macro_{key}": value for key, value in macro.items()},
            **{f"window5_{key}": value for key, value in window5.items()},
            **{f"window10_{key}": value for key, value in window10.items()},
            **{f"within_{key}": value for key, value in within.items()},
            **{f"song_mean_{key}": value for key, value in means.items()},
        },
    }
    (ROOT / "outputs/cross_fitted_oof_stack.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/oof_cross_fitted_stack_valence.npy", stacked)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
