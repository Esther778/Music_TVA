from __future__ import annotations

import json

import numpy as np

from cross_fitted_oof_stack import (
    ROOT,
    load_target_and_ids,
    optimize_weights,
)
from cross_validate_mert_layer_mix import macro_metrics, window_metrics
from train_valence_harmonic_context import scalar_metrics


def load_model_oof(prefix: str, song_ids: np.ndarray, fold_ids: np.ndarray):
    prediction = np.empty(len(song_ids), dtype=np.float32)
    for fold in range(5):
        fold_prediction = np.load(
            ROOT / f"outputs/{prefix}_fold{fold}/held_pred.npy"
        )
        mask = fold_ids == fold
        if len(fold_prediction) != mask.sum():
            raise ValueError(f"{prefix} fold {fold}: prediction length mismatch.")
        prediction[mask] = fold_prediction
    return prediction


def main() -> None:
    target, song_ids, fold_ids = load_target_and_ids()
    names = [
        "layer6_ridge",
        "global_local_ridge",
        "global_local_temporal",
        "embedding_adapter",
        "music_auxiliary",
    ]
    predictions = np.stack(
        [
            np.load(ROOT / "outputs/oof_layer6_valence.npy"),
            np.load(ROOT / "outputs/oof_global_local_valence.npy"),
            load_model_oof("global_local_temporal", song_ids, fold_ids),
            load_model_oof("global_local_embedding_adapter", song_ids, fold_ids),
            load_model_oof("global_local_music_auxiliary", song_ids, fold_ids),
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
        fold_metrics = scalar_metrics(held_prediction, target[held_mask])
        fold_rows.append(
            {
                "fold": fold,
                "weights": dict(zip(names, weights.tolist())),
                **fold_metrics,
            }
        )
        print(
            f"fold={fold} weights={weights.round(4).tolist()} "
            f"rmse={fold_metrics['rmse']:.6f} "
            f"pcc={fold_metrics['pcc']:.6f} ccc={fold_metrics['ccc']:.6f}"
        )
    micro = scalar_metrics(stacked, target)
    macro = macro_metrics(stacked, target, song_ids)
    window5 = window_metrics(stacked, target, song_ids, 10)
    window10 = window_metrics(stacked, target, song_ids, 20)
    centered_prediction, centered_target = [], []
    mean_prediction, mean_target = [], []
    for song_id in np.unique(song_ids):
        mask = song_ids == song_id
        pred_song = stacked[mask]
        target_song = target[mask]
        centered_prediction.append(pred_song - pred_song.mean())
        centered_target.append(target_song - target_song.mean())
        mean_prediction.append(float(pred_song.mean()))
        mean_target.append(float(target_song.mean()))
    within = scalar_metrics(
        np.concatenate(centered_prediction), np.concatenate(centered_target)
    )
    means = scalar_metrics(np.asarray(mean_prediction), np.asarray(mean_target))
    result = {
        "dataset": "DEAM original train split only",
        "method": "cross-fitted extended nonnegative stacking",
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
    (ROOT / "outputs/cross_fitted_extended_stack.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/oof_cross_fitted_extended_valence.npy", stacked)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
