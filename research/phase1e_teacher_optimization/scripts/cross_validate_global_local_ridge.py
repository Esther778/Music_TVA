from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cross_validate_mert_layer_mix import (
    ROOT,
    load_all,
    macro_metrics,
    window_metrics,
)
from train_valence_harmonic_context import scalar_metrics


def main() -> None:
    layer6, _, target, song_ids, fold_ids, frame_counts = load_all()
    rows = []
    oof_prediction = np.empty_like(target)
    for fold in range(5):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        train_song_ids = np.unique(song_ids[train_mask])
        held_song_ids = np.unique(song_ids[held_mask])

        global_train_x, global_train_y = [], []
        for song_id in train_song_ids:
            mask = song_ids == song_id
            global_train_x.append(layer6[mask].mean(axis=0))
            global_train_y.append(float(target[mask].mean()))
        global_train_x = np.asarray(global_train_x, dtype=np.float32)
        global_train_y = np.asarray(global_train_y, dtype=np.float32)
        global_mean = global_train_x.mean(axis=0, keepdims=True)
        global_std = global_train_x.std(axis=0, keepdims=True) + 1e-6
        global_model = Ridge(alpha=100.0, solver="lsqr", max_iter=2000, tol=1e-5)
        global_model.fit((global_train_x - global_mean) / global_std, global_train_y)

        local_train_x, local_train_y, local_weights = [], [], []
        for song_id in train_song_ids:
            mask = song_ids == song_id
            features = layer6[mask]
            labels = target[mask]
            local_train_x.append(features - features.mean(axis=0, keepdims=True))
            local_train_y.append(labels - labels.mean())
            local_weights.append(
                np.full(len(labels), 1.0 / len(labels), dtype=np.float32)
            )
        local_train_x = np.concatenate(local_train_x)
        local_train_y = np.concatenate(local_train_y)
        local_weights = np.concatenate(local_weights)
        local_weights *= len(local_weights) / local_weights.sum()
        local_mean = local_train_x.mean(axis=0, keepdims=True)
        local_std = local_train_x.std(axis=0, keepdims=True) + 1e-6
        local_model = Ridge(alpha=100.0, solver="lsqr", max_iter=2000, tol=1e-5)
        local_model.fit(
            (local_train_x - local_mean) / local_std,
            local_train_y,
            sample_weight=local_weights,
        )

        fold_predictions, fold_targets, fold_songs = [], [], []
        for song_id in held_song_ids:
            mask = song_ids == song_id
            features = layer6[mask]
            global_feature = features.mean(axis=0, keepdims=True)
            predicted_mean = float(
                global_model.predict(
                    (global_feature - global_mean) / global_std
                )[0]
            )
            centered_features = features - global_feature
            local_prediction = local_model.predict(
                (centered_features - local_mean) / local_std
            )
            local_prediction = local_prediction - local_prediction.mean()
            prediction = predicted_mean + local_prediction
            oof_prediction[mask] = prediction
            fold_predictions.append(prediction)
            fold_targets.append(target[mask])
            fold_songs.append(np.full(mask.sum(), song_id, dtype=np.int32))
        prediction = np.concatenate(fold_predictions)
        truth = np.concatenate(fold_targets)
        held_songs = np.concatenate(fold_songs)
        micro = scalar_metrics(prediction, truth)
        macro = macro_metrics(prediction, truth, held_songs)
        window5 = window_metrics(prediction, truth, held_songs, 10)
        window10 = window_metrics(prediction, truth, held_songs, 20)
        centered_prediction, centered_truth = [], []
        song_mean_prediction, song_mean_truth = [], []
        for song_id in held_song_ids:
            mask = held_songs == song_id
            pred_song = prediction[mask]
            truth_song = truth[mask]
            centered_prediction.append(pred_song - pred_song.mean())
            centered_truth.append(truth_song - truth_song.mean())
            song_mean_prediction.append(float(pred_song.mean()))
            song_mean_truth.append(float(truth_song.mean()))
        within = scalar_metrics(
            np.concatenate(centered_prediction), np.concatenate(centered_truth)
        )
        means = scalar_metrics(
            np.asarray(song_mean_prediction), np.asarray(song_mean_truth)
        )
        rows.append(
            {
                "fold": fold,
                **{f"micro_{key}": value for key, value in micro.items()},
                **{f"macro_{key}": value for key, value in macro.items()},
                **{f"window5_{key}": value for key, value in window5.items()},
                **{f"window10_{key}": value for key, value in window10.items()},
                **{f"within_{key}": value for key, value in within.items()},
                **{f"song_mean_{key}": value for key, value in means.items()},
            }
        )
        print(
            f"fold={fold} micro_rmse={micro['rmse']:.6f} "
            f"pcc={micro['pcc']:.6f} ccc={micro['ccc']:.6f} "
            f"within_pcc={within['pcc']:.6f}"
        )
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/global_local_ridge_five_fold.csv", index=False)
    summary = {
        "dataset": "DEAM original train split only",
        "folds": 5,
        "test_used": False,
        "legacy_validation_used": False,
        "global_alpha": 100.0,
        "local_alpha": 100.0,
        "metrics": {},
    }
    for column in table.columns:
        if column == "fold":
            continue
        summary["metrics"][f"{column}_mean"] = float(table[column].mean())
        summary["metrics"][f"{column}_std"] = float(table[column].std(ddof=1))
    (ROOT / "outputs/global_local_ridge_five_fold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/oof_global_local_valence.npy", oof_prediction)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
