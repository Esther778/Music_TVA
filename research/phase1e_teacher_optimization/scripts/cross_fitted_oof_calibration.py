from __future__ import annotations

import json

import numpy as np

from cross_fitted_oof_stack import ROOT, load_target_and_ids
from cross_validate_mert_layer_mix import macro_metrics, window_metrics
from train_valence_harmonic_context import scalar_metrics


def main() -> None:
    target, song_ids, fold_ids = load_target_and_ids()
    prediction = np.load(ROOT / "outputs/oof_cross_fitted_stack_valence.npy")
    calibrated = np.empty_like(prediction)
    folds = []
    for fold in range(5):
        train_mask = fold_ids != fold
        held_mask = fold_ids == fold
        train_prediction = prediction[train_mask]
        train_target = target[train_mask]
        covariance = np.mean(
            (train_prediction - train_prediction.mean())
            * (train_target - train_target.mean())
        )
        slope = max(
            float(covariance / max(train_prediction.var(), 1e-8)),
            0.0,
        )
        intercept = float(train_target.mean() - slope * train_prediction.mean())
        calibrated[held_mask] = slope * prediction[held_mask] + intercept
        fold_metrics = scalar_metrics(calibrated[held_mask], target[held_mask])
        folds.append(
            {
                "fold": fold,
                "slope": slope,
                "intercept": intercept,
                **fold_metrics,
            }
        )
        print(
            f"fold={fold} slope={slope:.6f} intercept={intercept:.6f} "
            f"rmse={fold_metrics['rmse']:.6f} ccc={fold_metrics['ccc']:.6f}"
        )
    micro = scalar_metrics(calibrated, target)
    macro = macro_metrics(calibrated, target, song_ids)
    window5 = window_metrics(calibrated, target, song_ids, 10)
    window10 = window_metrics(calibrated, target, song_ids, 20)
    centered_prediction, centered_target = [], []
    mean_prediction, mean_target = [], []
    for song_id in np.unique(song_ids):
        mask = song_ids == song_id
        pred_song = calibrated[mask]
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
        "method": "cross-fitted positive affine calibration",
        "calibration_training": "four other OOF folds only",
        "test_used": False,
        "legacy_validation_used": False,
        "folds": folds,
        "metrics": {
            **{f"micro_{key}": value for key, value in micro.items()},
            **{f"macro_{key}": value for key, value in macro.items()},
            **{f"window5_{key}": value for key, value in window5.items()},
            **{f"window10_{key}": value for key, value in window10.items()},
            **{f"within_{key}": value for key, value in within.items()},
            **{f"song_mean_{key}": value for key, value in means.items()},
        },
    }
    (ROOT / "outputs/cross_fitted_oof_calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/oof_cross_fitted_calibrated_valence.npy", calibrated)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
