from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cross_fitted_extended_stack import load_model_oof
from cross_fitted_oof_stack import (
    ROOT,
    load_target_and_ids,
    optimize_weights,
)
from cross_validate_mert_layer_mix import macro_metrics, window_metrics
from train_valence_harmonic_context import scalar_metrics


def grouped_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    song_ids: np.ndarray,
    fold_ids: np.ndarray,
    short_mask: np.ndarray,
) -> dict:
    result = {}
    for name, mask in (
        ("all", np.ones(len(target), dtype=bool)),
        ("short", short_mask),
        ("long", ~short_mask),
    ):
        result[name] = {
            **{f"micro_{key}": value for key, value in scalar_metrics(
                prediction[mask], target[mask]
            ).items()},
            **{f"macro_{key}": value for key, value in macro_metrics(
                prediction[mask], target[mask], song_ids[mask]
            ).items()},
            **{f"window5_{key}": value for key, value in window_metrics(
                prediction[mask], target[mask], song_ids[mask], 10
            ).items()},
            **{f"window10_{key}": value for key, value in window_metrics(
                prediction[mask], target[mask], song_ids[mask], 20
            ).items()},
            "songs": int(len(np.unique(song_ids[mask]))),
            "frames": int(mask.sum()),
        }
    return result


def main() -> None:
    target, song_ids, fold_ids = load_target_and_ids()
    manifest = pd.read_csv(ROOT / "data/five_fold_manifest.csv")
    frame_count = dict(zip(manifest["song_id"], manifest["frames"]))
    short_mask = np.asarray(
        [frame_count[int(song_id)] == 60 for song_id in song_ids],
        dtype=bool,
    )
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
            load_model_oof(
                "global_local_embedding_adapter", song_ids, fold_ids
            ),
            load_model_oof(
                "global_local_music_auxiliary", song_ids, fold_ids
            ),
        ]
    )
    stacked = np.empty_like(target)
    fold_rows = []
    for fold in range(5):
        meta_mask = fold_ids != fold
        held_mask = fold_ids == fold
        fold_result = {"fold": fold, "types": {}}
        for clip_type, type_mask in (
            ("short", short_mask),
            ("long", ~short_mask),
        ):
            train_mask = meta_mask & type_mask
            evaluation_mask = held_mask & type_mask
            weights = optimize_weights(
                predictions[:, train_mask], target[train_mask]
            )
            held_prediction = np.tensordot(
                weights,
                predictions[:, evaluation_mask],
                axes=(0, 0),
            )
            stacked[evaluation_mask] = held_prediction
            metrics = scalar_metrics(
                held_prediction, target[evaluation_mask]
            )
            fold_result["types"][clip_type] = {
                "meta_songs": int(
                    len(np.unique(song_ids[train_mask]))
                ),
                "held_songs": int(
                    len(np.unique(song_ids[evaluation_mask]))
                ),
                "weights": dict(zip(names, weights.tolist())),
                **metrics,
            }
            print(
                f"fold={fold} type={clip_type} "
                f"weights={weights.round(4).tolist()} "
                f"rmse={metrics['rmse']:.6f} "
                f"pcc={metrics['pcc']:.6f} "
                f"ccc={metrics['ccc']:.6f}"
            )
        fold_rows.append(fold_result)

    result = {
        "dataset": "DEAM original train split only",
        "method": "cross-fitted clip-type-specific nonnegative stacking",
        "clip_type_rule": "short: 60 dynamic frames; long: more than 60",
        "meta_training": "same clip type in four other OOF folds only",
        "test_used": False,
        "legacy_validation_used": False,
        "models": names,
        "folds": fold_rows,
        "metrics": grouped_metrics(
            stacked, target, song_ids, fold_ids, short_mask
        ),
    }
    output = ROOT / "outputs/cross_fitted_clip_type_stack.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(
        ROOT / "outputs/oof_cross_fitted_clip_type_stack_valence.npy",
        stacked.astype(np.float32),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
