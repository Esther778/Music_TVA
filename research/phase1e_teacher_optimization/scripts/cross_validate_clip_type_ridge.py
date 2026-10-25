from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge

from cross_validate_mert_layer_mix import ROOT, load_all
from train_valence_harmonic_context import scalar_metrics


ALPHA = 100.0


def fit_direct(
    features: np.ndarray,
    target: np.ndarray,
    song_ids: np.ndarray,
    frame_counts: dict[int, int],
    train_mask: np.ndarray,
    held_mask: np.ndarray,
) -> np.ndarray:
    mean = features[train_mask].mean(axis=0, keepdims=True)
    std = features[train_mask].std(axis=0, keepdims=True) + 1e-6
    train_x = (features[train_mask] - mean) / std
    held_x = (features[held_mask] - mean) / std
    weights = np.asarray(
        [1.0 / frame_counts[int(song)] for song in song_ids[train_mask]],
        dtype=np.float64,
    )
    weights *= len(weights) / weights.sum()
    model = Ridge(
        alpha=ALPHA, solver="lsqr", max_iter=2000, tol=1e-5
    )
    model.fit(train_x, target[train_mask], sample_weight=weights)
    return model.predict(held_x).astype(np.float32)


def fit_global_local(
    features: np.ndarray,
    target: np.ndarray,
    song_ids: np.ndarray,
    train_mask: np.ndarray,
    held_mask: np.ndarray,
) -> np.ndarray:
    train_songs = np.unique(song_ids[train_mask])
    global_x, global_y = [], []
    local_x, local_y, local_weight = [], [], []
    for song_id in train_songs:
        mask = train_mask & (song_ids == song_id)
        song_x = features[mask]
        song_y = target[mask]
        mean_x = song_x.mean(axis=0)
        mean_y = float(song_y.mean())
        global_x.append(mean_x)
        global_y.append(mean_y)
        local_x.append(song_x - mean_x)
        local_y.append(song_y - mean_y)
        local_weight.append(
            np.full(len(song_y), 1.0 / len(song_y), dtype=np.float64)
        )
    global_x = np.asarray(global_x, dtype=np.float32)
    global_y = np.asarray(global_y, dtype=np.float32)
    local_x = np.concatenate(local_x)
    local_y = np.concatenate(local_y)
    local_weight = np.concatenate(local_weight)
    local_weight *= len(local_weight) / local_weight.sum()
    global_mean = global_x.mean(axis=0, keepdims=True)
    global_std = global_x.std(axis=0, keepdims=True) + 1e-6
    local_mean = local_x.mean(axis=0, keepdims=True)
    local_std = local_x.std(axis=0, keepdims=True) + 1e-6
    global_model = Ridge(
        alpha=ALPHA, solver="lsqr", max_iter=2000, tol=1e-5
    )
    local_model = Ridge(
        alpha=ALPHA, solver="lsqr", max_iter=2000, tol=1e-5
    )
    global_model.fit((global_x - global_mean) / global_std, global_y)
    local_model.fit(
        (local_x - local_mean) / local_std,
        local_y,
        sample_weight=local_weight,
    )
    prediction = np.empty(held_mask.sum(), dtype=np.float32)
    held_indices = np.where(held_mask)[0]
    held_songs = song_ids[held_mask]
    for song_id in np.unique(held_songs):
        local_positions = np.where(held_songs == song_id)[0]
        original_positions = held_indices[local_positions]
        song_x = features[original_positions]
        mean_x = song_x.mean(axis=0, keepdims=True)
        global_prediction = float(
            global_model.predict(
                (mean_x - global_mean) / global_std
            )[0]
        )
        local_prediction = local_model.predict(
            ((song_x - mean_x) - local_mean) / local_std
        )
        prediction[local_positions] = (
            global_prediction + local_prediction
        )
    return prediction


def main() -> None:
    layer6, _, target, song_ids, fold_ids, frame_counts = load_all()
    short_mask = np.asarray(
        [frame_counts[int(song)] == 60 for song in song_ids],
        dtype=bool,
    )
    outputs = {
        "direct": np.empty_like(target),
        "global_local": np.empty_like(target),
    }
    rows = []
    for fold in range(5):
        for clip_type, type_mask in (
            ("short", short_mask),
            ("long", ~short_mask),
        ):
            train_mask = (fold_ids != fold) & type_mask
            held_mask = (fold_ids == fold) & type_mask
            predictions = {
                "direct": fit_direct(
                    layer6,
                    target,
                    song_ids,
                    frame_counts,
                    train_mask,
                    held_mask,
                ),
                "global_local": fit_global_local(
                    layer6,
                    target,
                    song_ids,
                    train_mask,
                    held_mask,
                ),
            }
            for model_name, prediction in predictions.items():
                outputs[model_name][held_mask] = prediction
                metrics = scalar_metrics(
                    prediction, target[held_mask]
                )
                rows.append(
                    {
                        "fold": fold,
                        "clip_type": clip_type,
                        "model": model_name,
                        "train_songs": int(
                            len(np.unique(song_ids[train_mask]))
                        ),
                        "held_songs": int(
                            len(np.unique(song_ids[held_mask]))
                        ),
                        **metrics,
                    }
                )
                print(
                    f"fold={fold} type={clip_type} model={model_name} "
                    f"rmse={metrics['rmse']:.6f} "
                    f"pcc={metrics['pcc']:.6f} "
                    f"ccc={metrics['ccc']:.6f}"
                )
    summary = {
        "dataset": "DEAM original train split only",
        "method": "clip-type-specific layer6 Ridge",
        "alpha": ALPHA,
        "clip_type_rule": "short: 60 frames; long: more than 60",
        "test_used": False,
        "legacy_validation_used": False,
        "folds": rows,
        "metrics": {},
    }
    for model_name, prediction in outputs.items():
        summary["metrics"][model_name] = {}
        for clip_type, mask in (
            ("short", short_mask),
            ("long", ~short_mask),
            ("all", np.ones(len(target), dtype=bool)),
        ):
            summary["metrics"][model_name][clip_type] = scalar_metrics(
                prediction[mask], target[mask]
            )
        np.save(
            ROOT
            / f"outputs/oof_clip_type_{model_name}_valence.npy",
            prediction.astype(np.float32),
        )
    (ROOT / "outputs/clip_type_ridge_five_fold.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
