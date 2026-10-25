from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from train_bilstm_full_song_transfer import metrics
from train_dual_scale_full_song_va import INDEX, PROTOCOL, ROOT, load_song


OUTPUT_DIR = ROOT / "outputs/full_song_global_local_ridge_cv"
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
RESIDUAL_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_full_song_features(rows: pd.DataFrame) -> dict[int, dict]:
    result = {}
    for row in rows.sort_values("song_id").itertuples():
        song_id = int(row.song_id)
        layer6, final, hand, target = load_song(song_id, row.sequence_path)
        frame_features = np.concatenate([layer6, final, hand], axis=1)
        global_features = np.concatenate(
            [
                layer6.mean(axis=0),
                layer6.std(axis=0),
                final.mean(axis=0),
                final.std(axis=0),
                hand.mean(axis=0),
                hand.std(axis=0),
            ]
        ).astype(np.float32)
        result[song_id] = {
            "frame_features": frame_features.astype(np.float32),
            "global_features": global_features,
            "target": target.astype(np.float32),
        }
    return result


def normalize(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (held - mean) / std


def fit_candidate(
    data: dict[int, dict],
    train_ids: list[int],
    held_ids: list[int],
    global_alpha: float,
    local_alpha: float,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_global_x = np.asarray(
        [data[song_id]["global_features"] for song_id in train_ids],
        dtype=np.float32,
    )
    train_global_y = np.asarray(
        [data[song_id]["target"].mean(axis=0) for song_id in train_ids],
        dtype=np.float32,
    )
    held_global_x = np.asarray(
        [data[song_id]["global_features"] for song_id in held_ids],
        dtype=np.float32,
    )
    norm_train_global_x, norm_held_global_x = normalize(
        train_global_x, held_global_x
    )
    global_model = Ridge(alpha=global_alpha, solver="lsqr")
    global_model.fit(norm_train_global_x, train_global_y)
    held_global_prediction = global_model.predict(
        norm_held_global_x
    ).astype(np.float32)

    train_local_x = []
    train_local_y = []
    train_local_weight = []
    for song_id in train_ids:
        features = data[song_id]["frame_features"]
        target = data[song_id]["target"]
        train_local_x.append(features - features.mean(axis=0, keepdims=True))
        train_local_y.append(target - target.mean(axis=0, keepdims=True))
        train_local_weight.append(
            np.full(len(target), 1.0 / len(target), dtype=np.float64)
        )
    train_local_x = np.concatenate(train_local_x).astype(np.float32)
    train_local_y = np.concatenate(train_local_y).astype(np.float32)
    train_local_weight = np.concatenate(train_local_weight)
    train_local_weight *= len(train_local_weight) / train_local_weight.sum()
    local_mean = train_local_x.mean(axis=0, keepdims=True)
    local_std = train_local_x.std(axis=0, keepdims=True) + 1e-6
    local_model = Ridge(alpha=local_alpha, solver="lsqr")
    local_model.fit(
        (train_local_x - local_mean) / local_std,
        train_local_y,
        sample_weight=train_local_weight,
    )

    predictions = []
    targets = []
    song_ids = []
    for index, song_id in enumerate(held_ids):
        features = data[song_id]["frame_features"]
        target = data[song_id]["target"]
        centered_features = features - features.mean(axis=0, keepdims=True)
        local_prediction = local_model.predict(
            (centered_features - local_mean) / local_std
        ).astype(np.float32)
        prediction = (
            held_global_prediction[index][None, :]
            + residual_scale * local_prediction
        )
        predictions.append(prediction.astype(np.float32))
        targets.append(target)
        song_ids.extend([song_id] * len(target))
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.asarray(song_ids, dtype=np.int32),
    )


def ids(rows: pd.DataFrame) -> list[int]:
    return [int(value) for value in rows.sort_values("song_id").song_id]


def main() -> None:
    protocol = pd.read_csv(PROTOCOL)
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    protocol = protocol.drop(columns=["sequence_path"]).merge(
        index, on="song_id"
    )
    full_rows = protocol[protocol.domain == "full_song"].copy()
    data = load_full_song_features(
        full_rows.drop_duplicates("song_id").copy()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    all_test_predictions = []
    all_test_targets = []
    all_test_song_ids = []
    for fold in sorted(protocol.outer_fold.unique()):
        part = protocol[protocol.outer_fold == fold].copy()
        train_ids = ids(
            part[(part.role == "train") & (part.domain == "full_song")]
        )
        validation_ids = ids(part[part.role == "validation"])
        test_ids = ids(part[part.role == "test"])
        best = None
        for global_alpha, local_alpha, residual_scale in product(
            ALPHAS, ALPHAS, RESIDUAL_SCALES
        ):
            prediction, target, _ = fit_candidate(
                data,
                train_ids,
                validation_ids,
                global_alpha,
                local_alpha,
                residual_scale,
            )
            result = metrics(prediction, target)
            score = result["rmse"]
            row = {
                "outer_fold": int(fold),
                "global_alpha": global_alpha,
                "local_alpha": local_alpha,
                "residual_scale": residual_scale,
                "validation_score": score,
                **{f"validation_{key}": value for key, value in result.items()},
            }
            if best is None or score < best["validation_score"]:
                best = row
        test_prediction, test_target, test_song_ids = fit_candidate(
            data,
            train_ids,
            test_ids,
            best["global_alpha"],
            best["local_alpha"],
            best["residual_scale"],
        )
        test_result = metrics(test_prediction, test_target)
        fold_row = {
            **best,
            "train_full_songs": len(train_ids),
            "validation_full_songs": len(validation_ids),
            "test_full_songs": len(test_ids),
            "test_song_ids": test_ids,
            **{f"test_{key}": value for key, value in test_result.items()},
        }
        fold_rows.append(fold_row)
        all_test_predictions.append(test_prediction)
        all_test_targets.append(test_target)
        all_test_song_ids.append(test_song_ids)
        print(
            f"fold={fold} val_rmse={best['validation_rmse']:.6f} "
            f"test_rmse={test_result['rmse']:.6f} "
            f"test_v_rmse={test_result['rmse_valence']:.6f} "
            f"test_a_rmse={test_result['rmse_arousal']:.6f} "
            f"scale={best['residual_scale']}"
        )

    test_prediction = np.concatenate(all_test_predictions)
    test_target = np.concatenate(all_test_targets)
    test_song_ids = np.concatenate(all_test_song_ids)
    overall = metrics(test_prediction, test_target)
    summary = {
        "experiment_id": "full_song_global_local_ridge_nested_cv",
        "test_used_for_hyperparameter_selection": False,
        "training_domain": "complete songs only",
        "model": (
            "Ridge song-level global baseline plus Ridge within-song residual"
        ),
        "alpha_grid": ALPHAS,
        "residual_scale_grid": RESIDUAL_SCALES,
        "selection_metric": "validation mean frame RMSE per outer fold",
        "folds": fold_rows,
        "overall_outer_test": overall,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    np.save(OUTPUT_DIR / "outer_test_pred.npy", test_prediction.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_target.npy", test_target.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_song_ids.npy", test_song_ids.astype(np.int32))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
