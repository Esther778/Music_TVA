from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from train_bilstm_full_song_transfer import metrics
from train_dual_scale_full_song_va import INDEX, PROTOCOL, ROOT, load_song


OUTPUT_DIR = ROOT / "outputs/full_song_global_ridge_cv"
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
FEATURE_SETS = ["hand", "mert_mean", "all_mean_std"]


def song_features(song_id: int, sequence_path: str | Path) -> dict:
    layer6, final, hand, target = load_song(song_id, sequence_path)
    return {
        "hand": np.concatenate([hand.mean(axis=0), hand.std(axis=0)]),
        "mert_mean": np.concatenate(
            [layer6.mean(axis=0), final.mean(axis=0)]
        ),
        "all_mean_std": np.concatenate(
            [
                layer6.mean(axis=0),
                layer6.std(axis=0),
                final.mean(axis=0),
                final.std(axis=0),
                hand.mean(axis=0),
                hand.std(axis=0),
            ]
        ),
        "target_mean": target.mean(axis=0),
        "target": target.astype(np.float32),
    }


def load_data(rows: pd.DataFrame) -> dict[int, dict]:
    data = {}
    for row in rows.drop_duplicates("song_id").sort_values("song_id").itertuples():
        data[int(row.song_id)] = song_features(int(row.song_id), row.sequence_path)
    return data


def ids(rows: pd.DataFrame) -> list[int]:
    return [int(value) for value in rows.sort_values("song_id").song_id]


def normalize(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (held - mean) / std


def predict_constant(
    data: dict[int, dict],
    train_ids: list[int],
    held_ids: list[int],
    feature_set: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray([data[sid][feature_set] for sid in train_ids])
    train_y = np.asarray([data[sid]["target_mean"] for sid in train_ids])
    held_x = np.asarray([data[sid][feature_set] for sid in held_ids])
    norm_train_x, norm_held_x = normalize(train_x, held_x)
    model = Ridge(alpha=alpha)
    model.fit(norm_train_x, train_y)
    held_mean_prediction = model.predict(norm_held_x).astype(np.float32)
    predictions = []
    targets = []
    song_ids = []
    for index, song_id in enumerate(held_ids):
        target = data[song_id]["target"]
        prediction = np.tile(held_mean_prediction[index], (len(target), 1))
        predictions.append(prediction.astype(np.float32))
        targets.append(target)
        song_ids.extend([song_id] * len(target))
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.asarray(song_ids, dtype=np.int32),
    )


def main() -> None:
    protocol = pd.read_csv(PROTOCOL)
    index = pd.read_csv(INDEX)[["song_id", "sequence_path"]]
    protocol = protocol.drop(columns=["sequence_path"]).merge(
        index, on="song_id"
    )
    full_rows = protocol[protocol.domain == "full_song"].copy()
    data = load_data(full_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    all_predictions = []
    all_targets = []
    all_song_ids = []
    for fold in sorted(protocol.outer_fold.unique()):
        part = protocol[protocol.outer_fold == fold].copy()
        train_ids = ids(
            part[(part.role == "train") & (part.domain == "full_song")]
        )
        validation_ids = ids(part[part.role == "validation"])
        test_ids = ids(part[part.role == "test"])
        best = None
        for feature_set, alpha in product(FEATURE_SETS, ALPHAS):
            prediction, target, _ = predict_constant(
                data, train_ids, validation_ids, feature_set, alpha
            )
            result = metrics(prediction, target)
            row = {
                "outer_fold": int(fold),
                "feature_set": feature_set,
                "alpha": alpha,
                **{f"validation_{key}": value for key, value in result.items()},
            }
            if best is None or row["validation_rmse"] < best["validation_rmse"]:
                best = row
        test_prediction, test_target, test_song_ids = predict_constant(
            data,
            train_ids,
            test_ids,
            best["feature_set"],
            best["alpha"],
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
        all_predictions.append(test_prediction)
        all_targets.append(test_target)
        all_song_ids.append(test_song_ids)
        print(
            f"fold={fold} feature={best['feature_set']} "
            f"alpha={best['alpha']} "
            f"val_rmse={best['validation_rmse']:.6f} "
            f"test_rmse={test_result['rmse']:.6f} "
            f"test_v_rmse={test_result['rmse_valence']:.6f} "
            f"test_a_rmse={test_result['rmse_arousal']:.6f}"
        )

    prediction = np.concatenate(all_predictions)
    target = np.concatenate(all_targets)
    song_ids = np.concatenate(all_song_ids)
    overall = metrics(prediction, target)
    summary = {
        "experiment_id": "full_song_global_ridge_nested_cv",
        "test_used_for_hyperparameter_selection": False,
        "training_domain": "complete songs only",
        "model": "Ridge predicts song-level mean VA; frame output is constant within song",
        "feature_sets": FEATURE_SETS,
        "alpha_grid": ALPHAS,
        "selection_metric": "validation mean frame RMSE per outer fold",
        "folds": fold_rows,
        "overall_outer_test": overall,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    np.save(OUTPUT_DIR / "outer_test_pred.npy", prediction.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_target.npy", target.astype(np.float32))
    np.save(OUTPUT_DIR / "outer_test_song_ids.npy", song_ids.astype(np.int32))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
