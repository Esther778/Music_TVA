from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"
FEATURE_ROOT = ROOT / "data" / "mert_multilayer"
PILOT_MANIFEST = ROOT / "data" / "multilayer_pilot_manifest.csv"


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pcc = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = 2 * covariance / np.maximum(
        prediction.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mean_pcc": float(np.mean(pcc)),
        "pcc_valence": float(pcc[0]),
        "pcc_arousal": float(pcc[1]),
        "mean_ccc": float(np.mean(ccc)),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
    }


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pilot = pd.read_csv(PILOT_MANIFEST)
    sequence_index = pd.read_csv(
        MERT_ROOT / "data" / "datasets" / "mert_va" / "sequence_index.csv"
    ).set_index("song_id")
    features = []
    targets = []
    layer_ids = None
    for song_id in (
        pilot[pilot["split"] == split].sort_values("song_id")["song_id"]
    ):
        song_id = int(song_id)
        with np.load(FEATURE_ROOT / f"{song_id}.npz") as multi:
            times = multi["time_sec"].astype(np.float32)
            embeddings = multi["embeddings"].astype(np.float32)
            layer_ids = multi["layers"].astype(int)
        sequence_path = Path(sequence_index.loc[song_id, "sequence_path"])
        with np.load(sequence_path) as sequence:
            target_times = sequence["time_sec"].astype(np.float32)
            target = sequence["Y"].astype(np.float32)
        indices = []
        for time_sec in target_times:
            match = np.where(np.isclose(times, time_sec, atol=1e-4))[0]
            if len(match) != 1:
                raise ValueError(
                    f"Could not align song {song_id} time {time_sec}."
                )
            indices.append(int(match[0]))
        features.append(embeddings[indices])
        targets.append(target)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(targets, axis=0),
        layer_ids,
    )


def main() -> None:
    train_features, train_target, layer_ids = load_split("train")
    val_features, val_target, val_layer_ids = load_split("val")
    if not np.array_equal(layer_ids, val_layer_ids):
        raise ValueError("Train and validation layer IDs differ.")
    candidates = []
    for layer_index, layer_id in enumerate(layer_ids):
        candidates.append((f"layer_{layer_id}", train_features[:, layer_index], val_features[:, layer_index]))
    candidates.append(
        (
            "mean_layers_3_6_9_12",
            train_features.mean(axis=1),
            val_features.mean(axis=1),
        )
    )

    rows = []
    for candidate_name, train_x, val_x in candidates:
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True) + 1e-6
        train_x = (train_x - mean) / std
        val_x = (val_x - mean) / std
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(train_x, train_target)
            prediction = model.predict(val_x)
            rows.append(
                {
                    "candidate": candidate_name,
                    "alpha": alpha,
                    **metrics(prediction, val_target),
                }
            )
    best_by_candidate = []
    for candidate_name, *_ in candidates:
        candidate_rows = [
            row for row in rows if row["candidate"] == candidate_name
        ]
        best_by_candidate.append(
            min(candidate_rows, key=lambda row: row["rmse"])
        )
    result = {
        "purpose": "pilot screening only",
        "train_songs": 100,
        "val_songs": 60,
        "test_used": False,
        "best_by_candidate": sorted(
            best_by_candidate, key=lambda row: row["rmse"]
        ),
        "all_trials": rows,
    }
    output = ROOT / "outputs" / "multilayer_pilot_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result["best_by_candidate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
