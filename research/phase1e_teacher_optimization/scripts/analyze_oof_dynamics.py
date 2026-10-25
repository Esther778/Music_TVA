from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_valence_harmonic_context import scalar_metrics


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/five_fold_manifest.csv").sort_values("song_id")
    index = pd.read_csv(INDEX).set_index("song_id")
    targets, songs = [], []
    for row in manifest.itertuples():
        with np.load(index.loc[int(row.song_id), "sequence_path"]) as data:
            target = data["Y"][:, 0].astype(np.float32)
        targets.append(target)
        songs.append(np.full(len(target), int(row.song_id), dtype=np.int32))
    target = np.concatenate(targets)
    song_ids = np.concatenate(songs)
    candidates = [
        "final",
        "mix_25_layer6",
        "mean_layer6_final",
        "mix_75_layer6",
        "layer6",
    ]
    result = {}
    for candidate in candidates:
        prediction = np.load(ROOT / f"outputs/oof_{candidate}_valence.npy")
        song_true_means, song_pred_means = [], []
        centered_true, centered_pred = [], []
        delta_true, delta_pred = [], []
        for song_id in np.unique(song_ids):
            mask = song_ids == song_id
            truth = target[mask]
            pred = prediction[mask]
            song_true_means.append(float(truth.mean()))
            song_pred_means.append(float(pred.mean()))
            centered_true.append(truth - truth.mean())
            centered_pred.append(pred - pred.mean())
            if len(truth) > 1:
                delta_true.append(np.diff(truth))
                delta_pred.append(np.diff(pred))
        result[candidate] = {
            "song_mean": scalar_metrics(
                np.asarray(song_pred_means), np.asarray(song_true_means)
            ),
            "within_song_centered": scalar_metrics(
                np.concatenate(centered_pred), np.concatenate(centered_true)
            ),
            "first_difference": scalar_metrics(
                np.concatenate(delta_pred), np.concatenate(delta_true)
            ),
            "prediction_std": float(prediction.std()),
            "target_std": float(target.std()),
        }
    output = ROOT / "outputs/oof_dynamic_decomposition.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
