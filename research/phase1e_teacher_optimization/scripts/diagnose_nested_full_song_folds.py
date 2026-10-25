from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_bilstm_full_song_transfer import INDEX, PROTOCOL, metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/nested_full_song_fold_diagnostics"


def load_targets(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    song_ids = []
    for row in rows.sort_values("song_id").itertuples():
        with np.load(row.sequence_path) as data:
            y = data["Y"].astype(np.float32)
        targets.append(y)
        song_ids.extend([int(row.song_id)] * len(y))
    return np.concatenate(targets), np.asarray(song_ids, dtype=np.int32)


def split_stats(name: str, rows: pd.DataFrame) -> dict:
    target, song_ids = load_targets(rows)
    song_means = []
    for song_id in np.unique(song_ids):
        song_means.append(target[song_ids == song_id].mean(axis=0))
    song_means = np.asarray(song_means)
    return {
        "name": name,
        "songs": int(len(rows)),
        "frames": int(len(target)),
        "song_ids": [int(value) for value in sorted(rows.song_id)],
        "frame_mean_valence": float(target[:, 0].mean()),
        "frame_mean_arousal": float(target[:, 1].mean()),
        "frame_std_valence": float(target[:, 0].std()),
        "frame_std_arousal": float(target[:, 1].std()),
        "frame_min_valence": float(target[:, 0].min()),
        "frame_max_valence": float(target[:, 0].max()),
        "frame_min_arousal": float(target[:, 1].min()),
        "frame_max_arousal": float(target[:, 1].max()),
        "song_mean_valence_mean": float(song_means[:, 0].mean()),
        "song_mean_valence_std": float(song_means[:, 0].std()),
        "song_mean_arousal_mean": float(song_means[:, 1].mean()),
        "song_mean_arousal_std": float(song_means[:, 1].std()),
    }


def mean_baseline(train_rows: pd.DataFrame, held_rows: pd.DataFrame) -> dict:
    train_target, _ = load_targets(train_rows)
    held_target, _ = load_targets(held_rows)
    mean = train_target.mean(axis=0)
    prediction = np.tile(mean, (len(held_target), 1)).astype(np.float32)
    return {
        "train_mean_valence": float(mean[0]),
        "train_mean_arousal": float(mean[1]),
        **metrics(prediction, held_target),
    }


def main() -> None:
    protocol = pd.read_csv(PROTOCOL)
    index = pd.read_csv(INDEX)[["song_id", "frames", "sequence_path"]]
    protocol = protocol.drop(columns=["sequence_path"]).merge(
        index, on="song_id"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = []
    for fold in sorted(protocol.outer_fold.unique()):
        part = protocol[protocol.outer_fold == fold].copy()
        train_all = part[part.role == "train"].copy()
        train_full = part[
            (part.role == "train") & (part.domain == "full_song")
        ].copy()
        validation = part[part.role == "validation"].copy()
        test = part[part.role == "test"].copy()
        stats = {
            "outer_fold": int(fold),
            "train_all": split_stats("train_all", train_all),
            "train_full": split_stats("train_full", train_full),
            "validation": split_stats("validation", validation),
            "test": split_stats("test", test),
            "mean_baseline_validation": mean_baseline(
                train_all, validation
            ),
            "mean_baseline_test": mean_baseline(train_all, test),
        }
        for split_name in ("validation", "test"):
            split = stats[split_name]
            rows.append(
                {
                    "outer_fold": int(fold),
                    "split": split_name,
                    "songs": split["songs"],
                    "frames": split["frames"],
                    "frame_mean_valence": split["frame_mean_valence"],
                    "frame_mean_arousal": split["frame_mean_arousal"],
                    "frame_std_valence": split["frame_std_valence"],
                    "frame_std_arousal": split["frame_std_arousal"],
                    "song_mean_valence_mean": split[
                        "song_mean_valence_mean"
                    ],
                    "song_mean_arousal_mean": split[
                        "song_mean_arousal_mean"
                    ],
                    "train_all_mean_valence": stats["train_all"][
                        "frame_mean_valence"
                    ],
                    "train_all_mean_arousal": stats["train_all"][
                        "frame_mean_arousal"
                    ],
                    "delta_valence_vs_train": split[
                        "frame_mean_valence"
                    ]
                    - stats["train_all"]["frame_mean_valence"],
                    "delta_arousal_vs_train": split[
                        "frame_mean_arousal"
                    ]
                    - stats["train_all"]["frame_mean_arousal"],
                    **{
                        f"mean_baseline_{key}": value
                        for key, value in stats[
                            f"mean_baseline_{split_name}"
                        ].items()
                    },
                }
            )
        summaries.append(stats)

    summary = {
        "diagnostic_id": "nested_full_song_fold_distribution_and_baseline",
        "test_used_for_training_or_selection": False,
        "folds": summaries,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "fold_split_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"saved={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
