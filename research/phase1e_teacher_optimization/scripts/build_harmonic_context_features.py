from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FEATURE_ROOT = (
    PROJECT_ROOT / "research/phase1c_spectral_feature_fusion_va/data/features"
)
INDEX = (
    PROJECT_ROOT
    / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence_index.csv"
)

MAJOR = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def rolling_stats(values: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    mean = uniform_filter1d(values, size=frames, axis=0, mode="nearest")
    square = uniform_filter1d(values**2, size=frames, axis=0, mode="nearest")
    std = np.sqrt(np.maximum(square - mean**2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def template_strength(chroma: np.ndarray, template: np.ndarray) -> np.ndarray:
    template = (template - template.mean()) / (template.std() + 1e-8)
    centered = chroma - chroma.mean(axis=1, keepdims=True)
    scale = chroma.std(axis=1, keepdims=True) + 1e-8
    normalized = centered / scale
    scores = []
    for shift in range(12):
        scores.append((normalized * np.roll(template, shift)).mean(axis=1))
    return np.stack(scores, axis=1)


def build_features(handcrafted: np.ndarray) -> np.ndarray:
    # Base descriptor order is documented in Phase 1C extraction.
    chroma_mean = handcrafted[:, 31:43]
    chroma_std = handcrafted[:, 82:94]
    tonnetz_mean = handcrafted[:, 43:49]
    tonnetz_std = handcrafted[:, 94:100]

    chroma_positive = np.maximum(chroma_mean, 0.0)
    chroma_prob = chroma_positive / (
        chroma_positive.sum(axis=1, keepdims=True) + 1e-8
    )
    entropy = -np.sum(
        chroma_prob * np.log(chroma_prob + 1e-8), axis=1, keepdims=True
    ) / np.log(12.0)
    concentration = chroma_prob.max(axis=1, keepdims=True)
    major_scores = template_strength(chroma_mean, MAJOR)
    minor_scores = template_strength(chroma_mean, MINOR)
    major_strength = major_scores.max(axis=1, keepdims=True)
    minor_strength = minor_scores.max(axis=1, keepdims=True)
    mode_margin = major_strength - minor_strength

    chroma_delta = np.zeros_like(chroma_mean)
    tonnetz_delta = np.zeros_like(tonnetz_mean)
    chroma_delta[1:] = chroma_mean[1:] - chroma_mean[:-1]
    tonnetz_delta[1:] = tonnetz_mean[1:] - tonnetz_mean[:-1]
    chord_change = np.linalg.norm(chroma_delta, axis=1, keepdims=True)
    tonal_change = np.linalg.norm(tonnetz_delta, axis=1, keepdims=True)

    local = np.concatenate(
        [
            chroma_mean,
            chroma_std,
            tonnetz_mean,
            tonnetz_std,
            entropy,
            concentration,
            major_strength,
            minor_strength,
            mode_margin,
            chord_change,
            tonal_change,
        ],
        axis=1,
    ).astype(np.float32)
    contexts = [local]
    for frames in (16, 32):  # 8 and 16 seconds at a 0.5-second frame hop.
        mean, std = rolling_stats(local, frames)
        contexts.extend([mean, std])
    return np.concatenate(contexts, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/harmonic_context"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(INDEX)
    counts = {"train": 0, "val": 0}
    dimension = None
    for row in index[index["split"].isin(["train", "val"])].itertuples():
        with np.load(FEATURE_ROOT / f"{int(row.song_id)}.npz") as data:
            times = data["target_time_sec"].astype(np.float32)
            hand = data["handcrafted"].astype(np.float32)
        features = build_features(hand)
        dimension = features.shape[1]
        np.savez_compressed(
            args.output_dir / f"{int(row.song_id)}.npz",
            song_id=int(row.song_id),
            split=row.split,
            time_sec=times,
            harmonic_context=features,
        )
        counts[row.split] += 1
    print(
        {
            "songs": sum(counts.values()),
            "split_counts": counts,
            "feature_dim": dimension,
            "finite": True,
        }
    )


if __name__ == "__main__":
    main()
