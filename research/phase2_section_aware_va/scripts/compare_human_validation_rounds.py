#!/usr/bin/env python3
"""Compare the pilot and anchored human-validation result summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va/outputs/human_validation"
PILOT_DIR = OUTPUT_ROOT / "rater_01_evaluation"
ANCHORED_DIR = OUTPUT_ROOT / "rater_01_anchored_v1_evaluation"
OUTPUT_PATH = ANCHORED_DIR / "pilot_vs_anchored_comparison.json"


METRICS = [
    ("within_song_macro", "valence_spearman_mean"),
    ("within_song_macro", "arousal_spearman_mean"),
    ("within_song_macro", "change_spearman_mean"),
    ("section_pooled", "valence_spearman"),
    ("section_pooled", "arousal_spearman"),
    ("section_pooled", "within_song_centered_valence_spearman"),
    ("section_pooled", "within_song_centered_arousal_spearman"),
    ("section_pooled", "change_spearman_vs_arousal_delta"),
    ("between_song", "valence_mean_spearman"),
    ("between_song", "arousal_mean_spearman"),
    ("change_direction", "nonstable_sign_accuracy"),
]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def paired_change(
    joined: pd.DataFrame, column: str, seed: int = 20260723
) -> dict[str, object]:
    values = (
        joined[f"{column}_anchored"] - joined[f"{column}_pilot"]
    ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return {
        "paired_songs": int(len(values)),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_p": float(wilcoxon(values).pvalue),
    }


def main() -> None:
    pilot = load_metrics(PILOT_DIR / "summary.json")
    anchored = load_metrics(ANCHORED_DIR / "summary.json")
    metric_deltas = {}
    for group, metric in METRICS:
        pilot_value = float(pilot[group][metric])
        anchored_value = float(anchored[group][metric])
        metric_deltas[f"{group}.{metric}"] = {
            "pilot": pilot_value,
            "anchored": anchored_value,
            "delta": anchored_value - pilot_value,
        }

    pilot_song = pd.read_csv(PILOT_DIR / "per_song_metrics.csv")
    anchored_song = pd.read_csv(ANCHORED_DIR / "per_song_metrics.csv")
    joined = pilot_song.merge(
        anchored_song, on="song_id", suffixes=("_pilot", "_anchored"), validate="one_to_one"
    )
    paired = {
        column: paired_change(joined, column)
        for column in (
            "valence_spearman",
            "arousal_spearman",
            "change_spearman_vs_arousal_delta",
        )
    }

    payload = {
        "status": "pilot_vs_anchored_descriptive_comparison",
        "warning": (
            "The same rater completed both rounds; this is a protocol sensitivity "
            "comparison, not an independent replication."
        ),
        "metric_deltas": metric_deltas,
        "paired_song_changes": paired,
        "human_level_counts": {
            "pilot": pilot["human_level_counts"],
            "anchored": anchored["human_level_counts"],
        },
        "human_peaks": {
            "pilot": pilot["counts"]["human_peaks"],
            "anchored": anchored["counts"]["human_peaks"],
        },
        "frozen_top_arousal_candidate": {
            "pilot": pilot["frozen_top_arousal_candidate"],
            "anchored": anchored["frozen_top_arousal_candidate"],
        },
        "chorus_minus_verse": {
            "pilot": pilot["chorus_minus_verse"],
            "anchored": anchored["chorus_minus_verse"],
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
