#!/usr/bin/env python3
"""Measure same-rater stability between immediate and delayed anchored reviews."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
ANNOTATION_DIR = PHASE_ROOT / "data/blind_validation/stage5/human_annotations"
OUTPUT_ROOT = PHASE_ROOT / "outputs/human_validation"
INITIAL_CSV = ANNOTATION_DIR / "blind_sample_15_human_annotations_anchored_v1_rater_01.csv"
DELAYED_CSV = (
    ANNOTATION_DIR
    / "blind_sample_15_human_annotations_anchored_v1_delayed_review_rater_01.csv"
)
INITIAL_SUMMARY = OUTPUT_ROOT / "rater_01_anchored_v1_evaluation/summary.json"
DELAYED_DIR = OUTPUT_ROOT / "rater_01_anchored_v1_delayed_review_evaluation"
DELAYED_SUMMARY = DELAYED_DIR / "summary.json"
OUTPUT_PATH = DELAYED_DIR / "initial_vs_delayed_review.json"


LEVEL_MAP = {"low": 0, "mid": 1, "high": 2}
CHANGE_MAP = {"down": -1, "stable": 0, "up": 1, "not_applicable": 9}


METRICS = [
    ("within_song_macro", "valence_spearman_mean"),
    ("within_song_macro", "arousal_spearman_mean"),
    ("section_pooled", "valence_spearman"),
    ("section_pooled", "arousal_spearman"),
    ("section_pooled", "within_song_centered_valence_spearman"),
    ("section_pooled", "within_song_centered_arousal_spearman"),
    ("between_song", "valence_mean_spearman"),
    ("between_song", "arousal_mean_spearman"),
    ("change_direction", "nonstable_sign_accuracy"),
]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def agreement(
    joined: pd.DataFrame,
    field: str,
    mapping: dict[str, int] | None,
    weights: str | None,
) -> dict[str, float | int]:
    initial = joined[f"{field}_initial"]
    delayed = joined[f"{field}_delayed"]
    initial_numeric = initial.map(mapping) if mapping else initial
    delayed_numeric = delayed.map(mapping) if mapping else delayed
    return {
        "rows": int(len(joined)),
        "exact_agreement": float(initial.eq(delayed).mean()),
        "changed_rows": int(initial.ne(delayed).sum()),
        "cohen_kappa": float(
            cohen_kappa_score(initial_numeric, delayed_numeric, weights=weights)
        ),
    }


def main() -> None:
    initial = pd.read_csv(INITIAL_CSV)
    delayed = pd.read_csv(DELAYED_CSV)
    fields = [
        "human_valence_level",
        "human_arousal_level",
        "human_change_from_prev",
        "is_climax",
    ]
    joined = initial[["human_annotation_id", *fields]].merge(
        delayed[["human_annotation_id", *fields]],
        on="human_annotation_id",
        suffixes=("_initial", "_delayed"),
        validate="one_to_one",
    )
    if len(joined) != 179:
        raise RuntimeError(f"Expected 179 paired rows, found {len(joined)}")

    initial_metrics = load_metrics(INITIAL_SUMMARY)
    delayed_metrics = load_metrics(DELAYED_SUMMARY)
    metric_deltas = {}
    for group, metric in METRICS:
        before = float(initial_metrics[group][metric])
        after = float(delayed_metrics[group][metric])
        metric_deltas[f"{group}.{metric}"] = {
            "initial": before,
            "delayed": after,
            "delta": after - before,
        }

    payload = {
        "status": "same_rater_delayed_review_stability",
        "primary_result_rule": (
            "Delayed review is primary because reduced continuous-listening fatigue was "
            "declared before recomputing model agreement; the initial anchored review is "
            "retained as sensitivity evidence."
        ),
        "warning": "Same-rater stability is not inter-rater reliability.",
        "changed_any_field_rows": int(
            pd.concat(
                [
                    joined[f"{field}_initial"].ne(joined[f"{field}_delayed"])
                    for field in fields
                ],
                axis=1,
            ).any(axis=1).sum()
        ),
        "agreement": {
            "valence": agreement(
                joined, "human_valence_level", LEVEL_MAP, "quadratic"
            ),
            "arousal": agreement(
                joined, "human_arousal_level", LEVEL_MAP, "quadratic"
            ),
            "change": agreement(
                joined, "human_change_from_prev", CHANGE_MAP, None
            ),
            "local_peak": agreement(joined, "is_climax", None, None),
        },
        "metric_deltas": metric_deltas,
        "initial_level_counts": initial_metrics["human_level_counts"],
        "delayed_level_counts": delayed_metrics["human_level_counts"],
        "initial_chorus_minus_verse": initial_metrics["chorus_minus_verse"],
        "delayed_chorus_minus_verse": delayed_metrics["chorus_minus_verse"],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
