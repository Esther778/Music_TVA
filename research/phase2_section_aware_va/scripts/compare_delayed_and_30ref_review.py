#!/usr/bin/env python3
"""Compare the delayed anchored-v1 review with the 30-reference review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
ANNOTATION_DIR = PHASE_ROOT / "data/blind_validation/stage5/human_annotations"
OUTPUT_ROOT = PHASE_ROOT / "outputs/human_validation"
BASELINE_CSV = (
    ANNOTATION_DIR
    / "blind_sample_15_human_annotations_anchored_v1_delayed_review_rater_01.csv"
)
CURRENT_CSV = (
    ANNOTATION_DIR
    / "blind_sample_15_human_annotations_anchored_v2_30refs_review_rater_01.csv"
)
BASELINE_SUMMARY = (
    OUTPUT_ROOT / "rater_01_anchored_v1_delayed_review_evaluation/summary.json"
)
CURRENT_DIR = OUTPUT_ROOT / "rater_01_anchored_v2_30refs_review_evaluation"
CURRENT_SUMMARY = CURRENT_DIR / "summary.json"
OUTPUT_PATH = CURRENT_DIR / "comparison_vs_delayed_v1.json"

LEVEL_MAP = {"low": 0, "mid": 1, "high": 2}
CHANGE_MAP = {"down": -1, "stable": 0, "up": 1, "not_applicable": 9}
FIELDS = [
    "human_valence_level",
    "human_arousal_level",
    "human_change_from_prev",
    "is_climax",
]
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


def agreement(
    joined: pd.DataFrame,
    field: str,
    mapping: dict[str, int] | None,
    weights: str | None,
) -> dict[str, float | int]:
    baseline = joined[f"{field}_baseline"]
    current = joined[f"{field}_current"]
    baseline_numeric = baseline.map(mapping) if mapping else baseline
    current_numeric = current.map(mapping) if mapping else current
    return {
        "rows": int(len(joined)),
        "exact_agreement": float(baseline.eq(current).mean()),
        "changed_rows": int(baseline.ne(current).sum()),
        "cohen_kappa": float(
            cohen_kappa_score(baseline_numeric, current_numeric, weights=weights)
        ),
    }


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV)
    current = pd.read_csv(CURRENT_CSV)
    joined = baseline[["human_annotation_id", *FIELDS]].merge(
        current[["human_annotation_id", *FIELDS]],
        on="human_annotation_id",
        suffixes=("_baseline", "_current"),
        validate="one_to_one",
    )
    if len(joined) != 179:
        raise RuntimeError(f"Expected 179 paired rows, found {len(joined)}")

    baseline_metrics = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))[
        "metrics"
    ]
    current_metrics = json.loads(CURRENT_SUMMARY.read_text(encoding="utf-8"))[
        "metrics"
    ]
    metric_deltas = {}
    for group, metric in METRICS:
        before = float(baseline_metrics[group][metric])
        after = float(current_metrics[group][metric])
        metric_deltas[f"{group}.{metric}"] = {
            "baseline": before,
            "current": after,
            "delta": after - before,
        }

    changed_masks = [
        joined[f"{field}_baseline"].ne(joined[f"{field}_current"])
        for field in FIELDS
    ]
    payload = {
        "status": "same_rater_30_reference_review_comparison",
        "baseline_csv": str(BASELINE_CSV),
        "current_csv": str(CURRENT_CSV),
        "changed_any_field_rows": int(pd.concat(changed_masks, axis=1).any(axis=1).sum()),
        "agreement": {
            "valence": agreement(joined, "human_valence_level", LEVEL_MAP, "quadratic"),
            "arousal": agreement(joined, "human_arousal_level", LEVEL_MAP, "quadratic"),
            "change": agreement(joined, "human_change_from_prev", CHANGE_MAP, None),
            "local_peak": agreement(joined, "is_climax", None, None),
        },
        "metric_deltas": metric_deltas,
        "warning": (
            "Same-rater calibrated review is sensitivity evidence, not inter-rater "
            "reliability."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
