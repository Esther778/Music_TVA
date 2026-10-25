#!/usr/bin/env python3
"""Freeze the final same-rater 30-reference human-validation result."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
ANNOTATION_DIR = PHASE_ROOT / "data/blind_validation/stage5/human_annotations"
EVALUATION_DIR = (
    PHASE_ROOT
    / "outputs/human_validation/rater_01_anchored_v2_30refs_review_evaluation"
)

ANNOTATION_CSV = (
    ANNOTATION_DIR
    / "blind_sample_15_human_annotations_anchored_v2_30refs_review_rater_01.csv"
)
BASELINE_CSV = (
    ANNOTATION_DIR
    / "blind_sample_15_human_annotations_anchored_v1_delayed_review_rater_01.csv"
)
ANCHORS_JSON = (
    PHASE_ROOT
    / "outputs/human_validation/calibration_candidates_v2_supplement/"
    "frozen_va_anchors_v2_30clips.json"
)
SUMMARY_JSON = EVALUATION_DIR / "summary.json"
COMPARISON_JSON = EVALUATION_DIR / "comparison_vs_delayed_v1.json"
PER_SONG_CSV = EVALUATION_DIR / "per_song_metrics.csv"
REPORT_MD = PHASE_ROOT / "docs/BLIND_VALIDATION_HUMAN_RESULTS_30REF_REVIEW_20260726.md"
STAGE3_MANIFEST = (
    PHASE_ROOT / "data/blind_validation/stage3/stage3_inference_manifest.json"
)
STAGE4_MANIFEST = PHASE_ROOT / "data/blind_validation/stage4/stage4_manifest.json"
STAGE5_MANIFEST = (
    PHASE_ROOT / "data/blind_validation/stage5/stage5_analysis_manifest.json"
)
ANALYSIS_SCRIPT = PHASE_ROOT / "scripts/analyze_blind_human_validation.py"
COMPARISON_SCRIPT = PHASE_ROOT / "scripts/compare_delayed_and_30ref_review.py"
FREEZE_SCRIPT = PHASE_ROOT / "scripts/freeze_human_validation_30refs.py"
OUTPUT_MANIFEST = (
    ANNOTATION_DIR / "frozen_human_validation_30refs_rater01_20260726.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_annotations() -> dict:
    with ANNOTATION_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sample_index",
        "song_id",
        "human_annotation_id",
        "section_order",
        "start_time_sec",
        "end_time_sec",
        "human_valence_level",
        "human_arousal_level",
        "human_change_from_prev",
        "is_climax",
        "rater_id",
    }
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise RuntimeError(f"Missing annotation columns: {missing_columns}")
    if len(rows) != 179:
        raise RuntimeError(f"Expected 179 rows, found {len(rows)}")
    if len({row["human_annotation_id"] for row in rows}) != 179:
        raise RuntimeError("Annotation IDs are not unique")
    if len({row["song_id"] for row in rows}) != 15:
        raise RuntimeError("Expected 15 unique songs")
    allowed = {
        "human_valence_level": {"low", "mid", "high"},
        "human_arousal_level": {"low", "mid", "high"},
        "human_change_from_prev": {"up", "down", "stable", "not_applicable"},
        "is_climax": {"0", "1"},
    }
    for field, values in allowed.items():
        invalid = sorted({row[field] for row in rows} - values)
        if invalid:
            raise RuntimeError(f"Invalid {field}: {invalid}")
    for row in rows:
        first = int(row["section_order"]) == 1
        is_na = row["human_change_from_prev"] == "not_applicable"
        if first != is_na:
            raise RuntimeError(
                f"Invalid first-section change: {row['human_annotation_id']}"
            )
    return {
        "rows": len(rows),
        "songs": len({row["song_id"] for row in rows}),
        "unique_annotation_ids": len(
            {row["human_annotation_id"] for row in rows}
        ),
        "rater_ids": sorted({row["rater_id"] for row in rows}),
        "valence_counts": dict(Counter(row["human_valence_level"] for row in rows)),
        "arousal_counts": dict(Counter(row["human_arousal_level"] for row in rows)),
        "change_counts": dict(
            Counter(row["human_change_from_prev"] for row in rows)
        ),
        "local_peak_count": sum(row["is_climax"] == "1" for row in rows),
        "status": "pass",
    }


def main() -> None:
    files = {
        "annotation_csv": ANNOTATION_CSV,
        "baseline_sensitivity_csv": BASELINE_CSV,
        "anchors_json": ANCHORS_JSON,
        "evaluation_summary": SUMMARY_JSON,
        "comparison_vs_delayed_v1": COMPARISON_JSON,
        "per_song_metrics": PER_SONG_CSV,
        "human_results_report": REPORT_MD,
        "stage3_inference_manifest": STAGE3_MANIFEST,
        "stage4_alignment_manifest": STAGE4_MANIFEST,
        "stage5_structure_manifest": STAGE5_MANIFEST,
        "analysis_script": ANALYSIS_SCRIPT,
        "comparison_script": COMPARISON_SCRIPT,
        "freeze_script": FREEZE_SCRIPT,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing freeze inputs: {missing}")

    annotation_audit = audit_annotations()
    anchors = load_json(ANCHORS_JSON)
    if anchors.get("total_anchor_count") != 30:
        raise RuntimeError("Expected 30 frozen anchors")
    if any(
        count != 5 for count in anchors["anchors_per_dimension_level"].values()
    ):
        raise RuntimeError("Expected five anchors per dimension-level group")
    if anchors.get("teacher_predictions_used") is not False:
        raise RuntimeError("Anchor library must remain model-free")

    summary = load_json(SUMMARY_JSON)
    if summary["audit"].get("status") != "pass":
        raise RuntimeError("Evaluation audit did not pass")
    if summary["audit"].get("rows") != 179:
        raise RuntimeError("Evaluation summary row count mismatch")
    comparison = load_json(COMPARISON_JSON)
    if comparison.get("status") != "same_rater_30_reference_review_comparison":
        raise RuntimeError("Unexpected comparison status")

    stage3 = load_json(STAGE3_MANIFEST)
    stage4 = load_json(STAGE4_MANIFEST)
    stage5 = load_json(STAGE5_MANIFEST)
    if stage4.get("status") != "pass" or stage5.get("status") != "pass":
        raise RuntimeError("Stage 4 or Stage 5 manifest did not pass")

    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_id": "human_validation_30refs_rater01_20260726",
        "status": "frozen_calibrated_same_rater_review",
        "paper_role": (
            "primary ordinal human-validation result; anchored-v1 delayed review "
            "retained as protocol-sensitivity evidence"
        ),
        "scope": "15 human-created Chinese popular songs; 179 frozen sections",
        "scientific_guardrails": {
            "teacher_frozen": True,
            "model_predictions_hidden_during_annotation": True,
            "anchor_teacher_predictions_used": False,
            "not_inter_rater_reliability": True,
            "ordinal_labels_not_continuous_va_ground_truth": True,
            "no_teacher_tuning_from_heldout_results": True,
        },
        "annotation_audit": annotation_audit,
        "anchor_audit": {
            "version": anchors["version"],
            "clips": anchors["total_anchor_count"],
            "unique_songs": anchors["unique_song_count"],
            "groups": anchors["anchors_per_dimension_level"],
            "blind_test_overlap_count": anchors["blind_test_overlap_count"],
        },
        "teacher": {
            "model_ids": stage3["model_ids"],
            "teacher_hashes": stage3["teacher_hashes"],
            "feature_backbone": stage3["feature_backbone"],
            "frame_hop_sec": stage3["frame_hop_sec"],
            "window_sec": stage3["teacher_window_sec"],
            "stride_sec": stage3["teacher_stride_sec"],
            "window_blend": stage3["teacher_window_blend"],
        },
        "pipeline_audit": {
            "stage3_songs": stage3["songs"],
            "stage3_frames": stage3["frames"],
            "stage4_status": stage4["status"],
            "stage4_unassigned_frames": stage4["unassigned_frames"],
            "stage4_overlap_frames": stage4["overlap_frames"],
            "stage5_status": stage5["status"],
            "stage5_statistical_unit": stage5["statistical_unit"],
        },
        "headline_metrics": {
            "pooled_valence_spearman": summary["metrics"]["section_pooled"][
                "valence_spearman"
            ],
            "pooled_arousal_spearman": summary["metrics"]["section_pooled"][
                "arousal_spearman"
            ],
            "within_song_macro_valence_spearman": summary["metrics"][
                "within_song_macro"
            ]["valence_spearman_mean"],
            "within_song_macro_arousal_spearman": summary["metrics"][
                "within_song_macro"
            ]["arousal_spearman_mean"],
            "between_song_valence_spearman": summary["metrics"]["between_song"][
                "valence_mean_spearman"
            ],
            "between_song_arousal_spearman": summary["metrics"]["between_song"][
                "arousal_mean_spearman"
            ],
            "change_sign_accuracy": summary["metrics"]["change_direction"][
                "nonstable_sign_accuracy"
            ],
            "top_arousal_exact_peak_hits": summary["metrics"][
                "frozen_top_arousal_candidate"
            ]["exact_human_peak_hits"],
            "top_arousal_adjacent_peak_hits": summary["metrics"][
                "frozen_top_arousal_candidate"
            ]["adjacent_human_peak_hits"],
        },
        "files": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
    }
    OUTPUT_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
