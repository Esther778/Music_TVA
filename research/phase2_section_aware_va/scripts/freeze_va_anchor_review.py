#!/usr/bin/env python3
"""Validate a completed anchor review and freeze the approved VA scale anchors."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
CANDIDATE_DIR = PHASE_ROOT / "outputs/human_validation/calibration_candidates"
DEFAULT_REVIEW = Path.home() / "Downloads/va_anchor_candidate_review.csv"
CANDIDATE_CSV = CANDIDATE_DIR / "va_anchor_candidates.csv"
ARCHIVE_DIR = CANDIDATE_DIR / "human_reviews"
ARCHIVED_REVIEW = ARCHIVE_DIR / "va_anchor_candidate_review_rater01_20260722.csv"
FROZEN_CSV = CANDIDATE_DIR / "frozen_va_anchors_v1.csv"
FROZEN_JSON = CANDIDATE_DIR / "frozen_va_anchors_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    candidate_rows = read_csv(CANDIDATE_CSV)
    review_rows = read_csv(args.review)
    candidate_by_id = {row["candidate_id"]: row for row in candidate_rows}
    review_by_id = {row["candidate_id"]: row for row in review_rows}

    if len(candidate_rows) != 12 or len(candidate_by_id) != 12:
        raise RuntimeError("Expected 12 unique source candidates")
    if len(review_rows) != 12 or len(review_by_id) != 12:
        raise RuntimeError("Expected 12 unique reviewed candidates")
    if set(candidate_by_id) != set(review_by_id):
        raise RuntimeError("Reviewed candidate IDs do not match the frozen candidate set")

    frozen_rows: list[dict[str, object]] = []
    for candidate_id, source in candidate_by_id.items():
        review = review_by_id[candidate_id]
        for field in (
            "dimension",
            "proposed_level",
            "song_id",
            "title",
            "artist",
        ):
            if review[field].strip() != source[field].strip():
                raise RuntimeError(f"Immutable field drift for {candidate_id}: {field}")
        for field in ("start_sec", "end_sec"):
            if abs(float(review[field]) - float(source[field])) > 1e-6:
                raise RuntimeError(f"Immutable field drift for {candidate_id}: {field}")

        judgment = review["judgment"].strip()
        corrected = review["corrected_level"].strip()
        if judgment not in {"符合", "不符合"}:
            raise RuntimeError(f"Unresolved judgment for {candidate_id}: {judgment!r}")
        if judgment == "符合":
            if corrected:
                raise RuntimeError(f"Unexpected correction for accepted candidate {candidate_id}")
            final_level = source["proposed_level"]
        else:
            if corrected not in {"low", "mid", "high"}:
                raise RuntimeError(f"Missing valid correction for {candidate_id}")
            final_level = corrected

        frozen_rows.append(
            {
                "anchor_id": candidate_id,
                "dimension": source["dimension"],
                "final_level": final_level,
                "original_proposed_level": source["proposed_level"],
                "human_judgment": judgment,
                "song_id": source["song_id"],
                "title": source["title"],
                "artist": source["artist"],
                "start_sec": float(source["start_sec"]),
                "end_sec": float(source["end_sec"]),
                "duration_sec": float(source["duration_sec"]),
                "clip_relative_path": source["clip_relative_path"],
                "clip_sha256": source["clip_sha256"],
                "review_notes": review["notes"].strip(),
            }
        )

    counts = {
        f"{dimension}_{level}": sum(
            row["dimension"] == dimension and row["final_level"] == level
            for row in frozen_rows
        )
        for dimension in ("valence", "arousal")
        for level in ("low", "mid", "high")
    }
    if any(value != 2 for value in counts.values()):
        raise RuntimeError(f"Each dimension-level must have exactly two anchors: {counts}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.review, ARCHIVED_REVIEW)
    with FROZEN_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frozen_rows[0]))
        writer.writeheader()
        writer.writerows(frozen_rows)

    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v1",
        "status": "frozen_for_anchored_reannotation",
        "reviewer_id": "rater01",
        "review_source_archive": str(ARCHIVED_REVIEW),
        "candidate_source": str(CANDIDATE_CSV),
        "teacher_predictions_used": False,
        "blind_test_overlap_count": 0,
        "anchors_per_dimension_level": counts,
        "anchors": frozen_rows,
    }
    FROZEN_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
