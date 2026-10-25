#!/usr/bin/env python3
"""Freeze the chat-confirmed supplemental candidates into a 30-anchor VA set."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from build_va_anchor_candidates import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
V1_DIR = PHASE_ROOT / "outputs/human_validation/calibration_candidates"
SUPPLEMENT_DIR = (
    PHASE_ROOT / "outputs/human_validation/calibration_candidates_v2_supplement"
)
V1_JSON = V1_DIR / "frozen_va_anchors_v1.json"
SUPPLEMENT_CSV = SUPPLEMENT_DIR / "va_anchor_supplement_candidates.csv"
REVIEW_DIR = SUPPLEMENT_DIR / "human_reviews"
REVIEW_CSV = REVIEW_DIR / "va_anchor_supplement_chat_confirmation_20260725.csv"
FROZEN_CSV = SUPPLEMENT_DIR / "frozen_va_anchors_v2_30clips.csv"
FROZEN_JSON = SUPPLEMENT_DIR / "frozen_va_anchors_v2_30clips.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    v1_payload = json.loads(V1_JSON.read_text(encoding="utf-8"))
    existing = v1_payload["anchors"]
    supplement = read_csv(SUPPLEMENT_CSV)
    if len(existing) != 12 or len(supplement) != 18:
        raise RuntimeError(
            f"Expected 12 existing and 18 supplemental anchors, found "
            f"{len(existing)} and {len(supplement)}"
        )
    if len({row["candidate_id"] for row in supplement}) != 18:
        raise RuntimeError("Supplement candidate IDs are not unique")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_rows = [
        {
            "candidate_id": row["candidate_id"],
            "dimension": row["dimension"],
            "proposed_level": row["proposed_level"],
            "song_id": row["song_id"],
            "title": row["title"],
            "artist": row["artist"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "judgment": "符合",
            "corrected_level": "",
            "notes": "用户于2026-07-25在聊天中确认18个补充候选全部符合",
            "confirmation_method": "user_chat_statement",
        }
        for row in supplement
    ]
    with REVIEW_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    anchors: list[dict[str, object]] = []
    for row in existing:
        anchors.append(
            {
                **row,
                "source_set": "v1_human_reviewed",
                "page_clip_relative_path": (
                    "calibration_candidates/" + str(row["clip_relative_path"])
                ),
                "confirmation_method": "exported_candidate_review_csv",
            }
        )
    for row in supplement:
        anchors.append(
            {
                "anchor_id": row["candidate_id"],
                "dimension": row["dimension"],
                "final_level": row["proposed_level"],
                "original_proposed_level": row["proposed_level"],
                "human_judgment": "符合",
                "song_id": row["song_id"],
                "title": row["title"],
                "artist": row["artist"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "duration_sec": float(row["duration_sec"]),
                "clip_relative_path": row["clip_relative_path"],
                "page_clip_relative_path": (
                    "calibration_candidates_v2_supplement/"
                    + row["clip_relative_path"]
                ),
                "clip_sha256": row["clip_sha256"],
                "review_notes": review_rows[0]["notes"],
                "source_set": "v2_supplement_chat_confirmed",
                "confirmation_method": "user_chat_statement",
            }
        )

    if len({str(row["song_id"]) for row in anchors}) != 30:
        raise RuntimeError("The 30-anchor set must use 30 unique songs")
    counts = {
        f"{dimension}_{level}": sum(
            row["dimension"] == dimension and row["final_level"] == level
            for row in anchors
        )
        for dimension in ("valence", "arousal")
        for level in ("low", "mid", "high")
    }
    if any(value != 5 for value in counts.values()):
        raise RuntimeError(f"Expected five anchors per group: {counts}")

    with FROZEN_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchors[0]))
        writer.writeheader()
        writer.writerows(anchors)

    payload = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v2_30clips",
        "status": "frozen_reference_library_not_yet_used_for_formal_annotations",
        "confirmation_method_supplement": "user_chat_statement",
        "confirmation_summary": (
            "User stated on 2026-07-25 that all 18 supplemental candidates fit "
            "their proposed levels; no exported review CSV was provided."
        ),
        "existing_anchor_count": len(existing),
        "supplement_anchor_count": len(supplement),
        "total_anchor_count": len(anchors),
        "anchors_per_dimension_level": counts,
        "unique_song_count": len({str(row["song_id"]) for row in anchors}),
        "teacher_predictions_used": False,
        "blind_test_overlap_count": 0,
        "review_csv": str(REVIEW_CSV),
        "review_csv_sha256": sha256(REVIEW_CSV),
        "frozen_csv": str(FROZEN_CSV),
        "anchors": anchors,
    }
    FROZEN_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
