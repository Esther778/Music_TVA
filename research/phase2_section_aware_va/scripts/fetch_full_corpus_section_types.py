#!/usr/bin/env python3
"""Reveal section types only after frozen full-corpus VA alignment passes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import quote, supabase_get
from align_blind_stage4_sections import canonical_sha256


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = PHASE2_ROOT / "data/full_corpus/stage1"
DEFAULT_STAGE3 = PHASE2_ROOT / "data/full_corpus/stage3"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/full_corpus/stage4"
CHINA_TZ = timezone(timedelta(hours=8))
FIELDS = [
    "human_annotation_id", "song_id", "section_order", "section_type",
    "start_time_sec", "end_time_sec", "annotation_kind", "is_approved",
    "confidence", "updated_at",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    boundaries_path = args.stage1_dir / "full_corpus_99_boundaries.json"
    protocol_path = args.stage1_dir / "full_corpus_99_protocol.json"
    alignment_manifest_path = args.stage3_dir / "full_corpus_alignment_manifest.json"
    frozen = json.loads(boundaries_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_manifest_path.read_text(encoding="utf-8"))
    if canonical_sha256(frozen) != protocol["boundaries_canonical_sha256"]:
        raise RuntimeError("Frozen full-corpus boundary hash mismatch")
    if alignment.get("status") != "pass":
        raise RuntimeError("Full-corpus alignment must pass before section types are revealed")

    frozen_by_id = {str(row["human_annotation_id"]): row for row in frozen}
    ids = sorted(frozen_by_id)
    fetched: list[dict[str, object]] = []
    for start in range(0, len(ids), 30):
        chunk = ids[start:start + 30]
        query = (
            f"select={quote(','.join(FIELDS))}"
            f"&human_annotation_id=in.({','.join(chunk)})"
            "&order=song_id.asc,section_order.asc,human_annotation_id.asc"
        )
        fetched.extend(supabase_get("human_annotations_readable", query))
    fetched.sort(key=lambda row: (str(row["song_id"]), int(row["section_order"]), str(row["human_annotation_id"])))

    fetched_ids = [str(row["human_annotation_id"]) for row in fetched]
    errors: list[str] = []
    if len(fetched) != len(frozen): errors.append(f"row_count:{len(fetched)}!={len(frozen)}")
    if len(set(fetched_ids)) != len(fetched_ids): errors.append("duplicate_annotation_ids")
    missing = sorted(set(ids) - set(fetched_ids))
    unexpected = sorted(set(fetched_ids) - set(ids))
    if missing: errors.append(f"missing_ids:{len(missing)}")
    if unexpected: errors.append(f"unexpected_ids:{len(unexpected)}")
    mismatches = []
    for row in fetched:
        source = frozen_by_id.get(str(row["human_annotation_id"]))
        if source is None:
            continue
        if (
            str(row["song_id"]) != str(source["song_id"])
            or int(row["section_order"]) != int(source["section_order"])
            or abs(float(row["start_time_sec"]) - float(source["start_time_sec"])) > 1e-8
            or abs(float(row["end_time_sec"]) - float(source["end_time_sec"])) > 1e-8
        ):
            mismatches.append(str(row["human_annotation_id"]))
    if mismatches: errors.append(f"boundary_mismatches:{len(mismatches)}")
    missing_types = [str(row["human_annotation_id"]) for row in fetched if not str(row.get("section_type") or "").strip()]
    if missing_types: errors.append(f"missing_section_types:{len(missing_types)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "full_corpus_99_frozen_section_types.json"
    labels_path.write_text(json.dumps(fetched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "fetched_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Full-corpus structural-label reveal after frozen Teacher inference and boundary alignment",
        "status": "pass" if not errors else "fail", "errors": errors,
        "requested_ids": len(ids), "fetched_rows": len(fetched),
        "missing_ids": missing, "unexpected_ids": unexpected,
        "boundary_mismatch_ids": mismatches, "missing_type_ids": missing_types,
        "fields_read": FIELDS, "human_va_fields_present": False,
        "human_change_fields_present": False, "human_climax_fields_present": False,
        "frozen_boundaries_canonical_sha256": canonical_sha256(frozen),
        "alignment_manifest_status": alignment.get("status"),
        "labels_path": str(labels_path.resolve()),
        "labels_canonical_sha256": canonical_sha256(fetched),
    }
    manifest_path = args.output_dir / "full_corpus_label_reveal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
