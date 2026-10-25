#!/usr/bin/env python3
"""Fetch held-out section types for the frozen blind sample after VA inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import quote, supabase_get


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2 = PHASE2_ROOT / "data/blind_validation/stage2"
DEFAULT_STAGE4 = PHASE2_ROOT / "data/blind_validation/stage4"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/blind_validation/stage5"
CHINA_TZ = timezone(timedelta(hours=8))
FIELDS = [
    "human_annotation_id",
    "song_id",
    "section_order",
    "section_type",
    "start_time_sec",
    "end_time_sec",
    "annotation_kind",
    "is_approved",
    "confidence",
    "updated_at",
]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage4-dir", type=Path, default=DEFAULT_STAGE4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    boundaries_path = args.stage2_dir / "blind_sample_15_boundaries.json"
    protocol_path = args.stage2_dir / "blind_sample_15_protocol.json"
    stage4_manifest_path = args.stage4_dir / "stage4_manifest.json"
    frozen = json.loads(boundaries_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage4 = json.loads(stage4_manifest_path.read_text(encoding="utf-8"))

    if canonical_sha256(frozen) != protocol["frozen_boundaries_sha256"]:
        raise RuntimeError("Frozen Stage 2 boundary hash mismatch")
    if stage4.get("status") != "pass":
        raise RuntimeError("Stage 4 must pass before held-out section types are fetched")

    frozen_by_id = {str(row["human_annotation_id"]): row for row in frozen}
    ids = sorted(frozen_by_id)
    fetched: list[dict[str, object]] = []
    for start in range(0, len(ids), 30):
        chunk = ids[start : start + 30]
        query = (
            f"select={quote(','.join(FIELDS))}"
            f"&human_annotation_id=in.({','.join(chunk)})"
            "&order=song_id.asc,section_order.asc,human_annotation_id.asc"
        )
        fetched.extend(supabase_get("human_annotations_readable", query))

    fetched.sort(key=lambda row: (str(row["song_id"]), int(row["section_order"]), str(row["human_annotation_id"])))
    errors: list[str] = []
    fetched_ids = [str(row["human_annotation_id"]) for row in fetched]
    if len(fetched) != len(frozen):
        errors.append(f"row_count:{len(fetched)}!={len(frozen)}")
    if len(set(fetched_ids)) != len(fetched_ids):
        errors.append("duplicate_annotation_ids")
    missing = sorted(set(ids) - set(fetched_ids))
    unexpected = sorted(set(fetched_ids) - set(ids))
    if missing:
        errors.append(f"missing_ids:{len(missing)}")
    if unexpected:
        errors.append(f"unexpected_ids:{len(unexpected)}")

    boundary_mismatches = []
    for row in fetched:
        frozen_row = frozen_by_id.get(str(row["human_annotation_id"]))
        if frozen_row is None:
            continue
        if (
            str(row["song_id"]) != str(frozen_row["song_id"])
            or int(row["section_order"]) != int(frozen_row["section_order"])
            or abs(float(row["start_time_sec"]) - float(frozen_row["start_time_sec"])) > 1e-8
            or abs(float(row["end_time_sec"]) - float(frozen_row["end_time_sec"])) > 1e-8
        ):
            boundary_mismatches.append(str(row["human_annotation_id"]))
    if boundary_mismatches:
        errors.append(f"boundary_mismatches:{len(boundary_mismatches)}")

    missing_types = [str(row["human_annotation_id"]) for row in fetched if not str(row.get("section_type") or "").strip()]
    if missing_types:
        errors.append(f"missing_section_types:{len(missing_types)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "blind_sample_15_heldout_section_types.json"
    labels_path.write_text(json.dumps(fetched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "fetched_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Stage 5 held-out structural-label reveal after frozen Stage 3/4 outputs",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "requested_ids": len(ids),
        "fetched_rows": len(fetched),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "boundary_mismatch_ids": boundary_mismatches,
        "missing_type_ids": missing_types,
        "fields_read": FIELDS,
        "human_va_fields_present": False,
        "human_change_fields_present": False,
        "human_climax_fields_present": False,
        "frozen_boundaries_canonical_sha256": canonical_sha256(frozen),
        "stage4_manifest_status": stage4.get("status"),
        "heldout_labels_path": str(labels_path.resolve()),
        "heldout_labels_canonical_sha256": canonical_sha256(fetched),
    }
    manifest_path = args.output_dir / "stage5_label_reveal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
