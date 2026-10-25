#!/usr/bin/env python3
"""Freeze the audited 99-song corpus and seed reusable blind-15 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from select_blind_validation_sample import (
    assert_stage1_passed,
    canonical_sha256,
    file_sha256,
)


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
DEFAULT_STAGE1 = PHASE2_ROOT / "data/blind_validation/stage1"
DEFAULT_BLIND_STAGE2 = PHASE2_ROOT / "data/blind_validation/stage2"
DEFAULT_BLIND_STAGE3 = PHASE2_ROOT / "data/blind_validation/stage3"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/full_corpus/stage1"
DEFAULT_INFERENCE_OUTPUT = PHASE2_ROOT / "data/full_corpus/stage2"
CHINA_TZ = timezone(timedelta(hours=8))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def link_verified(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(f"Existing reuse target differs from source: {target}")
        return "existing_verified"
    os.link(source, target)
    return "hard_linked"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--blind-stage2-dir", type=Path, default=DEFAULT_BLIND_STAGE2)
    parser.add_argument("--blind-stage3-dir", type=Path, default=DEFAULT_BLIND_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inference-output-dir", type=Path, default=DEFAULT_INFERENCE_OUTPUT)
    args = parser.parse_args()

    audit_path = args.stage1_dir / "stage1_input_audit.json"
    songs_path = args.stage1_dir / "raw/reviewed_songs_allowed.json"
    boundaries_path = args.stage1_dir / "raw/section_boundaries_allowed.json"
    blind_manifest_path = args.blind_stage2_dir / "blind_sample_15.csv"
    blind_stage3_manifest_path = args.blind_stage3_dir / "stage3_inference_manifest.json"

    audit = load_json(audit_path)
    assert_stage1_passed(audit)
    songs: list[dict[str, Any]] = load_json(songs_path)
    boundaries: list[dict[str, Any]] = load_json(boundaries_path)
    blind_rows = list(csv.DictReader(blind_manifest_path.open(newline="", encoding="utf-8")))
    blind_stage3_manifest = load_json(blind_stage3_manifest_path)

    by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for boundary in boundaries:
        by_song[str(boundary["song_id"])].append(boundary)
    for rows in by_song.values():
        rows.sort(key=lambda row: (int(row["section_order"]), str(row["human_annotation_id"])))

    blind_by_id = {str(row["song_id"]): row for row in blind_rows}
    blind_completed = {str(row["song_id"]): row for row in blind_stage3_manifest["completed"]}
    if len(blind_by_id) != 15 or len(blind_completed) != 15:
        raise RuntimeError("Frozen blind reuse source must contain exactly 15 unique songs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    for index, song in enumerate(sorted(songs, key=lambda row: str(row["id"])), start=1):
        song_id = str(song["id"])
        audio_path = PROJECT_ROOT / str(song["audio_path"]).lstrip("/")
        song_boundaries = by_song[song_id]
        audio_hash = file_sha256(audio_path)
        boundary_hash = canonical_sha256(song_boundaries)
        reusable = song_id in blind_by_id
        manifest_rows.append({
            "corpus_index": index,
            "sample_index": index,
            "song_id": song_id,
            "title": song["title"],
            "artist": song["artist"],
            "audio_path": song["audio_path"],
            "duration_sec": float(song["duration_sec"]),
            "section_count": len(song_boundaries),
            "audio_sha256": audio_hash,
            "boundary_sha256": boundary_hash,
            "blind15_reuse_eligible": reusable,
        })
        if not reusable:
            continue

        blind = blind_by_id[song_id]
        errors = []
        if audio_hash != blind["audio_sha256"]:
            errors.append("audio_hash")
        if boundary_hash != blind["boundary_sha256"]:
            errors.append("boundary_hash")
        if song_id not in blind_completed:
            errors.append("missing_stage3_completed")

        sources = {
            "feature": args.blind_stage3_dir / "features" / f"{song_id}.npz",
            "frame_va": args.blind_stage3_dir / "frame_va" / f"{song_id}_frame_va.csv",
            "metadata": args.blind_stage3_dir / "metadata" / f"{song_id}.json",
        }
        if not errors:
            metadata = load_json(sources["metadata"])
            if metadata.get("audio_sha256") != audio_hash:
                errors.append("metadata_audio_hash")
            if metadata.get("feature_sha256") != file_sha256(sources["feature"]):
                errors.append("metadata_feature_hash")
            if metadata.get("frame_va_sha256") != file_sha256(sources["frame_va"]):
                errors.append("metadata_frame_hash")
        if errors:
            raise RuntimeError(f"Blind-15 reuse audit failed for {song_id}: {errors}")

        targets = {
            "feature": args.inference_output_dir / "features" / sources["feature"].name,
            "frame_va": args.inference_output_dir / "frame_va" / sources["frame_va"].name,
            "metadata": args.inference_output_dir / "metadata" / sources["metadata"].name,
        }
        methods = {name: link_verified(sources[name], targets[name]) for name in sources}
        reuse_rows.append({
            "song_id": song_id,
            "title": song["title"],
            "audio_sha256": audio_hash,
            "boundary_sha256": boundary_hash,
            "feature_sha256": file_sha256(sources["feature"]),
            "frame_va_sha256": file_sha256(sources["frame_va"]),
            "metadata_sha256": file_sha256(sources["metadata"]),
            "reuse_status": "pass",
            "link_method": ";".join(sorted(set(methods.values()))),
        })

    if len(manifest_rows) != 99 or len(by_song) != 99 or len(boundaries) != 1097:
        raise RuntimeError(
            f"Unexpected frozen population: songs={len(manifest_rows)}, "
            f"boundary_songs={len(by_song)}, boundaries={len(boundaries)}"
        )
    if len(reuse_rows) != 15:
        raise RuntimeError(f"Expected 15 reusable blind songs, got {len(reuse_rows)}")

    manifest_path = args.output_dir / "full_corpus_99_manifest.csv"
    frozen_boundaries_path = args.output_dir / "full_corpus_99_boundaries.json"
    reuse_path = args.output_dir / "blind15_reuse_audit.csv"
    protocol_path = args.output_dir / "full_corpus_99_protocol.json"
    write_csv(manifest_path, manifest_rows, list(manifest_rows[0]))
    write_csv(reuse_path, reuse_rows, list(reuse_rows[0]))
    frozen_boundaries_path.write_text(
        json.dumps(boundaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Frozen full-corpus audio inference input after blind human validation",
        "songs": len(manifest_rows),
        "sections": len(boundaries),
        "blind15_reused": len(reuse_rows),
        "songs_requiring_inference": len(manifest_rows) - len(reuse_rows),
        "source_stage1_audit": str(audit_path.resolve()),
        "source_stage1_audit_sha256": file_sha256(audit_path),
        "source_songs_sha256": file_sha256(songs_path),
        "source_boundaries_sha256": file_sha256(boundaries_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_canonical_sha256": canonical_sha256(manifest_rows),
        "boundaries_sha256": file_sha256(frozen_boundaries_path),
        "boundaries_canonical_sha256": canonical_sha256(boundaries),
        "reuse_audit_sha256": file_sha256(reuse_path),
        "section_types_read": False,
        "human_emotion_answers_read": False,
        "notes": [
            "Boundary rows are canonical; the stale denormalized human_annotation_count field is not used.",
            "Blind-15 feature, frame-VA, and metadata files are hard-linked after hash verification.",
        ],
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
