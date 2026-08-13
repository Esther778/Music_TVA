#!/usr/bin/env python3
"""Freeze top60 quota 100-song inputs for VA Teacher inference."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
MANIFESTS = PHASE2 / "data/manifests"
DEFAULT_SONGS = MANIFESTS / "top60_quota_100_reviewed_manifest_20260813.csv"
DEFAULT_SECTIONS = MANIFESTS / "top60_quota_100_human_annotations_readable_20260813.csv"
DEFAULT_OUTPUT = PHASE2 / "data/top60_quota_100_20260813/stage1"
CHINA_TZ = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    songs = read_csv(DEFAULT_SONGS)
    sections = read_csv(DEFAULT_SECTIONS)
    if len(songs) != 100:
        raise RuntimeError(f"Expected 100 songs, got {len(songs)}")

    by_song: dict[str, list[dict[str, Any]]] = {}
    for row in sections:
        by_song.setdefault(str(row["song_id"]), []).append(row)
    if len(by_song) != 100:
        raise RuntimeError(f"Expected sections for 100 songs, got {len(by_song)}")

    manifest_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, song in enumerate(songs, start=1):
        song_id = str(song["song_id"])
        audio_rel = str(song["audio_path"]).lstrip("/")
        audio_path = ROOT / audio_rel
        if not audio_path.is_file():
            errors.append(f"{song_id}:missing_audio:{audio_rel}")
            continue
        rows = sorted(by_song.get(song_id, []), key=lambda item: int(item["section_order"]))
        if not rows:
            errors.append(f"{song_id}:missing_sections")
            continue
        starts = [float(row["start_time_sec"]) for row in rows]
        ends = [float(row["end_time_sec"]) for row in rows]
        if abs(starts[0]) > 1e-8:
            errors.append(f"{song_id}:first_start")
        if any(end <= start for start, end in zip(starts, ends)):
            errors.append(f"{song_id}:nonpositive_section")
        if len(rows) > 1 and any(abs(starts[i] - ends[i - 1]) > 1e-8 for i in range(1, len(rows))):
            errors.append(f"{song_id}:continuity")
        orders = [int(row["section_order"]) for row in rows]
        if orders != list(range(1, len(rows) + 1)):
            errors.append(f"{song_id}:section_order")
        for row in rows:
            boundary_rows.append(
                {
                    "song_id": song_id,
                    "title": song["title"],
                    "artist": song["artist"],
                    "human_annotation_id": row["human_annotation_id"],
                    "section_order": int(row["section_order"]),
                    "section_type": row["section_type"],
                    "start_time_sec": float(row["start_time_sec"]),
                    "end_time_sec": float(row["end_time_sec"]),
                    "duration_sec": float(row["duration_sec"]),
                }
            )
        manifest_rows.append(
            {
                "corpus_index": index,
                "sample_index": index,
                "selected_order": int(song["selected_order"]),
                "rank": int(song["rank"]),
                "quota_artist": song["quota_artist"],
                "song_id": song_id,
                "title": song["title"],
                "artist": song["artist"],
                "audio_path": "/" + audio_rel,
                "duration_sec": float(song["duration_sec"]),
                "section_count": len(rows),
                "audio_sha256": sha256_file(audio_path),
                "boundary_sha256": canonical_sha256(boundary_rows[-len(rows):]),
            }
        )

    if errors:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False, indent=2))

    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = DEFAULT_OUTPUT / "top60_quota_100_manifest.csv"
    boundaries_path = DEFAULT_OUTPUT / "top60_quota_100_boundaries.json"
    protocol_path = DEFAULT_OUTPUT / "top60_quota_100_protocol.json"
    write_csv(manifest_path, manifest_rows, list(manifest_rows[0]))
    boundaries_path.write_text(json.dumps(boundary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Top60 quota 100-song VA Teacher frozen inputs",
        "songs": len(manifest_rows),
        "sections": len(boundary_rows),
        "source_song_manifest": str(DEFAULT_SONGS.relative_to(ROOT)),
        "source_section_manifest": str(DEFAULT_SECTIONS.relative_to(ROOT)),
        "source_song_manifest_sha256": sha256_file(DEFAULT_SONGS),
        "source_section_manifest_sha256": sha256_file(DEFAULT_SECTIONS),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": canonical_sha256(manifest_rows),
        "boundaries_sha256": sha256_file(boundaries_path),
        "boundaries_canonical_sha256": canonical_sha256(boundary_rows),
        "section_types_frozen": True,
        "human_emotion_answers_read": False,
        "old_97_99_outputs_modified": False,
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
