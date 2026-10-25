#!/usr/bin/env python3
"""Build section-level audio clips for the blind human validation workbook."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/data/blind_validation/stage2/blind_sample_15.csv"
)
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/data/blind_validation/stage5/"
    "blind_sample_15_human_emotion_annotation_template.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/outputs/human_validation/audio_clips"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/outputs/human_validation/"
    "blind_sample_15_audio_clip_audit.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "research/phase2_section_aware_va/outputs/human_validation/"
    "blind_sample_15_audio_clip_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration(path: Path, ffprobe: str) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def resolve_audio_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        project_relative = PROJECT_ROOT / raw_path.lstrip("/")
        if project_relative.exists():
            return project_relative
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_sample(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["song_id"]: row for row in rows}
    if len(rows) != 15 or len(result) != 15:
        raise RuntimeError(f"Expected 15 unique sample songs, found {len(rows)}/{len(result)}")
    return result


def load_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 179:
        raise RuntimeError(f"Expected 179 section rows, found {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    sample = load_sample(args.sample)
    annotations = load_annotations(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    section_counts: dict[str, int] = defaultdict(int)
    audit_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for row in annotations:
        song_id = row["song_id"]
        song = sample.get(song_id)
        if song is None:
            raise RuntimeError(f"Missing song metadata for {song_id}")

        sample_index = int(row["sample_index"])
        section_order = int(row["section_order"])
        section_type = row["section_type"].strip().lower().replace(" ", "_")
        start_sec = float(row["start_time_sec"])
        end_sec = float(row["end_time_sec"])
        requested_duration = end_sec - start_sec
        if requested_duration <= 0:
            raise RuntimeError(
                f"Invalid interval {song_id} section {section_order}: {start_sec}-{end_sec}"
            )

        source = resolve_audio_path(song["audio_path"])
        if not source.exists():
            raise FileNotFoundError(source)

        song_dir = args.output_dir / f"sample_{sample_index:02d}"
        song_dir.mkdir(parents=True, exist_ok=True)
        clip_name = f"section_{section_order:02d}_{section_type}.mp3"
        clip_path = song_dir / clip_name

        if args.force or not clip_path.exists():
            subprocess.run(
                [
                    args.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start_sec:.3f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{requested_duration:.3f}",
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    "apad=pad_dur=1",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    str(clip_path),
                ],
                check=True,
            )

        measured_duration = probe_duration(clip_path, args.ffprobe)
        duration_error = measured_duration - requested_duration
        status = "pass" if abs(duration_error) <= 0.08 else "fail"
        if status == "fail":
            failures.append(
                f"{song_id} section {section_order}: duration error {duration_error:.6f}s"
            )
        section_counts[song_id] += 1
        audit_rows.append(
            {
                "sample_index": sample_index,
                "song_id": song_id,
                "title": row["title"],
                "artist": row["artist"],
                "human_annotation_id": row["human_annotation_id"],
                "section_order": section_order,
                "section_type": row["section_type"],
                "requested_start_sec": f"{start_sec:.3f}",
                "requested_end_sec": f"{end_sec:.3f}",
                "requested_duration_sec": f"{requested_duration:.3f}",
                "measured_duration_sec": f"{measured_duration:.6f}",
                "duration_error_sec": f"{duration_error:.6f}",
                "clip_relative_path": str(clip_path.relative_to(args.output_dir.parent)),
                "clip_sha256": sha256(clip_path),
                "status": status,
            }
        )

    for song_id, song in sample.items():
        expected = int(song["section_count"])
        actual = section_counts[song_id]
        if actual != expected:
            failures.append(f"{song_id}: expected {expected} clips, found {actual}")

    fieldnames = list(audit_rows[0])
    with args.audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_path": str(args.sample),
        "annotation_path": str(args.annotations),
        "output_dir": str(args.output_dir),
        "audit_path": str(args.audit),
        "songs": len(sample),
        "clips": len(audit_rows),
        "passed": sum(row["status"] == "pass" for row in audit_rows),
        "failed": len(failures),
        "bitrate_kbps": 128,
        "sample_rate_hz": 44100,
        "channels": 2,
        "duration_tolerance_sec": 0.08,
        "failures": failures,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if failures:
        raise RuntimeError("Audio clip audit failed; see manifest")


if __name__ == "__main__":
    main()
