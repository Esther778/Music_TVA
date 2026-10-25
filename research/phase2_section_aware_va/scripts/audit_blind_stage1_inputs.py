#!/usr/bin/env python3
"""Audit blind-safe Supabase inputs without reading section or VA answers."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
DEFAULT_INPUT = PHASE2_ROOT / "data/blind_validation/stage1/raw"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/blind_validation/stage1/stage1_input_audit.json"


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON list: {path}")
    return data


def local_audio_path(value: object) -> Path:
    path = str(value or "").strip()
    if path.startswith("/"):
        path = path[1:]
    return PROJECT_ROOT / path


def as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def probe_audio_duration(path: Path) -> float | None:
    """Use the macOS audio inspector for records missing database duration."""
    try:
        result = subprocess.run(
            ["afinfo", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"estimated duration:\s*([0-9.]+) sec", result.stdout)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    songs = load_json(args.input_dir / "reviewed_songs_allowed.json")
    sections = load_json(args.input_dir / "section_boundaries_allowed.json")
    song_ids = {str(row.get("id")) for row in songs}
    by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_boundaries: list[dict[str, Any]] = []
    annotation_ids: list[str] = []

    for row in sections:
        song_id = str(row.get("song_id"))
        by_song[song_id].append(row)
        annotation_ids.append(str(row.get("human_annotation_id")))
        start = as_float(row.get("start_time_sec"))
        end = as_float(row.get("end_time_sec"))
        if start is None or end is None or start < 0 or end <= start:
            invalid_boundaries.append({
                "song_id": song_id,
                "section_order": row.get("section_order"),
                "start_time_sec": row.get("start_time_sec"),
                "end_time_sec": row.get("end_time_sec"),
            })

    audio_exists = [local_audio_path(row.get("audio_path")).is_file() for row in songs]
    section_counts = [len(by_song.get(str(row.get("id")), [])) for row in songs]
    durations = [
        value for row in songs
        if (value := as_float(row.get("duration_sec"))) is not None
    ]
    count_mismatches = []
    for row, actual in zip(songs, section_counts):
        expected = row.get("human_annotation_count")
        if expected not in (None, "") and int(expected) != actual:
            count_mismatches.append({
                "song_id": row.get("id"),
                "summary_count": int(expected),
                "boundary_count": actual,
            })

    boundary_gap_rows: list[dict[str, Any]] = []
    boundary_overlap_rows: list[dict[str, Any]] = []
    first_section_not_zero: list[dict[str, Any]] = []
    final_section_duration_mismatches: list[dict[str, Any]] = []
    section_duration_mismatches: list[dict[str, Any]] = []
    section_order_issues: list[dict[str, Any]] = []
    song_by_id = {str(row.get("id")): row for row in songs}

    for song_id, rows in by_song.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row.get("section_order") or 0),
                as_float(row.get("start_time_sec")) or 0.0,
            ),
        )
        orders = [int(row.get("section_order") or 0) for row in ordered]
        expected_orders = (
            list(range(min(orders), min(orders) + len(orders))) if orders else []
        )
        if len(set(orders)) != len(orders) or orders != expected_orders:
            section_order_issues.append({"song_id": song_id, "orders": orders})

        if ordered:
            first_start = as_float(ordered[0].get("start_time_sec"))
            if first_start is not None and abs(first_start) > 0.05:
                first_section_not_zero.append({
                    "song_id": song_id,
                    "first_start_time_sec": first_start,
                })

        for previous, current in zip(ordered, ordered[1:]):
            previous_end = as_float(previous.get("end_time_sec"))
            current_start = as_float(current.get("start_time_sec"))
            if previous_end is None or current_start is None:
                continue
            delta = current_start - previous_end
            issue = {
                "song_id": song_id,
                "previous_section_order": previous.get("section_order"),
                "next_section_order": current.get("section_order"),
                "delta_sec": delta,
            }
            if delta > 0.05:
                boundary_gap_rows.append(issue)
            elif delta < -0.05:
                boundary_overlap_rows.append(issue)

        for row in ordered:
            start = as_float(row.get("start_time_sec"))
            end = as_float(row.get("end_time_sec"))
            stated_duration = as_float(row.get("duration_sec"))
            if start is None or end is None or stated_duration is None:
                continue
            calculated_duration = end - start
            if abs(calculated_duration - stated_duration) > 0.05:
                section_duration_mismatches.append({
                    "song_id": song_id,
                    "section_order": row.get("section_order"),
                    "calculated_duration_sec": calculated_duration,
                    "stored_duration_sec": stated_duration,
                })

        song_duration = as_float(song_by_id.get(song_id, {}).get("duration_sec"))
        if ordered and song_duration is not None and song_duration > 0:
            final_end = as_float(ordered[-1].get("end_time_sec"))
            if final_end is not None and abs(final_end - song_duration) > 0.5:
                final_section_duration_mismatches.append({
                    "song_id": song_id,
                    "final_section_end_sec": final_end,
                    "song_duration_sec": song_duration,
                    "delta_sec": final_end - song_duration,
                })

    nonpositive_song_durations = []
    for row in songs:
        duration = as_float(row.get("duration_sec"))
        if duration is not None and duration > 0:
            continue
        path = local_audio_path(row.get("audio_path"))
        nonpositive_song_durations.append({
            "song_id": row.get("id"),
            "title": row.get("title"),
            "artist": row.get("artist"),
            "database_duration_sec": duration,
            "local_audio_duration_sec": probe_audio_duration(path) if path.is_file() else None,
            "audio_path": str(row.get("audio_path") or ""),
        })

    allowed_song_fields = sorted({key for row in songs for key in row})
    allowed_boundary_fields = sorted({key for row in sections for key in row})
    forbidden_seen = sorted(
        set(allowed_song_fields + allowed_boundary_fields)
        & {
            "section_type", "confidence", "qc_status", "is_approved",
            "before_json", "after_json", "valence", "arousal",
            "human_valence_level", "human_arousal_level", "is_climax",
        }
    )

    report = {
        "protocol": "stage-1 blind-safe read-only audit",
        "reviewed_songs": len(songs),
        "unique_song_ids": len(song_ids),
        "section_boundary_rows": len(sections),
        "songs_with_boundaries": len(song_ids & set(by_song)),
        "songs_missing_boundaries": sorted(song_ids - set(by_song)),
        "boundary_song_ids_outside_reviewed_set": sorted(set(by_song) - song_ids),
        "invalid_boundary_rows": invalid_boundaries,
        "section_continuity": {
            "tolerance_sec": 0.05,
            "section_order_issues": section_order_issues,
            "first_section_not_zero": first_section_not_zero,
            "boundary_gap_rows": boundary_gap_rows,
            "boundary_overlap_rows": boundary_overlap_rows,
            "section_duration_mismatches": section_duration_mismatches,
            "final_section_vs_song_duration_tolerance_sec": 0.5,
            "final_section_duration_mismatches": final_section_duration_mismatches,
        },
        "unique_human_annotation_ids": len(set(annotation_ids)),
        "duplicate_human_annotation_id_rows": len(annotation_ids) - len(set(annotation_ids)),
        "local_audio_exists": sum(audio_exists),
        "local_audio_missing": len(audio_exists) - sum(audio_exists),
        "section_count": {
            "min": min(section_counts) if section_counts else 0,
            "median": statistics.median(section_counts) if section_counts else 0,
            "max": max(section_counts) if section_counts else 0,
        },
        "duration_sec": {
            "min": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "nonpositive_song_durations": nonpositive_song_durations,
        "status_distribution": dict(Counter(str(row.get("status") or "") for row in songs)),
        "is_reviewed_distribution": dict(Counter(str(row.get("is_reviewed")) for row in songs)),
        "source_distribution": dict(Counter(str(row.get("source") or "") for row in songs)),
        "summary_count_mismatch_count": len(count_mismatches),
        "summary_count_mismatches": count_mismatches,
        "allowed_song_fields": allowed_song_fields,
        "allowed_boundary_fields": allowed_boundary_fields,
        "forbidden_answer_fields_seen": forbidden_seen,
        "blind_boundary": {
            "frame_va_model_may_read": ["audio file only"],
            "section_aggregation_may_read_after_prediction_freeze": allowed_boundary_fields,
            "must_remain_hidden_until_evaluation": [
                "section_type if used as an evaluation target",
                "human VA level",
                "change direction judgment",
                "climax judgment",
                "confidence and review history",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
