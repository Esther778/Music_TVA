#!/usr/bin/env python3
"""Select and freeze a blind-safe 15-song validation sample."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
DEFAULT_INPUT = PHASE2_ROOT / "data/blind_validation/stage1/raw"
DEFAULT_AUDIT = PHASE2_ROOT / "data/blind_validation/stage1/stage1_input_audit.json"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/blind_validation/stage2"
DEFAULT_SEED = 20260720
CHINA_TZ = timezone(timedelta(hours=8))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_stage1_passed(audit: dict[str, Any]) -> None:
    continuity = audit["section_continuity"]
    failures = {
        "songs_missing_boundaries": audit["songs_missing_boundaries"],
        "invalid_boundary_rows": audit["invalid_boundary_rows"],
        "local_audio_missing": audit["local_audio_missing"],
        "nonpositive_song_durations": audit["nonpositive_song_durations"],
        "section_order_issues": continuity["section_order_issues"],
        "first_section_not_zero": continuity["first_section_not_zero"],
        "boundary_gap_rows": continuity["boundary_gap_rows"],
        "boundary_overlap_rows": continuity["boundary_overlap_rows"],
        "section_duration_mismatches": continuity["section_duration_mismatches"],
        "final_section_duration_mismatches": continuity["final_section_duration_mismatches"],
        "forbidden_answer_fields_seen": audit["forbidden_answer_fields_seen"],
    }
    active = {key: value for key, value in failures.items() if value}
    if active:
        raise RuntimeError(f"Stage 1 audit has unresolved failures: {active}")


def section_bucket(section_count: int) -> tuple[int, str]:
    if section_count <= 10:
        return 0, "low_08_10"
    if section_count <= 12:
        return 1, "mid_11_12"
    return 2, "high_13_plus"


def choose_unique_sample(
    candidates_by_cell: dict[tuple[int, int], list[dict[str, Any]]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cells = sorted(candidates_by_cell, key=lambda cell: (len(candidates_by_cell[cell]), cell))
    shuffled: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for cell in cells:
        rows = sorted(candidates_by_cell[cell], key=lambda row: str(row["id"]))
        rng.shuffle(rows)
        shuffled[cell] = rows

    selected: list[dict[str, Any]] = []
    used_artists: set[str] = set()
    used_titles: set[str] = set()

    def search(index: int) -> bool:
        if index == len(cells):
            return True
        cell = cells[index]
        for row in shuffled[cell]:
            artist = normalize(row["artist"])
            title = normalize(row["title"])
            if artist in used_artists or title in used_titles:
                continue
            selected.append(row)
            used_artists.add(artist)
            used_titles.add(title)
            if search(index + 1):
                return True
            selected.pop()
            used_artists.remove(artist)
            used_titles.remove(title)
        return False

    if not search(0):
        raise RuntimeError("Could not select one song per stratum with unique artists and titles")
    return sorted(selected, key=lambda row: (row["duration_bin"], row["section_bin"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "song_id",
        "title",
        "artist",
        "audio_path",
        "duration_sec",
        "section_count",
        "duration_stratum",
        "section_stratum",
        "selection_seed",
        "audio_sha256",
        "boundary_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    audit = load_json(args.audit)
    assert_stage1_passed(audit)
    songs: list[dict[str, Any]] = load_json(args.input_dir / "reviewed_songs_allowed.json")
    boundaries: list[dict[str, Any]] = load_json(args.input_dir / "section_boundaries_allowed.json")

    by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in boundaries:
        by_song[str(row["song_id"])].append(row)
    for rows in by_song.values():
        rows.sort(key=lambda row: (int(row["section_order"]), str(row["human_annotation_id"])))

    durations = sorted(float(row["duration_sec"]) for row in songs)
    duration_cuts = [durations[index] for index in (19, 39, 59, 79)]
    duration_labels = ["d1_shortest", "d2_short", "d3_medium", "d4_long", "d5_longest"]
    candidates_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for song in songs:
        song_id = str(song["id"])
        duration_bin = bisect.bisect_right(duration_cuts, float(song["duration_sec"]))
        section_bin, section_label = section_bucket(len(by_song[song_id]))
        candidate = dict(song)
        candidate.update({
            "duration_bin": duration_bin,
            "duration_stratum": duration_labels[duration_bin],
            "section_bin": section_bin,
            "section_stratum": section_label,
        })
        candidates_by_cell[(duration_bin, section_bin)].append(candidate)

    expected_cells = {(duration_bin, section_bin) for duration_bin in range(5) for section_bin in range(3)}
    missing_cells = expected_cells - set(candidates_by_cell)
    if missing_cells:
        raise RuntimeError(f"Empty strata prevent 5x3 sampling: {sorted(missing_cells)}")

    selected = choose_unique_sample(candidates_by_cell, args.seed)
    frozen_at = datetime.now(CHINA_TZ).isoformat(timespec="seconds")
    manifest_rows: list[dict[str, Any]] = []
    frozen_boundaries: list[dict[str, Any]] = []

    for index, song in enumerate(selected, start=1):
        song_id = str(song["id"])
        audio_path = PROJECT_ROOT / str(song["audio_path"]).lstrip("/")
        song_boundaries = by_song[song_id]
        manifest_rows.append({
            "sample_index": index,
            "song_id": song_id,
            "title": song["title"],
            "artist": song["artist"],
            "audio_path": song["audio_path"],
            "duration_sec": float(song["duration_sec"]),
            "section_count": len(song_boundaries),
            "duration_stratum": song["duration_stratum"],
            "section_stratum": song["section_stratum"],
            "selection_seed": args.seed,
            "audio_sha256": file_sha256(audio_path),
            "boundary_sha256": canonical_sha256(song_boundaries),
        })
        frozen_boundaries.extend(song_boundaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "blind_sample_15.csv"
    boundaries_path = args.output_dir / "blind_sample_15_boundaries.json"
    protocol_path = args.output_dir / "blind_sample_15_protocol.json"
    write_csv(manifest_path, manifest_rows)
    boundaries_path.write_text(
        json.dumps(frozen_boundaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "protocol": "blind validation stage 2 stratified frozen sample",
        "frozen_at": frozen_at,
        "selection_seed": args.seed,
        "population_song_count": len(songs),
        "sample_song_count": len(manifest_rows),
        "selection_inputs": [
            "song_id",
            "title",
            "artist",
            "audio_path",
            "duration_sec",
            "section boundary count",
        ],
        "selection_excluded": [
            "audio content",
            "section type",
            "lyrics",
            "review history",
            "VA predictions",
            "human VA/change/climax judgments",
        ],
        "duration_cut_points_sec": duration_cuts,
        "duration_strata": duration_labels,
        "section_strata": ["low_08_10", "mid_11_12", "high_13_plus"],
        "sampling_rule": "one song per 5x3 duration-section-count cell, with unique normalized artist and title",
        "manifest_sha256": canonical_sha256(manifest_rows),
        "frozen_boundaries_sha256": canonical_sha256(frozen_boundaries),
        "candidate_cell_counts": {
            f"duration_{d + 1}_section_{s + 1}": len(candidates_by_cell[(d, s)])
            for d in range(5)
            for s in range(3)
        },
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path),
        "boundaries": str(boundaries_path),
        "protocol": str(protocol_path),
        "songs": manifest_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
