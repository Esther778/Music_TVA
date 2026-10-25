#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from import_local_results import request, update


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely export, verify, and update lyric text only.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export songs, lyric lines, and section snapshots.")
    export_parser.add_argument("--out", required=True, help="Snapshot JSON path.")

    apply_parser = subparsers.add_parser("apply", help="Apply reviewed lyric text corrections.")
    apply_parser.add_argument("--input", required=True, help="CSV with song_id,line_index,corrected_text.")
    apply_parser.add_argument("--snapshot", required=True, help="Snapshot JSON from export.")
    apply_parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing.")
    apply_parser.add_argument("--limit", type=int, default=0, help="Apply at most N changed rows.")

    verify_parser = subparsers.add_parser("verify", help="Verify timings and sections against a snapshot.")
    verify_parser.add_argument("--snapshot", required=True, help="Snapshot JSON from export.")

    args = parser.parse_args()
    if args.command == "export":
        export_snapshot(Path(args.out))
    elif args.command == "apply":
        apply_corrections(Path(args.input), Path(args.snapshot), dry_run=args.dry_run, limit=args.limit)
    elif args.command == "verify":
        verify_snapshot(Path(args.snapshot))


def export_snapshot(path: Path) -> None:
    snapshot = fetch_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "exported "
        f"{len(snapshot['songs'])} songs, "
        f"{len(snapshot['lyrics'])} lyric lines, "
        f"{len(snapshot['sections'])} sections to {path}"
    )


def apply_corrections(input_path: Path, snapshot_path: Path, *, dry_run: bool, limit: int) -> None:
    snapshot = read_snapshot(snapshot_path)
    current = fetch_snapshot()
    ensure_static_data_unchanged(snapshot, current)

    corrections = read_corrections(input_path)
    lyric_by_key = {(row["song_id"], int(row["line_index"])): row for row in current["lyrics"]}
    snapshot_text_by_key = {(row["song_id"], int(row["line_index"])): row["text"] for row in snapshot["lyrics"]}

    changed = []
    skipped_same = 0
    for row in corrections:
        key = (row["song_id"], int(row["line_index"]))
        existing = lyric_by_key.get(key)
        if existing is None:
            raise SystemExit(f"correction references missing lyric line: {key}")
        if key not in snapshot_text_by_key:
            raise SystemExit(f"correction references line not present in snapshot: {key}")
        corrected = row["corrected_text"].strip()
        if not corrected:
            raise SystemExit(f"empty corrected_text is not allowed: {key}")
        if corrected == existing["text"]:
            skipped_same += 1
            continue
        changed.append((existing, corrected))

    if limit:
        changed = changed[:limit]

    print(f"review rows: {len(corrections)}")
    print(f"changed rows: {len(changed)}")
    print(f"unchanged rows: {skipped_same}")
    print_song_summary(changed)

    if dry_run:
        print("dry-run only; no database writes")
        return

    for existing, corrected in changed:
        update(
            "lyric_lines",
            {"text": corrected},
            {"song_id": f"eq.{existing['song_id']}", "line_index": f"eq.{existing['line_index']}"},
        )
    after = fetch_snapshot()
    ensure_static_data_unchanged(snapshot, after)
    print(f"applied {len(changed)} lyric text updates")


def verify_snapshot(snapshot_path: Path) -> None:
    snapshot = read_snapshot(snapshot_path)
    current = fetch_snapshot()
    ensure_static_data_unchanged(snapshot, current)
    changed_text = 0
    before_text = {(row["song_id"], int(row["line_index"])): row["text"] for row in snapshot["lyrics"]}
    for row in current["lyrics"]:
        key = (row["song_id"], int(row["line_index"]))
        if before_text.get(key) != row["text"]:
            changed_text += 1
    print(f"verified static data unchanged; lyric text changed rows: {changed_text}")


def fetch_snapshot() -> dict[str, list[dict]]:
    return {
        "songs": paged_get("songs", "select=id,title,artist,audio_path,status&order=id.asc"),
        "lyrics": paged_get(
            "lyric_lines",
            "select=song_id,line_index,start_time_sec,end_time_sec,text,source&order=song_id.asc,line_index.asc",
        ),
        "sections": paged_get(
            "auto_sections",
            "select=song_id,run_id,section_index,start_time_sec,end_time_sec,section_type,structure_label&order=song_id.asc,run_id.asc,section_index.asc",
        ),
    }


def paged_get(table: str, query: str, *, page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    separator = "&" if query else ""
    while True:
        endpoint = f"{query}{separator}limit={page_size}&offset={offset}"
        page = request("GET", f"{table}?{endpoint}", None, {})
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def read_snapshot(path: Path) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_corrections(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("song_id") and row.get("line_index")]
    required = {"song_id", "line_index", "corrected_text"}
    if not rows:
        raise SystemExit("correction CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"correction CSV missing columns: {', '.join(sorted(missing))}")
    keys = [(row["song_id"], int(row["line_index"])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate correction keys: {duplicates[:10]}")
    return rows


def ensure_static_data_unchanged(before: dict[str, list[dict]], after: dict[str, list[dict]]) -> None:
    before_lyrics = lyric_static_map(before["lyrics"])
    after_lyrics = lyric_static_map(after["lyrics"])
    if before_lyrics != after_lyrics:
        raise SystemExit("lyric timing/index/source drift detected")
    if section_static_rows(before["sections"]) != section_static_rows(after["sections"]):
        raise SystemExit("auto section drift detected")


def lyric_static_map(rows: list[dict]) -> dict[tuple[str, int], tuple[float, float, str]]:
    return {
        (row["song_id"], int(row["line_index"])): (
            float(row["start_time_sec"]),
            float(row["end_time_sec"]),
            row.get("source") or "",
        )
        for row in rows
    }


def section_static_rows(rows: list[dict]) -> list[tuple]:
    return [
        (
            row["song_id"],
            row["run_id"],
            int(row["section_index"]),
            float(row["start_time_sec"]),
            float(row["end_time_sec"]),
            row.get("section_type") or "",
            row.get("structure_label") or "",
        )
        for row in rows
    ]


def print_song_summary(changed: list[tuple[dict, str]]) -> None:
    counts = Counter(row["song_id"] for row, _ in changed)
    for song_id, count in counts.most_common():
        print(f"{song_id}: {count} changed lines")


if __name__ == "__main__":
    main()
