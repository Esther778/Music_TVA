#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from import_local_results import SUPABASE_KEY, SUPABASE_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply lyric text corrections with one curl PATCH per line.")
    parser.add_argument("--input", required=True, help="CSV with song_id,line_index,corrected_text.")
    parser.add_argument("--skip-first-line-index", type=int, default=-1, help="Skip one already-applied line index.")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    applied = 0
    for row in rows:
        line_index = int(row["line_index"])
        if line_index == args.skip_first_line_index:
            continue
        patch_text(row["song_id"], line_index, row["corrected_text"])
        applied += 1
    print(f"applied {applied} lyric text patches")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("song_id") and row.get("line_index")]
    if not rows:
        raise SystemExit("input CSV is empty")
    for column in ("song_id", "line_index", "corrected_text"):
        if column not in rows[0]:
            raise SystemExit(f"missing required column: {column}")
    return rows


def patch_text(song_id: str, line_index: int, text: str) -> None:
    endpoint = f"{SUPABASE_URL}/rest/v1/lyric_lines?song_id=eq.{song_id}&line_index=eq.{line_index}"
    command = [
        "curl",
        "-sS",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "-X",
        "PATCH",
        "-H",
        f"apikey: {SUPABASE_KEY}",
        "-H",
        f"Authorization: Bearer {SUPABASE_KEY}",
        "-H",
        "Content-Type: application/json",
        "-H",
        "Prefer: return=minimal",
        "--data-binary",
        json.dumps({"text": text}, ensure_ascii=False),
        endpoint,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
