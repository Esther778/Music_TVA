#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import urllib.parse
from pathlib import Path


SUPABASE_URL = "https://itcarzyhlkiovskfdqut.supabase.co"
SUPABASE_KEY = "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN"
DEFAULT_MANIFEST = Path(__file__).with_name("songs_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local section CSV outputs into Supabase.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Song import manifest CSV.")
    parser.add_argument("--replace", action="store_true", help="Delete existing song rows before import.")
    args = parser.parse_args()

    for song in read_manifest(Path(args.manifest)):
        sections_path = Path(song["sections_csv"])
        if not sections_path.exists():
            print(f"skip missing sections: {sections_path}")
            continue
        if args.replace:
            delete("songs", {"id": f"eq.{song['id']}"})
        duration = section_duration(sections_path)
        upsert(
            "songs",
            [
                {
                    "id": song["id"],
                    "title": song["title"],
                    "artist": song["artist"],
                    "audio_path": song["audio_path"],
                    "duration_sec": duration,
                    "source": "local_csv",
                    "status": "pending_review",
                }
            ],
            on_conflict="id",
        )
        run = insert(
            "model_runs",
            [
                {
                    "song_id": song["id"],
                    "model_version": "rule_pipeline",
                    "pipeline_version": song["pipeline_version"],
                    "params": {"sections_csv": song["sections_csv"]},
                    "notes": "Imported from local pipeline output.",
                }
            ],
        )[0]
        import_sections(song["id"], run["id"], sections_path)
        lyrics_path = Path(song["lyrics_csv"])
        if lyrics_path.exists():
            import_lyrics(song["id"], lyrics_path)
        print(f"imported {song['id']} run={run['id']}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("id")]


def import_sections(song_id: str, run_id: str, path: Path) -> None:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = parse_time(row.get("start_time") or row.get("start_time_sec"))
            end = parse_time(row.get("end_time") or row.get("end_time_sec"))
            boundary_confidence = first_number(row, "boundary_confidence", "acoustic_role_score")
            type_confidence = first_number(row, "type_confidence", "final_confidence")
            rows.append(
                {
                    "run_id": run_id,
                    "song_id": song_id,
                    "section_index": int(row["section_index"]),
                    "start_time_sec": start,
                    "end_time_sec": end,
                    "section_type": normalize_type(row["section_type"]),
                    "structure_label": row.get("structure_label") or "",
                    "lyric_evidence": row.get("lyric_evidence") or "",
                    "vocal_evidence": row.get("vocal_evidence") or "",
                    "acoustic_evidence": row.get("acoustic_evidence") or "",
                    "boundary_confidence": boundary_confidence,
                    "type_confidence": type_confidence,
                    "need_human_review": parse_bool(row.get("need_human_review")),
                    "raw": row,
                }
            )
    if rows:
        insert("auto_sections", rows)


def import_lyrics(song_id: str, path: Path) -> None:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            text = row.get("text", "").strip()
            if not text:
                continue
            rows.append(
                {
                    "song_id": song_id,
                    "line_index": index,
                    "start_time_sec": parse_time(row["start_time"]),
                    "end_time_sec": parse_time(row["end_time"]),
                    "text": text,
                    "source": "whisper_or_timed_lyrics",
                }
            )
    if rows:
        upsert("lyric_lines", rows, on_conflict="song_id,line_index")


def section_duration(path: Path) -> float:
    last = 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = max(last, parse_time(row.get("end_time") or row.get("end_time_sec")))
    return round(last, 2)


def first_number(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return None


def normalize_type(value: str) -> str:
    return value.strip().replace("-", "_")


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_time(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = value.strip()
    if ":" not in text:
        return float(text)
    minutes, seconds = text.split(":", 1)
    return int(minutes) * 60 + float(seconds)


def insert(table: str, payload: list[dict]) -> list[dict]:
    return request("POST", table, payload, {"Prefer": "return=representation"})


def upsert(table: str, payload: list[dict], on_conflict: str) -> list[dict]:
    return request(
        "POST",
        f"{table}?on_conflict={urllib.parse.quote(on_conflict)}",
        payload,
        {"Prefer": "resolution=merge-duplicates,return=representation"},
    )


def delete(table: str, filters: dict[str, str]) -> None:
    query = urllib.parse.urlencode(filters)
    request("DELETE", f"{table}?{query}", None, {"Prefer": "return=minimal"})


def request(method: str, endpoint: str, payload: list[dict] | None, extra_headers: dict[str, str]) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    command = [
        "curl",
        "-sS",
        "-X",
        method,
        "-H",
        f"apikey: {SUPABASE_KEY}",
        "-H",
        f"Authorization: Bearer {SUPABASE_KEY}",
        "-H",
        "Content-Type: application/json",
    ]
    for key, value in extra_headers.items():
        command.extend(["-H", f"{key}: {value}"])
    if payload is not None:
        command.extend(["--data-binary", json.dumps(payload, ensure_ascii=False)])
    command.append(url)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    body = completed.stdout
    if not body:
        return []
    return json.loads(body)


if __name__ == "__main__":
    main()
