#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
    parser.add_argument("--songs-dir", default="songs", help="Directory of local MP3 files to register.")
    parser.add_argument("--skip-song-discovery", action="store_true", help="Only import songs listed in the manifest.")
    parser.add_argument("--dedupe-existing", action="store_true", help="Delete duplicate database song rows, preserving reviewed rows first.")
    parser.add_argument("--register-pending", action="store_true", help="Allow inserting songs without model output. Disabled by default for QC-ready imports.")
    parser.add_argument(
        "--force-model-output",
        action="store_true",
        help="For existing songs, insert a new latest model run and set the song back to pending_review without deleting human history.",
    )
    args = parser.parse_args()

    songs = read_manifest(Path(args.manifest))
    if not args.skip_song_discovery:
        songs.extend(discover_audio_songs(Path(args.songs_dir), songs))
    existing_songs = get("songs", "select=id,title,artist,audio_path,status")
    if args.dedupe_existing:
        existing_songs = dedupe_existing_songs(existing_songs)
    existing_by_id = {song["id"]: song for song in existing_songs}
    existing_by_audio_path = {song.get("audio_path"): song for song in existing_songs if song.get("audio_path")}

    for song in songs:
        sections_path = Path(song["sections_csv"]) if song["sections_csv"] else None
        existing_song = existing_by_id.get(song["id"]) or existing_by_audio_path.get(song["audio_path"])
        if not args.replace and existing_song:
            import_missing_model_output(existing_song["id"], song, sections_path, force=args.force_model_output)
            continue
        if args.replace:
            delete("songs", {"id": f"eq.{song['id']}"})
        if not sections_path or not sections_path.is_file():
            if not args.register_pending:
                print(f"skip song without model output {song['id']}")
                continue
            duration = audio_duration(Path(song["local_audio_path"]))
        else:
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
        if not sections_path or not sections_path.is_file():
            print(f"registered pending song without sections {song['id']}")
            continue
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
        lyrics_path = Path(song["lyrics_csv"]) if song["lyrics_csv"] else None
        if lyrics_path and lyrics_path.is_file():
            import_lyrics(song["id"], lyrics_path)
        print(f"imported {song['id']} run={run['id']}")


def import_missing_model_output(song_id: str, song: dict[str, str], sections_path: Path | None, *, force: bool = False) -> None:
    if not sections_path or not sections_path.is_file():
        print(f"preserved existing {song_id}")
        return
    if has_auto_sections(song_id) and not force:
        print(f"preserved existing {song_id}")
        return
    update("songs", {"status": "pending_review"}, {"id": f"eq.{song_id}"})
    run = insert(
        "model_runs",
        [
            {
                "song_id": song_id,
                "model_version": "rule_pipeline",
                "pipeline_version": song["pipeline_version"],
                "params": {"sections_csv": song["sections_csv"]},
                "notes": "Imported from local pipeline output.",
            }
        ],
    )[0]
    import_sections(song_id, run["id"], sections_path)
    lyrics_path = Path(song["lyrics_csv"]) if song["lyrics_csv"] else None
    if lyrics_path and lyrics_path.is_file():
        if force:
            delete("lyric_lines", {"song_id": f"eq.{song_id}"})
        import_lyrics(song_id, lyrics_path)
    print(f"imported {'new' if force else 'missing'} sections {song_id} run={run['id']}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("id")]
    for row in rows:
        row["local_audio_path"] = str(Path("songs") / Path(row["audio_path"]).name)
    return rows


def discover_audio_songs(songs_dir: Path, manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    known_audio_paths = {song["audio_path"] for song in manifest}
    seen_content_hashes: set[str] = set()
    seen_identities = {(normalize_identity(song["artist"]), normalize_identity(song["title"])) for song in manifest}
    songs = []
    for path in sorted(songs_dir.glob("*.mp3")):
        audio_path = f"/songs/{path.name}"
        if audio_path in known_audio_paths:
            continue
        content_hash = file_hash(path)
        artist, title = filename_metadata(path)
        identity = (normalize_identity(artist), normalize_identity(title))
        if content_hash in seen_content_hashes or identity in seen_identities:
            print(f"skip duplicate audio {path}")
            continue
        seen_content_hashes.add(content_hash)
        seen_identities.add(identity)
        song_id = f"local_{hashlib.sha1(f'{artist}|{title}'.encode('utf-8')).hexdigest()[:12]}"
        sections_csv = Path("outputs/functional_sections") / f"{song_id}_agentic_sections.csv"
        lyrics_csv = Path("outputs/functional_sections") / f"{path.stem}_whisper_segments.csv"
        songs.append(
            {
                "id": song_id,
                "title": title,
                "artist": artist,
                "audio_path": audio_path,
                "local_audio_path": str(path),
                "sections_csv": str(sections_csv) if sections_csv.is_file() else "",
                "lyrics_csv": str(lyrics_csv) if lyrics_csv.is_file() else "",
                "pipeline_version": "agentic_nlp_acoustic",
            }
        )
    return songs


def filename_metadata(path: Path) -> tuple[str, str]:
    stem = re.sub(r"^\d+[.]", "", path.stem).strip()
    parts = re.split(r"\s*-\s*", stem, maxsplit=1)
    if len(parts) != 2:
        return "unknown", stem
    artist, title = parts
    if artist == "起风了" and title == "买辣椒也用券":
        return title, artist
    return artist.replace("_", " / "), title


def normalize_identity(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_duration(path: Path) -> float:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return round(float(completed.stdout.strip()), 2)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


def dedupe_existing_songs(songs: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    kept_by_audio_path: dict[str, dict[str, str]] = {}
    kept_by_identity: dict[str, dict[str, str]] = {}
    for song in sorted(songs, key=lambda item: (item.get("status") != "reviewed", item["id"])):
        audio_path = normalize_identity(song.get("audio_path") or "")
        identity = f"{normalize_identity(song.get('artist') or '')}|{normalize_identity(song.get('title') or '')}"
        duplicate_of = kept_by_audio_path.get(audio_path) or (kept_by_identity.get(identity) if identity != "|" else None)
        if duplicate_of:
            delete("songs", {"id": f"eq.{song['id']}"})
            print(f"deleted duplicate database song {song['id']} kept={duplicate_of['id']}")
            continue
        kept.append(song)
        if audio_path:
            kept_by_audio_path[audio_path] = song
        if identity != "|":
            kept_by_identity[identity] = song
    return kept


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


def has_auto_sections(song_id: str) -> bool:
    query = f"select=id&song_id=eq.{urllib.parse.quote(song_id)}&limit=1"
    return bool(get("auto_sections", query))


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
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "intern": "interlude",
        "instrumental": "interlude",
        "instrumental_bridge": "interlude",
        "间奏": "interlude",
    }
    return aliases.get(normalized, normalized)


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


def update(table: str, payload: dict, filters: dict[str, str]) -> list[dict]:
    query = urllib.parse.urlencode(filters)
    return request("PATCH", f"{table}?{query}", payload, {"Prefer": "return=representation"})


def delete(table: str, filters: dict[str, str]) -> None:
    query = urllib.parse.urlencode(filters)
    request("DELETE", f"{table}?{query}", None, {"Prefer": "return=minimal"})


def get(table: str, query: str) -> list[dict]:
    return request("GET", f"{table}?{query}", None, {})


def request(method: str, endpoint: str, payload: list[dict] | dict | None, extra_headers: dict[str, str]) -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    command = [
        "curl",
        "-sS",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
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
