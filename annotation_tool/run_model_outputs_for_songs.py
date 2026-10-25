#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "functional_sections"
DEFAULT_SONGS_DIR = ROOT / "songs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model section outputs for local songs before database import.")
    parser.add_argument("--songs-dir", default=str(DEFAULT_SONGS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--song-list", default="", help="Optional newline-delimited list of audio filenames to process.")
    parser.add_argument(
        "--filename-order",
        choices=["artist-title", "title-artist"],
        default="artist-title",
        help="How to parse filenames separated by '-'.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N songs. 0 means no limit.")
    parser.add_argument("--only-missing", action="store_true", default=True, help="Skip songs that already have section output.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing section outputs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing later songs if one song fails.")
    parser.add_argument("--no-lyric-search", action="store_true", help="Skip external lyric search and use transcript fallback.")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--write-manifest", default="", help="Optional CSV manifest for generated ready songs.")
    parser.add_argument("--import-after-each", action="store_true", help="Import each successful model output into Supabase immediately.")
    args = parser.parse_args()

    songs_dir = Path(args.songs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    failed = []
    failed_imports = []
    for audio in iter_audio_files(songs_dir, args.song_list):
        artist, title = filename_metadata(audio, args.filename_order)
        song_id = song_id_for(artist, title)
        sections_csv = output_dir / f"{song_id}_agentic_sections.csv"
        lyrics_csv = output_dir / f"{audio.stem}_whisper_segments.csv"
        if sections_csv.is_file() and args.only_missing and not args.force:
            row = manifest_row(song_id, title, artist, audio, sections_csv, lyrics_csv)
            processed.append(row)
            if args.import_after_each:
                try:
                    import_ready_song(row, output_dir / f"{song_id}_ready_manifest.csv")
                except subprocess.CalledProcessError as error:
                    failed_imports.append((audio.name, str(error)))
                    print(f"failed import {audio.name}: {error}", flush=True)
                    if not args.continue_on_error:
                        raise
            if args.limit and len(processed) >= args.limit:
                break
            continue
        command = [
            sys.executable,
            str(ROOT / "hybrid_structure" / "agentic" / "orchestrator.py"),
            "--audio",
            str(audio),
            "--title",
            title,
            "--track-name",
            title,
            "--artist-name",
            artist,
            "--song-id",
            song_id,
            "--work-dir",
            str(output_dir),
            "--output",
            str(sections_csv),
            "--reuse-transcript",
            "--candidate-output",
            str(output_dir / f"{song_id}_boundary_candidates.csv"),
            "--model",
            args.model,
            "--language",
            args.language,
        ]
        if args.no_lyric_search:
            command.append("--no-lyric-search")
        print(f"running model output {song_id} {artist} - {title}", flush=True)
        try:
            env = os.environ.copy()
            env.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache_model_run")
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as error:
            failed.append((audio.name, str(error)))
            print(f"failed model output {audio.name}: {error}", flush=True)
            if args.continue_on_error:
                continue
            raise
        row = manifest_row(song_id, title, artist, audio, sections_csv, lyrics_csv)
        processed.append(row)
        if args.import_after_each:
            try:
                import_ready_song(row, output_dir / f"{song_id}_ready_manifest.csv")
            except subprocess.CalledProcessError as error:
                failed_imports.append((audio.name, str(error)))
                print(f"failed import {audio.name}: {error}", flush=True)
                if not args.continue_on_error:
                    raise
        if args.limit and len(processed) >= args.limit:
            break

    if args.write_manifest:
        write_manifest(Path(args.write_manifest), processed)
    print(f"generated {len(processed)} model outputs")
    if failed:
        print(f"failed {len(failed)} model outputs")
        for name, error in failed:
            print(f"failed: {name}: {error}")
    if failed_imports:
        print(f"failed {len(failed_imports)} imports")
        for name, error in failed_imports:
            print(f"failed import: {name}: {error}")


def manifest_row(song_id: str, title: str, artist: str, audio: Path, sections_csv: Path, lyrics_csv: Path) -> dict[str, str]:
    return {
        "id": song_id,
        "title": title,
        "artist": artist,
        "audio_path": f"/songs/{audio.name}",
        "sections_csv": str(sections_csv.relative_to(ROOT)),
        "lyrics_csv": str(lyrics_csv.relative_to(ROOT)) if lyrics_csv.is_file() else "",
        "pipeline_version": "agentic_nlp_acoustic",
    }


def iter_audio_files(songs_dir: Path, song_list: str) -> list[Path]:
    if not song_list:
        return sorted(songs_dir.glob("*.mp3"))
    paths = []
    with Path(song_list).open(encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            path = songs_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def filename_metadata(path: Path, filename_order: str) -> tuple[str, str]:
    stem = re.sub(r"^\d+[.]", "", path.stem).strip()
    parts = re.split(r"\s*-\s*", stem, maxsplit=1)
    if len(parts) != 2:
        return "unknown", stem
    if filename_order == "title-artist":
        title, artist = parts
    else:
        artist, title = parts
    if artist == "起风了" and title == "买辣椒也用券":
        return title, artist
    return artist.replace("_", " / "), title


def song_id_for(artist: str, title: str) -> str:
    return f"local_{hashlib.sha1(f'{artist}|{title}'.encode('utf-8')).hexdigest()[:12]}"


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "title", "artist", "audio_path", "sections_csv", "lyrics_csv", "pipeline_version"],
        )
        writer.writeheader()
        writer.writerows(rows)


def import_ready_song(row: dict[str, str], manifest_path: Path) -> None:
    write_manifest(manifest_path, [row])
    command = [
        sys.executable,
        str(ROOT / "annotation_tool" / "import_local_results.py"),
        "--manifest",
        str(manifest_path),
        "--skip-song-discovery",
    ]
    last_error = None
    for attempt in range(1, 4):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            print(f"import attempt {attempt}/3 failed for {row['id']}", flush=True)
    raise last_error


if __name__ == "__main__":
    main()
