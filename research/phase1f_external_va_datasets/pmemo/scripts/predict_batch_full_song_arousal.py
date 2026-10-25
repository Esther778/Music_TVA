#!/usr/bin/env python3
"""Batch complete-song Arousal prediction."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = ROOT / "scripts" / "predict_full_song_arousal.py"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def slug(path: Path) -> str:
    safe = path.stem
    for char in '/\\:*?"<>| ':
        safe = safe.replace(char, "_")
    return safe


def collect_audio(audio_dir: Path | None, manifest: Path | None) -> list[Path]:
    if manifest:
        rows = []
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "audio_path" not in reader.fieldnames:
                raise SystemExit("Manifest must include an audio_path column.")
            for row in reader:
                rows.append(Path(row["audio_path"]))
        return rows
    if not audio_dir:
        raise SystemExit("Provide --audio-dir or --manifest.")
    return sorted([p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    audios = collect_audio(args.audio_dir, args.manifest)
    if args.limit:
        audios = audios[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, audio in enumerate(audios, start=1):
        output = args.output_dir / f"{slug(audio)}_arousal.csv"
        if output.exists() and args.skip_existing:
            status = "skipped"
            error = ""
        else:
            command = [
                sys.executable,
                str(PREDICT_SCRIPT),
                "--audio",
                str(audio),
                "--output",
                str(output),
                "--device",
                args.device,
            ]
            try:
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                status = "ok"
                error = completed.stderr.strip()
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = (exc.stderr or exc.stdout or str(exc)).strip()
        rows.append({"audio_path": str(audio), "output_path": str(output), "status": status, "error": error})
        print(f"[{index}/{len(audios)}] {status} {audio} -> {output}", flush=True)

    report = args.output_dir / "batch_report.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "output_path", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"batch_report={report}")


if __name__ == "__main__":
    main()
