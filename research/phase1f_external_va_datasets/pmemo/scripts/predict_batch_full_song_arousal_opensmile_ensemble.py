#!/usr/bin/env python3
"""Batch complete-song Arousal prediction with the OpenSMILE ensemble."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("predict_full_song_arousal_opensmile_ensemble.py")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}


def safe_stem(path: Path) -> str:
    keep = []
    for ch in path.stem:
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "audio"


def discover_audio(audio_dir: Path) -> list[Path]:
    return sorted(p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def read_manifest(path: Path) -> list[Path]:
    import pandas as pd

    df = pd.read_csv(path)
    if "audio_path" not in df.columns:
        raise ValueError("manifest must include audio_path column")
    return [Path(str(x)) for x in df["audio_path"].dropna().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    audio_paths = discover_audio(args.audio_dir) if args.audio_dir else read_manifest(args.manifest)
    if args.limit:
        audio_paths = audio_paths[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, audio in enumerate(audio_paths, start=1):
        out = args.output_dir / f"{safe_stem(audio)}_arousal_opensmile_ensemble.csv"
        cmd = [
            str(Path(".venv/bin/python")),
            str(SCRIPT),
            "--audio",
            str(audio),
            "--output",
            str(out),
            "--device",
            args.device,
        ]
        try:
            subprocess.run(cmd, check=True)
            status = "ok"
            error = ""
            print(f"[{i}/{len(audio_paths)}] ok {audio} -> {out}", flush=True)
        except Exception as exc:  # noqa: BLE001 - batch report should preserve failures
            status = "failed"
            error = repr(exc)
            print(f"[{i}/{len(audio_paths)}] failed {audio}: {error}", flush=True)
        rows.append({"audio_path": str(audio), "output_path": str(out), "status": status, "error": error})

    report = args.output_dir / "batch_report.csv"
    with report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "output_path", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"batch_report={report}")


if __name__ == "__main__":
    main()
