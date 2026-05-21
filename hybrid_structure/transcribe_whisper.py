#!/usr/bin/env python3
"""Transcribe a song with faster-whisper and write timestamped lyric segments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start_time: float
    end_time: float
    text: str
    avg_logprob: float
    no_speech_prob: float


def format_timestamp(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:05.2f}"


def transcribe(audio_path: Path, model_size: str, language: str | None, vad_filter: bool) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 700},
    )
    rows: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        rows.append(
            TranscriptSegment(
                start_time=round(float(segment.start), 2),
                end_time=round(float(segment.end), 2),
                text=text,
                avg_logprob=round(float(segment.avg_logprob), 4),
                no_speech_prob=round(float(segment.no_speech_prob), 4),
            )
        )
    return rows


def write_csv(path: Path, rows: list[TranscriptSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start_time", "end_time", "text", "avg_logprob", "no_speech_prob"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "start_time": format_timestamp(row.start_time),
                    "end_time": format_timestamp(row.end_time),
                    "text": row.text,
                    "avg_logprob": row.avg_logprob,
                    "no_speech_prob": row.no_speech_prob,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a song into timestamped lyric segments.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="small", help="faster-whisper model size, e.g. tiny/base/small/medium.")
    parser.add_argument("--language", default="zh", help="Language code, or empty string for auto.")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filtering. Often better for singing.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    rows = transcribe(Path(args.audio), args.model, args.language or None, vad_filter=not args.no_vad)
    write_csv(Path(args.output), rows)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        print(f"{format_timestamp(row.start_time)}-{format_timestamp(row.end_time)} {row.text}")


if __name__ == "__main__":
    main()
