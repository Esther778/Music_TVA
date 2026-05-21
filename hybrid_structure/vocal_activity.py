#!/usr/bin/env python3
"""Detect low-vocal regions from a separated Demucs vocals stem."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np

import segment_hybrid as base


def detect_low_vocal_regions(
    vocals_path: Path,
    min_duration: float = 5.0,
    smooth_seconds: float = 1.0,
    percentile: float = 28.0,
    peak_ratio: float = 0.055,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    y, sr = librosa.load(vocals_path, sr=22050, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length).reshape(-1)
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    smooth_frames = max(3, int(round(smooth_seconds * sr / hop_length)))
    kernel = np.ones(smooth_frames) / smooth_frames
    smooth = np.convolve(rms, kernel, mode="same")
    threshold = max(float(np.percentile(smooth, percentile)), float(np.percentile(smooth, 95) * peak_ratio))

    inactive = smooth < threshold
    regions: list[tuple[float, float]] = []
    start: float | None = None
    for time, is_inactive in zip(times, inactive):
        current = float(time)
        if is_inactive and start is None:
            start = current
        elif not is_inactive and start is not None:
            if current - start >= min_duration:
                regions.append((round(start, 2), round(current, 2)))
            start = None
    if start is not None and float(times[-1]) - start >= min_duration:
        regions.append((round(start, 2), round(float(times[-1]), 2)))

    stats = {
        "threshold": threshold,
        "p28": float(np.percentile(smooth, 28)),
        "p50": float(np.percentile(smooth, 50)),
        "p95": float(np.percentile(smooth, 95)),
        "duration": float(librosa.get_duration(y=y, sr=sr)),
    }
    return regions, stats


def write_csv(path: Path, song_id: str, regions: list[tuple[float, float]], stats: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["song_id", "region_index", "start_time", "end_time", "duration", "reason", "threshold"],
        )
        writer.writeheader()
        for index, (start, end) in enumerate(regions, start=1):
            writer.writerow(
                {
                    "song_id": song_id,
                    "region_index": index,
                    "start_time": base.format_timestamp(start),
                    "end_time": base.format_timestamp(end),
                    "duration": base.format_timestamp(end - start),
                    "reason": "low separated-vocal RMS",
                    "threshold": round(stats["threshold"], 8),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect low-vocal regions from a Demucs vocals stem.")
    parser.add_argument("--vocals", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-duration", type=float, default=5.0)
    args = parser.parse_args()

    regions, stats = detect_low_vocal_regions(Path(args.vocals), min_duration=args.min_duration)
    write_csv(Path(args.output), args.song_id, regions, stats)
    print(f"Vocals: {args.vocals}")
    print(f"Output: {args.output}")
    print(
        "threshold="
        f"{stats['threshold']:.8f} p28={stats['p28']:.8f} p50={stats['p50']:.8f} p95={stats['p95']:.8f}"
    )
    for index, (start, end) in enumerate(regions, start=1):
        print(f"{index:02d} {base.format_timestamp(start)}-{base.format_timestamp(end)} {base.format_timestamp(end - start)}")


if __name__ == "__main__":
    main()
