#!/usr/bin/env python3
"""Build model-free VA anchor candidates from songs outside the blind test set."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
SONGS_PATH = PHASE_ROOT / "data/blind_validation/stage1/raw/reviewed_songs_allowed.json"
BOUNDARIES_PATH = (
    PHASE_ROOT / "data/blind_validation/stage1/raw/section_boundaries_allowed.json"
)
BLIND_SAMPLE_PATH = PHASE_ROOT / "data/blind_validation/stage2/blind_sample_15.csv"
TEMPLATE_PATH = PHASE_ROOT / "web/va_anchor_candidate_review_template.html"
OUTPUT_DIR = PHASE_ROOT / "outputs/human_validation/calibration_candidates"
CLIP_DIR = OUTPUT_DIR / "audio_clips"
MANIFEST_CSV = OUTPUT_DIR / "va_anchor_candidates.csv"
MANIFEST_JSON = OUTPUT_DIR / "va_anchor_candidates_manifest.json"
OUTPUT_HTML = OUTPUT_DIR / "va_anchor_candidate_review.html"


# These intervals are contained in reviewed section boundaries. The labels are
# hypotheses for human review, not ground truth and not model predictions.
CANDIDATES = [
    {
        "candidate_id": "V_LOW_01",
        "dimension": "valence",
        "proposed_level": "low",
        "song_id": "local_d6d63a144a77",
        "start_sec": 194.0,
        "end_sec": 214.0,
        "selection_basis": "消极情绪候选；取人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_LOW_02",
        "dimension": "valence",
        "proposed_level": "low",
        "song_id": "local_d8999279ff23",
        "start_sec": 70.0,
        "end_sec": 89.0,
        "selection_basis": "消极情绪候选；取人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_MID_01",
        "dimension": "valence",
        "proposed_level": "mid",
        "song_id": "local_ef6d5a9855b1",
        "start_sec": 54.0,
        "end_sec": 74.0,
        "selection_basis": "中性或正负混合情绪候选；取完整且稳定的演唱片段",
    },
    {
        "candidate_id": "V_MID_02",
        "dimension": "valence",
        "proposed_level": "mid",
        "song_id": "local_4d1ec59ae156",
        "start_sec": 46.5,
        "end_sec": 62.0,
        "selection_basis": "中性或正负混合情绪候选；取完整人工质检 section",
    },
    {
        "candidate_id": "V_HIGH_01",
        "dimension": "valence",
        "proposed_level": "high",
        "song_id": "local_dee323b18c24",
        "start_sec": 67.0,
        "end_sec": 87.0,
        "selection_basis": "积极情绪候选；取人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "V_HIGH_02",
        "dimension": "valence",
        "proposed_level": "high",
        "song_id": "local_3913ac8a270a",
        "start_sec": 54.0,
        "end_sec": 74.0,
        "selection_basis": "积极情绪候选；取人工质检边界内的代表性演唱段",
    },
    {
        "candidate_id": "A_LOW_01",
        "dimension": "arousal",
        "proposed_level": "low",
        "song_id": "local_715bd8149a37",
        "start_sec": 33.5,
        "end_sec": 50.0,
        "selection_basis": "低唤醒候选；平静、低强度且非静音的完整 section",
    },
    {
        "candidate_id": "A_LOW_02",
        "dimension": "arousal",
        "proposed_level": "low",
        "song_id": "local_e46cad906ab3",
        "start_sec": 33.5,
        "end_sec": 53.5,
        "selection_basis": "低唤醒候选；取人工质检边界内的平稳演唱段",
    },
    {
        "candidate_id": "A_MID_01",
        "dimension": "arousal",
        "proposed_level": "mid",
        "song_id": "local_bb4362dfd88c",
        "start_sec": 59.0,
        "end_sec": 79.0,
        "selection_basis": "中等唤醒候选；节奏和能量适中且持续稳定",
    },
    {
        "candidate_id": "A_MID_02",
        "dimension": "arousal",
        "proposed_level": "mid",
        "song_id": "local_de1d1e9e4570",
        "start_sec": 52.0,
        "end_sec": 72.0,
        "selection_basis": "中等唤醒候选；经基础声学强度筛查后取人工质检边界内的稳定段",
    },
    {
        "candidate_id": "A_HIGH_01",
        "dimension": "arousal",
        "proposed_level": "high",
        "song_id": "local_7c3286a38f8f",
        "start_sec": 118.5,
        "end_sec": 138.5,
        "selection_basis": "高唤醒候选；强节奏、高能量的代表性演唱段",
    },
    {
        "candidate_id": "A_HIGH_02",
        "dimension": "arousal",
        "proposed_level": "high",
        "song_id": "local_3a80df5f5397",
        "start_sec": 199.0,
        "end_sec": 219.0,
        "selection_basis": "高唤醒候选；强节奏、高能量的代表性演唱段",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_audio_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        project_relative = PROJECT_ROOT / raw_path.lstrip("/")
        if project_relative.exists():
            return project_relative
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def acoustic_descriptors(path: Path) -> dict[str, float]:
    audio, sample_rate = librosa.load(path, sr=22050, mono=True)
    if audio.size == 0:
        raise RuntimeError(f"Empty audio clip: {path}")
    rms = librosa.feature.rms(y=audio)[0]
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    mean_rms = max(float(np.mean(rms)), 1e-12)
    return {
        "rms_dbfs": 20.0 * math.log10(mean_rms),
        "onset_strength_mean": float(np.mean(onset)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "zero_crossing_rate": float(np.mean(zcr)),
    }


def load_blind_ids() -> set[str]:
    with BLIND_SAMPLE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return {row["song_id"] for row in csv.DictReader(handle)}


def main() -> None:
    songs = json.loads(SONGS_PATH.read_text(encoding="utf-8"))
    boundaries = json.loads(BOUNDARIES_PATH.read_text(encoding="utf-8"))
    songs_by_id = {row["id"]: row for row in songs}
    blind_ids = load_blind_ids()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for candidate in CANDIDATES:
        song_id = candidate["song_id"]
        if song_id in blind_ids:
            raise RuntimeError(f"Anchor candidate leaks blind-test song: {song_id}")
        song = songs_by_id.get(song_id)
        if song is None:
            raise RuntimeError(f"Unknown song_id: {song_id}")

        start_sec = float(candidate["start_sec"])
        end_sec = float(candidate["end_sec"])
        containing = [
            row
            for row in boundaries
            if row["song_id"] == song_id
            and float(row["start_time_sec"]) <= start_sec + 1e-6
            and float(row["end_time_sec"]) >= end_sec - 1e-6
        ]
        if len(containing) != 1:
            raise RuntimeError(
                f"{candidate['candidate_id']} must be inside exactly one reviewed boundary; "
                f"found {len(containing)}"
            )

        source = resolve_audio_path(song["audio_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        duration = end_sec - start_sec
        clip_name = f"{candidate['candidate_id'].lower()}_{song_id}.mp3"
        clip_path = CLIP_DIR / clip_name
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_sec:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(clip_path),
            ],
            check=True,
        )
        measured_duration = probe_duration(clip_path)
        duration_error = measured_duration - duration
        if abs(duration_error) > 0.08:
            failures.append(
                f"{candidate['candidate_id']}: duration error {duration_error:.6f}s"
            )

        descriptors = acoustic_descriptors(clip_path)
        rows.append(
            {
                **candidate,
                "title": song["title"],
                "artist": song["artist"],
                "section_order": int(containing[0]["section_order"]),
                "duration_sec": round(duration, 3),
                "measured_duration_sec": round(measured_duration, 6),
                "duration_error_sec": round(duration_error, 6),
                "source_audio": str(source.relative_to(PROJECT_ROOT)),
                "clip_relative_path": f"audio_clips/{clip_name}",
                "clip_sha256": sha256(clip_path),
                **{key: round(value, 6) for key, value in descriptors.items()},
                "status": "pass" if abs(duration_error) <= 0.08 else "fail",
            }
        )

    fieldnames = list(rows[0])
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at_utc": created_at,
        "purpose": "Human confirmation of model-free cross-song VA scale anchors",
        "status": "candidate_only_not_frozen",
        "blind_test_song_count": len(blind_ids),
        "candidate_count": len(rows),
        "candidate_song_count": len({row["song_id"] for row in rows}),
        "blind_test_overlap_count": len(blind_ids & {str(row["song_id"]) for row in rows}),
        "teacher_predictions_used": False,
        "reviewed_boundaries_used": True,
        "duration_tolerance_sec": 0.08,
        "failures": failures,
        "csv_path": str(MANIFEST_CSV),
        "html_path": str(OUTPUT_HTML),
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("__DATA_JSON__") != 1:
        raise RuntimeError("Template must contain exactly one __DATA_JSON__ placeholder")
    html = template.replace(
        "__DATA_JSON__",
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Candidate audio audit failed")


if __name__ == "__main__":
    main()
