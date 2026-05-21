#!/usr/bin/env python3
"""MSAF-based music section segmentation for MER/VA analysis.

This script intentionally uses MSAF as the only segmentation backend. The old
librosa-based pipeline has been removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Section:
    section_id: int
    start_time: float
    end_time: float
    duration: float
    label: str
    structure_label: str = ""


@dataclass
class LyricLine:
    time: float
    text: str


def round_time(value: float) -> float:
    """Keep timestamps compact and easy to align with VA time series."""
    return round(float(value), 2)


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS.xx for human-readable CSV review."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def parse_timestamp(value: str) -> float:
    """Parse S, M:S, or H:M:S timestamps into seconds."""
    value = value.strip()
    if not value:
        raise ValueError("empty timestamp")
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported timestamp: {value}")


def require_msaf():
    """Import MSAF with a clear install message for this project environment."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
        # MSAF 0.1.80 expects scipy.inf, which was removed in newer SciPy.
        # Patch it before importing MSAF so the original MSAF backend can run.
        import numpy as np
        import scipy

        if not hasattr(scipy, "inf"):
            scipy.inf = np.inf  # type: ignore[attr-defined]
        if not hasattr(scipy.signal, "gaussian"):
            scipy.signal.gaussian = scipy.signal.windows.gaussian  # type: ignore[attr-defined]
        import msaf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "MSAF is not installed in this Python environment.\n"
            "Install it with:\n"
            "  python -m pip install msaf\n"
            "or, for this project virtualenv:\n"
            "  .venv/bin/python -m pip install msaf"
        ) from exc
    return msaf


def cleanup_msaf_artifacts(input_path: Path) -> None:
    """Remove MSAF side files so the project output stays focused."""
    Path(".features_msaf_tmp.json").unlink(missing_ok=True)
    jams_path = Path("estimations") / f"{input_path.stem}.jams"
    jams_path.unlink(missing_ok=True)
    estimations_dir = Path("estimations")
    if estimations_dir.exists() and not any(estimations_dir.iterdir()):
        estimations_dir.rmdir()
    shutil.rmtree(".matplotlib-cache", ignore_errors=True)


def load_lyrics(lyrics_path: Path | None) -> list[LyricLine]:
    """Load timestamped lyrics from LRC or CSV.

    Lyrics are optional soft evidence. A lyric line timestamp marks vocal text
    onset, not necessarily the musical section boundary itself, so later code
    searches nearby acoustic novelty peaks instead of cutting exactly there.
    """
    if lyrics_path is None:
        return []
    if not lyrics_path.exists():
        raise SystemExit(f"Lyrics file not found: {lyrics_path}")

    if lyrics_path.suffix.lower() == ".lrc":
        lines: list[LyricLine] = []
        pattern = re.compile(r"\[(\d{1,2}:\d{2}(?:\.\d+)?)\](.*)")
        for raw_line in lyrics_path.read_text(encoding="utf-8").splitlines():
            matches = pattern.findall(raw_line)
            if not matches:
                continue
            for timestamp, text in matches:
                text = text.strip()
                if text:
                    lines.append(LyricLine(round_time(parse_timestamp(timestamp)), text))
        return sorted(lines, key=lambda item: item.time)

    if lyrics_path.suffix.lower() == ".csv":
        lines = []
        with lyrics_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            time_field = next((name for name in ("time", "start_time", "start", "timestamp") if name in reader.fieldnames), None)
            text_field = next((name for name in ("text", "lyric", "lyrics", "line") if name in reader.fieldnames), None)
            if time_field is None or text_field is None:
                raise SystemExit("Lyrics CSV must include time/start_time and text/lyric columns.")
            for row in reader:
                text = (row.get(text_field) or "").strip()
                if text:
                    lines.append(LyricLine(round_time(parse_timestamp(row[time_field])), text))
        return sorted(lines, key=lambda item: item.time)

    raise SystemExit("Lyrics file must be .lrc or .csv.")


def normalize_lyric_text(text: str) -> str:
    """Normalize lyric text for simple repetition cues."""
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def lyric_boundary_candidates(lyrics: list[LyricLine], song_duration: float) -> list[float]:
    """Return likely section-related lyric transition times.

    The candidates include vocal entry after an instrumental intro, unusually
    long lyric gaps, and repeated line starts. These are intentionally broad
    hints; acoustic novelty still decides the final boundary.
    """
    if len(lyrics) < 2:
        return []

    import numpy as np

    candidates: set[float] = set()
    first_vocal = lyrics[0].time
    if 5.0 <= first_vocal <= song_duration * 0.18:
        candidates.add(first_vocal)

    gaps = np.diff([line.time for line in lyrics])
    positive_gaps = [gap for gap in gaps if gap > 0]
    if positive_gaps:
        median_gap = float(np.median(positive_gaps))
        gap_threshold = max(4.0, median_gap * 1.8)
        for previous, current in zip(lyrics[:-1], lyrics[1:]):
            if current.time - previous.time >= gap_threshold:
                candidates.add(current.time)

    seen: dict[str, float] = {}
    for line in lyrics:
        normalized = normalize_lyric_text(line.text)
        if len(normalized) < 4:
            continue
        if normalized in seen and line.time - seen[normalized] > 20.0:
            candidates.add(line.time)
        else:
            seen[normalized] = line.time

    return sorted(
        round_time(time)
        for time in candidates
        if 3.0 < time < song_duration - 3.0
    )


def normalize_boundaries(boundaries: Iterable[float]) -> list[float]:
    """Sort, de-duplicate, and clamp negative boundaries from MSAF."""
    clean = sorted({round_time(max(0.0, float(boundary))) for boundary in boundaries})
    if not clean:
        raise ValueError("MSAF returned no section boundaries.")
    if clean[0] != 0.0:
        clean.insert(0, 0.0)
    return clean


def msaf_labels_to_names(labels: Iterable[object], count: int) -> list[str]:
    """Convert MSAF labels to stable CSV labels.

    MSAF labels can be numeric cluster IDs or strings depending on the selected
    algorithm. For MER/VA alignment, we preserve useful label repetition as
    A/B/C... when possible. If labels are missing, we use Segment_N.
    """
    raw_labels = list(labels)
    if len(raw_labels) < count:
        return [f"Segment_{index}" for index in range(1, count + 1)]

    mapping: dict[object, str] = {}
    next_letter = ord("A")
    names: list[str] = []
    for label in raw_labels[:count]:
        if label not in mapping:
            if next_letter <= ord("Z"):
                mapping[label] = chr(next_letter)
                next_letter += 1
            else:
                mapping[label] = f"Segment_{len(mapping) + 1}"
        names.append(mapping[label])
    return names


def merge_short_boundaries(boundaries: list[float], min_section_seconds: float) -> list[float]:
    """Merge tiny MSAF edge fragments that are not useful for VA alignment."""
    if len(boundaries) <= 2:
        return boundaries

    merged = [boundaries[0]]
    for boundary in boundaries[1:-1]:
        if boundary - merged[-1] >= min_section_seconds:
            merged.append(boundary)

    if boundaries[-1] - merged[-1] < min_section_seconds and len(merged) > 1:
        merged.pop()
    merged.append(boundaries[-1])
    return merged


def make_sections(boundaries: list[float], labels: Iterable[object], min_section_seconds: float) -> list[Section]:
    """Convert MSAF boundaries into section rows.

    Each row can later be joined to frame-level or window-level VA predictions
    where section.start_time <= va_time < section.end_time.
    """
    clean = merge_short_boundaries(normalize_boundaries(boundaries), min_section_seconds)
    section_count = max(0, len(clean) - 1)
    label_names = msaf_labels_to_names(labels, section_count)

    sections: list[Section] = []
    for index, (start, end) in enumerate(zip(clean[:-1], clean[1:]), start=1):
        start = round_time(start)
        end = round_time(end)
        if end <= start:
            continue
        sections.append(
            Section(
                section_id=index,
                start_time=start,
                end_time=end,
                duration=round_time(end - start),
                label=label_names[index - 1],
            )
        )
    return sections


def segment_with_msaf(
    input_path: Path,
    boundaries_id: str,
    labels_id: str,
    min_section_seconds: float,
) -> list[Section]:
    """Run MSAF and return normalized section rows."""
    msaf = require_msaf()
    boundaries, labels = msaf.process(
        str(input_path),
        boundaries_id=boundaries_id,
        labels_id=labels_id,
    )
    cleanup_msaf_artifacts(input_path)
    print(f"Segmentation backend: MSAF ({boundaries_id}/{labels_id})")
    return make_sections(boundaries, labels, min_section_seconds)


def load_analysis_features(input_path: Path) -> dict[str, object]:
    """Load audio and compute MIR features used by semantic post-processing."""
    import librosa
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    y, sr = librosa.load(input_path, sr=22050, mono=True)
    hop_length = 512
    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length).reshape(1, -1)
    harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length).reshape(1, -1)
    percussive_rms = librosa.feature.rms(y=percussive, hop_length=hop_length).reshape(1, -1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(1, -1)
    harmonic_ratio = harmonic_rms / (rms + 1e-8)
    percussive_ratio = percussive_rms / (rms + 1e-8)
    vocal_proxy = harmonic_ratio * (1.0 - np.clip(percussive_ratio, 0.0, 1.0))
    features = np.vstack([chroma, mfcc, rms, centroid, onset, harmonic_ratio, percussive_ratio, vocal_proxy])
    scaled = StandardScaler().fit_transform(features.T)
    novelty = np.r_[0.0, np.linalg.norm(np.diff(scaled, axis=0), axis=1)]
    times = librosa.frames_to_time(np.arange(features.shape[1]), sr=sr, hop_length=hop_length)
    return {
        "y": y,
        "sr": sr,
        "times": times,
        "features": features,
        "novelty": novelty,
        "rms_curve": rms.reshape(-1),
        "onset_curve": onset.reshape(-1),
        "vocal_proxy_curve": vocal_proxy.reshape(-1),
        "harmonic_ratio_curve": harmonic_ratio.reshape(-1),
        "percussive_ratio_curve": percussive_ratio.reshape(-1),
    }


def energy_for_sections(analysis: dict[str, object], sections: list[Section]) -> list[float]:
    """Compute RMS energy per section for chorus/bridge heuristics."""
    import numpy as np

    y = analysis["y"]
    sr = analysis["sr"]
    energies: list[float] = []
    for section in sections:
        start = int(section.start_time * sr)
        end = int(section.end_time * sr)
        energies.append(float(np.sqrt(np.mean(y[start:end] ** 2))) if end > start else 0.0)
    return energies


def frame_curve_for_sections(analysis: dict[str, object], sections: list[Section], curve_name: str) -> list[float]:
    """Average a frame-level curve inside each candidate section."""
    import numpy as np

    times = analysis["times"]
    curve = analysis[curve_name]
    values: list[float] = []
    for section in sections:
        mask = (times >= section.start_time) & (times < section.end_time)
        values.append(float(np.mean(curve[mask])) if np.any(mask) else 0.0)
    return values


def lyric_stats_for_sections(lyrics: list[LyricLine], sections: list[Section]) -> list[dict[str, float]]:
    """Summarize lyric density and repeated text inside each section."""
    normalized_counts: dict[str, int] = {}
    for line in lyrics:
        normalized = normalize_lyric_text(line.text)
        if len(normalized) >= 4:
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1

    stats: list[dict[str, float]] = []
    for section in sections:
        lines = [line for line in lyrics if section.start_time <= line.time < section.end_time]
        normalized = [normalize_lyric_text(line.text) for line in lines]
        repeated = [text for text in normalized if len(text) >= 4 and normalized_counts.get(text, 0) > 1]
        char_count = sum(len(text) for text in normalized)
        stats.append(
            {
                "line_count": float(len(lines)),
                "char_count": float(char_count),
                "repeat_count": float(len(repeated)),
                "repeat_ratio": float(len(repeated) / len(lines)) if lines else 0.0,
            }
        )
    return stats


def feature_vectors_for_sections(analysis: dict[str, object], sections: list[Section]) -> list[object]:
    """Average chroma/MFCC/energy descriptors per section for repetition estimates."""
    import numpy as np

    times = analysis["times"]
    features = analysis["features"]
    vectors: list[object] = []
    for section in sections:
        mask = (times >= section.start_time) & (times < section.end_time)
        if np.any(mask):
            vectors.append(np.mean(features[:, mask], axis=1))
        else:
            vectors.append(np.zeros(features.shape[0]))
    return vectors


def section_has_lyrics(lyrics: list[LyricLine], section: Section) -> bool:
    """Return whether timestamped lyric lines fall inside a section."""
    return any(section.start_time <= line.time < section.end_time for line in lyrics)


def best_internal_boundary(
    analysis: dict[str, object],
    section: Section,
    min_section_seconds: float,
    novelty_floor: float,
    lyric_boundaries: list[float] | None = None,
    lyric_boundary_tolerance: float = 2.5,
    lyric_boundary_weight: float = 0.35,
) -> float | None:
    """Find a strong novelty peak inside an over-long MSAF section.

    When timestamped lyrics are available, lyric transitions softly boost
    nearby acoustic novelty peaks. This accounts for the common offset where
    arrangement/melody changes slightly before or after the sung lyric line.
    """
    import numpy as np
    from scipy.signal import find_peaks

    times = analysis["times"]
    novelty = analysis["novelty"]
    start = section.start_time + min_section_seconds
    end = section.end_time - min_section_seconds
    if end <= start:
        return None

    indices = np.flatnonzero((times >= start) & (times <= end))
    if len(indices) < 3:
        return None

    local = novelty[indices]
    peaks, _properties = find_peaks(local, prominence=max(0.1, float(np.std(local)) * 0.4))
    if len(peaks) == 0:
        return None

    scores = local[peaks].astype(float)
    if lyric_boundaries:
        lyric_array = np.array(
            [
                time
                for time in lyric_boundaries
                if section.start_time + min_section_seconds <= time <= section.end_time - min_section_seconds
            ]
        )
        if len(lyric_array) > 0:
            peak_times = times[indices[peaks]]
            distances = np.min(np.abs(peak_times[:, None] - lyric_array[None, :]), axis=1)
            proximity = np.exp(-(distances / max(lyric_boundary_tolerance, 0.1)) ** 2)
            scores = scores + lyric_boundary_weight * max(float(np.std(local)), 0.1) * proximity

    best = int(peaks[int(np.argmax(scores))])
    if local[best] < novelty_floor:
        return None
    return round_time(float(times[indices[best]]))


def refine_boundaries_by_novelty(
    analysis: dict[str, object],
    sections: list[Section],
    min_section_seconds: float,
    long_section_factor: float,
    lyric_boundaries: list[float] | None = None,
    lyric_boundary_tolerance: float = 2.5,
    lyric_boundary_weight: float = 0.35,
) -> list[Section]:
    """Add internal boundaries only when a region is long for this song."""
    import numpy as np

    if len(sections) < 3:
        return sections

    semantic_min_seconds = max(min_section_seconds, 8.0)
    median_duration = float(np.median([section.duration for section in sections]))
    long_threshold = max(semantic_min_seconds * 2.0, median_duration * long_section_factor)
    novelty_floor = float(np.percentile(analysis["novelty"], 78))

    clean = sorted({sections[0].start_time, *(section.end_time for section in sections)})
    for _pass in range(3):
        changed = False
        next_boundaries = [clean[0]]
        for start, end in zip(clean[:-1], clean[1:]):
            section = Section(0, start, end, round_time(end - start), "")
            if section.duration > long_threshold:
                boundary = best_internal_boundary(
                    analysis,
                    section,
                    semantic_min_seconds,
                    novelty_floor,
                    lyric_boundaries=lyric_boundaries,
                    lyric_boundary_tolerance=lyric_boundary_tolerance,
                    lyric_boundary_weight=lyric_boundary_weight,
                )
                if boundary is not None and start < boundary < end:
                    next_boundaries.append(boundary)
                    changed = True
            next_boundaries.append(end)
        clean = sorted({round_time(boundary) for boundary in next_boundaries})
        if not changed:
            break

    return [
        Section(
            section_id=index,
            start_time=start,
            end_time=end,
            duration=round_time(end - start),
            label=f"Segment_{index}",
        )
        for index, (start, end) in enumerate(zip(clean[:-1], clean[1:]), start=1)
        if end > start
    ]


def cosine_similarity(a: object, b: object) -> float:
    """Small local cosine similarity helper."""
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def semantic_labels_from_mir(
    analysis: dict[str, object],
    sections: list[Section],
    lyrics: list[LyricLine] | None = None,
) -> list[str]:
    """Infer coarse pop section types from acoustic, vocal, lyric, and position cues.

    The labeling logic treats pop structure as functional roles rather than
    purely mechanical segments. Long high-energy runs often contain the
    pre-chorus build followed by the chorus release, so the first part of a
    sustained high-energy run can be labeled as Pre-chorus instead of being
    collapsed into Chorus. Lyrics and vocal activity are soft functional cues:
    repeated lyric text supports Chorus, lyric density supports Verse, and low
    vocal activity supports Intro/Outro/Interlude-like roles.
    """
    import numpy as np

    if not sections:
        return []

    song_duration = sections[-1].end_time
    energies = np.array(energy_for_sections(analysis, sections))
    vocal_activity = np.array(frame_curve_for_sections(analysis, sections, "vocal_proxy_curve"))
    onset_activity = np.array(frame_curve_for_sections(analysis, sections, "onset_curve"))
    lyric_stats = lyric_stats_for_sections(lyrics or [], sections)
    repeat_ratios = np.array([item["repeat_ratio"] for item in lyric_stats])
    lyric_counts = np.array([item["line_count"] for item in lyric_stats])
    median_energy = float(np.median(energies))
    median_vocal = float(np.median(vocal_activity))
    high_energy = max(float(np.percentile(energies, 62)), median_energy * 1.25)
    high_vocal = max(float(np.percentile(vocal_activity, 58)), median_vocal * 1.05)
    low_vocal = min(float(np.percentile(vocal_activity, 35)), median_vocal * 0.92)

    labels = ["Verse" for _ in sections]
    if sections[0].start_time <= 0.01 and (
        sections[0].end_time <= song_duration * 0.12
        or vocal_activity[0] <= low_vocal
        or lyric_counts[0] == 0
    ):
        labels[0] = "Intro"

    for i, section in enumerate(sections):
        if section.start_time >= song_duration * 0.88 or (
            i >= len(sections) - 2
            and energies[i] < median_energy * 0.9
            and (vocal_activity[i] <= median_vocal or lyric_counts[i] == 0)
        ):
            labels[i] = "Outro"

    high_mask = [
        labels[i] not in {"Intro", "Outro"}
        and sections[i].start_time >= song_duration * 0.18
        and sections[i].start_time <= song_duration * 0.86
        and (
            energies[i] >= high_energy
            or (energies[i] >= median_energy * 1.08 and vocal_activity[i] >= high_vocal)
            or repeat_ratios[i] >= 0.45
        )
        for i in range(len(sections))
    ]

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(high_mask):
        if not high_mask[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(high_mask) and high_mask[index + 1]:
            index += 1
        runs.append((start, index))
        index += 1

    chorus_runs = runs[:3]
    for start, end in chorus_runs:
        run_length = end - start + 1
        chorus_start = start
        if run_length >= 4:
            labels[start] = "Pre-chorus"
            labels[start + 1] = "Pre-chorus"
            chorus_start = start + 2
        elif run_length >= 3:
            labels[start] = "Pre-chorus"
            chorus_start = start + 1
        elif start - 1 > 0 and labels[start - 1] == "Verse":
            labels[start - 1] = "Pre-chorus"

        for chorus_index in range(chorus_start, end + 1):
            if labels[chorus_index] not in {"Outro", "Pre-chorus"}:
                labels[chorus_index] = "Chorus"

        bridge_index = end + 1
        if bridge_index < len(sections) and labels[bridge_index] == "Verse":
            if bridge_index < len(sections) - 2 and energies[bridge_index] < max(energies[start : end + 1]) * 0.88:
                labels[bridge_index] = "Bridge"

    if lyrics:
        for index, repeat_ratio in enumerate(repeat_ratios):
            if labels[index] not in {"Intro", "Outro"} and repeat_ratio >= 0.45 and energies[index] >= median_energy:
                labels[index] = "Chorus"

        for index in range(1, len(labels)):
            if labels[index] == "Chorus" and labels[index - 1] == "Verse":
                if lyric_counts[index - 1] > 0 and energies[index - 1] >= median_energy * 0.9:
                    labels[index - 1] = "Pre-chorus"

    for index, label in enumerate(labels):
        if label == "Verse" and vocal_activity[index] <= low_vocal and energies[index] <= median_energy * 0.95:
            if index == 0:
                labels[index] = "Intro"
            elif index >= len(labels) - 2 or sections[index].start_time >= song_duration * 0.82:
                labels[index] = "Outro"
            elif labels[index - 1] == "Chorus":
                labels[index] = "Bridge"

    for index in range(1, len(labels)):
        if labels[index] == "Pre-chorus" and labels[index - 1] == "Bridge":
            labels[index] = "Verse"

    for index in range(1, len(labels) - 1):
        if (
            labels[index] == "Verse"
            and labels[index - 1] == "Bridge"
            and labels[index + 1] == "Chorus"
            and sections[index].start_time >= song_duration * 0.70
        ):
            labels[index] = "Chorus"

    for index, label in enumerate(labels):
        if label == "Bridge" and sections[index].start_time >= song_duration * 0.86:
            labels[index] = "Outro"

    return labels


def merge_adjacent_semantic_sections(sections: list[Section]) -> list[Section]:
    """Merge adjacent sections when they are a single semantic event."""
    if not sections:
        return []

    mergeable = {"Pre-chorus", "Chorus", "Outro"}
    merged = [sections[0]]
    for section in sections[1:]:
        previous = merged[-1]
        if section.label == previous.label and section.label in mergeable:
            previous.end_time = section.end_time
            previous.duration = round_time(previous.end_time - previous.start_time)
        else:
            merged.append(section)

    for index, section in enumerate(merged, start=1):
        section.section_id = index
    return merged


def resolve_transition_fragments(
    analysis: dict[str, object],
    sections: list[Section],
    lyrics: list[LyricLine] | None = None,
) -> list[Section]:
    """Absorb short between-line fragments into the nearer musical function.

    If an acoustic boundary falls between two lyric lines, the between-line
    material often belongs functionally to either the previous lyric section or
    the next lyric section. We keep the acoustic boundary candidates available
    during analysis, but for final section labels we reassign short ambiguous
    fragments by comparing their section-level audio vector to neighboring
    sections. Timestamped lyrics tighten this rule to lyric-free fragments;
    without timestamps, it still acts as a conservative acoustic fallback.
    """
    import numpy as np

    if len(sections) < 3:
        return sections

    resolved = [
        Section(s.section_id, s.start_time, s.end_time, s.duration, s.label, s.structure_label)
        for s in sections
    ]
    vectors = feature_vectors_for_sections(analysis, resolved)
    median_duration = float(np.median([section.duration for section in resolved]))
    short_fragment_seconds = max(6.0, min(10.0, median_duration * 0.75))

    for index in range(1, len(resolved) - 1):
        current = resolved[index]
        previous = resolved[index - 1]
        following = resolved[index + 1]
        if current.duration > short_fragment_seconds:
            continue
        if current.label in {"Intro", "Outro"}:
            continue
        if lyrics and section_has_lyrics(lyrics, current):
            continue

        previous_similarity = cosine_similarity(vectors[index], vectors[index - 1])
        following_similarity = cosine_similarity(vectors[index], vectors[index + 1])
        if following_similarity - previous_similarity >= 0.03:
            current.label = following.label
        elif previous_similarity - following_similarity >= 0.03:
            current.label = previous.label

    for index, section in enumerate(resolved, start=1):
        section.section_id = index
        section.structure_label = structure_label_for_semantic_label(section.label)
    return resolved


def structure_label_for_semantic_label(label: str) -> str:
    """Map semantic pop labels to reusable functional section classes."""
    return {
        "Intro": "A",
        "Outro": "A",
        "Verse": "B",
        "Pre-chorus": "C",
        "Chorus": "D",
        "Bridge": "E",
    }.get(label, "")


def apply_mir_pop_labels(
    input_path: Path,
    msaf_sections: list[Section],
    min_section_seconds: float,
    long_section_factor: float,
    lyrics_path: Path | None = None,
    lyric_boundary_tolerance: float = 2.5,
    lyric_boundary_weight: float = 0.35,
) -> list[Section]:
    """MIR post-processing for section-aware weak labels.

    This step uses adaptive novelty splitting and section-level audio features.
    It avoids fixed phrase lengths so it can generalize better across pop songs
    with different tempo, meter, and arrangement density.
    """
    analysis = load_analysis_features(input_path)
    lyrics = load_lyrics(lyrics_path)
    lyric_boundaries = lyric_boundary_candidates(lyrics, msaf_sections[-1].end_time) if lyrics else []
    refined = refine_boundaries_by_novelty(
        analysis,
        msaf_sections,
        min_section_seconds=min_section_seconds,
        long_section_factor=long_section_factor,
        lyric_boundaries=lyric_boundaries,
        lyric_boundary_tolerance=lyric_boundary_tolerance,
        lyric_boundary_weight=lyric_boundary_weight,
    )
    labels = semantic_labels_from_mir(analysis, refined, lyrics=lyrics)
    for index, (section, label) in enumerate(zip(refined, labels), start=1):
        section.section_id = index
        section.label = label
        section.structure_label = structure_label_for_semantic_label(label)
    refined = resolve_transition_fragments(analysis, refined, lyrics=lyrics)
    merged = merge_adjacent_semantic_sections(refined)
    for section in merged:
        section.structure_label = structure_label_for_semantic_label(section.label)
    return merged


def write_csv(sections: Iterable[Section], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section_id", "start_time", "end_time", "duration", "label", "structure_label"],
        )
        writer.writeheader()
        for section in sections:
            writer.writerow(
                {
                    "section_id": section.section_id,
                    "start_time": format_timestamp(section.start_time),
                    "end_time": format_timestamp(section.end_time),
                    "duration": format_timestamp(section.duration),
                    "label": section.label,
                    "structure_label": section.structure_label,
                }
            )


def write_json(sections: Iterable[Section], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(section) for section in sections]
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def default_json_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a song into structural sections with MSAF.")
    parser.add_argument("--input", required=True, help="Input .mp3 or .wav file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--json-output", help="Optional JSON output path. Defaults to output path with .json suffix.")
    parser.add_argument("--boundaries-id", default="sf", help="MSAF boundary algorithm, e.g. sf, cnmf, foote.")
    parser.add_argument("--labels-id", default="fmc2d", help="MSAF label algorithm, e.g. fmc2d, cnmf, scluster.")
    parser.add_argument("--min-section-seconds", type=float, default=3.0, help="Merge MSAF sections shorter than this.")
    parser.add_argument("--mode", choices=["mir-pop", "raw-msaf"], default="mir-pop", help="Output MIR pop labels or raw MSAF labels.")
    parser.add_argument("--lyrics", help="Optional timestamped lyrics file (.lrc or CSV with time/text columns).")
    parser.add_argument(
        "--lyric-boundary-tolerance",
        type=float,
        default=2.5,
        help="Seconds around lyric transition hints where acoustic novelty peaks may be boosted.",
    )
    parser.add_argument(
        "--lyric-boundary-weight",
        type=float,
        default=0.35,
        help="Soft weight for lyric transition hints. Use 0 to disable lyric influence.",
    )
    parser.add_argument(
        "--long-section-factor",
        type=float,
        default=1.5,
        help="Split sections longer than median_duration * this value when novelty supports it.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    json_path = Path(args.json_output) if args.json_output else default_json_path(output_path)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if input_path.suffix.lower() not in {".mp3", ".wav"}:
        raise SystemExit(f"Unsupported audio format: {input_path.suffix}. Please use .mp3 or .wav.")

    sections = segment_with_msaf(input_path, args.boundaries_id, args.labels_id, args.min_section_seconds)
    if args.mode == "mir-pop":
        sections = apply_mir_pop_labels(
            input_path,
            sections,
            min_section_seconds=args.min_section_seconds,
            long_section_factor=args.long_section_factor,
            lyrics_path=Path(args.lyrics) if args.lyrics else None,
            lyric_boundary_tolerance=args.lyric_boundary_tolerance,
            lyric_boundary_weight=args.lyric_boundary_weight,
        )
    write_csv(sections, output_path)
    write_json(sections, json_path)
    print(f"Saved CSV: {output_path.resolve()}")
    print(f"Saved JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
