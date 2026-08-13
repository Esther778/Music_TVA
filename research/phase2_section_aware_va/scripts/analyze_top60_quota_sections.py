#!/usr/bin/env python3
"""Analyze the reviewed QQ top-60 quota corpus from readable annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE2_ROOT / "data/manifests/top60_quota_100_reviewed_manifest_20260813.csv"
DEFAULT_ANNOTATIONS = PHASE2_ROOT / "data/manifests/top60_quota_100_human_annotations_readable_20260813.csv"
DEFAULT_OUTPUT = PHASE2_ROOT / "analysis/top60_quota_100_20260813"
CHINA_TZ = timezone(timedelta(hours=8))


def normalize_type(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"


def repeated_type_summary(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (song_id, section_type), group in sections.groupby(["song_id", "section_type"]):
        if len(group) < 2:
            continue
        rows.append(
            {
                "song_id": song_id,
                "song_title": group["song_title"].iloc[0],
                "song_artist": group["song_artist"].iloc[0],
                "section_type": section_type,
                "repeat_count": int(len(group)),
                "first_start_sec": float(group["start_time_sec"].min()),
                "last_end_sec": float(group["end_time_sec"].max()),
                "mean_duration_sec": float(group["duration_sec"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["section_type", "repeat_count", "song_title"], ascending=[True, False, True])


def transition_summary(sections: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for _, group in sections.sort_values(["song_id", "section_order"]).groupby("song_id"):
        types = group["section_type"].tolist()
        for left, right in zip(types, types[1:]):
            pairs.append(f"{left}->{right}")
    counts = Counter(pairs)
    rows = [{"transition": key, "count": value} for key, value in counts.most_common()]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"song_id": str})
    sections = pd.read_csv(args.annotations, dtype={"song_id": str})
    sections["section_type"] = sections["section_type"].map(normalize_type)
    numeric_cols = ["section_order", "start_time_sec", "end_time_sec", "duration_sec"]
    for column in numeric_cols:
        sections[column] = pd.to_numeric(sections[column], errors="coerce")
    sections = sections.sort_values(["song_id", "section_order", "start_time_sec"])

    errors: list[str] = []
    if len(manifest) != 100:
        errors.append(f"manifest_rows:{len(manifest)}")
    if manifest["song_id"].nunique() != 100:
        errors.append("manifest_song_id_unique")
    if sections["song_id"].nunique() != 100:
        errors.append(f"annotation_songs:{sections['song_id'].nunique()}")
    if sections[["start_time_sec", "end_time_sec"]].isna().any().any():
        errors.append("missing_boundary_values")
    if (sections["end_time_sec"] <= sections["start_time_sec"]).any():
        errors.append("non_positive_section_duration")
    missing_audio = manifest[manifest["audio_exists"].astype(str).str.lower() != "true"]
    if len(missing_audio):
        errors.append(f"missing_audio:{len(missing_audio)}")

    song_rows = []
    continuity_errors = []
    for song_id, group in sections.groupby("song_id"):
        ordered = group.sort_values("section_order")
        gaps = []
        overlaps = []
        previous_end = None
        for row in ordered.itertuples():
            if previous_end is not None:
                delta = float(row.start_time_sec) - float(previous_end)
                if delta > 0.01:
                    gaps.append(delta)
                elif delta < -0.01:
                    overlaps.append(delta)
            previous_end = row.end_time_sec
        if gaps or overlaps:
            continuity_errors.append({"song_id": song_id, "gaps": gaps, "overlaps": overlaps})
        type_counts = Counter(ordered["section_type"])
        song_rows.append(
            {
                "song_id": song_id,
                "title": ordered["song_title"].iloc[0],
                "artist": ordered["song_artist"].iloc[0],
                "section_count": int(len(ordered)),
                "duration_sec": float(ordered["end_time_sec"].max()),
                "unique_section_types": int(len(type_counts)),
                "dominant_section_type": type_counts.most_common(1)[0][0],
                "chorus_count": int(type_counts.get("chorus", 0)),
                "verse_count": int(type_counts.get("verse", 0)),
                "bridge_count": int(type_counts.get("bridge", 0)),
                "pre_chorus_count": int(type_counts.get("pre_chorus", 0)),
                "intro_count": int(type_counts.get("intro", 0)),
                "outro_count": int(type_counts.get("outro", 0)),
            }
        )
    if continuity_errors:
        errors.append(f"continuity_errors:{len(continuity_errors)}")

    song_features = pd.DataFrame(song_rows).sort_values(["artist", "title"])
    type_summary = (
        sections.groupby("section_type")
        .agg(section_count=("section_type", "size"), song_count=("song_id", "nunique"), mean_duration_sec=("duration_sec", "mean"), median_duration_sec=("duration_sec", "median"))
        .reset_index()
        .sort_values("section_count", ascending=False)
    )
    type_summary["song_coverage"] = type_summary["song_count"] / len(manifest)

    artist_summary = (
        manifest.groupby("quota_artist")
        .agg(song_count=("song_id", "nunique"), mean_sections=("readable_section_count", "mean"))
        .reset_index()
        .sort_values(["song_count", "quota_artist"], ascending=[False, True])
    )
    repeat_summary = repeated_type_summary(sections)
    transitions = transition_summary(sections)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    song_path = args.output_dir / "top60_quota_100_song_structure_summary.csv"
    type_path = args.output_dir / "top60_quota_100_section_type_summary.csv"
    artist_path = args.output_dir / "top60_quota_100_artist_summary.csv"
    repeat_path = args.output_dir / "top60_quota_100_repeated_section_summary.csv"
    transition_path = args.output_dir / "top60_quota_100_transition_summary.csv"
    continuity_path = args.output_dir / "top60_quota_100_continuity_issues.json"
    manifest_path = args.output_dir / "top60_quota_100_analysis_manifest.json"

    song_features.to_csv(song_path, index=False)
    type_summary.to_csv(type_path, index=False)
    artist_summary.to_csv(artist_path, index=False)
    repeat_summary.to_csv(repeat_path, index=False)
    transitions.to_csv(transition_path, index=False)
    continuity_path.write_text(json.dumps(continuity_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    analysis_manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "input_manifest": str(args.manifest.resolve()),
        "input_annotations": str(args.annotations.resolve()),
        "songs": int(len(manifest)),
        "sections": int(len(sections)),
        "unique_artists_by_quota": int(manifest["quota_artist"].nunique()),
        "mean_sections_per_song": float(song_features["section_count"].mean()),
        "median_sections_per_song": float(song_features["section_count"].median()),
        "section_type_counts": {row["section_type"]: int(row["section_count"]) for _, row in type_summary.iterrows()},
        "outputs": {
            "song_structure_summary": str(song_path.resolve()),
            "section_type_summary": str(type_path.resolve()),
            "artist_summary": str(artist_path.resolve()),
            "repeated_section_summary": str(repeat_path.resolve()),
            "transition_summary": str(transition_path.resolve()),
            "continuity_issues": str(continuity_path.resolve()),
        },
        "note": "This analysis uses reviewed human section boundaries/types only; it does not run or reuse frame-level VA prediction.",
    }
    manifest_path.write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis_manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
