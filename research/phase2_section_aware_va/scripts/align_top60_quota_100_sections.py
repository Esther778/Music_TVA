#!/usr/bin/env python3
"""Align top60 quota 100-song pseudo-VA frames to reviewed sections."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from align_blind_stage4_sections import (
    CHINA_TZ,
    align_frames,
    build_section_features,
    build_song_features,
    canonical_sha256,
    sha256_file,
    validate_boundaries,
)


PHASE2 = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = PHASE2 / "data/top60_quota_100_20260813/stage1"
DEFAULT_STAGE2 = PHASE2 / "data/top60_quota_100_20260813/stage2"
DEFAULT_OUTPUT = PHASE2 / "data/top60_quota_100_20260813/stage3"


def include_exact_final_boundary_frames(
    aligned: pd.DataFrame,
    boundaries: pd.DataFrame,
    audit_rows: list[dict[str, object]],
) -> int:
    aligned["assigned_on_final_boundary"] = False
    audit_by_song = {str(row["song_id"]): row for row in audit_rows}
    reassigned = 0
    for song_id, song_boundaries in boundaries.groupby("song_id"):
        final = song_boundaries.sort_values("section_order").iloc[-1]
        mask = (
            (aligned["song_id"] == str(song_id))
            & (aligned["section_assignment_status"] == "outside_frozen_boundaries")
            & np.isclose(aligned["time_sec"], float(final["end_time_sec"]), atol=1e-8)
        )
        count = int(mask.sum())
        if not count:
            continue
        aligned.loc[mask, "section_order"] = int(final["section_order"])
        aligned.loc[mask, "human_annotation_id"] = str(final["human_annotation_id"])
        aligned.loc[mask, "section_start_sec"] = float(final["start_time_sec"])
        aligned.loc[mask, "section_end_sec"] = float(final["end_time_sec"])
        aligned.loc[mask, "section_duration_sec"] = float(final["duration_sec"])
        aligned.loc[mask, "section_assignment_status"] = "assigned"
        aligned.loc[mask, "assigned_on_final_boundary"] = True
        audit = audit_by_song[str(song_id)]
        audit["assigned_frames"] = int(audit["assigned_frames"]) + count
        audit["unassigned_frames"] = int(audit["unassigned_frames"]) - count
        audit["final_boundary_frames_included"] = count
        reassigned += count
    for row in audit_rows:
        row.setdefault("final_boundary_frames_included", 0)
    return reassigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.stage1_dir / "top60_quota_100_manifest.csv"
    boundaries_path = args.stage1_dir / "top60_quota_100_boundaries.json"
    protocol_path = args.stage1_dir / "top60_quota_100_protocol.json"
    frames_path = args.stage2_dir / "top60_quota_100_frame_va.csv"
    frame_audit_path = args.stage2_dir / "top60_quota_100_output_audit_summary.json"

    manifest = pd.read_csv(manifest_path, dtype={"song_id": str})
    boundary_records = json.loads(boundaries_path.read_text(encoding="utf-8"))
    boundaries = pd.DataFrame(boundary_records)
    boundaries["song_id"] = boundaries["song_id"].astype(str)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frame_audit = json.loads(frame_audit_path.read_text(encoding="utf-8"))
    frames = pd.read_csv(frames_path, dtype={"song_id": str})

    if frame_audit.get("overall_status") != "pass":
        raise RuntimeError("Frame output audit must pass before section alignment")
    if canonical_sha256(boundary_records) != protocol["boundaries_canonical_sha256"]:
        raise RuntimeError("Frozen boundary canonical hash mismatch")
    boundary_errors = validate_boundaries(boundaries, manifest)
    if boundary_errors:
        raise RuntimeError(f"Frozen boundary validation failed: {boundary_errors}")

    aligned, audit_rows = align_frames(frames, boundaries, manifest)
    final_boundary_frames = include_exact_final_boundary_frames(aligned, boundaries, audit_rows)
    trailing_unassigned = aligned[aligned["section_assignment_status"] != "assigned"].copy()
    trailing_frame_count = 0
    trailing_max_overrun_sec = 0.0
    if not trailing_unassigned.empty:
        final_by_song = boundaries.groupby("song_id")["end_time_sec"].max().astype(float)
        trailing_unassigned["final_boundary_sec"] = trailing_unassigned["song_id"].map(final_by_song)
        trailing_unassigned["overrun_sec"] = trailing_unassigned["time_sec"].astype(float) - trailing_unassigned["final_boundary_sec"].astype(float)
        invalid = trailing_unassigned[trailing_unassigned["overrun_sec"] < -1e-8]
        if not invalid.empty:
            raise RuntimeError(f"Non-trailing unassigned frames: {invalid[['song_id','time_sec','final_boundary_sec']].to_dict('records')}")
        trailing_frame_count = int(len(trailing_unassigned))
        trailing_max_overrun_sec = float(trailing_unassigned["overrun_sec"].max())
    section_features = build_section_features(aligned, boundaries, manifest)
    # Reveal section types from the frozen 100-song boundary snapshot only after
    # frame inference and alignment have passed.
    section_type_by_id = boundaries.set_index("human_annotation_id")["section_type"].astype(str)
    section_features["section_type"] = section_features["human_annotation_id"].map(section_type_by_id)
    section_features["previous_section_type"] = section_features.groupby("song_id")["section_type"].shift(1)
    section_features["next_section_type"] = section_features.groupby("song_id")["section_type"].shift(-1)
    song_features = build_song_features(aligned, section_features, manifest)
    peak_a = section_features.loc[section_features.groupby("song_id")["mean_arousal"].idxmax()]
    peak_v = section_features.loc[section_features.groupby("song_id")["mean_valence"].idxmax()]
    peak_a = peak_a.set_index("song_id")[["section_type", "section_order"]]
    peak_v = peak_v.set_index("song_id")[["section_type", "section_order"]]
    song_features["arousal_peak_section_type"] = song_features["song_id"].map(peak_a["section_type"])
    song_features["arousal_peak_section_order"] = song_features["song_id"].map(peak_a["section_order"])
    song_features["valence_peak_section_type"] = song_features["song_id"].map(peak_v["section_type"])
    song_features["valence_peak_section_order"] = song_features["song_id"].map(peak_v["section_order"])
    audit = pd.DataFrame(audit_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "top60_quota_100_frame_section_alignment.csv"
    section_path = args.output_dir / "top60_quota_100_section_features.csv"
    song_path = args.output_dir / "top60_quota_100_song_structure_features.csv"
    audit_path = args.output_dir / "top60_quota_100_alignment_audit.csv"
    aligned.to_csv(aligned_path, index=False)
    section_features.to_csv(section_path, index=False)
    song_features.to_csv(song_path, index=False)
    audit.to_csv(audit_path, index=False)

    errors: list[str] = []
    if len(aligned) != len(frames): errors.append("aligned_frame_count")
    if len(section_features) != len(boundaries): errors.append("section_feature_count")
    if len(song_features) != len(manifest): errors.append("song_feature_count")
    if int(audit["overlap_frames"].sum()) != 0: errors.append("overlap_frames")
    if int(audit["unassigned_frames"].sum()) != trailing_frame_count: errors.append("unexpected_unassigned_frames")
    if not (audit["sections"] == audit["sections_with_frames"]).all(): errors.append("sections_without_frames")
    previous = ["change_from_prev_valence", "change_from_prev_arousal", "contrast_from_prev"]
    following = ["change_to_next_valence", "change_to_next_arousal", "contrast_to_next"]
    core_numeric = section_features.select_dtypes(include=np.number).drop(columns=previous + following)
    if not np.isfinite(core_numeric.to_numpy()).all(): errors.append("nonfinite_section_features")
    first = section_features.groupby("song_id").cumcount() == 0
    last = section_features.groupby("song_id").cumcount(ascending=False) == 0
    if not section_features.loc[first, previous].isna().all().all(): errors.append("first_previous_transition")
    if not np.isfinite(section_features.loc[~first, previous].to_numpy()).all(): errors.append("internal_previous_transition")
    if not section_features.loc[last, following].isna().all().all(): errors.append("last_next_transition")
    if not np.isfinite(section_features.loc[~last, following].to_numpy()).all(): errors.append("internal_next_transition")

    summary = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Top60 quota 100-song frame-to-section alignment",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "songs": len(manifest),
        "sections": len(boundaries),
        "frames": len(frames),
        "assigned_frames": int(audit["assigned_frames"].sum()),
        "unassigned_frames": int(audit["unassigned_frames"].sum()),
        "trailing_unassigned_frames_excluded": trailing_frame_count,
        "trailing_unassigned_max_overrun_sec": trailing_max_overrun_sec,
        "overlap_frames": int(audit["overlap_frames"].sum()),
        "sections_without_frames": int((audit["sections"] - audit["sections_with_frames"]).sum()),
        "final_boundary_frames_included": final_boundary_frames,
        "interval_rule": "internal sections [start,end); final section [start,end]",
        "frozen_boundaries_canonical_sha256": canonical_sha256(boundary_records),
        "frame_va_sha256": sha256_file(frames_path),
        "section_features_sha256": sha256_file(section_path),
        "song_features_sha256": sha256_file(song_path),
        "outputs": {
            "frame_section_alignment": str(aligned_path.resolve()),
            "section_features": str(section_path.resolve()),
            "song_structure_features": str(song_path.resolve()),
            "alignment_audit": str(audit_path.resolve()),
        },
    }
    summary_path = args.output_dir / "top60_quota_100_alignment_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
