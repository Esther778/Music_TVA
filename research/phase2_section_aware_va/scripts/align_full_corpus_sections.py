#!/usr/bin/env python3
"""Align full-corpus pseudo-VA frames to the frozen section boundaries."""

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


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = PHASE2_ROOT / "data/full_corpus/stage1"
DEFAULT_STAGE2 = PHASE2_ROOT / "data/full_corpus/stage2"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/full_corpus/stage3"


def include_exact_final_boundary_frames(
    aligned: pd.DataFrame,
    boundaries: pd.DataFrame,
    audit_rows: list[dict[str, object]],
) -> int:
    """Use [start, end) internally and [start, end] for the final section."""
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

    manifest_path = args.stage1_dir / "full_corpus_99_manifest.csv"
    boundaries_path = args.stage1_dir / "full_corpus_99_boundaries.json"
    protocol_path = args.stage1_dir / "full_corpus_99_protocol.json"
    frames_path = args.stage2_dir / "full_corpus_99_frame_va.csv"
    inference_audit_path = args.stage2_dir / "full_corpus_output_audit_summary.json"
    manifest = pd.read_csv(manifest_path, dtype={"song_id": str})
    boundary_records = json.loads(boundaries_path.read_text(encoding="utf-8"))
    boundaries = pd.DataFrame(boundary_records)
    boundaries["song_id"] = boundaries["song_id"].astype(str)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    inference_audit = json.loads(inference_audit_path.read_text(encoding="utf-8"))
    frames = pd.read_csv(frames_path, dtype={"song_id": str})

    if inference_audit.get("overall_status") != "pass":
        raise RuntimeError("Full-corpus inference audit must pass before section alignment")
    if canonical_sha256(boundary_records) != protocol["boundaries_canonical_sha256"]:
        raise RuntimeError("Frozen full-corpus boundary canonical hash mismatch")
    boundary_errors = validate_boundaries(boundaries, manifest)
    if boundary_errors:
        raise RuntimeError(f"Frozen boundary validation failed: {boundary_errors}")

    aligned, audit_rows = align_frames(frames, boundaries, manifest)
    final_boundary_frames = include_exact_final_boundary_frames(aligned, boundaries, audit_rows)
    section_features = build_section_features(aligned, boundaries, manifest)
    song_features = build_song_features(aligned, section_features, manifest)
    audit = pd.DataFrame(audit_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "full_corpus_99_frame_section_alignment.csv"
    section_path = args.output_dir / "full_corpus_99_section_features.csv"
    song_path = args.output_dir / "full_corpus_99_song_structure_features.csv"
    audit_path = args.output_dir / "full_corpus_alignment_audit.csv"
    aligned.to_csv(aligned_path, index=False)
    section_features.to_csv(section_path, index=False)
    song_features.to_csv(song_path, index=False)
    audit.to_csv(audit_path, index=False)

    errors: list[str] = []
    if len(aligned) != len(frames): errors.append("aligned_frame_count")
    if len(section_features) != len(boundaries): errors.append("section_feature_count")
    if len(song_features) != len(manifest): errors.append("song_feature_count")
    if int(audit["overlap_frames"].sum()) != 0: errors.append("overlap_frames")
    if int(audit["unassigned_frames"].sum()) != 0: errors.append("unassigned_frames")
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
        "protocol": "Frozen full-corpus boundary alignment without reading section types or human emotion answers",
        "status": "pass" if not errors else "fail", "errors": errors,
        "songs": len(manifest), "sections": len(boundaries), "frames": len(frames),
        "assigned_frames": int(audit["assigned_frames"].sum()),
        "unassigned_frames": int(audit["unassigned_frames"].sum()),
        "overlap_frames": int(audit["overlap_frames"].sum()),
        "sections_without_frames": int((audit["sections"] - audit["sections_with_frames"]).sum()),
        "final_boundary_frames_included": final_boundary_frames,
        "interval_rule": "internal sections [start,end); final section [start,end]",
        "frozen_boundaries_canonical_sha256": canonical_sha256(boundary_records),
        "frame_va_sha256": sha256_file(frames_path),
        "section_types_read": False, "human_emotion_answers_read": False,
        "outputs": {
            "frame_section_alignment": {"path": str(aligned_path.resolve()), "sha256": sha256_file(aligned_path)},
            "section_features": {"path": str(section_path.resolve()), "sha256": sha256_file(section_path)},
            "song_structure_features": {"path": str(song_path.resolve()), "sha256": sha256_file(song_path)},
            "alignment_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
    }
    summary_path = args.output_dir / "full_corpus_alignment_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
