#!/usr/bin/env python3
"""Align frozen blind-sample pseudo-VA frames to frozen section boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2 = PHASE2_ROOT / "data/blind_validation/stage2"
DEFAULT_STAGE3 = PHASE2_ROOT / "data/blind_validation/stage3"
DEFAULT_OUTPUT = PHASE2_ROOT / "data/blind_validation/stage4"
CHINA_TZ = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slope(times: pd.Series, values: pd.Series) -> float:
    x = times.to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x - x[0], y, 1)[0])


def quadrant(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "positive_high_arousal"
    if valence >= 0 and arousal < 0:
        return "positive_low_arousal"
    if valence < 0 and arousal >= 0:
        return "negative_high_arousal"
    return "negative_low_arousal"


def validate_boundaries(boundaries: pd.DataFrame, manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if len(boundaries) != int(manifest["section_count"].sum()):
        errors.append("boundary_total_count")
    for song in manifest.itertuples(index=False):
        rows = boundaries[boundaries["song_id"] == song.song_id].sort_values("section_order")
        if len(rows) != int(song.section_count):
            errors.append(f"{song.song_id}:section_count")
            continue
        orders = rows["section_order"].to_numpy(dtype=int)
        starts = rows["start_time_sec"].to_numpy(dtype=float)
        ends = rows["end_time_sec"].to_numpy(dtype=float)
        if not np.array_equal(orders, np.arange(1, len(rows) + 1)):
            errors.append(f"{song.song_id}:section_order")
        if not np.isclose(starts[0], 0.0, atol=1e-8):
            errors.append(f"{song.song_id}:first_start")
        if np.any(ends <= starts):
            errors.append(f"{song.song_id}:nonpositive_section")
        if len(rows) > 1 and not np.allclose(starts[1:], ends[:-1], atol=1e-8):
            errors.append(f"{song.song_id}:continuity")
    return errors


def align_frames(
    frames: pd.DataFrame,
    boundaries: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    aligned_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for song in manifest.sort_values("sample_index").itertuples(index=False):
        song_frames = frames[frames["song_id"] == song.song_id].sort_values("time_sec").copy()
        song_sections = boundaries[boundaries["song_id"] == song.song_id].sort_values("section_order").copy()
        starts = song_sections["start_time_sec"].to_numpy(dtype=float)
        ends = song_sections["end_time_sec"].to_numpy(dtype=float)
        times = song_frames["time_sec"].to_numpy(dtype=float)
        assignment = np.full(len(song_frames), -1, dtype=int)
        match_counts = np.zeros(len(song_frames), dtype=int)
        for section_index, (start, end) in enumerate(zip(starts, ends)):
            mask = (times >= start) & (times < end)
            assignment[mask] = section_index
            match_counts[mask] += 1

        overlap_frames = int((match_counts > 1).sum())
        valid = assignment >= 0
        song_frames["section_order"] = -1
        song_frames["human_annotation_id"] = ""
        song_frames["section_start_sec"] = np.nan
        song_frames["section_end_sec"] = np.nan
        song_frames["section_duration_sec"] = np.nan
        if np.any(valid):
            selected = song_sections.iloc[assignment[valid]]
            song_frames.loc[valid, "section_order"] = selected["section_order"].to_numpy(dtype=int)
            song_frames.loc[valid, "human_annotation_id"] = selected["human_annotation_id"].astype(str).to_numpy()
            song_frames.loc[valid, "section_start_sec"] = selected["start_time_sec"].to_numpy(dtype=float)
            song_frames.loc[valid, "section_end_sec"] = selected["end_time_sec"].to_numpy(dtype=float)
            song_frames.loc[valid, "section_duration_sec"] = selected["duration_sec"].to_numpy(dtype=float)
        song_frames["section_assignment_status"] = np.where(valid, "assigned", "outside_frozen_boundaries")
        aligned_parts.append(song_frames)
        audit_rows.append(
            {
                "sample_index": int(song.sample_index),
                "song_id": song.song_id,
                "title": song.title,
                "frames": len(song_frames),
                "assigned_frames": int(valid.sum()),
                "unassigned_frames": int((~valid).sum()),
                "overlap_frames": overlap_frames,
                "sections": len(song_sections),
                "sections_with_frames": int(song_frames.loc[valid, "section_order"].nunique()),
                "final_boundary_sec": float(ends[-1]),
                "last_frame_sec": float(times[-1]),
            }
        )
    return pd.concat(aligned_parts, ignore_index=True), audit_rows


def build_section_features(aligned: pd.DataFrame, boundaries: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_by_id = manifest.set_index("song_id")
    assigned = aligned[aligned["section_assignment_status"] == "assigned"]
    for boundary in boundaries.sort_values(["song_id", "section_order"]).itertuples(index=False):
        group = assigned[
            (assigned["song_id"] == boundary.song_id)
            & (assigned["section_order"] == int(boundary.section_order))
        ].sort_values("time_sec")
        if group.empty:
            raise RuntimeError(f"Frozen section has no 0.5-second frame: {boundary.human_annotation_id}")
        song = manifest_by_id.loc[boundary.song_id]
        valence = group["valence"]
        arousal = group["arousal"]
        times = group["time_sec"]
        peak_v_index = valence.idxmax()
        trough_v_index = valence.idxmin()
        peak_a_index = arousal.idxmax()
        trough_a_index = arousal.idxmin()
        song_end = float(boundaries[boundaries["song_id"] == boundary.song_id]["end_time_sec"].max())
        mean_v = float(valence.mean())
        mean_a = float(arousal.mean())
        rows.append(
            {
                "sample_index": int(song["sample_index"]),
                "song_id": boundary.song_id,
                "title": song["title"],
                "artist": song["artist"],
                "human_annotation_id": boundary.human_annotation_id,
                "section_order": int(boundary.section_order),
                "section_type": "withheld_until_stage5",
                "section_start_sec": float(boundary.start_time_sec),
                "section_end_sec": float(boundary.end_time_sec),
                "section_duration_sec": float(boundary.duration_sec),
                "normalized_start": float(boundary.start_time_sec) / song_end,
                "normalized_end": float(boundary.end_time_sec) / song_end,
                "normalized_midpoint": (float(boundary.start_time_sec) + float(boundary.end_time_sec)) / (2 * song_end),
                "frame_count": len(group),
                "mean_valence": mean_v,
                "std_valence": float(valence.std(ddof=0)),
                "min_valence": float(valence.min()),
                "max_valence": float(valence.max()),
                "entry_valence": float(valence.iloc[0]),
                "exit_valence": float(valence.iloc[-1]),
                "delta_valence": float(valence.iloc[-1] - valence.iloc[0]),
                "slope_valence_per_sec": slope(times, valence),
                "peak_valence": float(valence.loc[peak_v_index]),
                "peak_valence_time_sec": float(group.loc[peak_v_index, "time_sec"]),
                "trough_valence": float(valence.loc[trough_v_index]),
                "trough_valence_time_sec": float(group.loc[trough_v_index, "time_sec"]),
                "mean_arousal": mean_a,
                "std_arousal": float(arousal.std(ddof=0)),
                "min_arousal": float(arousal.min()),
                "max_arousal": float(arousal.max()),
                "entry_arousal": float(arousal.iloc[0]),
                "exit_arousal": float(arousal.iloc[-1]),
                "delta_arousal": float(arousal.iloc[-1] - arousal.iloc[0]),
                "slope_arousal_per_sec": slope(times, arousal),
                "peak_arousal": float(arousal.loc[peak_a_index]),
                "peak_arousal_time_sec": float(group.loc[peak_a_index, "time_sec"]),
                "trough_arousal": float(arousal.loc[trough_a_index]),
                "trough_arousal_time_sec": float(group.loc[trough_a_index, "time_sec"]),
                "mean_va_radius": float(np.hypot(mean_v, mean_a)),
                "mean_va_quadrant": quadrant(mean_v, mean_a),
            }
        )

    features = pd.DataFrame(rows).sort_values(["sample_index", "section_order"]).reset_index(drop=True)
    for dimension in ("valence", "arousal"):
        mean_col = f"mean_{dimension}"
        features[f"change_from_prev_{dimension}"] = features.groupby("song_id")[mean_col].diff()
        features[f"change_to_next_{dimension}"] = features.groupby("song_id")[mean_col].shift(-1) - features[mean_col]
    features["contrast_from_prev"] = np.hypot(
        features["change_from_prev_valence"], features["change_from_prev_arousal"]
    )
    features["contrast_to_next"] = np.hypot(
        features["change_to_next_valence"], features["change_to_next_arousal"]
    )
    return features


def build_song_features(
    aligned: pd.DataFrame,
    sections: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for song in manifest.sort_values("sample_index").itertuples(index=False):
        frame_group = aligned[
            (aligned["song_id"] == song.song_id)
            & (aligned["section_assignment_status"] == "assigned")
        ].sort_values("time_sec")
        section_group = sections[sections["song_id"] == song.song_id].sort_values("section_order")
        final_boundary = float(section_group["section_end_sec"].max())
        climax = section_group.loc[section_group["mean_arousal"].idxmax()]
        valence_peak = section_group.loc[section_group["mean_valence"].idxmax()]
        contrasts = section_group.dropna(subset=["contrast_from_prev"])
        max_contrast = contrasts.loc[contrasts["contrast_from_prev"].idxmax()]
        frame_a_peak = frame_group.loc[frame_group["arousal"].idxmax()]
        rows.append(
            {
                "sample_index": int(song.sample_index),
                "song_id": song.song_id,
                "title": song.title,
                "artist": song.artist,
                "duration_sec": final_boundary,
                "section_count": len(section_group),
                "assigned_frame_count": len(frame_group),
                "mean_valence": float(frame_group["valence"].mean()),
                "mean_arousal": float(frame_group["arousal"].mean()),
                "global_valence_range": float(frame_group["valence"].max() - frame_group["valence"].min()),
                "global_arousal_range": float(frame_group["arousal"].max() - frame_group["arousal"].min()),
                "start_section_mean_valence": float(section_group.iloc[0]["mean_valence"]),
                "end_section_mean_valence": float(section_group.iloc[-1]["mean_valence"]),
                "net_section_valence_change": float(section_group.iloc[-1]["mean_valence"] - section_group.iloc[0]["mean_valence"]),
                "start_section_mean_arousal": float(section_group.iloc[0]["mean_arousal"]),
                "end_section_mean_arousal": float(section_group.iloc[-1]["mean_arousal"]),
                "net_section_arousal_change": float(section_group.iloc[-1]["mean_arousal"] - section_group.iloc[0]["mean_arousal"]),
                "climax_section_order_by_mean_arousal": int(climax["section_order"]),
                "climax_section_mean_arousal": float(climax["mean_arousal"]),
                "climax_section_time_ratio": float(climax["normalized_midpoint"]),
                "frame_peak_arousal": float(frame_a_peak["arousal"]),
                "frame_peak_arousal_time_ratio": float(frame_a_peak["time_sec"] / final_boundary),
                "max_mean_valence_section_order": int(valence_peak["section_order"]),
                "max_mean_valence": float(valence_peak["mean_valence"]),
                "max_section_contrast": float(max_contrast["contrast_from_prev"]),
                "max_contrast_transition_to_section": int(max_contrast["section_order"]),
                "section_va_path_length": float(contrasts["contrast_from_prev"].sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.stage2_dir / "blind_sample_15.csv"
    boundaries_path = args.stage2_dir / "blind_sample_15_boundaries.json"
    protocol_path = args.stage2_dir / "blind_sample_15_protocol.json"
    frame_path = args.stage3_dir / "blind_sample_15_frame_va.csv"
    stage3_audit_path = args.stage3_dir / "stage3_output_audit_summary.json"

    manifest = pd.read_csv(manifest_path)
    boundary_records = json.loads(boundaries_path.read_text(encoding="utf-8"))
    boundaries = pd.DataFrame(boundary_records)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage3_audit = json.loads(stage3_audit_path.read_text(encoding="utf-8"))
    frames = pd.read_csv(frame_path)

    if stage3_audit.get("overall_status") != "pass":
        raise RuntimeError("Stage 3 audit must pass before section alignment")
    if canonical_sha256(boundary_records) != protocol["frozen_boundaries_sha256"]:
        raise RuntimeError("Frozen boundary canonical hash mismatch")
    boundary_errors = validate_boundaries(boundaries, manifest)
    if boundary_errors:
        raise RuntimeError(f"Frozen boundary validation failed: {boundary_errors}")

    aligned, song_audit_rows = align_frames(frames, boundaries, manifest)
    section_features = build_section_features(aligned, boundaries, manifest)
    song_features = build_song_features(aligned, section_features, manifest)
    audit = pd.DataFrame(song_audit_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "blind_sample_15_frame_section_alignment.csv"
    section_path = args.output_dir / "blind_sample_15_section_features.csv"
    song_path = args.output_dir / "blind_sample_15_song_structure_features.csv"
    audit_path = args.output_dir / "stage4_alignment_audit.csv"
    aligned.to_csv(aligned_path, index=False)
    section_features.to_csv(section_path, index=False)
    song_features.to_csv(song_path, index=False)
    audit.to_csv(audit_path, index=False)

    errors = []
    if len(aligned) != len(frames):
        errors.append("aligned_frame_count")
    if len(section_features) != len(boundaries):
        errors.append("section_feature_count")
    if len(song_features) != len(manifest):
        errors.append("song_feature_count")
    if int(audit["overlap_frames"].sum()) != 0:
        errors.append("overlap_frames")
    if int(audit["unassigned_frames"].sum()) != 0:
        errors.append("unassigned_frames")
    if not (audit["sections"] == audit["sections_with_frames"]).all():
        errors.append("sections_without_frames")
    transition_prev = ["change_from_prev_valence", "change_from_prev_arousal", "contrast_from_prev"]
    transition_next = ["change_to_next_valence", "change_to_next_arousal", "contrast_to_next"]
    core_numeric = section_features.select_dtypes(include=np.number).drop(columns=transition_prev + transition_next)
    if not np.isfinite(core_numeric.to_numpy()).all():
        errors.append("nonfinite_section_features")
    first_sections = section_features.groupby("song_id").cumcount() == 0
    last_sections = section_features.groupby("song_id").cumcount(ascending=False) == 0
    if not section_features.loc[first_sections, transition_prev].isna().all().all():
        errors.append("first_section_previous_transition")
    if not np.isfinite(section_features.loc[~first_sections, transition_prev].to_numpy()).all():
        errors.append("internal_previous_transition")
    if not section_features.loc[last_sections, transition_next].isna().all().all():
        errors.append("last_section_next_transition")
    if not np.isfinite(section_features.loc[~last_sections, transition_next].to_numpy()).all():
        errors.append("internal_next_transition")

    summary = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "protocol": "Stage 4 frozen-boundary alignment without section types or human emotion answers",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "songs": len(manifest),
        "sections": len(boundaries),
        "frames": len(frames),
        "assigned_frames": int(audit["assigned_frames"].sum()),
        "unassigned_frames": int(audit["unassigned_frames"].sum()),
        "overlap_frames": int(audit["overlap_frames"].sum()),
        "sections_without_frames": int((audit["sections"] - audit["sections_with_frames"]).sum()),
        "frozen_boundaries_canonical_sha256": canonical_sha256(boundary_records),
        "stage3_frame_va_sha256": sha256_file(frame_path),
        "section_types_read": False,
        "human_emotion_answers_read": False,
        "outputs": {
            "frame_section_alignment": {"path": str(aligned_path.resolve()), "sha256": sha256_file(aligned_path)},
            "section_features": {"path": str(section_path.resolve()), "sha256": sha256_file(section_path)},
            "song_structure_features": {"path": str(song_path.resolve()), "sha256": sha256_file(song_path)},
            "alignment_audit": {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)},
        },
    }
    summary_path = args.output_dir / "stage4_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
