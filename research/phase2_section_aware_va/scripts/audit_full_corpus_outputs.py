#!/usr/bin/env python3
"""Audit frozen Teacher frame outputs for the complete 99-song corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict_blind_sample_final_teacher import MODEL_IDS, sha256_file


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE2_ROOT / "data/full_corpus/stage1/full_corpus_99_manifest.csv"
DEFAULT_STAGE2 = PHASE2_ROOT / "data/full_corpus/stage2"
EXPECTED_WINDOW = {"window_sec": 30.0, "stride_sec": 15.0, "blend": "Hann weighted overlap-add"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_STAGE2)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"song_id": str})
    combined_path = args.stage2_dir / "full_corpus_99_frame_va.csv"
    combined = pd.read_csv(combined_path, dtype={"song_id": str})
    audit_rows: list[dict[str, object]] = []
    rebuilt_parts: list[pd.DataFrame] = []

    for row in manifest.itertuples(index=False):
        song_id = str(row.song_id)
        frame_path = args.stage2_dir / "frame_va" / f"{song_id}_frame_va.csv"
        feature_path = args.stage2_dir / "features" / f"{song_id}.npz"
        metadata_path = args.stage2_dir / "metadata" / f"{song_id}.json"
        errors: list[str] = []
        for required in (frame_path, feature_path, metadata_path):
            if not required.is_file():
                errors.append(f"missing:{required.name}")
        if errors:
            audit_rows.append({"song_id": song_id, "title": row.title, "status": "fail", "errors": ";".join(errors)})
            continue

        frames = pd.read_csv(frame_path, dtype={"song_id": str})
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(feature_path) as features:
            time_len = len(features["time_sec"])
            feature_shape = features["mert_hidden6"].shape
            smile_shape = features["opensmile_sequence"].shape

        if metadata.get("audio_sha256") != str(row.audio_sha256):
            errors.append("audio_hash")
        if sha256_file(frame_path) != metadata.get("frame_va_sha256"):
            errors.append("frame_hash")
        if sha256_file(feature_path) != metadata.get("feature_sha256"):
            errors.append("feature_hash")
        if len(frames) != int(metadata.get("frames", -1)):
            errors.append("metadata_frame_count")
        if time_len != len(frames) or feature_shape[0] != len(frames) or smile_shape[0] != len(frames):
            errors.append("feature_frame_count")
        if frames["song_id"].nunique() != 1 or str(frames["song_id"].iloc[0]) != song_id:
            errors.append("song_id")
        if frames["title"].nunique() != 1 or str(frames["title"].iloc[0]) != str(row.title):
            errors.append("title")
        if frames["artist"].nunique() != 1 or str(frames["artist"].iloc[0]) != str(row.artist):
            errors.append("artist")
        if frames["valence_teacher_model_id"].nunique() != 1 or frames["valence_teacher_model_id"].iloc[0] != MODEL_IDS["valence"]:
            errors.append("valence_model_id")
        if frames["arousal_teacher_model_id"].nunique() != 1 or frames["arousal_teacher_model_id"].iloc[0] != MODEL_IDS["arousal"]:
            errors.append("arousal_model_id")
        if float(metadata.get("teacher_window_sec", -1)) != EXPECTED_WINDOW["window_sec"]:
            errors.append("teacher_window_sec")
        if float(metadata.get("teacher_stride_sec", -1)) != EXPECTED_WINDOW["stride_sec"]:
            errors.append("teacher_stride_sec")
        if metadata.get("teacher_window_blend") != EXPECTED_WINDOW["blend"]:
            errors.append("teacher_window_blend")

        times = frames["time_sec"].to_numpy(dtype=float)
        va = frames[["valence", "arousal"]].to_numpy(dtype=float)
        va01 = frames[["valence_0_1", "arousal_0_1"]].to_numpy(dtype=float)
        duration = float(metadata["duration_sec"])
        manifest_duration = float(row.duration_sec)
        duration_difference = abs(duration - manifest_duration)
        if duration_difference > 1.0:
            errors.append("decoded_manifest_duration_mismatch")
        expected_times = np.arange(0.5, duration, 0.5)
        if len(times) != len(expected_times) or not np.allclose(times, expected_times, atol=1e-8):
            errors.append("time_axis")
        if not np.isfinite(va).all() or not np.isfinite(va01).all():
            errors.append("nonfinite_va")
        if np.nanmin(va) < -1 or np.nanmax(va) > 1:
            errors.append("va_range")
        if not np.allclose(va01, (va + 1) / 2, atol=1e-7):
            errors.append("va01_transform")

        valence_std = float(frames["valence"].std())
        arousal_std = float(frames["arousal"].std())
        collapse_warning = valence_std < 0.005 or arousal_std < 0.005
        audit_rows.append({
            "song_id": song_id, "title": row.title,
            "status": "pass" if not errors else "fail", "frames": len(frames),
            "duration_sec": duration, "manifest_duration_sec": manifest_duration,
            "duration_abs_difference_sec": duration_difference,
            "last_frame_sec": float(times[-1]),
            "valence_mean": float(frames["valence"].mean()), "valence_std": valence_std,
            "valence_min": float(frames["valence"].min()), "valence_max": float(frames["valence"].max()),
            "arousal_mean": float(frames["arousal"].mean()), "arousal_std": arousal_std,
            "arousal_min": float(frames["arousal"].min()), "arousal_max": float(frames["arousal"].max()),
            "collapse_warning": collapse_warning, "errors": ";".join(errors),
        })
        rebuilt_parts.append(frames)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.stage2_dir / "full_corpus_output_audit.csv"
    audit.to_csv(audit_path, index=False)
    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)
    same_shape = combined.shape == rebuilt.shape and combined.columns.equals(rebuilt.columns)
    if same_shape:
        numeric = combined.select_dtypes(include=np.number).columns
        text = combined.columns.difference(numeric)
        numeric_diff = float(np.max(np.abs(combined[numeric].to_numpy() - rebuilt[numeric].to_numpy())))
        text_match = bool((combined[text].astype(str) == rebuilt[text].astype(str)).to_numpy().all())
    else:
        numeric_diff, text_match = float("inf"), False
    combined_match = bool(same_shape and numeric_diff <= 5e-8 and text_match)
    overall = bool(
        len(audit) == len(manifest) == 99
        and (audit["status"] == "pass").all()
        and not audit["collapse_warning"].fillna(False).any()
        and combined_match
    )
    summary = {
        "manifest_songs": int(len(manifest)), "audited_songs": int(len(audit)),
        "passed_songs": int((audit["status"] == "pass").sum()),
        "failed_songs": int((audit["status"] == "fail").sum()),
        "collapse_warnings": int(audit["collapse_warning"].fillna(False).sum()),
        "combined_rows": int(len(combined)), "rebuilt_rows": int(len(rebuilt)),
        "combined_matches_manifest_order_concat": combined_match,
        "combined_numeric_max_abs_diff": numeric_diff, "combined_numeric_tolerance": 5e-8,
        "combined_sha256": sha256_file(combined_path), "audit_csv": str(audit_path.resolve()),
        "audit_csv_sha256": sha256_file(audit_path), "overall_status": "pass" if overall else "fail",
    }
    summary_path = args.stage2_dir / "full_corpus_output_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
