#!/usr/bin/env python3
"""Audit frozen Stage 3 frame-level pseudo-VA outputs without section labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
DEFAULT_MANIFEST = PHASE2_ROOT / "data/blind_validation/stage2/blind_sample_15.csv"
DEFAULT_STAGE3 = PHASE2_ROOT / "data/blind_validation/stage3"
EXPECTED_MODEL_IDS = {
    "valence": "pmemo_final_clean_valence_teacher_v1",
    "arousal": "pmemo_final_clean_arousal_teacher_v1",
}
EXPECTED_WINDOW = {"window_sec": 30.0, "stride_sec": 15.0, "blend": "Hann weighted overlap-add"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    combined_path = args.stage3_dir / "blind_sample_15_frame_va.csv"
    combined = pd.read_csv(combined_path)
    audit_rows: list[dict[str, object]] = []
    per_song_frames: list[pd.DataFrame] = []

    for row in manifest.itertuples(index=False):
        song_id = str(row.song_id)
        audio_path = PROJECT_ROOT / str(row.audio_path).lstrip("/")
        frame_path = args.stage3_dir / "frame_va" / f"{song_id}_frame_va.csv"
        feature_path = args.stage3_dir / "features" / f"{song_id}.npz"
        metadata_path = args.stage3_dir / "metadata" / f"{song_id}.json"
        errors: list[str] = []

        for required in (audio_path, frame_path, feature_path, metadata_path):
            if not required.is_file():
                errors.append(f"missing:{required.name}")
        if errors:
            audit_rows.append({"song_id": song_id, "title": row.title, "status": "fail", "errors": ";".join(errors)})
            continue

        frames = pd.read_csv(frame_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(feature_path) as features:
            feature_shapes = {key: value.shape for key, value in features.items()}

        if sha256_file(audio_path) != str(row.audio_sha256):
            errors.append("audio_hash")
        if sha256_file(frame_path) != metadata.get("frame_va_sha256"):
            errors.append("frame_hash")
        if sha256_file(feature_path) != metadata.get("feature_sha256"):
            errors.append("feature_hash")
        if len(frames) != int(metadata.get("frames", -1)):
            errors.append("metadata_frame_count")
        if any(shape[0] != len(frames) for shape in feature_shapes.values() if len(shape) >= 1):
            errors.append("feature_frame_count")
        if frames["song_id"].nunique() != 1 or str(frames["song_id"].iloc[0]) != song_id:
            errors.append("song_id")
        if frames["title"].nunique() != 1 or str(frames["title"].iloc[0]) != str(row.title):
            errors.append("title")
        if frames["artist"].nunique() != 1 or str(frames["artist"].iloc[0]) != str(row.artist):
            errors.append("artist")
        if (
            frames["valence_teacher_model_id"].nunique() != 1
            or frames["valence_teacher_model_id"].iloc[0] != EXPECTED_MODEL_IDS["valence"]
        ):
            errors.append("valence_model_id")
        if (
            frames["arousal_teacher_model_id"].nunique() != 1
            or frames["arousal_teacher_model_id"].iloc[0] != EXPECTED_MODEL_IDS["arousal"]
        ):
            errors.append("arousal_model_id")
        if metadata.get("valence_teacher_model_id") != EXPECTED_MODEL_IDS["valence"]:
            errors.append("metadata_valence_model_id")
        if metadata.get("arousal_teacher_model_id") != EXPECTED_MODEL_IDS["arousal"]:
            errors.append("metadata_arousal_model_id")
        if float(metadata.get("teacher_window_sec", -1)) != EXPECTED_WINDOW["window_sec"]:
            errors.append("teacher_window_sec")
        if float(metadata.get("teacher_stride_sec", -1)) != EXPECTED_WINDOW["stride_sec"]:
            errors.append("teacher_stride_sec")
        if metadata.get("teacher_window_blend") != EXPECTED_WINDOW["blend"]:
            errors.append("teacher_window_blend")

        times = frames["time_sec"].to_numpy(dtype=float)
        va = frames[["valence", "arousal"]].to_numpy(dtype=float)
        va01 = frames[["valence_0_1", "arousal_0_1"]].to_numpy(dtype=float)
        actual_duration = float(metadata["duration_sec"])
        expected_times = np.arange(0.5, actual_duration, 0.5)
        if len(times) != len(expected_times) or not np.allclose(times, expected_times, atol=1e-8):
            errors.append("time_axis")
        if not np.isfinite(va).all() or not np.isfinite(va01).all():
            errors.append("nonfinite_va")
        if np.nanmin(va) < -1.0 or np.nanmax(va) > 1.0:
            errors.append("va_range")
        if np.nanmin(va01) < 0.0 or np.nanmax(va01) > 1.0:
            errors.append("va01_range")
        if not np.allclose(va01, (va + 1.0) / 2.0, atol=1e-7):
            errors.append("va01_transform")

        valence_std = float(frames["valence"].std())
        arousal_std = float(frames["arousal"].std())
        collapse_warning = valence_std < 0.005 or arousal_std < 0.005
        audit_rows.append(
            {
                "song_id": song_id,
                "title": row.title,
                "status": "pass" if not errors else "fail",
                "frames": len(frames),
                "duration_sec": actual_duration,
                "last_frame_sec": float(times[-1]),
                "valence_mean": float(frames["valence"].mean()),
                "valence_std": valence_std,
                "valence_min": float(frames["valence"].min()),
                "valence_max": float(frames["valence"].max()),
                "arousal_mean": float(frames["arousal"].mean()),
                "arousal_std": arousal_std,
                "arousal_min": float(frames["arousal"].min()),
                "arousal_max": float(frames["arousal"].max()),
                "collapse_warning": collapse_warning,
                "errors": ";".join(errors),
            }
        )
        per_song_frames.append(frames)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.stage3_dir / "stage3_output_audit.csv"
    audit.to_csv(audit_path, index=False)

    rebuilt = pd.concat(per_song_frames, ignore_index=True)
    numeric_columns = combined.select_dtypes(include=np.number).columns
    text_columns = combined.columns.difference(numeric_columns)
    numeric_max_abs_diff = float(
        np.max(np.abs(combined[numeric_columns].to_numpy() - rebuilt[numeric_columns].to_numpy()))
    )
    text_matches = bool((combined[text_columns] == rebuilt[text_columns]).to_numpy().all())
    combined_matches = bool(
        combined.shape == rebuilt.shape
        and combined.columns.equals(rebuilt.columns)
        and text_matches
        and numeric_max_abs_diff <= 5e-8
    )
    summary = {
        "manifest_songs": int(len(manifest)),
        "audited_songs": int(len(audit)),
        "passed_songs": int((audit["status"] == "pass").sum()),
        "failed_songs": int((audit["status"] == "fail").sum()),
        "collapse_warnings": int(audit.get("collapse_warning", pd.Series(dtype=bool)).fillna(False).sum()),
        "combined_rows": int(len(combined)),
        "rebuilt_rows": int(len(rebuilt)),
        "combined_matches_manifest_order_concat": combined_matches,
        "combined_numeric_max_abs_diff": numeric_max_abs_diff,
        "combined_numeric_tolerance": 5e-8,
        "combined_sha256": sha256_file(combined_path),
        "audit_csv": str(audit_path.resolve()),
        "audit_csv_sha256": sha256_file(audit_path),
        "overall_status": "pass"
        if len(audit) == len(manifest)
        and (audit["status"] == "pass").all()
        and not audit["collapse_warning"].fillna(False).any()
        and combined_matches
        else "fail",
    }
    summary_path = args.stage3_dir / "stage3_output_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["overall_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
