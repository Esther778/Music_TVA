#!/usr/bin/env python3
"""Build normalized PMEmo manifests for VA Teacher experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019"
DEFAULT_OUT = ROOT / "data" / "processed"


def build(extracted_dir: Path, output_dir: Path) -> None:
    annotations = extracted_dir / "annotations" / "dynamic_annotations.csv"
    static_annotations = extracted_dir / "annotations" / "static_annotations.csv"
    metadata = extracted_dir / "metadata.csv"
    dynamic_features = extracted_dir / "features" / "dynamic_features.csv"
    chorus_dir = extracted_dir / "chorus"

    dyn = pd.read_csv(annotations)
    stat = pd.read_csv(static_annotations)
    meta = pd.read_csv(metadata)
    feat_index = pd.read_csv(dynamic_features, usecols=["musicId", "frameTime"])

    dyn = dyn.rename(
        columns={
            "Arousal(mean)": "arousal",
            "Valence(mean)": "valence",
            "frameTime": "time_sec",
        }
    )
    stat = stat.rename(
        columns={
            "Arousal(mean)": "static_arousal",
            "Valence(mean)": "static_valence",
        }
    )

    merged = dyn.merge(meta, on="musicId", how="left").merge(stat, on="musicId", how="left")
    merged["dataset"] = "PMEmo2019"
    merged["song_id"] = merged["musicId"].astype(int).map(lambda x: f"pmemo_{x}")
    merged["audio_path"] = merged["musicId"].astype(int).map(lambda x: str(chorus_dir / f"{x}.mp3"))
    merged["label_range"] = "0_1"
    merged["first_15s_removed"] = True
    merged["frame_step_sec"] = 0.5
    merged["has_dynamic_feature_row"] = merged[["musicId", "time_sec"]].merge(
        feat_index.rename(columns={"frameTime": "time_sec"}).assign(has_dynamic_feature_row=True),
        on=["musicId", "time_sec"],
        how="left",
    )["has_dynamic_feature_row"].fillna(False)

    frame_cols = [
        "dataset",
        "song_id",
        "musicId",
        "fileName",
        "title",
        "artist",
        "album",
        "audio_path",
        "duration",
        "chorus_start_time",
        "chorus_end_time",
        "time_sec",
        "valence",
        "arousal",
        "static_valence",
        "static_arousal",
        "label_range",
        "first_15s_removed",
        "frame_step_sec",
        "has_dynamic_feature_row",
    ]
    frame_manifest = merged[frame_cols].sort_values(["musicId", "time_sec"])

    song_rows = []
    for music_id, group in frame_manifest.groupby("musicId", sort=True):
        audio_path = Path(group["audio_path"].iloc[0])
        song_rows.append(
            {
                "dataset": "PMEmo2019",
                "song_id": f"pmemo_{int(music_id)}",
                "musicId": int(music_id),
                "fileName": group["fileName"].iloc[0],
                "title": group["title"].iloc[0],
                "artist": group["artist"].iloc[0],
                "album": group["album"].iloc[0],
                "audio_path": str(audio_path),
                "audio_exists": audio_path.exists(),
                "metadata_duration": group["duration"].iloc[0],
                "chorus_start_time": group["chorus_start_time"].iloc[0],
                "chorus_end_time": group["chorus_end_time"].iloc[0],
                "frame_count": int(group.shape[0]),
                "label_time_min": float(group["time_sec"].min()),
                "label_time_max": float(group["time_sec"].max()),
                "mean_valence": float(group["valence"].mean()),
                "mean_arousal": float(group["arousal"].mean()),
                "static_valence": float(group["static_valence"].iloc[0]),
                "static_arousal": float(group["static_arousal"].iloc[0]),
                "all_frames_have_dynamic_features": bool(group["has_dynamic_feature_row"].all()),
            }
        )
    song_manifest = pd.DataFrame(song_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_manifest.to_csv(output_dir / "pmemo2019_dynamic_frame_manifest.csv", index=False)
    song_manifest.to_csv(output_dir / "pmemo2019_song_manifest.csv", index=False)

    summary = {
        "frame_rows": int(frame_manifest.shape[0]),
        "dynamic_songs": int(frame_manifest["musicId"].nunique()),
        "song_rows": int(song_manifest.shape[0]),
        "all_audio_exists": bool(song_manifest["audio_exists"].all()),
        "all_frames_have_dynamic_features": bool(song_manifest["all_frames_have_dynamic_features"].all()),
        "valence_min": float(frame_manifest["valence"].min()),
        "valence_max": float(frame_manifest["valence"].max()),
        "arousal_min": float(frame_manifest["arousal"].min()),
        "arousal_max": float(frame_manifest["arousal"].max()),
        "time_min": float(frame_manifest["time_sec"].min()),
        "time_max": float(frame_manifest["time_sec"].max()),
        "frame_count_min": int(song_manifest["frame_count"].min()),
        "frame_count_max": int(song_manifest["frame_count"].max()),
        "frame_count_mean": float(song_manifest["frame_count"].mean()),
    }
    pd.Series(summary).to_json(output_dir / "pmemo2019_manifest_summary.json", force_ascii=False, indent=2)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.extracted_dir, args.output_dir)


if __name__ == "__main__":
    main()
