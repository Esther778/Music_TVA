#!/usr/bin/env python3
"""Add canonical DEAM-scale labels to PMEmo manifests.

PMEmo released dynamic labels are in [0, 1]. The current DEAM Teacher work uses
[-1, 1] as the canonical dynamic VA scale, so this script writes both scales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_MANIFEST = ROOT / "data" / "processed" / "pmemo2019_dynamic_frame_manifest.csv"
DEFAULT_SONG_MANIFEST = ROOT / "data" / "processed" / "pmemo2019_song_manifest.csv"


def to_deam_scale(series: pd.Series) -> pd.Series:
    return series.astype(float) * 2.0 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--song-manifest", type=Path, default=DEFAULT_SONG_MANIFEST)
    args = parser.parse_args()

    frame = pd.read_csv(args.frame_manifest)
    song = pd.read_csv(args.song_manifest)

    frame["pmemo_valence_0_1"] = frame["valence"].astype(float)
    frame["pmemo_arousal_0_1"] = frame["arousal"].astype(float)
    frame["valence_deam_scale"] = to_deam_scale(frame["pmemo_valence_0_1"])
    frame["arousal_deam_scale"] = to_deam_scale(frame["pmemo_arousal_0_1"])
    frame["canonical_label_scale"] = "minus1_1"
    frame["scale_conversion"] = "pmemo_0_1_to_deam_minus1_1: y = 2*x - 1"

    song["pmemo_mean_valence_0_1"] = song["mean_valence"].astype(float)
    song["pmemo_mean_arousal_0_1"] = song["mean_arousal"].astype(float)
    song["mean_valence_deam_scale"] = to_deam_scale(song["pmemo_mean_valence_0_1"])
    song["mean_arousal_deam_scale"] = to_deam_scale(song["pmemo_mean_arousal_0_1"])
    song["pmemo_static_valence_0_1"] = song["static_valence"].astype(float)
    song["pmemo_static_arousal_0_1"] = song["static_arousal"].astype(float)
    song["static_valence_deam_scale"] = to_deam_scale(song["pmemo_static_valence_0_1"])
    song["static_arousal_deam_scale"] = to_deam_scale(song["pmemo_static_arousal_0_1"])
    song["canonical_label_scale"] = "minus1_1"

    frame_out = args.frame_manifest.with_name("pmemo2019_dynamic_frame_manifest_normalized.csv")
    song_out = args.song_manifest.with_name("pmemo2019_song_manifest_normalized.csv")
    frame.to_csv(frame_out, index=False)
    song.to_csv(song_out, index=False)

    summary = {
        "frame_manifest": str(frame_out),
        "song_manifest": str(song_out),
        "frame_rows": int(frame.shape[0]),
        "song_rows": int(song.shape[0]),
        "valence_deam_min": float(frame["valence_deam_scale"].min()),
        "valence_deam_max": float(frame["valence_deam_scale"].max()),
        "arousal_deam_min": float(frame["arousal_deam_scale"].min()),
        "arousal_deam_max": float(frame["arousal_deam_scale"].max()),
    }
    pd.Series(summary).to_json(
        args.frame_manifest.with_name("pmemo2019_label_normalization_summary.json"),
        force_ascii=False,
        indent=2,
    )
    print(summary)


if __name__ == "__main__":
    main()
