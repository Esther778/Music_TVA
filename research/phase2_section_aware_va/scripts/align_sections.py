#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import PHASE2_ROOT, quote, read_csv, supabase_get


ANNOTATION_FIELDS = [
    "song_id",
    "song_title",
    "song_artist",
    "section_order",
    "section_type",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "song_status",
    "is_approved",
    "confidence",
    "human_annotation_id",
    "updated_at",
]


def fetch_annotations(song_ids: list[str], annotations_json: str = "") -> pd.DataFrame:
    if annotations_json:
        with Path(annotations_json).open(encoding="utf-8") as handle:
            return pd.DataFrame(json.load(handle))
    rows = []
    select = quote(",".join(ANNOTATION_FIELDS))
    for start in range(0, len(song_ids), 20):
        chunk = song_ids[start : start + 20]
        ids = ",".join(quote(song_id) for song_id in chunk)
        query = (
            f"select={select}"
            f"&song_id=in.({ids})"
            "&song_status=eq.reviewed"
            "&order=song_id.asc,section_order.asc"
        )
        rows.extend(supabase_get("human_annotations_readable", query))
    return pd.DataFrame(rows)


def align_song(frame_df: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    if sections.empty:
        frame_df = frame_df.copy()
        frame_df["section_index"] = -1
        frame_df["section_type"] = "unassigned"
        frame_df["section_start_sec"] = np.nan
        frame_df["section_end_sec"] = np.nan
        return frame_df

    ordered = sections.sort_values(["start_time_sec", "end_time_sec"]).reset_index(drop=True)
    starts = ordered["start_time_sec"].astype(float).to_numpy()
    ends = ordered["end_time_sec"].astype(float).to_numpy()
    section_orders = ordered["section_order"].astype(int).to_numpy()
    section_types = ordered["section_type"].astype(str).to_numpy()

    section_index = []
    section_type = []
    section_start = []
    section_end = []
    for time_sec in frame_df["time_sec"].astype(float):
        matches = np.where((starts <= time_sec) & (time_sec < ends))[0]
        if len(matches) == 0 and len(ends) > 0 and math.isclose(time_sec, float(ends[-1]), abs_tol=0.25):
            matches = np.array([len(ends) - 1])
        if len(matches) == 0:
            section_index.append(-1)
            section_type.append("unassigned")
            section_start.append(np.nan)
            section_end.append(np.nan)
        else:
            idx = int(matches[0])
            section_index.append(int(section_orders[idx]))
            section_type.append(str(section_types[idx]))
            section_start.append(float(starts[idx]))
            section_end.append(float(ends[idx]))

    aligned = frame_df.copy()
    aligned["section_index"] = section_index
    aligned["section_type"] = section_type
    aligned["section_start_sec"] = section_start
    aligned["section_end_sec"] = section_end
    return aligned


def slope(time_values: pd.Series, value_values: pd.Series) -> float:
    if len(time_values) < 2:
        return 0.0
    x = time_values.astype(float).to_numpy()
    y = value_values.astype(float).to_numpy()
    x = x - x[0]
    if np.allclose(x, 0):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def section_features(aligned: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    valid = aligned[aligned["section_index"] >= 0].copy()
    feature_rows = []
    for (song_id, section_index), group in valid.groupby(["song_id", "section_index"], sort=True):
        group = group.sort_values("time_sec")
        split_row = splits[splits["id"] == song_id].iloc[0]
        n_sections = int(valid[valid["song_id"] == song_id]["section_index"].nunique())
        section_type = str(group["section_type"].iloc[0])
        section_start = float(group["section_start_sec"].iloc[0])
        section_end = float(group["section_end_sec"].iloc[0])
        duration = section_end - section_start
        valence = group["valence"].astype(float)
        arousal = group["arousal"].astype(float)
        times = group["time_sec"].astype(float)
        peak_v_idx = int(np.argmax(np.abs(valence - valence.mean())))
        peak_a_idx = int(np.argmax(np.abs(arousal - arousal.mean())))
        feature_rows.append(
            {
                "song_id": song_id,
                "title": split_row["title"],
                "artist": split_row["artist"],
                "split": split_row["split"],
                "selected_order": int(split_row["selected_order"]),
                "batch_index": int(split_row["batch_index"]),
                "section_index": int(section_index),
                "section_type": section_type,
                "section_start_sec": section_start,
                "section_end_sec": section_end,
                "section_duration_sec": duration,
                "section_position": int(section_index) / max(n_sections - 1, 1),
                "n_sections": n_sections,
                "frame_count": len(group),
                "mean_valence": float(valence.mean()),
                "std_valence": float(valence.std(ddof=0)),
                "min_valence": float(valence.min()),
                "max_valence": float(valence.max()),
                "entry_valence": float(valence.iloc[0]),
                "exit_valence": float(valence.iloc[-1]),
                "delta_valence": float(valence.iloc[-1] - valence.iloc[0]),
                "slope_valence": slope(times, valence),
                "peak_valence_time_rel": float((times.iloc[peak_v_idx] - section_start) / max(duration, 1e-8)),
                "mean_arousal": float(arousal.mean()),
                "std_arousal": float(arousal.std(ddof=0)),
                "min_arousal": float(arousal.min()),
                "max_arousal": float(arousal.max()),
                "entry_arousal": float(arousal.iloc[0]),
                "exit_arousal": float(arousal.iloc[-1]),
                "delta_arousal": float(arousal.iloc[-1] - arousal.iloc[0]),
                "slope_arousal": slope(times, arousal),
                "peak_arousal_time_rel": float((times.iloc[peak_a_idx] - section_start) / max(duration, 1e-8)),
            }
        )

    features = pd.DataFrame(feature_rows).sort_values(["selected_order", "section_index"])
    for column in ["mean_valence", "mean_arousal"]:
        features[f"prev_{column}"] = features.groupby("song_id")[column].shift(1)
        features[f"next_{column}"] = features.groupby("song_id")[column].shift(-1)
        features[f"prev_delta_{column}"] = features[column] - features[f"prev_{column}"]
        features[f"next_delta_{column}"] = features[f"next_{column}"] - features[column]
    return features.fillna(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Phase 1 frame-level VA to reviewed section annotations.")
    parser.add_argument("--splits", default=str(PHASE2_ROOT / "data" / "manifests" / "phase2_initial_70_splits.csv"))
    parser.add_argument("--annotations-json", default="", help="Optional local JSON snapshot from human_annotations_readable.")
    args = parser.parse_args()

    splits = pd.DataFrame(read_csv(Path(args.splits)))
    song_ids = splits["id"].tolist()
    annotations = fetch_annotations(song_ids, args.annotations_json)
    if annotations.empty:
        raise SystemExit("No reviewed human annotations found for the selected songs.")
    annotations["section_order"] = annotations["section_order"].astype(int)
    annotations["start_time_sec"] = annotations["start_time_sec"].astype(float)
    annotations["end_time_sec"] = annotations["end_time_sec"].astype(float)

    frame_dir = PHASE2_ROOT / "data" / "frame_va"
    out_dir = PHASE2_ROOT / "data" / "section_va"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_aligned = []
    for _, song in splits.sort_values("selected_order").iterrows():
        frame_path = frame_dir / f"{song['id']}_frame_va.csv"
        if not frame_path.is_file():
            print(f"missing_frame_va {song['id']} {frame_path}")
            continue
        frame_df = pd.read_csv(frame_path)
        song_sections = annotations[annotations["song_id"] == song["id"]]
        aligned = align_song(frame_df, song_sections)
        aligned.to_csv(out_dir / f"{song['id']}_frame_section_va.csv", index=False)
        all_aligned.append(aligned)
        print(f"aligned {song['id']} frames={len(aligned)} sections={song_sections['section_order'].nunique()}")

    if not all_aligned:
        raise SystemExit("No frame VA files were aligned.")
    combined = pd.concat(all_aligned, ignore_index=True)
    combined.to_csv(out_dir / "phase2_initial_70_frame_section_va.csv", index=False)
    features = section_features(combined, splits)
    features.to_csv(out_dir / "phase2_initial_70_section_features.csv", index=False)
    print(f"combined_frames={len(combined)} section_features={len(features)}")


if __name__ == "__main__":
    main()
