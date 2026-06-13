#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from common import PHASE2_ROOT, local_audio_path, reviewed_song_query, supabase_get, write_csv


SONG_FIELDS = [
    "selected_order",
    "id",
    "title",
    "artist",
    "audio_path",
    "local_audio_path",
    "audio_exists",
    "duration_sec",
    "duration_label",
    "source",
    "status",
    "created_at",
    "updated_at",
    "latest_run_id",
    "auto_section_count",
    "human_annotation_count",
    "is_reviewed",
]

SPLIT_FIELDS = SONG_FIELDS + ["split", "batch_index"]


def assign_splits(rows: list[dict[str, str]], seed: int) -> list[str]:
    n_total = len(rows)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 reviewed-song manifests and 8:1:1 splits.")
    parser.add_argument("--limit", type=int, default=70, help="Number of reviewed songs to use for the first run.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--date", default=date.today().isoformat().replace("-", ""), help="Date tag for output files.")
    parser.add_argument("--reviewed-json", default="", help="Optional local JSON snapshot from song_review_summary.")
    args = parser.parse_args()

    select = ",".join(
        [
            "id",
            "title",
            "artist",
            "audio_path",
            "duration_sec",
            "duration_label",
            "source",
            "status",
            "created_at",
            "updated_at",
            "latest_run_id",
            "auto_section_count",
            "human_annotation_count",
            "is_reviewed",
        ]
    )
    if args.reviewed_json:
        with Path(args.reviewed_json).open(encoding="utf-8") as handle:
            reviewed = json.load(handle)
    else:
        reviewed = supabase_get("song_review_summary", reviewed_song_query(select))
    if len(reviewed) < args.limit:
        raise SystemExit(f"Need {args.limit} reviewed songs, found {len(reviewed)}.")

    rows: list[dict[str, str]] = []
    for index, row in enumerate(reviewed[: args.limit], start=1):
        audio_path = row.get("audio_path") or ""
        local_path = local_audio_path(audio_path)
        rows.append(
            {
                "selected_order": str(index),
                "id": str(row.get("id") or ""),
                "title": str(row.get("title") or ""),
                "artist": str(row.get("artist") or ""),
                "audio_path": audio_path,
                "local_audio_path": str(local_path),
                "audio_exists": str(local_path.is_file()).lower(),
                "duration_sec": str(row.get("duration_sec") or ""),
                "duration_label": str(row.get("duration_label") or ""),
                "source": str(row.get("source") or ""),
                "status": str(row.get("status") or ""),
                "created_at": str(row.get("created_at") or ""),
                "updated_at": str(row.get("updated_at") or ""),
                "latest_run_id": str(row.get("latest_run_id") or ""),
                "auto_section_count": str(row.get("auto_section_count") or ""),
                "human_annotation_count": str(row.get("human_annotation_count") or ""),
                "is_reviewed": str(row.get("is_reviewed")).lower(),
            }
        )

    labels = assign_splits(rows, args.seed)
    split_rows = []
    for row, split in zip(rows, labels):
        split_row = dict(row)
        split_row["split"] = split
        split_row["batch_index"] = str((int(row["selected_order"]) + 9) // 10)
        split_rows.append(split_row)

    manifest_dir = PHASE2_ROOT / "data" / "manifests"
    write_csv(manifest_dir / f"reviewed_songs_{args.date}.csv", rows, SONG_FIELDS)
    write_csv(manifest_dir / f"phase2_initial_{args.limit}_manifest.csv", rows, SONG_FIELDS)
    write_csv(manifest_dir / f"phase2_initial_{args.limit}_splits.csv", split_rows, SPLIT_FIELDS)

    missing_audio = [row for row in rows if row["audio_exists"] != "true"]
    split_counts = {label: labels.count(label) for label in ["train", "val", "test"]}
    print(f"reviewed_count={len(reviewed)} selected={len(rows)} split_counts={split_counts}")
    print(f"missing_audio={len(missing_audio)}")
    for row in missing_audio[:10]:
        print(f"missing {row['id']} {row['local_audio_path']}")


if __name__ == "__main__":
    main()
