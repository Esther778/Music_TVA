#!/usr/bin/env python3
"""Build a GPT-readable data-only package for the 100-song top60 quota corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
DATA = PHASE2 / "data/top60_quota_100_20260813"
ANALYSIS = PHASE2 / "analysis/top60_quota_100_20260813"
VA_MUSIC = ANALYSIS / "va_music"
EXPORTS = ROOT / "exports"
PACKAGE_ID = "gpt_top60_quota_100_section_va_data_package_20260820"
PACKAGE = EXPORTS / PACKAGE_ID
ZIP_PATH = EXPORTS / f"{PACKAGE_ID}.zip"
CHINA_TZ = timezone(timedelta(hours=8))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, rel: str) -> Path:
    dst = PACKAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_json(rel: str, data: dict | list) -> Path:
    dst = PACKAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dst


def write_text(rel: str, text: str) -> Path:
    dst = PACKAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst


def write_csv(rel: str, rows: list[dict], fieldnames: list[str]) -> Path:
    dst = PACKAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dst


def add_core_files() -> None:
    # 02 corpus inventory
    copy_file(DATA / "stage1/top60_quota_100_manifest.csv", "02_corpus_inventory/top60_quota_100_manifest.csv")
    copy_file(DATA / "stage1/top60_quota_100_boundaries.json", "02_corpus_inventory/top60_quota_100_boundaries.json")
    copy_file(DATA / "stage1/top60_quota_100_protocol.json", "02_corpus_inventory/top60_quota_100_protocol.json")
    for name in [
        "top60_quota_100_artist_summary.csv",
        "top60_quota_100_continuity_issues.json",
        "top60_quota_100_analysis_manifest.json",
    ]:
        src = ANALYSIS / name
        if src.exists():
            copy_file(src, f"02_corpus_inventory/{name}")

    # 03/04/05/06 model outputs and aligned features
    copy_file(DATA / "stage2/top60_quota_100_frame_va.csv", "03_frame_level_va/top60_quota_100_frame_va.csv")
    copy_file(
        DATA / "stage3/top60_quota_100_frame_section_alignment.csv",
        "04_frame_section_alignment/top60_quota_100_frame_section_alignment.csv",
    )
    copy_file(DATA / "stage3/top60_quota_100_section_features.csv", "05_section_level/top60_quota_100_section_features.csv")
    copy_file(
        DATA / "stage3/top60_quota_100_song_structure_features.csv",
        "06_song_level/top60_quota_100_song_structure_features.csv",
    )

    # 07 paper-level statistics
    for src in sorted((ANALYSIS / "paper_statistics").glob("*")):
        if src.is_file():
            copy_file(src, f"07_paper_statistics/{src.name}")

    # 09 precomputed VA/MIR/MER tables
    for src in sorted((VA_MUSIC / "tables").glob("*.csv")):
        copy_file(src, f"09_derived_tables/{src.name}")
    copy_file(VA_MUSIC / "TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md", "09_derived_tables/TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md")

    # 10 provenance and audits
    for src, rel in [
        (DATA / "stage2/top60_quota_100_inference_manifest.json", "10_protocols_and_audits/top60_quota_100_inference_manifest.json"),
        (DATA / "stage2/top60_quota_100_output_audit_summary.json", "10_protocols_and_audits/top60_quota_100_output_audit_summary.json"),
        (DATA / "stage2/top60_quota_100_output_audit.csv", "10_protocols_and_audits/top60_quota_100_output_audit.csv"),
        (DATA / "stage2/reuse_audit.csv", "10_protocols_and_audits/reuse_audit.csv"),
        (DATA / "stage2/inference_log.csv", "10_protocols_and_audits/inference_log.csv"),
        (DATA / "stage3/top60_quota_100_alignment_manifest.json", "10_protocols_and_audits/top60_quota_100_alignment_manifest.json"),
        (DATA / "stage3/top60_quota_100_alignment_audit.csv", "10_protocols_and_audits/top60_quota_100_alignment_audit.csv"),
        (DATA / "supabase_sql_batch_manifest.json", "10_protocols_and_audits/supabase_sql_batch_manifest.json"),
        (VA_MUSIC / "va_music_analysis_manifest.json", "10_protocols_and_audits/va_music_analysis_manifest.json"),
    ]:
        if src.exists():
            copy_file(src, rel)


def add_analysis_ready_and_extra_derivatives() -> None:
    manifest = pd.read_csv(DATA / "stage1/top60_quota_100_manifest.csv")
    sections = pd.read_csv(DATA / "stage3/top60_quota_100_section_features.csv")
    songs = pd.read_csv(DATA / "stage3/top60_quota_100_song_structure_features.csv")
    frames = pd.read_csv(DATA / "stage2/top60_quota_100_frame_va.csv")
    paper_songs = pd.read_csv(ANALYSIS / "paper_statistics/paper_song_level_features.csv")

    # Analysis-ready compact tables.
    sections.merge(
        manifest[["song_id", "rank", "quota_artist", "selected_order", "audio_sha256", "boundary_sha256"]],
        on="song_id",
        how="left",
    ).to_csv(PACKAGE / "08_analysis_ready/section_analysis_ready.csv", index=False)
    songs.merge(
        paper_songs[["song_id", "quota_artist", "rank", "rank_band", "unique_section_types", "chorus_count", "verse_count", "bridge_count", "pre_chorus_count"]],
        on="song_id",
        how="left",
    ).to_csv(PACKAGE / "08_analysis_ready/song_analysis_ready.csv", index=False)

    mapping = pd.DataFrame(
        [
            {"raw_section_type": "intro", "normalized_section_type": "intro", "mapping_rule": "canonical"},
            {"raw_section_type": "verse", "normalized_section_type": "verse", "mapping_rule": "canonical"},
            {"raw_section_type": "pre_chorus", "normalized_section_type": "pre_chorus", "mapping_rule": "canonical"},
            {"raw_section_type": "chorus", "normalized_section_type": "chorus", "mapping_rule": "canonical"},
            {"raw_section_type": "bridge", "normalized_section_type": "bridge", "mapping_rule": "canonical"},
            {"raw_section_type": "interlude", "normalized_section_type": "interlude", "mapping_rule": "canonical"},
            {"raw_section_type": "outro", "normalized_section_type": "outro", "mapping_rule": "canonical"},
        ]
    )
    mapping.to_csv(PACKAGE / "08_analysis_ready/section_name_mapping.csv", index=False)

    # Normalized 101-point complete-song trajectories, one row per song/time point.
    trajectory_rows = []
    grid = [i / 100 for i in range(101)]
    for song_id, group in frames.groupby("song_id", sort=False):
        group = group.sort_values("time_sec")
        max_time = float(group["time_sec"].max())
        if max_time <= 0:
            continue
        title = group["title"].iloc[0]
        artist = group["artist"].iloc[0]
        for point in grid:
            trajectory_rows.append(
                {
                    "song_id": song_id,
                    "title": title,
                    "artist": artist,
                    "normalized_time": point,
                    "valence": float(group["valence"].iloc[0]) if len(group) == 1 else float(np.interp(point, group["time_sec"].to_numpy() / max_time, group["valence"].to_numpy())),
                    "arousal": float(group["arousal"].iloc[0]) if len(group) == 1 else float(np.interp(point, group["time_sec"].to_numpy() / max_time, group["arousal"].to_numpy())),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(PACKAGE / "08_analysis_ready/normalized_complete_song_trajectories_101pts.csv", index=False)

    # Extra derived tables that mirror the first 97-song package style.
    adjacent = []
    sequences = []
    for song_id, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order").reset_index(drop=True)
        types = group["section_type"].tolist()
        collapsed = [t for i, t in enumerate(types) if i == 0 or t != types[i - 1]]
        sequences.append(
            {
                "song_id": song_id,
                "title": group["title"].iloc[0],
                "artist": group["artist"].iloc[0],
                "raw_sequence": " > ".join(types),
                "collapsed_sequence": " > ".join(collapsed),
                "raw_section_count": len(types),
                "collapsed_section_count": len(collapsed),
            }
        )
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            cur = group.iloc[idx]
            dv = cur["mean_valence"] - prev["mean_valence"]
            da = cur["mean_arousal"] - prev["mean_arousal"]
            adjacent.append(
                {
                    "song_id": song_id,
                    "title": cur["title"],
                    "artist": cur["artist"],
                    "from_order": int(prev["section_order"]),
                    "to_order": int(cur["section_order"]),
                    "from_type": prev["section_type"],
                    "to_type": cur["section_type"],
                    "transition": f"{prev['section_type']}->{cur['section_type']}",
                    "delta_valence": dv,
                    "delta_arousal": da,
                    "va_distance": (dv ** 2 + da ** 2) ** 0.5,
                    "normalized_position": cur["normalized_midpoint"],
                    "direction_t0.03": direction(dv, da, 0.03),
                    "direction_t0.05": direction(dv, da, 0.05),
                    "direction_t0.07": direction(dv, da, 0.07),
                }
            )

    pd.DataFrame(adjacent).to_csv(PACKAGE / "09_derived_tables/adjacent_section_transitions_gpt.csv", index=False)
    pd.DataFrame(sequences).to_csv(PACKAGE / "09_derived_tables/musicological_collapsed_form_sequences_by_song.csv", index=False)
    pd.DataFrame(sequences).groupby("collapsed_sequence").agg(
        song_count=("song_id", "count"),
        example_song_id=("song_id", "first"),
        example_title=("title", "first"),
        example_artist=("artist", "first"),
        mean_collapsed_section_count=("collapsed_section_count", "mean"),
    ).reset_index().sort_values(["song_count", "collapsed_sequence"], ascending=[False, True]).to_csv(
        PACKAGE / "09_derived_tables/musicological_collapsed_form_sequence_summary.csv", index=False
    )

    section_type_means = sections.groupby(["song_id", "title", "artist", "section_type"]).agg(
        section_count=("section_type", "size"),
        mean_valence=("mean_valence", "mean"),
        mean_arousal=("mean_arousal", "mean"),
        duration_sec=("section_duration_sec", "sum"),
    ).reset_index()
    section_type_means.to_csv(PACKAGE / "09_derived_tables/song_by_section_type_means.csv", index=False)
    section_type_means.merge(manifest[["song_id", "rank", "quota_artist"]], on="song_id", how="left").to_csv(
        PACKAGE / "09_derived_tables/song_by_section_type_means_with_rank.csv", index=False
    )


def direction(delta_v: float, delta_a: float, threshold: float) -> str:
    v = "valence_up" if delta_v > threshold else "valence_down" if delta_v < -threshold else "valence_flat"
    a = "arousal_up" if delta_a > threshold else "arousal_down" if delta_a < -threshold else "arousal_flat"
    return f"{v}|{a}"


def add_index_files() -> None:
    created_at = datetime.now(CHINA_TZ).isoformat(timespec="seconds")
    package_contents = [
        {"directory": "00_index", "category": "index", "description": "Package scope, contents, schemas, and file manifest"},
        {"directory": "02_corpus_inventory", "category": "corpus", "description": "100-song top60 quota manifest, boundaries, protocol, artist coverage"},
        {"directory": "03_frame_level_va", "category": "frame_va", "description": "Teacher frame-level valence/arousal predictions"},
        {"directory": "04_frame_section_alignment", "category": "alignment", "description": "Frame rows aligned to human-QC section boundaries"},
        {"directory": "05_section_level", "category": "section", "description": "Section-level VA and transition-local features"},
        {"directory": "06_song_level", "category": "song", "description": "Song-level VA and structure features"},
        {"directory": "07_paper_statistics", "category": "paper_stats", "description": "Paper-oriented MIR/statistical tables and notes"},
        {"directory": "08_analysis_ready", "category": "analysis_ready", "description": "Clean merged GPT-ready tables and normalized trajectories"},
        {"directory": "09_derived_tables", "category": "derived", "description": "VA/MIR/MER derived tables, transitions, sequences, repeated sections, peaks"},
        {"directory": "10_protocols_and_audits", "category": "provenance", "description": "Teacher, alignment, Supabase, and audit manifests"},
    ]
    write_csv("00_index/PACKAGE_CONTENTS.csv", package_contents, ["directory", "category", "description"])
    write_json(
        "00_index/PACKAGE_SCOPE.json",
        {
            "package_id": PACKAGE_ID,
            "created_at": created_at,
            "purpose": "data-only export for independent GPT-assisted analysis",
            "corpus": "QQ Music top60 quota 100-song Chinese pop corpus",
            "song_count": 100,
            "section_count": 1046,
            "frame_count": 51448,
            "frame_hop_sec": 0.5,
            "primary_analysis_scope": "100 songs selected under rank quota rule: ranks 1-10 target 3 songs, 11-30 target 2 songs, 31-60 target 1 song",
            "not_included": ["audio files", "intermediate .npz feature caches", "Supabase SQL batch fallback files"],
            "teacher_models": ["pmemo_final_clean_valence_teacher_v1", "pmemo_final_clean_arousal_teacher_v1"],
            "supabase_tables": [
                "paper_top60_quota_100_va_runs",
                "paper_top60_quota_100_va_frames",
                "paper_top60_quota_100_section_va",
                "paper_top60_quota_100_song_va",
            ],
            "source_commit": git_commit(),
        },
    )
    write_json(
        "00_index/PACKAGE_SUMMARY.json",
        {
            "package_id": PACKAGE_ID,
            "created_at": created_at,
            "headline": "GPT-readable data package for the 100-song top60 quota VA/MIR/MER corpus",
            "recommended_start_files": [
                "00_index/PACKAGE_SCOPE.json",
                "00_index/PACKAGE_CONTENTS.csv",
                "08_analysis_ready/song_analysis_ready.csv",
                "08_analysis_ready/section_analysis_ready.csv",
                "09_derived_tables/TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md",
                "07_paper_statistics/PAPER_STATISTICS_SUMMARY_ZH.md",
            ],
        },
    )
    write_text(
        "README_FOR_GPT.md",
        f"""# GPT analysis data package: top60 quota 100 songs

This is a data-only package. It contains no audio and no intermediate model feature cache.

Use `00_index/PACKAGE_SCOPE.json` first, then `00_index/PACKAGE_CONTENTS.csv`.

Best starting tables:

- `08_analysis_ready/song_analysis_ready.csv`
- `08_analysis_ready/section_analysis_ready.csv`
- `08_analysis_ready/normalized_complete_song_trajectories_101pts.csv`
- `09_derived_tables/va_paired_section_tests.csv`
- `09_derived_tables/va_transition_summary.csv`
- `09_derived_tables/musicological_collapsed_form_sequences_by_song.csv`
- `07_paper_statistics/PAPER_STATISTICS_SUMMARY_ZH.md`
- `09_derived_tables/TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md`

Interpretation boundary: VA values are teacher-model pseudo-labels for musicological analysis, not direct human emotion ground truth.
Generated at {datetime.now(CHINA_TZ).isoformat(timespec='seconds')}.
""",
    )


def git_commit() -> str:
    head = ROOT / ".git/HEAD"
    if not head.exists():
        return "unknown"
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = ROOT / ".git" / text[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else "unknown"
    return text


def build_catalogs() -> None:
    file_rows = []
    csv_rows = []
    json_catalog = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path == ZIP_PATH:
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        file_rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=0)
            with path.open("r", encoding="utf-8", errors="replace") as f:
                row_count = max(sum(1 for _ in f) - 1, 0)
            csv_rows.append({"path": rel, "rows": row_count, "columns": len(df.columns), "column_names": "|".join(df.columns)})
        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                json_catalog.append({"path": rel, "top_level_type": type(data).__name__, "keys": "|".join(data.keys()) if isinstance(data, dict) else ""})
            except json.JSONDecodeError:
                json_catalog.append({"path": rel, "top_level_type": "invalid_json", "keys": ""})
    write_csv("00_index/FILE_MANIFEST.csv", file_rows, ["path", "bytes", "sha256"])
    write_csv("00_index/CSV_SCHEMA_CATALOG.csv", csv_rows, ["path", "rows", "columns", "column_names"])
    write_json("00_index/JSON_SCHEMA_CATALOG.json", json_catalog)


def zip_package() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PACKAGE.parent))


def main() -> None:
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for rel in [
        "08_analysis_ready",
        "09_derived_tables",
    ]:
        (PACKAGE / rel).mkdir(parents=True, exist_ok=True)
    add_core_files()
    add_analysis_ready_and_extra_derivatives()
    add_index_files()
    build_catalogs()
    zip_package()
    print(json.dumps({"package": str(PACKAGE), "zip": str(ZIP_PATH), "zip_bytes": ZIP_PATH.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
