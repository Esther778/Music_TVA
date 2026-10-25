#!/usr/bin/env python3
"""Audit a local PMEmo dataset drop before normalization/training.

The script is intentionally read-only with respect to raw data. It recursively
scans the PMEmo raw directory, summarizes audio/table files, detects likely VA
and timestamp columns, and writes compact audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "audit"

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
TABLE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".mat"}

ID_HINTS = ("song", "music", "clip", "track", "audio", "id", "filename", "file")
TIME_HINTS = ("time", "timestamp", "sec", "second", "frame", "sample")
VALENCE_HINTS = ("valence", "val", "v_mean", "mean_v", "pleasure")
AROUSAL_HINTS = ("arousal", "aro", "a_mean", "mean_a", "energy")


@dataclass
class AudioRecord:
    path: str
    suffix: str
    size_mb: float
    duration_sec: float | None
    duration_source: str | None
    error: str | None


@dataclass
class TableRecord:
    path: str
    suffix: str
    size_mb: float
    rows: int | None
    columns: list[str]
    id_columns: list[str]
    time_columns: list[str]
    valence_columns: list[str]
    arousal_columns: list[str]
    numeric_ranges: dict[str, dict[str, float | int | None]]
    likely_time_step: dict[str, float | None]
    error: str | None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 6)


def _contains_any(name: str, hints: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in hints)


def _try_audio_duration(path: Path) -> tuple[float | None, str | None, str | None]:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        if info.frames and info.samplerate:
            return round(float(info.frames) / float(info.samplerate), 6), "soundfile", None
    except Exception as exc:
        soundfile_error = f"soundfile: {exc}"
    else:
        soundfile_error = "soundfile: missing frames or samplerate"

    try:
        import librosa  # type: ignore

        duration = librosa.get_duration(path=str(path))
        if duration and math.isfinite(duration):
            return round(float(duration), 6), "librosa", None
    except Exception as exc:
        return None, None, f"{soundfile_error}; librosa: {exc}"

    return None, None, soundfile_error


def _read_table(path: Path) -> tuple[Any | None, str | None]:
    suffix = path.suffix.lower()
    try:
        import pandas as pd  # type: ignore

        if suffix == ".csv":
            return pd.read_csv(path), None
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t"), None
        if suffix == ".txt":
            try:
                return pd.read_csv(path, sep=None, engine="python"), None
            except Exception:
                return pd.read_csv(path, sep="\t"), None
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path), None
        if suffix == ".json":
            return pd.read_json(path), None
        if suffix == ".mat":
            from scipy.io import loadmat  # type: ignore

            mat = loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            return {"mat_keys": keys}, None
    except Exception as exc:
        return None, str(exc)

    return None, f"unsupported table suffix: {suffix}"


def _table_record(path: Path) -> TableRecord:
    obj, error = _read_table(path)
    if error:
        return TableRecord(
            path=_rel(path),
            suffix=path.suffix.lower(),
            size_mb=_size_mb(path),
            rows=None,
            columns=[],
            id_columns=[],
            time_columns=[],
            valence_columns=[],
            arousal_columns=[],
            numeric_ranges={},
            likely_time_step={},
            error=error,
        )

    if isinstance(obj, dict):
        columns = list(obj.get("mat_keys", []))
        return TableRecord(
            path=_rel(path),
            suffix=path.suffix.lower(),
            size_mb=_size_mb(path),
            rows=None,
            columns=columns,
            id_columns=[c for c in columns if _contains_any(c, ID_HINTS)],
            time_columns=[c for c in columns if _contains_any(c, TIME_HINTS)],
            valence_columns=[c for c in columns if _contains_any(c, VALENCE_HINTS)],
            arousal_columns=[c for c in columns if _contains_any(c, AROUSAL_HINTS)],
            numeric_ranges={},
            likely_time_step={},
            error=None,
        )

    df = obj
    columns = [str(c) for c in df.columns]
    id_cols = [c for c in columns if _contains_any(c, ID_HINTS)]
    time_cols = [c for c in columns if _contains_any(c, TIME_HINTS)]
    val_cols = [c for c in columns if _contains_any(c, VALENCE_HINTS)]
    aro_cols = [c for c in columns if _contains_any(c, AROUSAL_HINTS)]

    numeric_ranges: dict[str, dict[str, float | int | None]] = {}
    likely_time_step: dict[str, float | None] = {}
    for col in columns:
        try:
            values = df[col]
            numeric = values.dropna().astype(float)
        except Exception:
            continue
        if numeric.empty:
            continue
        numeric_ranges[col] = {
            "count": int(numeric.shape[0]),
            "min": round(float(numeric.min()), 6),
            "max": round(float(numeric.max()), 6),
            "mean": round(float(numeric.mean()), 6),
            "std": round(float(numeric.std()), 6) if numeric.shape[0] > 1 else 0.0,
        }
        if col in time_cols and numeric.shape[0] >= 3:
            diffs = numeric.sort_values().diff().dropna()
            diffs = diffs[diffs > 0]
            likely_time_step[col] = round(float(diffs.median()), 6) if not diffs.empty else None

    return TableRecord(
        path=_rel(path),
        suffix=path.suffix.lower(),
        size_mb=_size_mb(path),
        rows=int(df.shape[0]),
        columns=columns,
        id_columns=id_cols,
        time_columns=time_cols,
        valence_columns=val_cols,
        arousal_columns=aro_cols,
        numeric_ranges=numeric_ranges,
        likely_time_step=likely_time_step,
        error=None,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def audit(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_dir = raw_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in raw_dir.rglob("*") if p.is_file()]
    audio_files = sorted([p for p in all_files if p.suffix.lower() in AUDIO_EXTS])
    table_files = sorted([p for p in all_files if p.suffix.lower() in TABLE_EXTS])

    audio_records: list[AudioRecord] = []
    for path in audio_files:
        duration, source, error = _try_audio_duration(path)
        audio_records.append(
            AudioRecord(
                path=_rel(path),
                suffix=path.suffix.lower(),
                size_mb=_size_mb(path),
                duration_sec=duration,
                duration_source=source,
                error=error,
            )
        )

    table_records = [_table_record(path) for path in table_files]

    durations = [r.duration_sec for r in audio_records if r.duration_sec is not None]
    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir.resolve()),
        "file_count": len(all_files),
        "audio_count": len(audio_records),
        "table_count": len(table_records),
        "audio_duration_sec": {
            "min": round(min(durations), 6) if durations else None,
            "max": round(max(durations), 6) if durations else None,
            "mean": round(sum(durations) / len(durations), 6) if durations else None,
        },
        "tables_with_valence_candidates": sum(bool(r.valence_columns) for r in table_records),
        "tables_with_arousal_candidates": sum(bool(r.arousal_columns) for r in table_records),
        "tables_with_time_candidates": sum(bool(r.time_columns) for r in table_records),
        "audio_errors": sum(bool(r.error) for r in audio_records),
        "table_errors": sum(bool(r.error) for r in table_records),
    }

    payload = {
        "summary": summary,
        "audio": [asdict(r) for r in audio_records],
        "tables": [asdict(r) for r in table_records],
    }

    (output_dir / "pmemo_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_csv(
        output_dir / "pmemo_audio_inventory.csv",
        [asdict(r) for r in audio_records],
        ["path", "suffix", "size_mb", "duration_sec", "duration_source", "error"],
    )
    _write_csv(
        output_dir / "pmemo_table_inventory.csv",
        [
            {
                "path": r.path,
                "suffix": r.suffix,
                "size_mb": r.size_mb,
                "rows": r.rows,
                "columns": "|".join(r.columns),
                "id_columns": "|".join(r.id_columns),
                "time_columns": "|".join(r.time_columns),
                "valence_columns": "|".join(r.valence_columns),
                "arousal_columns": "|".join(r.arousal_columns),
                "error": r.error,
            }
            for r in table_records
        ],
        [
            "path",
            "suffix",
            "size_mb",
            "rows",
            "columns",
            "id_columns",
            "time_columns",
            "valence_columns",
            "arousal_columns",
            "error",
        ],
    )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = audit(args.raw_dir, args.output_dir)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
