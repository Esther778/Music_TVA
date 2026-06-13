#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any


PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE2_ROOT.parents[1]
PHASE1_ROOT = PROJECT_ROOT / "research" / "phase1_va_prediction" / "openl3_va_baseline"
SONGS_DIR = PROJECT_ROOT / "songs"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://itcarzyhlkiovskfdqut.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_IG4iiZuIDAoxhzZSPZnRRA_xSl0VRfN")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def supabase_get(table: str, query: str) -> list[dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{query}"
    command = [
        "curl",
        "-sS",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "-H",
        f"apikey: {SUPABASE_KEY}",
        "-H",
        f"Authorization: Bearer {SUPABASE_KEY}",
        url,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        return []
    return json.loads(completed.stdout)


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_audio_path(audio_path: str) -> Path:
    path = audio_path.strip()
    if path.startswith("/"):
        path = path[1:]
    return PROJECT_ROOT / path


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def reviewed_song_query(select: str, limit: int | None = None) -> str:
    parts = [
        f"select={quote(select)}",
        "status=eq.reviewed",
        "order=updated_at.asc,id.asc",
    ]
    if limit is not None:
        parts.append(f"limit={limit}")
    return "&".join(parts)

