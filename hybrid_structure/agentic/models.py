from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LyricSourceReport:
    transcript_csv: Path
    source: str
    line_count: int
    note: str


@dataclass
class LyricFunctionReport:
    role_count: dict[str, int]
    paragraph_count: int
    note: str


@dataclass
class BoundaryCandidate:
    candidate_time: float
    source: str
    confidence: float
    reason: str


@dataclass
class AcousticBoundaryReport:
    duration: float
    candidates: list[BoundaryCandidate]
    note: str


@dataclass
class VocalActivityReport:
    source: str
    low_vocal_regions: list[tuple[float, float]]
    note: str

