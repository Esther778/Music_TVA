from __future__ import annotations

import csv
from pathlib import Path

import functional_section_pipeline as pipeline

from .models import AcousticBoundaryReport, BoundaryCandidate


class AcousticBoundaryAgent:
    """Extract acoustic features and produce boundary candidates only."""

    def run(
        self,
        audio: Path,
        lines: list[pipeline.LyricLine],
        output_candidates: Path | None = None,
    ) -> tuple[dict[str, object], AcousticBoundaryReport]:
        features = pipeline.load_acoustic_features(audio)
        candidates = self._lyric_candidates(lines)
        candidates.extend(self._novelty_candidates(features))
        candidates = self._merge_candidates(candidates)
        if output_candidates:
            self.write_candidates(output_candidates, candidates)
        return features, AcousticBoundaryReport(
            duration=float(features["duration"]),
            candidates=candidates,
            note="声学变化只作为边界候选，不单独决定 section 类型",
        )

    def _lyric_candidates(self, lines: list[pipeline.LyricLine]) -> list[BoundaryCandidate]:
        candidates: list[BoundaryCandidate] = []
        if not lines:
            return candidates
        candidates.append(BoundaryCandidate(lines[0].start, "vocal_enter", 0.82, "第一次完整主唱进入"))
        context = pipeline.structure_context(lines)
        for index in range(1, len(lines)):
            gap = lines[index].start - lines[index - 1].end
            if gap >= context.long_gap_seconds:
                candidates.append(
                    BoundaryCandidate(lines[index - 1].end, "lyric_gap", 0.78, "连续歌词之间出现长无歌词空隙")
                )
                candidates.append(
                    BoundaryCandidate(lines[index].start, "vocal_reenter", 0.76, "长空隙后主唱重新进入")
                )
            elif gap >= context.paragraph_gap_seconds:
                candidates.append(BoundaryCandidate(lines[index].start, "lyric_paragraph", 0.62, "歌词段落边界"))
        return candidates

    def _novelty_candidates(self, features: dict[str, object]) -> list[BoundaryCandidate]:
        import numpy as np

        times = features["times"]
        novelty = features["novelty"]
        if len(novelty) < 3:
            return []
        threshold = float(np.quantile(novelty, 0.9))
        candidates: list[BoundaryCandidate] = []
        for index in range(1, len(novelty) - 1):
            value = float(novelty[index])
            if value < threshold:
                continue
            if value >= float(novelty[index - 1]) and value >= float(novelty[index + 1]):
                candidates.append(
                    BoundaryCandidate(
                        float(times[index]),
                        "acoustic_novelty",
                        min(0.9, 0.45 + value * 0.55),
                        "频谱/节奏/音色综合 novelty 峰值",
                    )
                )
        return candidates

    def _merge_candidates(self, candidates: list[BoundaryCandidate], window: float = 2.0) -> list[BoundaryCandidate]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: item.candidate_time)
        merged: list[BoundaryCandidate] = []
        group: list[BoundaryCandidate] = [ordered[0]]
        for item in ordered[1:]:
            if item.candidate_time - group[-1].candidate_time <= window:
                group.append(item)
            else:
                merged.append(self._merge_group(group))
                group = [item]
        merged.append(self._merge_group(group))
        return merged

    def _merge_group(self, group: list[BoundaryCandidate]) -> BoundaryCandidate:
        best = max(group, key=lambda item: item.confidence)
        sources = "+".join(sorted({item.source for item in group}))
        confidence = min(0.95, max(item.confidence for item in group) + 0.04 * (len(group) - 1))
        reasons = "；".join(dict.fromkeys(item.reason for item in group))
        return BoundaryCandidate(round(best.candidate_time, 2), sources, round(confidence, 2), reasons)

    def write_candidates(self, path: Path, candidates: list[BoundaryCandidate]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_time", "source", "confidence", "reason"])
            writer.writeheader()
            for item in candidates:
                writer.writerow(
                    {
                        "candidate_time": pipeline.base.format_timestamp(item.candidate_time),
                        "source": item.source,
                        "confidence": item.confidence,
                        "reason": item.reason,
                    }
                )

