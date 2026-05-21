from __future__ import annotations

import functional_section_pipeline as pipeline

from .models import LyricFunctionReport


class LyricFunctionAgent:
    """Infer section function from timed lyric/vocal lines."""

    def run(self, lines: list[pipeline.LyricLine], title: str) -> tuple[list[str], LyricFunctionReport]:
        roles = pipeline.initial_line_roles(lines, title)
        context = pipeline.structure_context(lines)
        paragraphs = pipeline.group_lyric_paragraphs(lines, context)
        counts: dict[str, int] = {}
        for role in roles:
            counts[role] = counts.get(role, 0) + 1
        return roles, LyricFunctionReport(
            role_count=counts,
            paragraph_count=len(paragraphs),
            note="歌词/人声功能优先，结构位置和重复关系辅助",
        )

