from __future__ import annotations

import functional_section_pipeline as pipeline


class FusionAgent:
    """Fuse lyric function, vocal gaps, and acoustic boundary evidence."""

    def run(
        self,
        lines: list[pipeline.LyricLine],
        roles: list[str],
        features: dict[str, object],
        title: str,
        low_vocal_regions: list[tuple[float, float]],
        min_instrumental_gap: float,
    ) -> list[pipeline.SectionCandidate]:
        context = pipeline.structure_context(lines)
        sections = pipeline.make_role_blocks(lines, roles, min_instrumental_gap)
        sections = pipeline.add_intro_outro(sections, float(features["duration"]))
        pipeline.refine_instrumental_labels(sections, float(features["duration"]))
        sections = pipeline.merge_adjacent_same_type(sections)
        pipeline.snap_section_boundaries(sections, features)
        pipeline.promote_functional_bridges(sections, float(features["duration"]))
        sections = pipeline.split_overlong_repeated_sections(sections, lines, context)
        if low_vocal_regions:
            sections = pipeline.apply_low_vocal_regions(sections, low_vocal_regions, float(features["duration"]))
            pipeline.promote_functional_bridges(sections, float(features["duration"]))
            sections = pipeline.split_overlong_repeated_sections(sections, lines, context)
        pipeline.refresh_section_evidence(sections, features, title)
        return sections
