from __future__ import annotations

from pathlib import Path

import vocal_activity

from .models import VocalActivityReport


class VocalActivityAgent:
    """Detect low-vocal regions from an optional separated vocals stem."""

    def run(self, vocals_stem: Path | None) -> VocalActivityReport:
        if not vocals_stem:
            return VocalActivityReport("none", [], "未提供 vocals stem，跳过低人声校正")
        low_regions, _stats = vocal_activity.detect_low_vocal_regions(vocals_stem, min_duration=5.0)
        return VocalActivityReport(str(vocals_stem), low_regions, "使用 Demucs vocals stem 检测 intro/bridge/outro 候选")

