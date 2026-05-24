from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import functional_section_pipeline as pipeline


@dataclass
class TypeDecision:
    section_index: int
    old_type: str
    new_type: str
    lyric_function_score: float
    repetition_score: float
    position_score: float
    neighbor_score: float
    acoustic_profile_score: float
    template_score: float
    final_score: float
    reason: str


def refine_section_types(sections: list[pipeline.SectionCandidate], duration: float) -> tuple[list[pipeline.SectionCandidate], list[TypeDecision]]:
    decisions: list[TypeDecision] = []
    total = max(duration, 1.0)
    for index, section in enumerate(sections):
        old_type = section.section_type
        previous_type = previous_core_type(sections, index)
        next_type = next_core_type(sections, index)
        previous_sung_type = previous_sung_core_type(sections, index)
        next_sung_type = next_sung_core_type(sections, index)
        position = section.start / total
        has_lyrics = has_complete_lyrics(section)

        new_type, reason = choose_type(
            section,
            old_type,
            previous_type,
            next_type,
            previous_sung_type,
            next_sung_type,
            position,
            has_lyrics,
        )
        scores = score_type(section, new_type, previous_type, next_type, position, has_lyrics)
        final_score = round(
            0.35 * scores["lyric_function"]
            + 0.20 * scores["repetition"]
            + 0.15 * scores["position"]
            + 0.15 * scores["neighbor"]
            + 0.10 * scores["acoustic_profile"]
            + 0.05 * scores["template"],
            2,
        )
        if new_type != old_type:
            section.section_type = new_type
            section.lyric_evidence = append_reason(section.lyric_evidence, reason)
            section.need_human_review = True
        section.type_confidence = max(section.type_confidence, final_score)
        decisions.append(
            TypeDecision(
                section_index=index + 1,
                old_type=old_type,
                new_type=new_type,
                lyric_function_score=scores["lyric_function"],
                repetition_score=scores["repetition"],
                position_score=scores["position"],
                neighbor_score=scores["neighbor"],
                acoustic_profile_score=scores["acoustic_profile"],
                template_score=scores["template"],
                final_score=final_score,
                reason=reason,
            )
        )
    return sections, decisions


def choose_type(
    section: pipeline.SectionCandidate,
    old_type: str,
    previous_type: str,
    next_type: str,
    previous_sung_type: str,
    next_sung_type: str,
    position: float,
    has_lyrics: bool,
) -> tuple[str, str]:
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}".lower()

    if old_type in {"intro", "outro"} and has_lyrics:
        if position < 0.18:
            return "verse", "有完整歌词，intro/outro 硬约束降为开场 verse"
        if "祝福" in section.lyric_evidence or position > 0.72:
            return "chorus", "有完整歌词的尾声按 sung coda/chorus 功能处理，不标 outro"
        return "verse", "有完整歌词，不能标 intro/outro"

    if not has_lyrics:
        if position < 0.15 and next_sung_type in {"verse", "pre_chorus", "chorus"}:
            return "intro", "早段主歌前无完整歌词器乐，按 intro 处理"
        if position >= 0.82 or old_type == "outro":
            return "outro", "无完整歌词且在结尾，标 outro"
        return "bridge", "中段无完整歌词，标 bridge"

    if old_type == "pre_chorus" and next_type != "chorus":
        if previous_type == "chorus" and next_sung_type == "bridge":
            return "bridge", "副歌后复现/延展材料不导向 chorus，且后续进入 bridge，按连接/转折功能标 bridge"
        if position >= 0.45:
            return "bridge", "pre_chorus 未导向 chorus 且位于中后段，更像 bridge/transition"
        return "verse", "pre_chorus 未导向 chorus，降为 verse"

    if old_type == "pre_chorus" and previous_type == "chorus" and next_type == "chorus" and position >= 0.25:
        return "bridge", "位于 chorus 循环之间的非 hook 连接材料，按 bridge/transition 处理"

    if old_type == "verse" and previous_type == "chorus" and next_type == "chorus" and position >= 0.25:
        return "bridge", "位于 chorus 循环之间的非 hook 新材料，按 bridge/transition 处理"

    if old_type == "bridge" and next_type not in {"chorus", "outro"} and position < 0.35 and has_lyrics:
        return "verse", "早段有歌词材料且不导向 chorus，按 verse 处理"

    if old_type == "chorus":
        return "chorus", "保留 chorus：hook/final chorus/coda 功能优先"

    return old_type, "保留原类型：全局结构未提供更强反证"


def score_type(
    section: pipeline.SectionCandidate,
    section_type: str,
    previous_type: str,
    next_type: str,
    position: float,
    has_lyrics: bool,
) -> dict[str, float]:
    evidence = f"{section.lyric_evidence} {section.acoustic_evidence}".lower()
    lyric = 0.65
    repetition = 0.45
    position_score = 0.55
    neighbor = 0.55
    acoustic = 0.55
    template = 0.55

    if section_type == "chorus":
        lyric += 0.15 * float(any(word in evidence for word in ("hook", "副歌", "final", "coda", "祝福")))
        repetition += 0.25 * float(any(word in evidence for word in ("复现", "重复", "周期", "变体")))
        neighbor += 0.10 * float(previous_type in {"verse", "pre_chorus", "bridge", "chorus"})
        template += 0.10 * float(position >= 0.2)
    elif section_type == "pre_chorus":
        lyric += 0.15 * float(any(word in evidence for word in ("蓄力", "连接", "转向", "铺垫")))
        neighbor += 0.25 * float(next_type == "chorus")
        template += 0.15 * float(next_type == "chorus" and previous_type in {"verse", "chorus", "bridge"})
    elif section_type == "bridge":
        lyric += 0.15 * float(any(word in evidence for word in ("转折", "连接", "新材料", "对比", "器乐")))
        position_score += 0.15 * float(position >= 0.35)
        neighbor += 0.20 * float(next_type in {"chorus", "outro"})
        acoustic += 0.15 * float("无歌词" in section.lyric_evidence or "主唱退出" in section.vocal_evidence)
    elif section_type == "verse":
        lyric += 0.15 * float(any(word in evidence for word in ("叙事", "背景", "开场")))
        position_score += 0.10 * float(position < 0.45)
        neighbor += 0.10 * float(next_type in {"chorus", "pre_chorus", "bridge"})
    elif section_type in {"intro", "outro"}:
        lyric += 0.20 * float(not has_lyrics)
        acoustic += 0.15 * float("主唱退出" in section.vocal_evidence or "低人声" in section.acoustic_evidence)
        position_score += 0.20 * float((section_type == "intro" and position < 0.05) or (section_type == "outro" and position > 0.75))

    return {
        "lyric_function": round(min(0.95, lyric), 2),
        "repetition": round(min(0.95, repetition), 2),
        "position": round(min(0.95, position_score), 2),
        "neighbor": round(min(0.95, neighbor), 2),
        "acoustic_profile": round(min(0.95, acoustic), 2),
        "template": round(min(0.95, template), 2),
    }


def previous_core_type(sections: list[pipeline.SectionCandidate], index: int) -> str:
    for cursor in range(index - 1, -1, -1):
        return sections[cursor].section_type
    return ""


def next_core_type(sections: list[pipeline.SectionCandidate], index: int) -> str:
    for cursor in range(index + 1, len(sections)):
        return sections[cursor].section_type
    return ""


def previous_sung_core_type(sections: list[pipeline.SectionCandidate], index: int) -> str:
    for cursor in range(index - 1, -1, -1):
        if has_complete_lyrics(sections[cursor]):
            return sections[cursor].section_type
    return ""


def next_sung_core_type(sections: list[pipeline.SectionCandidate], index: int) -> str:
    for cursor in range(index + 1, len(sections)):
        if has_complete_lyrics(sections[cursor]):
            return sections[cursor].section_type
    return ""


def has_complete_lyrics(section: pipeline.SectionCandidate) -> bool:
    evidence = f"{section.lyric_evidence} {section.vocal_evidence} {section.acoustic_evidence}"
    no_complete_lyrics_markers = (
        "无歌词",
        "第一句歌词前",
        "主唱未正式进入",
        "主唱退出",
        "低人声",
        "弱主唱",
        "器乐",
        "instrumental",
    )
    if any(marker in evidence for marker in no_complete_lyrics_markers):
        return False
    return bool(
        section.text
        or "完整主唱" in section.vocal_evidence
        or ("主唱" in section.vocal_evidence and "退出" not in section.vocal_evidence)
    )


def append_reason(existing: str, reason: str) -> str:
    if not existing:
        return reason
    if reason in existing:
        return existing
    return f"{existing}；{reason}"


def write_type_decisions(path: Path, decisions: list[TypeDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "section_index",
        "old_type",
        "new_type",
        "lyric_function_score",
        "repetition_score",
        "position_score",
        "neighbor_score",
        "acoustic_profile_score",
        "template_score",
        "final_score",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.__dict__)
