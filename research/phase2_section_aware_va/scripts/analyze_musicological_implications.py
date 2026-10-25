#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate the 97-song statistical results into bounded musicological claims."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
ANALYSIS = PHASE2 / "analysis/full_corpus_97"
DATA = ANALYSIS / "data"
TABLES = ANALYSIS / "tables"
FIGURES = ANALYSIS / "figures"
REPORTS = ANALYSIS / "reports"
TYPE_ORDER = ["intro", "verse", "pre_chorus", "chorus", "bridge", "interlude", "outro"]
TYPE_LABELS = {
    "intro": "Intro", "verse": "Verse", "pre_chorus": "Pre-chorus",
    "chorus": "Chorus", "bridge": "Bridge", "interlude": "Interlude", "outro": "Outro",
}
COLORS = {
    "intro": "#4C78A8", "verse": "#72B7B2", "pre_chorus": "#F2CF5B",
    "chorus": "#E45756", "bridge": "#B279A2", "interlude": "#59A14F", "outro": "#9D755D",
}


def collapse_adjacent(sequence: list[str]) -> list[str]:
    result: list[str] = []
    for item in sequence:
        if not result or result[-1] != item:
            result.append(item)
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_function_profiles(sections: pd.DataFrame) -> pd.DataFrame:
    profile = sections.groupby("section_type").agg(
        section_count=("song_id", "size"),
        song_count=("song_id", "nunique"),
        mean_position=("normalized_midpoint", "mean"),
        median_position=("normalized_midpoint", "median"),
        mean_duration_sec=("section_duration_sec", "mean"),
        median_duration_sec=("section_duration_sec", "median"),
        mean_valence=("mean_valence", "mean"),
        median_valence=("mean_valence", "median"),
        mean_arousal=("mean_arousal", "mean"),
        median_arousal=("mean_arousal", "median"),
    ).reindex(TYPE_ORDER).reset_index()
    profile["song_coverage"] = profile.song_count / sections.song_id.nunique()
    profile["descriptive_formal_role"] = profile.section_type.map({
        "intro": "opening establishment",
        "verse": "core-cycle initiation/narrative space",
        "pre_chorus": "optional buildup within the core cycle",
        "chorus": "core-cycle arrival and recurrent focal section",
        "bridge": "late-cycle contrast outside the recurring core",
        "interlude": "instrumental hinge between rotations",
        "outro": "closure and post-peak dissipation",
    })
    return profile


def build_form_grammar(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    motif_rows = []
    for song_id, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order")
        raw = group.section_type.tolist()
        collapsed = collapse_adjacent(raw)
        detail_rows.append({
            "song_id": song_id, "title": group.title.iloc[0], "artist": group.artist.iloc[0],
            "raw_sequence": " > ".join(raw), "collapsed_sequence": " > ".join(collapsed),
            "raw_section_count": len(raw), "collapsed_section_count": len(collapsed),
        })
        for length in (2, 3, 4):
            for index in range(len(collapsed) - length + 1):
                motif_rows.append({
                    "song_id": song_id, "motif_length": length,
                    "motif": " > ".join(collapsed[index:index + length]),
                })
    detail = pd.DataFrame(detail_rows)
    summary = detail.groupby("collapsed_sequence").agg(
        song_count=("song_id", "nunique"),
        mean_raw_sections=("raw_section_count", "mean"),
        collapsed_section_count=("collapsed_section_count", "first"),
    ).reset_index().sort_values(["song_count", "collapsed_section_count"], ascending=[False, True])
    summary["song_fraction"] = summary.song_count / len(detail)
    motifs = pd.DataFrame(motif_rows).groupby(["motif_length", "motif"]).agg(
        event_count=("song_id", "size"), song_count=("song_id", "nunique")
    ).reset_index().sort_values(["motif_length", "song_count", "event_count"], ascending=[True, False, False])
    motifs["song_fraction"] = motifs.song_count / len(detail)
    return detail, summary, motifs


def build_transition_profiles(transitions: pd.DataFrame) -> pd.DataFrame:
    profile = transitions.groupby(["transition", "from_type", "to_type"]).agg(
        event_count=("song_id", "size"), song_count=("song_id", "nunique"),
        mean_delta_valence=("delta_valence", "mean"),
        mean_delta_arousal=("delta_arousal", "mean"),
        median_delta_valence=("delta_valence", "median"),
        median_delta_arousal=("delta_arousal", "median"),
        mean_va_distance=("va_distance", "mean"),
    ).reset_index().sort_values(["song_count", "event_count"], ascending=False)
    return profile


def build_chorus_peak_occurrences(sections: pd.DataFrame) -> pd.DataFrame:
    peaks = sections.loc[sections.groupby("song_id").mean_arousal.idxmax()].copy()
    chorus = peaks[peaks.section_type == "chorus"].copy()
    chorus["is_last_chorus_peak"] = chorus.is_last_type_occurrence.astype(bool)
    return chorus[[
        "song_id", "title", "artist", "section_order", "type_occurrence_index",
        "type_occurrence_count", "is_last_chorus_peak", "normalized_midpoint", "mean_arousal",
    ]].sort_values("mean_arousal", ascending=False)


def build_case_studies(sections: pd.DataFrame, transitions: pd.DataFrame,
                       song_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    song_transition = transitions.groupby(["song_id", "transition"])[
        ["delta_valence", "delta_arousal"]
    ].mean().reset_index()
    arousal = song_transition.pivot(index="song_id", columns="transition", values="delta_arousal")
    if {"verse->pre_chorus", "pre_chorus->chorus"}.issubset(arousal.columns):
        score = arousal[["verse->pre_chorus", "pre_chorus->chorus"]].dropna().sum(axis=1)
        for label, series in (("strong_staged_buildup", score.nlargest(3)),
                              ("weak_or_subverted_staged_buildup", score.nsmallest(3))):
            for song_id, value in series.items():
                meta = song_features.set_index("song_id").loc[song_id]
                rows.append({"song_id": song_id, "title": meta.title, "artist": meta.artist,
                             "case_category": label, "primary_value": value,
                             "listening_question": "Does the Verse-Pre-chorus-Chorus rotation perceptually build toward an arrival?"})
    release = transitions[transitions.transition == "chorus->outro"].groupby("song_id").delta_arousal.mean().nsmallest(3)
    for song_id, value in release.items():
        meta = song_features.set_index("song_id").loc[song_id]
        rows.append({"song_id": song_id, "title": meta.title, "artist": meta.artist,
                     "case_category": "strong_chorus_to_outro_release", "primary_value": value,
                     "listening_question": "Does the Outro function as dissipation, closure, or abrupt withdrawal?"})
    peaks = sections.loc[sections.groupby("song_id").mean_arousal.idxmax()]
    for section_type, label in (("bridge", "bridge_as_arousal_peak"), ("outro", "outro_as_arousal_peak")):
        for _, row in peaks[peaks.section_type == section_type].nlargest(3, "mean_arousal").iterrows():
            rows.append({"song_id": row.song_id, "title": row.title, "artist": row.artist,
                         "case_category": label, "primary_value": row.mean_arousal,
                         "listening_question": "Is this a genuine formal climax, a production peak, or a Teacher artifact?"})
    for _, row in song_features.nsmallest(3, "section_va_spearman").iterrows():
        rows.append({"song_id": row.song_id, "title": row.title, "artist": row.artist,
                     "case_category": "valence_arousal_separation", "primary_value": row.section_va_spearman,
                     "listening_question": "Which musical dimensions raise energy while preserving or lowering valence?"})
    return pd.DataFrame(rows).drop_duplicates(["song_id", "case_category"])


def build_evidence_matrix() -> pd.DataFrame:
    return pd.DataFrame([
        {"musicological_claim": "Verse-Pre-chorus-Chorus can form an initiation-buildup-arrival trajectory", "evidence_level": "direct pattern support", "data_evidence": "Arousal rises Verse->Pre-chorus and Pre-chorus->Chorus", "interpretive_boundary": "VA cannot identify the harmonic, timbral, rhythmic, or production mechanism"},
        {"musicological_claim": "Chorus functions more consistently as an energetic arrival than as a positive-valence arrival", "evidence_level": "direct pattern support", "data_evidence": "Chorus-Verse Arousal lift is larger and more universal than Valence lift", "interpretive_boundary": "Teacher bias remains possible; 15-song human evidence supports only the tendency"},
        {"musicological_claim": "Later rotations accumulate affective intensity", "evidence_level": "direct pattern support", "data_evidence": "Later Chorus, Verse, Pre-chorus, and Interlude all tend to rise", "interpretive_boundary": "No evidence that the accumulation is uniquely caused by Chorus repetition"},
        {"musicological_claim": "Repeated formal identity coexists with emotional variation", "evidence_level": "direct pattern support", "data_evidence": "First-last repeated sections change; Chorus is not more VA-consistent than Verse", "interpretive_boundary": "Section labels do not encode orchestration or exact thematic identity"},
        {"musicological_claim": "Interlude often acts as a rotation hinge rather than a simple low-energy break", "evidence_level": "indirect support", "data_evidence": "High coverage and common Chorus->Interlude->Verse motif; little mean A drop entering Interlude", "interpretive_boundary": "Instrumental content and production must be confirmed by listening"},
        {"musicological_claim": "Bridge is defined by contextual contrast, not one fixed VA direction", "evidence_level": "indirect support", "data_evidence": "Late position and heterogeneous entering/exiting VA directions", "interpretive_boundary": "Only 32 songs contain Bridge; label ambiguity is substantial"},
        {"musicological_claim": "Outro usually releases local peak energy without returning fully to the opening state", "evidence_level": "direct pattern support", "data_evidence": "Large Chorus->Outro drop but positive mean start-end Arousal change", "interpretive_boundary": "Outro boundaries may include fades or silence and need case-level listening"},
        {"musicological_claim": "Song progression combines formal rotations with late-song affective accumulation", "evidence_level": "direct plus synthesis", "data_evidence": "Low path directness, repeated rotations, late average peak, and local release", "interpretive_boundary": "This is a corpus-level description, not a new universal form type"},
        {"musicological_claim": "Some songs may use terminally climactic organization", "evidence_level": "candidate only", "data_evidence": "Late non-Chorus Arousal peaks exist", "interpretive_boundary": "Terminal climax requires thematic/formal listening evidence and cannot be inferred from VA alone"},
    ])


def make_figures(profile: pd.DataFrame, transitions: pd.DataFrame, motifs: pd.DataFrame,
                 trajectories: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for _, row in profile.iterrows():
        size = 80 + 500 * row.song_coverage
        ax.scatter(row.mean_position, row.mean_arousal, s=size, color=COLORS[row.section_type],
                   edgecolor="white", linewidth=1.2, alpha=.9)
        ax.text(row.mean_position + .012, row.mean_arousal + .012, TYPE_LABELS[row.section_type], fontsize=10)
    ax.axhline(0, color="#777", lw=.8, ls="--")
    ax.set(xlabel="Mean normalized position", ylabel="Mean section Arousal",
           title="Functional sections across normalized song time")
    ax.text(.01, .98, "Bubble size = song coverage", transform=ax.transAxes, va="top", fontsize=9)
    save_figure(fig, "12_musicological_section_function_map")

    selected = [
        "intro->verse", "verse->pre_chorus", "pre_chorus->chorus", "verse->chorus",
        "chorus->interlude", "interlude->verse", "chorus->bridge", "bridge->chorus", "chorus->outro",
    ]
    subset = transitions[transitions.transition.isin(selected)].groupby("transition").agg(
        delta_valence=("delta_valence", "mean"), delta_arousal=("delta_arousal", "mean"),
        songs=("song_id", "nunique")
    ).reindex(selected).dropna().reset_index()
    long = subset.melt(id_vars=["transition", "songs"], value_vars=["delta_valence", "delta_arousal"],
                       var_name="dimension", value_name="mean_change")
    long.dimension = long.dimension.map({"delta_valence": "Valence", "delta_arousal": "Arousal"})
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(data=long, x="transition", y="mean_change", hue="dimension",
                palette={"Valence": "#4C78A8", "Arousal": "#E45756"}, ax=ax)
    ax.axhline(0, color="#555", lw=.8)
    ax.set(xlabel="Adjacent formal transition", ylabel="Mean pseudo-VA change",
           title="Local affective functions of common section transitions")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, "13_musicological_transition_functions")

    top = motifs[motifs.motif_length == 3].head(10).sort_values("song_count")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(top.motif, top.song_count, color="#59A14F")
    axes[0].set(xlabel="Songs", title="Most common collapsed three-section motifs")
    mean = trajectories.groupby("normalized_time")[["valence", "arousal"]].mean()
    axes[1].plot(mean.index, mean.valence, label="Valence", color="#4C78A8", lw=2.4)
    axes[1].plot(mean.index, mean.arousal, label="Arousal", color="#E45756", lw=2.4)
    axes[1].axhline(0, color="#777", lw=.8, ls="--")
    axes[1].set(xlabel="Normalized song time", ylabel="Mean pseudo-VA",
                title="Corpus-average late accumulation and release")
    axes[1].legend()
    save_figure(fig, "14_formal_grammar_and_average_arc")


def write_report(profile: pd.DataFrame, form_summary: pd.DataFrame, motifs: pd.DataFrame,
                 transition_profile: pd.DataFrame, chorus_peaks: pd.DataFrame,
                 features: pd.DataFrame, repeats: pd.DataFrame,
                 evidence: pd.DataFrame, cases: pd.DataFrame) -> None:
    p = profile.set_index("section_type")
    tr = transition_profile.set_index("transition")
    chorus_verse = pd.read_csv(TABLES / "section_type_paired_tests_holm.csv")
    chorus_verse = chorus_verse[(chorus_verse.weighting == "unweighted") &
                                (chorus_verse.comparison == "Chorus - Verse")].set_index("dimension")
    repeat_change = pd.read_csv(TABLES / "repeated_section_change_contrasts_holm.csv")
    repeat_change = repeat_change[repeat_change.comparison == "Chorus change - Verse change"].set_index("dimension")
    repeat_pivot = repeats.pivot(index="song_id", columns="section_type", values="mean_pairwise_va_distance")
    repeat_pair = repeat_pivot[["verse", "chorus"]].dropna()
    repeat_distance_diff = repeat_pair.chorus - repeat_pair.verse
    repeat_distance_p = stats.wilcoxon(repeat_distance_diff).pvalue
    path_directness_mean = features.path_directness.mean()
    path_directness_median = features.path_directness.median()
    net_v = features.net_valence_change.mean()
    net_a = features.net_arousal_change.mean()
    net_a_positive = (features.net_arousal_change > 0).mean()
    chorus_peak_last = chorus_peaks.is_last_chorus_peak.mean()
    top_form = form_summary.iloc[0]
    top_three = motifs[motifs.motif_length == 3].head(5)

    lines = [
        "# 97首中文流行歌曲情绪结构的音乐学解释报告", "",
        f"报告时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", "",
        "当前科研进度：统计分析已经完成；本报告进一步把 section-level pseudo-VA 结果解释为流行音乐形式功能。报告不分析歌词，也不把 VA 变化直接归因于和声、调式、速度、音色或配器。", "",
        "## 一、分析立场：从数值差异到形式功能", "",
        "本研究真正能够讨论的不是“哪类段落数值更大”，而是不同段落在完整歌曲时间进程中承担怎样的情绪功能。参考流行音乐形式理论，可把常见段落理解为启动、蓄力、到达、对比、再导向和收束等功能。但这些理论主要来自欧美流行/摇滚研究，在本中文歌曲便利样本中只能作为解释框架，不能直接视为文化普适规则。", "",
        "本报告采用三层证据：**直接支持**表示数据与功能描述一致；**间接支持**表示数据允许该解释但存在替代解释；**候选/推测**表示必须通过听觉、和声或制作分析才能确认。", "",
        "## 二、样本中的流行歌曲形式语法", "",
        f"Verse 出现在 `{int(p.loc['verse','song_count'])}/97` 首，Chorus `{int(p.loc['chorus','song_count'])}/97`，Intro `{int(p.loc['intro','song_count'])}/97`，Outro `{int(p.loc['outro','song_count'])}/97`。这说明样本的共同骨架是 Intro—Verse/Chorus 循环—Outro。Pre-chorus 只覆盖 `{int(p.loc['pre_chorus','song_count'])}/97`，Bridge `{int(p.loc['bridge','song_count'])}/97`，属于可选功能模块。", "",
        f"Interlude 覆盖 `{int(p.loc['interlude','song_count'])}/97`，在本样本中非常常见。相邻同类标签折叠后仍有 `{len(form_summary)}` 种完整结构；最常见结构 `{top_form.collapsed_sequence}` 也只有 `{int(top_form.song_count)}/97` 首。这表示歌曲没有单一固定模板，但共享局部语法。", "",
        "最常见的三段式局部语法包括：", "",
    ]
    for _, row in top_three.iterrows():
        lines.append(f"- `{row.motif}`：{int(row.song_count)} 首。")
    lines.extend(["",
        "音乐学含义：形式稳定性主要存在于局部轮转和功能关系，而不是完整段落序列完全一致。论文应强调“共享局部语法下的整体结构多样性”。", "",
        "## 三、Verse—Pre-chorus—Chorus：情绪目的性", "",
        f"Verse 的平均 Arousal 为 `{p.loc['verse','mean_arousal']:.3f}`，Pre-chorus `{p.loc['pre_chorus','mean_arousal']:.3f}`，Chorus `{p.loc['chorus','mean_arousal']:.3f}`。局部转换中 Verse→Pre-chorus Arousal 平均增加 `{tr.loc['verse->pre_chorus','mean_delta_arousal']:+.3f}`，Pre-chorus→Chorus 再增加 `{tr.loc['pre_chorus->chorus','mean_delta_arousal']:+.3f}`。", "",
        "这与 Nobile 所说的 initiation—buildup—arrival 或 teleological trajectory 高度一致：Verse 建立循环，Pre-chorus 延长并加强指向性，Chorus 构成到达。但本数据只能证明情绪结果形成阶段性上升，不能证明这种目的性由属和弦、半终止、旋律上行或配器叠加中的哪一种机制产生。", "",
        f"Chorus 相对 Verse 的 Arousal 差 `{chorus_verse.loc['arousal','mean_difference']:+.3f}`，显著大于 Valence 差 `{chorus_verse.loc['valence','mean_difference']:+.3f}`。因此副歌首先表现为**能量和到达感**，其次才是积极性变化。把副歌直接称为“更快乐”并不严谨。", "",
        "## 四、重复不是复制：循环中的后程累积", "",
        f"后续 Chorus 的 Valence/Arousal 平均提高，但 Chorus 相对重复 Verse 的额外变化为 `{repeat_change.loc['valence','mean_difference']:+.3f}/{repeat_change.loc['arousal','mean_difference']:+.3f}`，Holm p=`{repeat_change.loc['valence','p_holm']:.3f}/{repeat_change.loc['arousal','p_holm']:.3f}`。重复 Chorus 与重复 Verse 的内部 VA 离散度差为 `{repeat_distance_diff.mean():+.3f}`，Wilcoxon p=`{repeat_distance_p:.3f}`。", "",
        "因此，副歌并没有表现出更严格的情绪恒定性，也没有获得独有的后续强化。更合理的音乐学解释是：流行歌曲通过 Verse/Chorus 轮转维持可识别形式身份，同时通过演唱强度、织体、音区、配器或制作层次的逐轮变化形成全曲累积。后半句属于待验证机制，不能作为本数据的直接结论。", "",
        "## 五、Interlude：轮转铰链而非单纯降温", "",
        f"Interlude 的平均位置为 `{p.loc['interlude','mean_position']:.3f}`，常见路径是 Chorus→Interlude（{int(tr.loc['chorus->interlude','song_count'])} 首）和 Interlude→Verse（{int(tr.loc['interlude->verse','song_count'])} 首）。进入 Interlude 时 Arousal 平均变化仅 `{tr.loc['chorus->interlude','mean_delta_arousal']:+.3f}`，随后进入 Verse 下降 `{tr.loc['interlude->verse','mean_delta_arousal']:+.3f}`。", "",
        "这提示 Interlude 经常维持或重新分配副歌后的能量，并把歌曲导向下一次 Verse/Chorus 轮转。把所有间奏理解为“休息段”不符合数据；更接近“轮转铰链”或“再导向区域”。是否由器乐独奏、音色替换或和声延长构成，需要逐首听审。", "",
        "## 六、Bridge：对比功能没有固定情绪方向", "",
        f"Bridge 只出现在 `{int(p.loc['bridge','song_count'])}/97` 首，平均位置 `{p.loc['bridge','mean_position']:.3f}`，描述性平均 V/A 为 `{p.loc['bridge','mean_valence']:.3f}/{p.loc['bridge','mean_arousal']:.3f}`。进入 Bridge 的平均变化为 `{tr.loc['chorus->bridge','mean_delta_valence']:+.3f}/{tr.loc['chorus->bridge','mean_delta_arousal']:+.3f}`，Bridge→Chorus 为 `{tr.loc['bridge->chorus','mean_delta_valence']:+.3f}/{tr.loc['bridge->chorus','mean_delta_arousal']:+.3f}`，且方向分布高度分散。", "",
        "这与“Bridge 位于核心 Verse/Chorus 循环之外、负责制造对比”的理论相容，但不支持 Bridge 必然降温、必然抬升或必然成为高潮。对比可以来自情绪方向、织体、节奏、和声或音色中的任一维度；VA 只捕捉其中一部分。", "",
        "## 七、Outro：局部释放但不完全返回起点", "",
        f"Chorus→Outro 的 Valence/Arousal 平均下降 `{tr.loc['chorus->outro','mean_delta_valence']:+.3f}/{tr.loc['chorus->outro','mean_delta_arousal']:+.3f}`，是最稳定的局部收束之一。但整首歌曲的起点到终点，平均净变化仍为 Valence `{net_v:+.3f}`、Arousal `{net_a:+.3f}`，且 `{net_a_positive:.1%}` 的歌曲结束 Arousal 高于开始。", "",
        "因此 Outro 更像高潮后的**部分释放和余韵**，而不是回到开头情绪原点。流行歌曲的收束可以完成形式闭合，同时保留高于开场的情绪残余。淡出、尾音和静音是否影响数值，需要在代表歌曲中复核。", "",
        "## 八、完整歌曲不是单向曲线，而是循环中的累积", "",
        f"平均归一化轨迹确实在歌曲约 75%–80% 处到达高位后回落，但歌曲平均 VA 路径直接度只有 `{path_directness_mean:.3f}`（中位数 `{path_directness_median:.3f}`）。也就是说，起点到终点的直线位移只占实际 VA 行走路径的一小部分。", "",
        "音乐学上，这更接近“重复—对比—回归”的循环运动，而不是从头到尾持续上升的单一 crescendo。多个 Verse/Chorus 轮转会反复离开并返回熟悉区域，同时整体基线逐渐提高。可以把它概括为**循环中的后程累积**，但暂时不应命名为新的普遍曲式类型。", "",
        "## 九、高潮与终点高潮的边界", "",
        f"最大 Arousal 位于 Chorus 的 53 首中，`{chorus_peak_last:.1%}` 出现在最后一次 Chorus。它支持后置强化，但仍有大量歌曲由 Bridge、Interlude 或 Outro 达到模型高点。", "",
        "Osborn 的 terminally climactic form 指向一种晚期新材料取代预期终极副歌的形式。当前数据只能筛选“晚期非 Chorus 高点”候选，不能据此判定终点高潮曲式，因为该判断还需要确认主题材料是否新出现、结构期待是否被替代，以及听觉上是否构成真正高潮。", "",
        "## 十、可用于论文的音乐学主张", "",
        "1. 本样本呈现以 Verse/Chorus 轮转为核心、Pre-chorus/Bridge 为可选功能模块、Interlude 作为常见轮转铰链的形式语法。",
        "2. Chorus 的情绪功能主要是 Arousal 到达，而不是无条件的 Valence 正向化。",
        "3. 后续副歌增强属于全曲后程累积，不是副歌独有的重复效应。",
        "4. Bridge 的意义在于上下文对比，而不是固定的高低情绪方向。",
        "5. Outro 经常执行局部释放，但完整歌曲通常保留高于开场的唤醒残余。",
        "6. 本样本的完整歌曲情绪进程同时呈现段落轮转、后程累积和高潮后的局部收束。", "",
        "## 十一、下一步个案分析", "",
        f"已输出 `{len(cases)}` 条候选个案，包括强/弱 Verse—Pre-chorus—Chorus 蓄力、Bridge 或 Outro 高点、强 Chorus→Outro 释放和 V/A 分离。下一步应从每类选择 2–3 首逐首听审，并增加时间轴图、section 边界和必要的响度/音色/节奏描述。个案分析用于解释统计规律和反例，不用于重新筛选样本。", "",
        "## 参考文献", "",
        "- Temperley (2018), Form, The Musical Language of Rock: https://doi.org/10.1093/oso/9780190653774.003.0008",
        "- Nobile (2020), Prechoruses, Bridges, and Auxiliary Sections: https://doi.org/10.1093/oso/9780190948351.003.0004",
        "- Nobile (2022), Teleology in Verse-Prechorus-Chorus Form: https://mtosmt.org/issues/mto.22.28.3/mto.22.28.3.nobile.html",
        "- de Clercq (2017), Embracing Ambiguity in Pop/Rock Form: https://www.mtosmt.org/issues/mto.17.23.3/mto.17.23.3.de_clercq.php",
        "- Osborn (2013), Terminally Climactic Forms: https://doi.org/10.1525/mts.2013.35.1.23",
        "- Nieto et al. (2019), The Harmonix Set: https://archives.ismir.net/ismir2019/paper/000068.pdf", "",
    ])
    (REPORTS / "MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sections = pd.read_csv(DATA / "cleaned_section_analysis.csv", dtype={"song_id": str})
    transitions = pd.read_csv(TABLES / "adjacent_section_transitions.csv", dtype={"song_id": str})
    features = pd.read_csv(TABLES / "song_level_structure_features.csv", dtype={"song_id": str})
    repeats = pd.read_csv(TABLES / "repeated_section_song_details.csv", dtype={"song_id": str})
    trajectories = pd.read_csv(DATA / "normalized_complete_song_trajectories_101pts.csv", dtype={"song_id": str})

    profile = build_function_profiles(sections)
    form_detail, form_summary, motifs = build_form_grammar(sections)
    transition_profile = build_transition_profiles(transitions)
    chorus_peaks = build_chorus_peak_occurrences(sections)
    cases = build_case_studies(sections, transitions, features)
    evidence = build_evidence_matrix()

    outputs = {
        "musicological_section_function_profiles.csv": profile,
        "musicological_collapsed_form_sequences_by_song.csv": form_detail,
        "musicological_collapsed_form_sequence_summary.csv": form_summary,
        "musicological_formal_transition_motifs.csv": motifs,
        "musicological_transition_profiles.csv": transition_profile,
        "musicological_chorus_peak_occurrences.csv": chorus_peaks,
        "musicological_case_study_candidates.csv": cases,
        "musicological_evidence_matrix.csv": evidence,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLES / name, index=False)

    make_figures(profile, transitions, motifs, trajectories)
    write_report(profile, form_summary, motifs, transition_profile, chorus_peaks,
                 features, repeats, evidence, cases)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass", "songs": int(sections.song_id.nunique()),
        "musicological_tables": list(outputs),
        "figures": [
            "12_musicological_section_function_map",
            "13_musicological_transition_functions",
            "14_formal_grammar_and_average_arc",
        ],
        "report": "reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md",
        "claim_boundary": "VA describes affective formal outcomes; causal musical mechanisms require additional features or listening analysis",
    }
    (ANALYSIS / "musicological_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
