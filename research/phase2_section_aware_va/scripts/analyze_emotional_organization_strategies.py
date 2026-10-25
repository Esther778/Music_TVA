#!/usr/bin/env python3
"""Translate section-level pseudo-VA into auditable musical-emotion functions.

This is a descriptive interpretation layer. It does not add inferential tests,
cluster songs into fixed types, or infer harmony/tempo/timbre from VA values.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/phase2_section_aware_va/analysis/full_corpus_97"
DATA = BASE / "data"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

SECTIONS_PATH = DATA / "cleaned_section_analysis.csv"
STABLE_THRESHOLD = 0.05
STRONG_THRESHOLD = 0.10
RETURN_THRESHOLD = 0.10


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / name, index=False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def safe_slope(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return math.nan
    return float(np.polyfit(valid["x"], valid["y"], 1)[0])


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(x.corr(y, method="spearman"))


def va_distance(v1: float, a1: float, v2: float, a2: float) -> float:
    return float(np.hypot(v1 - v2, a1 - a2))


def emotion_region(v: float, a: float) -> str:
    if v >= 0 and a >= 0:
        return "positive_high_arousal"
    if v >= 0 and a < 0:
        return "positive_low_arousal"
    if v < 0 and a >= 0:
        return "negative_high_arousal"
    return "negative_low_arousal"


REGION_ZH = {
    "positive_high_arousal": "偏积极、高唤醒区域",
    "positive_low_arousal": "偏积极、低唤醒区域",
    "negative_high_arousal": "偏消极、高唤醒区域",
    "negative_low_arousal": "偏消极、低唤醒区域",
}


def classify_prechorus(d1: float, d2: float) -> str:
    t = STABLE_THRESHOLD
    if d1 > t and d2 > t:
        return "smooth_buildup"
    if d1 > t and abs(d2) <= t:
        return "delayed_release"
    if abs(d1) <= t and d2 > t:
        return "sudden_release"
    if d1 > t and d2 < -t:
        return "false_buildup"
    if d1 < -t and d2 > t:
        return "dip_then_release"
    return "other_or_weak_pattern"


def classify_chorus_entry(dv: float, da: float) -> str:
    t = STABLE_THRESHOLD
    if dv > t and da > t:
        return "positive_release"
    if dv < -t and da > t:
        return "painful_release"
    if abs(dv) <= t and da > t:
        return "intensity_release"
    if dv > t and da < -t:
        return "soothing_release"
    if abs(dv) <= t and abs(da) <= t:
        return "continuation"
    if da < -t and dv <= t:
        return "reverse_chorus"
    return "mixed_or_valence_led"


def build_function_events(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre_events: list[dict] = []
    chorus_events: list[dict] = []
    bridge_events: list[dict] = []
    outro_events: list[dict] = []

    for song_id, song in sections.groupby("song_id", sort=False):
        song = song.sort_values("section_order").reset_index(drop=True)
        meta = song.iloc[0][["song_id", "title", "artist"]].to_dict()

        for i, row in song.iterrows():
            section_type = row["normalized_section_type"]
            if section_type == "pre_chorus" and 0 < i < len(song) - 1:
                previous = song.iloc[i - 1]
                following = song.iloc[i + 1]
                if previous["normalized_section_type"] == "verse" and following["normalized_section_type"] == "chorus":
                    d1 = float(row["mean_arousal"] - previous["mean_arousal"])
                    d2 = float(following["mean_arousal"] - row["mean_arousal"])
                    pre_events.append({
                        **meta,
                        "pre_chorus_order": int(row["section_order"]),
                        "verse_to_pre_arousal": d1,
                        "pre_to_chorus_arousal": d2,
                        "verse_to_pre_valence": float(row["mean_valence"] - previous["mean_valence"]),
                        "pre_to_chorus_valence": float(following["mean_valence"] - row["mean_valence"]),
                        "buildup_pattern": classify_prechorus(d1, d2),
                    })

            if section_type == "chorus" and i > 0:
                previous = song.iloc[i - 1]
                dv = float(row["mean_valence"] - previous["mean_valence"])
                da = float(row["mean_arousal"] - previous["mean_arousal"])
                chorus_events.append({
                    **meta,
                    "chorus_order": int(row["section_order"]),
                    "previous_type": previous["normalized_section_type"],
                    "delta_valence": dv,
                    "delta_arousal": da,
                    "va_distance": float(np.hypot(dv, da)),
                    "release_mode": classify_chorus_entry(dv, da),
                    "from_region": emotion_region(float(previous["mean_valence"]), float(previous["mean_arousal"])),
                    "to_region": emotion_region(float(row["mean_valence"]), float(row["mean_arousal"])),
                })

            if section_type == "bridge":
                prior = song.iloc[:i]
                prior_core = prior[prior["normalized_section_type"].isin(["verse", "chorus"])]
                prior_chorus = prior[prior["normalized_section_type"] == "chorus"]
                next_chorus = song.iloc[i + 1:]
                next_chorus = next_chorus[next_chorus["normalized_section_type"] == "chorus"]
                prev_row = song.iloc[i - 1] if i > 0 else None
                dv_in = float(row["mean_valence"] - prev_row["mean_valence"]) if prev_row is not None else math.nan
                da_in = float(row["mean_arousal"] - prev_row["mean_arousal"]) if prev_row is not None else math.nan
                if len(prior_core):
                    core_v = float(prior_core["mean_valence"].mean())
                    core_a = float(prior_core["mean_arousal"].mean())
                    core_distance = va_distance(float(row["mean_valence"]), float(row["mean_arousal"]), core_v, core_a)
                else:
                    core_distance = math.nan
                bridge_to_prior_chorus = math.nan
                next_to_prior_chorus = math.nan
                if len(prior_chorus):
                    chorus_v = float(prior_chorus["mean_valence"].mean())
                    chorus_a = float(prior_chorus["mean_arousal"].mean())
                    bridge_to_prior_chorus = va_distance(float(row["mean_valence"]), float(row["mean_arousal"]), chorus_v, chorus_a)
                    if len(next_chorus):
                        nxt = next_chorus.iloc[0]
                        next_to_prior_chorus = va_distance(float(nxt["mean_valence"]), float(nxt["mean_arousal"]), chorus_v, chorus_a)
                functions: list[str] = []
                if not math.isnan(da_in) and da_in > STABLE_THRESHOLD:
                    functions.append("intensifying")
                if not math.isnan(da_in) and da_in < -STABLE_THRESHOLD:
                    functions.append("cooling")
                if not math.isnan(dv_in) and abs(dv_in) > STABLE_THRESHOLD:
                    functions.append("valence_redirecting")
                if not math.isnan(core_distance) and core_distance > STRONG_THRESHOLD:
                    functions.append("contextual_deviation")
                if not math.isnan(next_to_prior_chorus) and not math.isnan(bridge_to_prior_chorus) and next_to_prior_chorus < bridge_to_prior_chorus:
                    functions.append("deviation_then_return")
                if not functions:
                    functions.append("connecting_or_weak_change")
                bridge_events.append({
                    **meta,
                    "bridge_order": int(row["section_order"]),
                    "incoming_delta_valence": dv_in,
                    "incoming_delta_arousal": da_in,
                    "distance_from_prior_core": core_distance,
                    "distance_from_prior_chorus": bridge_to_prior_chorus,
                    "next_chorus_distance_from_prior_chorus": next_to_prior_chorus,
                    "function_descriptors": "|".join(functions),
                })

        outros = song[song["normalized_section_type"] == "outro"]
        if len(outros):
            row = outros.iloc[-1]
            i = int(song.index[song["section_order"] == row["section_order"]][0])
            previous = song.iloc[i - 1] if i > 0 else None
            intro = song[song["normalized_section_type"] == "intro"]
            dv_in = float(row["mean_valence"] - previous["mean_valence"]) if previous is not None else math.nan
            da_in = float(row["mean_arousal"] - previous["mean_arousal"]) if previous is not None else math.nan
            intro_outro_distance = math.nan
            if len(intro):
                intro_outro_distance = va_distance(
                    float(intro["mean_valence"].mean()), float(intro["mean_arousal"].mean()),
                    float(row["mean_valence"]), float(row["mean_arousal"]),
                )
            functions = []
            if not math.isnan(da_in) and da_in < -STRONG_THRESHOLD:
                functions.append("rapid_release")
            if float(row["delta_arousal"]) < -STABLE_THRESHOLD:
                functions.append("gradual_dissipation_within_outro")
            if float(row["mean_arousal"]) > float(song["mean_arousal"].mean()) + STABLE_THRESHOLD:
                functions.append("high_level_ending")
            if not math.isnan(intro_outro_distance) and intro_outro_distance <= RETURN_THRESHOLD:
                functions.append("emotional_return")
            if not math.isnan(intro_outro_distance) and intro_outro_distance >= 2 * RETURN_THRESHOLD:
                functions.append("open_or_migrated_ending")
            if not math.isnan(dv_in) and dv_in > STABLE_THRESHOLD and da_in < -STABLE_THRESHOLD:
                functions.append("positive_closure")
            if not math.isnan(dv_in) and dv_in < -STABLE_THRESHOLD and da_in < -STABLE_THRESHOLD:
                functions.append("negative_settling")
            if not functions:
                functions.append("weak_or_mixed_closure")
            outro_events.append({
                **meta,
                "outro_order": int(row["section_order"]),
                "previous_type": previous["normalized_section_type"] if previous is not None else "START",
                "incoming_delta_valence": dv_in,
                "incoming_delta_arousal": da_in,
                "within_outro_delta_valence": float(row["delta_valence"]),
                "within_outro_delta_arousal": float(row["delta_arousal"]),
                "intro_outro_va_distance": intro_outro_distance,
                "closure_descriptors": "|".join(functions),
            })

    functional = pd.concat([
        pd.DataFrame(pre_events).assign(event_family="pre_chorus_buildup"),
        pd.DataFrame(chorus_events).assign(event_family="chorus_release"),
    ], ignore_index=True, sort=False)
    return functional, pd.DataFrame(bridge_events), pd.DataFrame(outro_events)


def build_region_transitions(sections: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, song in sections.groupby("song_id", sort=False):
        song = song.sort_values("section_order").reset_index(drop=True)
        for i in range(1, len(song)):
            left, right = song.iloc[i - 1], song.iloc[i]
            source = emotion_region(float(left["mean_valence"]), float(left["mean_arousal"]))
            target = emotion_region(float(right["mean_valence"]), float(right["mean_arousal"]))
            records.append({
                "song_id": right["song_id"], "title": right["title"], "artist": right["artist"],
                "from_order": int(left["section_order"]), "to_order": int(right["section_order"]),
                "from_type": left["normalized_section_type"], "to_type": right["normalized_section_type"],
                "from_region": source, "to_region": target, "crosses_region": source != target,
                "delta_valence": float(right["mean_valence"] - left["mean_valence"]),
                "delta_arousal": float(right["mean_arousal"] - left["mean_arousal"]),
            })
    return pd.DataFrame(records)


def cosine_consistency(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return math.nan
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1e-9]
    if len(vectors) < 2:
        return math.nan
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = []
    for i in range(len(unit)):
        for j in range(i + 1, len(unit)):
            similarities.append(float(np.dot(unit[i], unit[j])))
    return float(np.mean(similarities))


def build_song_profiles(
    sections: pd.DataFrame,
    function_events: pd.DataFrame,
    bridge_events: pd.DataFrame,
    outro_events: pd.DataFrame,
    region_transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for song_id, song in sections.groupby("song_id", sort=False):
        song = song.sort_values("section_order").reset_index(drop=True)
        weights = song["section_duration_sec"].to_numpy(float)
        mean_v = float(np.average(song["mean_valence"], weights=weights))
        mean_a = float(np.average(song["mean_arousal"], weights=weights))
        dv = song["mean_valence"].diff().dropna().to_numpy(float)
        da = song["mean_arousal"].diff().dropna().to_numpy(float)
        distances = np.hypot(dv, da)
        path_length = float(distances.sum())
        top2 = float(np.sort(distances)[-2:].sum()) if len(distances) >= 2 else float(distances.sum())
        late = song[song["normalized_midpoint"] >= 0.5]

        chorus = song[song["normalized_section_type"] == "chorus"]
        verse = song[song["normalized_section_type"] == "verse"]
        chorus_entries = []
        for i in song.index[song["normalized_section_type"] == "chorus"]:
            if i > 0:
                chorus_entries.append([
                    float(song.loc[i, "mean_valence"] - song.loc[i - 1, "mean_valence"]),
                    float(song.loc[i, "mean_arousal"] - song.loc[i - 1, "mean_arousal"]),
                ])
        chorus_entries_array = np.asarray(chorus_entries, dtype=float) if chorus_entries else np.empty((0, 2))

        repeat_distances = []
        for section_type in ["verse", "chorus"]:
            subset = song[song["normalized_section_type"] == section_type][["mean_valence", "mean_arousal"]].to_numpy(float)
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    repeat_distances.append(float(np.linalg.norm(subset[i] - subset[j])))

        song_regions = region_transitions[region_transitions["song_id"] == song_id]
        song_bridges = bridge_events[bridge_events["song_id"] == song_id] if len(bridge_events) else pd.DataFrame()
        song_outros = outro_events[outro_events["song_id"] == song_id] if len(outro_events) else pd.DataFrame()

        peak_a_idx = int(song["mean_arousal"].idxmax())
        peak_v_idx = int(song["mean_valence"].idxmax())
        max_transition_index = int(np.argmax(distances)) + 1 if len(distances) else 0
        row = {
            "song_id": song_id, "title": song.iloc[0]["title"], "artist": song.iloc[0]["artist"],
            "section_count": len(song),
            "baseline_valence": mean_v, "baseline_arousal": mean_a,
            "baseline_region": emotion_region(mean_v, mean_a),
            "valence_range": float(song["mean_valence"].max() - song["mean_valence"].min()),
            "arousal_range": float(song["mean_arousal"].max() - song["mean_arousal"].min()),
            "mean_section_contrast": float(distances.mean()) if len(distances) else 0.0,
            "max_section_contrast": float(distances.max()) if len(distances) else 0.0,
            "transition_concentration_max_share": float(distances.max() / path_length) if path_length > 0 else 0.0,
            "transition_concentration_top2_share": float(top2 / path_length) if path_length > 0 else 0.0,
            "whole_song_valence_slope": safe_slope(song["normalized_midpoint"], song["mean_valence"]),
            "whole_song_arousal_slope": safe_slope(song["normalized_midpoint"], song["mean_arousal"]),
            "late_valence_slope": safe_slope(late["normalized_midpoint"], late["mean_valence"]),
            "late_arousal_slope": safe_slope(late["normalized_midpoint"], late["mean_arousal"]),
            "chorus_first_last_valence": float(chorus.iloc[-1]["mean_valence"] - chorus.iloc[0]["mean_valence"]) if len(chorus) >= 2 else math.nan,
            "chorus_first_last_arousal": float(chorus.iloc[-1]["mean_arousal"] - chorus.iloc[0]["mean_arousal"]) if len(chorus) >= 2 else math.nan,
            "verse_first_last_valence": float(verse.iloc[-1]["mean_valence"] - verse.iloc[0]["mean_valence"]) if len(verse) >= 2 else math.nan,
            "verse_first_last_arousal": float(verse.iloc[-1]["mean_arousal"] - verse.iloc[0]["mean_arousal"]) if len(verse) >= 2 else math.nan,
            "chorus_entry_direction_consistency": cosine_consistency(chorus_entries_array),
            "chorus_entry_amplitude_cv": float(np.std(np.linalg.norm(chorus_entries_array, axis=1)) / np.mean(np.linalg.norm(chorus_entries_array, axis=1))) if len(chorus_entries_array) >= 2 and np.mean(np.linalg.norm(chorus_entries_array, axis=1)) > 0 else math.nan,
            "mean_repeated_core_va_distance": float(np.mean(repeat_distances)) if repeat_distances else math.nan,
            "intro_outro_va_distance": float(song_outros.iloc[0]["intro_outro_va_distance"]) if len(song_outros) else math.nan,
            "bridge_return_gain": float((song_bridges["distance_from_prior_chorus"] - song_bridges["next_chorus_distance_from_prior_chorus"]).mean()) if len(song_bridges) else math.nan,
            "arousal_peak_position": float(song.loc[peak_a_idx, "normalized_midpoint"]),
            "valence_peak_position": float(song.loc[peak_v_idx, "normalized_midpoint"]),
            "largest_transition_position": float(song.iloc[max_transition_index]["normalized_midpoint"]) if len(song) else math.nan,
            "closure_from_arousal_peak": float(song["mean_arousal"].max() - song.iloc[-1]["mean_arousal"]),
            "closure_from_valence_peak": float(song["mean_valence"].max() - song.iloc[-1]["mean_valence"]),
            "section_va_coupling_spearman": safe_spearman(song["mean_valence"], song["mean_arousal"]),
            "cross_region_transition_rate": float(song_regions["crosses_region"].mean()) if len(song_regions) else 0.0,
            "start_end_va_distance": va_distance(float(song.iloc[0]["mean_valence"]), float(song.iloc[0]["mean_arousal"]), float(song.iloc[-1]["mean_valence"]), float(song.iloc[-1]["mean_arousal"])),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def dominant_value(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    counts = frame[column].dropna().value_counts()
    return str(counts.index[0]) if len(counts) else None


PRE_ZH = {
    "smooth_buildup": "连续蓄力后到达副歌",
    "delayed_release": "主要蓄力发生在预副歌，副歌延续高位",
    "sudden_release": "预副歌变化有限，能量主要在副歌入口释放",
    "false_buildup": "预副歌升高后，副歌反而回落",
    "dip_then_release": "预副歌先回落，再进入副歌释放",
    "other_or_weak_pattern": "蓄力方向较弱或混合",
}

CHORUS_ZH = {
    "positive_release": "Valence 与 Arousal 同时上升的积极释放",
    "painful_release": "Arousal 上升而 Valence 下降的高强度负向释放",
    "intensity_release": "Valence 基本稳定、Arousal 上升的强度释放",
    "soothing_release": "Valence 上升而 Arousal 下降的舒缓释放",
    "continuation": "两个维度均变化较小的延续型进入",
    "reverse_chorus": "进入副歌后整体降温",
    "mixed_or_valence_led": "以 Valence 变化或混合方向为主的进入",
}


def portrait_for_song(
    song: pd.DataFrame,
    profile: pd.Series,
    function_events: pd.DataFrame,
    bridge_events: pd.DataFrame,
    outro_events: pd.DataFrame,
) -> tuple[str, str]:
    song_id = profile["song_id"]
    first = song.sort_values("section_order").iloc[0]
    clauses = [
        f"模型估计显示，歌曲从{REGION_ZH[emotion_region(float(first['mean_valence']), float(first['mean_arousal']))]}开始，"
        f"全曲基础位置位于{REGION_ZH[profile['baseline_region']]}。"
    ]
    song_pre = function_events[(function_events["song_id"] == song_id) & (function_events["event_family"] == "pre_chorus_buildup")]
    song_chorus = function_events[(function_events["song_id"] == song_id) & (function_events["event_family"] == "chorus_release")]
    pre_mode = dominant_value(song_pre, "buildup_pattern")
    chorus_mode = dominant_value(song_chorus, "release_mode")
    if pre_mode:
        clauses.append(f"Verse—Pre-chorus—Chorus 路径以“{PRE_ZH[pre_mode]}”最常见。")
    if chorus_mode:
        clauses.append(f"副歌入口主要表现为{CHORUS_ZH[chorus_mode]}。")

    repeat_descriptions = []
    for section_name, label in [("verse", "主歌"), ("chorus", "副歌")]:
        dv_key = f"{section_name}_first_last_valence"
        da_key = f"{section_name}_first_last_arousal"
        if math.isnan(profile[dv_key]) or math.isnan(profile[da_key]):
            continue
        dv_change = float(profile[dv_key])
        da_change = float(profile[da_key])
        if abs(dv_change) <= STABLE_THRESHOLD and abs(da_change) <= STABLE_THRESHOLD:
            repeat_descriptions.append(f"后续{label}基本确认首次出现的 VA 区域")
        elif abs(dv_change) <= STABLE_THRESHOLD and da_change > STABLE_THRESHOLD:
            repeat_descriptions.append(f"后续{label}保持 Valence 底色并提高 Arousal {da_change:+.3f}")
        elif dv_change > STABLE_THRESHOLD and da_change > STABLE_THRESHOLD:
            repeat_descriptions.append(f"后续{label}在 V/A 两个维度共同升级 ({dv_change:+.3f}/{da_change:+.3f})")
        elif da_change < -STABLE_THRESHOLD:
            repeat_descriptions.append(f"后续{label}降低 Arousal {da_change:+.3f}，呈现减弱或重释")
        else:
            repeat_descriptions.append(f"后续{label}主要改变 Valence {dv_change:+.3f}，呈现情绪重释")
    if repeat_descriptions:
        clauses.append("重复组织方面，" + "；".join(repeat_descriptions) + "。")

    song_bridge = bridge_events[bridge_events["song_id"] == song_id] if len(bridge_events) else pd.DataFrame()
    if len(song_bridge):
        descriptors = Counter("|".join(song_bridge["function_descriptors"]).split("|")).most_common(2)
        bridge_map = {
            "intensifying": "继续强化", "cooling": "暂时降温", "valence_redirecting": "改变 Valence 方向",
            "contextual_deviation": "偏离此前核心情绪区域", "deviation_then_return": "偏离后由副歌回归",
            "connecting_or_weak_change": "主要承担连接",
        }
        clauses.append("Bridge 主要呈现" + "、".join(bridge_map[key] for key, _ in descriptors) + "。")

    song_outro = outro_events[outro_events["song_id"] == song_id] if len(outro_events) else pd.DataFrame()
    if len(song_outro):
        outro_map = {
            "rapid_release": "快速卸力", "gradual_dissipation_within_outro": "尾奏内部渐进消散",
            "high_level_ending": "相对全曲保持高位结束", "emotional_return": "回到接近开场的 VA 区域",
            "open_or_migrated_ending": "停在明显不同于开场的新情绪位置", "positive_closure": "正向收束",
            "negative_settling": "消极沉降", "weak_or_mixed_closure": "混合或弱收束",
        }
        descriptors = song_outro.iloc[0]["closure_descriptors"].split("|")
        clauses.append("结尾表现为" + "、".join(outro_map[item] for item in descriptors) + "。")

    clauses.append(
        f"能量高点位于归一化时间 {profile['arousal_peak_position']:.2f}，最大段落转折位于 {profile['largest_transition_position']:.2f}；"
        "这些位置分别表示模型估计的 Arousal 高点和 VA 变化边界，不直接等同于听觉高潮。"
    )
    evidence = (
        f"baseline_VA={profile['baseline_valence']:.3f}/{profile['baseline_arousal']:.3f}; "
        f"range_VA={profile['valence_range']:.3f}/{profile['arousal_range']:.3f}; "
        f"mean_contrast={profile['mean_section_contrast']:.3f}; "
        f"VA_coupling={profile['section_va_coupling_spearman']:.3f}"
    )
    return "".join(clauses), evidence


def build_portraits(
    sections: pd.DataFrame,
    profiles: pd.DataFrame,
    function_events: pd.DataFrame,
    bridge_events: pd.DataFrame,
    outro_events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, profile in profiles.iterrows():
        song = sections[sections["song_id"] == profile["song_id"]]
        portrait, evidence = portrait_for_song(song, profile, function_events, bridge_events, outro_events)
        rows.append({
            "song_id": profile["song_id"], "title": profile["title"], "artist": profile["artist"],
            "emotional_organization_portrait_zh": portrait,
            "traceable_numeric_basis": evidence,
            "interpretive_boundary": "基于 section-level pseudo-VA；不推断歌词、和声、调式、节奏、音色或配器机制",
        })
    return pd.DataFrame(rows)


def count_descriptors(frame: pd.DataFrame, column: str, family: str) -> list[dict]:
    counter: Counter = Counter()
    for value in frame[column].dropna():
        counter.update(str(value).split("|"))
    total = len(frame)
    return [
        {"summary_family": family, "pattern": key, "event_count": value, "event_share": value / total if total else math.nan}
        for key, value in counter.most_common()
    ]


def build_corpus_summary(
    function_events: pd.DataFrame,
    bridge_events: pd.DataFrame,
    outro_events: pd.DataFrame,
    region_transitions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pre = function_events[function_events["event_family"] == "pre_chorus_buildup"]
    chorus = function_events[function_events["event_family"] == "chorus_release"]
    for family, frame, column in [
        ("pre_chorus_buildup", pre, "buildup_pattern"),
        ("chorus_release", chorus, "release_mode"),
    ]:
        for key, count in frame[column].value_counts().items():
            rows.append({"summary_family": family, "pattern": key, "event_count": int(count), "event_share": float(count / len(frame))})
    rows.extend(count_descriptors(bridge_events, "function_descriptors", "bridge_function"))
    rows.extend(count_descriptors(outro_events, "closure_descriptors", "outro_closure"))
    transition_counts = region_transitions.groupby(["from_region", "to_region"]).size().sort_values(ascending=False)
    for (source, target), count in transition_counts.items():
        rows.append({
            "summary_family": "va_region_transition", "pattern": f"{source}->{target}",
            "event_count": int(count), "event_share": float(count / len(region_transitions)),
        })
    return pd.DataFrame(rows)


def make_figures(
    function_events: pd.DataFrame,
    profiles: pd.DataFrame,
    region_transitions: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

    pre = function_events[function_events["event_family"] == "pre_chorus_buildup"]
    chorus = function_events[function_events["event_family"] == "chorus_release"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pre_order = pre["buildup_pattern"].value_counts().index
    chorus_order = chorus["release_mode"].value_counts().index
    sns.countplot(data=pre, y="buildup_pattern", order=pre_order, color="#4C78A8", ax=axes[0])
    sns.countplot(data=chorus, y="release_mode", order=chorus_order, color="#E45756", ax=axes[1])
    axes[0].set(title="Pre-chorus buildup functions", xlabel="Events", ylabel="")
    axes[1].set(title="Chorus entry release functions", xlabel="Events", ylabel="")
    fig.tight_layout()
    save_figure(fig, "15_buildup_and_release_function_counts")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    scatter = ax.scatter(
        profiles["mean_section_contrast"], profiles["transition_concentration_top2_share"],
        c=profiles["whole_song_arousal_slope"], cmap="coolwarm", s=65, alpha=0.8,
        edgecolor="white", linewidth=0.5,
    )
    ax.set(
        title="Continuous emotional-organization strategies across songs",
        xlabel="Mean adjacent-section VA contrast",
        ylabel="Share of VA path concentrated in two largest transitions",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Whole-song Arousal slope")
    fig.tight_layout()
    save_figure(fig, "16_continuous_emotional_strategy_space")

    order = ["positive_high_arousal", "positive_low_arousal", "negative_high_arousal", "negative_low_arousal"]
    matrix = pd.crosstab(region_transitions["from_region"], region_transitions["to_region"]).reindex(index=order, columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Transitions"}, ax=ax)
    ax.set(title="Section-to-section movement in the VA plane", xlabel="Destination region", ylabel="Source region")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, "17_va_region_transition_matrix")


def write_report(
    sections: pd.DataFrame,
    function_events: pd.DataFrame,
    bridge_events: pd.DataFrame,
    outro_events: pd.DataFrame,
    profiles: pd.DataFrame,
    region_transitions: pd.DataFrame,
) -> None:
    pre = function_events[function_events["event_family"] == "pre_chorus_buildup"]
    chorus = function_events[function_events["event_family"] == "chorus_release"]
    pre_counts = pre["buildup_pattern"].value_counts()
    chorus_counts = chorus["release_mode"].value_counts()
    crossed = float(region_transitions["crosses_region"].mean())
    arousal_led = float((profiles["arousal_range"] > profiles["valence_range"] * 1.25).mean())
    concentrated = float((profiles["transition_concentration_top2_share"] >= 0.5).mean())
    section_regions = sections.apply(lambda row: emotion_region(float(row["mean_valence"]), float(row["mean_arousal"])), axis=1)
    region_counts = section_regions.value_counts()
    valid_bridge_returns = bridge_events.dropna(subset=["next_chorus_distance_from_prior_chorus", "distance_from_prior_chorus"])
    bridge_return = float((valid_bridge_returns["next_chorus_distance_from_prior_chorus"] < valid_bridge_returns["distance_from_prior_chorus"]).mean()) if len(valid_bridge_returns) else math.nan
    intro_outro = outro_events["intro_outro_va_distance"].dropna()
    return_share = float((intro_outro <= RETURN_THRESHOLD).mean()) if len(intro_outro) else math.nan

    lines = [
        "# 97首中文流行歌曲的音乐情感组织与段落功能",
        "",
        f"报告时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "当前科研进度：完成 section 统计与第一层音乐学解释；本报告进一步描述歌曲如何通过蓄力、释放、偏离、回归、重复和收束组织情感体验。",
        "",
        "## 一、证据范围与方法",
        "",
        f"分析覆盖 `{sections['song_id'].nunique()}` 首歌曲、`{len(sections)}` 个 section。所有结果来自冻结 Teacher 的 section-level pseudo-VA。阈值 `±{STABLE_THRESHOLD:.2f}` 仅用于把局部变化描述为上升、下降或近似稳定，不是心理学类别边界，也没有根据结果调优。",
        "",
        "本轮不增加显著性检验，也不建立互斥的歌曲类型。输出是可复算的功能事件、连续策略指标和逐歌画像。高/低 VA 区域是二维情感空间中的可能听觉解释，不是精确离散情绪标签。",
        "",
        "## 二、Pre-chorus 如何组织蓄力",
        "",
        f"数据中共有 `{len(pre)}` 个严格相邻的 Verse→Pre-chorus→Chorus 事件。主要模式为：" + "；".join(f"{PRE_ZH[k]} `{v}/{len(pre)}`" for k, v in pre_counts.items()) + "。",
        "",
        "这一区分比单纯报告 Pre-chorus 平均 Arousal 更接近听觉过程：有些歌曲把能量分两级推进，有些把主要变化留到副歌入口，也有少数路径在预副歌抬升后不继续释放。这里的“蓄力”是结构性情感变化候选，不能直接等同于和声张力。",
        "",
        "## 三、Chorus 的释放不是单一现象",
        "",
        f"共有 `{len(chorus)}` 次副歌入口。释放方式分布为：" + "；".join(f"{CHORUS_ZH[k]} `{v}/{len(chorus)}`" for k, v in chorus_counts.items()) + "。",
        "",
        "因此，副歌可以提高积极性与能量，也可以在 Valence 基本稳定时集中释放强度，或者把偏消极情绪推向更高唤醒。副歌的共同形式作用更接近情感集中、放大或公开化，而不是统一变得快乐。",
        "",
        "## 四、Bridge 的偏离与回归",
        "",
        f"共有 `{len(bridge_events)}` 个 Bridge 事件，其中 `{len(valid_bridge_returns)}` 个同时具有可比较的前序 Chorus 中心和后续 Chorus；后者有 `{bridge_return:.1%}` 的下一次 Chorus 比 Bridge 更接近此前 Chorus 中心。",
        "",
        "这为“偏离—回归”提供描述性依据，但 Bridge 的方向并不统一：它可以继续强化、暂时降温、改变 Valence 方向，也可以只承担连接。是否发生主题、和声、织体或配器对比，必须逐首听审。",
        "",
        "## 五、Outro 如何结束情绪",
        "",
        f"在具有 Intro—Outro 可比性的歌曲中，`{return_share:.1%}` 的 Outro 回到距 Intro 不超过 `{RETURN_THRESHOLD:.2f}` 的 VA 区域。其余歌曲可能形成部分回归或新的结束位置。",
        "",
        "收束已拆分为快速卸力、尾奏内部渐进消散、高位结束、情绪回归、迁移式结束、正向收束和消极沉降等可并存描述。它们不是互斥曲式类别，而是同一首歌可能同时采用的结束策略。",
        "",
        "## 六、二维 VA 中的情绪性质变化",
        "",
        f"全部相邻 section 转换中，`{crossed:.1%}` 跨越了以 VA=0 划分的情感区域，其余主要在同一区域内改变程度。`{arousal_led:.1%}` 的歌曲 Arousal 范围至少是 Valence 范围的 1.25 倍，提示许多歌曲主要通过唤醒度而非情绪正负方向组织段落差异。",
        "",
        "跨区域并不自动等于戏剧性转折；同一区域内的大幅移动也可能形成强烈听觉对比。报告因此同时保留区域迁移、连续 VA 距离和两个维度各自的变化。",
        "",
        f"象限结果对 Teacher 的零点校准高度敏感：负 Valence、高 Arousal 区域仅有 `{int(region_counts.get('negative_high_arousal', 0))}/{len(sections)}` 个 section。因此象限分析只作辅助描述，不能据此推断中文流行歌曲缺少某类情绪，也不能取代连续 VA 结果。",
        "",
        "## 七、连续的情感组织策略",
        "",
        f"`{concentrated:.1%}` 的歌曲有至少一半 VA 路径集中在两个最大 section 转换中；其余歌曲的变化更分散。每首歌还记录了基础 VA 位置、V/A 动态范围、平均对比度、全曲与后半段斜率、副歌入口方向一致性、重复核心段距离、Bridge 回归增益、Intro—Outro 距离、三种高点位置、收束强度和 VA 耦合。",
        "",
        "这些指标描述歌曲在连续维度上的组织策略，不用于把歌曲硬分成少数类别。",
        "",
        "## 八、重复段落的音乐情感意义",
        "",
        "逐歌画像分别记录首次与末次 Verse/Chorus 的 V/A 变化、副歌入口方向一致性和重复核心段距离。由此可以区分情绪确认、强化、重释、升级或耗散，但这些术语只有在对应数值满足规则时才写入描述。",
        "",
        "## 九、逐歌情感画像",
        "",
        "已为 97 首歌曲生成规则驱动的中文情感组织画像。每条画像同时保存数值依据，并明确它只描述 pseudo-VA 结果。画像不使用歌词，不虚构听感，也不推断调式、和声、节奏、音色或配器。",
        "",
        "## 十、论文使用建议",
        "",
        "1. 主结果以功能事件和连续策略为核心，不把歌曲聚成互斥类型。",
        "2. 选择个案时依据预先输出的数值规则，并同时展示符合趋势和反例歌曲。",
        "3. 将“张力—释放”写成宏观情感组织解释框架，不等同于和声张力。",
        "4. 论文中把 Teacher 输出统一称为 pseudo-VA，人工验证只支撑趋势一致性。",
        "5. 下一步优先对代表个案做模型盲的听觉复核，再补充可直接测量的响度、节奏或音色描述。",
    ]
    (REPORTS / "EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    sections = pd.read_csv(SECTIONS_PATH).sort_values(["sample_index", "section_order"]).reset_index(drop=True)
    function_events, bridge_events, outro_events = build_function_events(sections)
    region_transitions = build_region_transitions(sections)
    profiles = build_song_profiles(sections, function_events, bridge_events, outro_events, region_transitions)
    portraits = build_portraits(sections, profiles, function_events, bridge_events, outro_events)
    summary = build_corpus_summary(function_events, bridge_events, outro_events, region_transitions)

    if profiles["song_id"].nunique() != 97 or len(profiles) != 97:
        raise ValueError("Song strategy profile must contain exactly 97 unique songs")
    if portraits["song_id"].nunique() != 97 or len(portraits) != 97:
        raise ValueError("Song portrait output must contain exactly 97 unique songs")
    if len(region_transitions) != len(sections) - sections["song_id"].nunique():
        raise ValueError("Region transition count does not equal sections minus songs")

    save_csv(function_events, "emotional_function_events.csv")
    save_csv(
        function_events[function_events["event_family"] == "pre_chorus_buildup"].dropna(axis=1, how="all"),
        "prechorus_buildup_events.csv",
    )
    save_csv(
        function_events[function_events["event_family"] == "chorus_release"].dropna(axis=1, how="all"),
        "chorus_release_events.csv",
    )
    save_csv(bridge_events, "bridge_deviation_and_return_events.csv")
    save_csv(outro_events, "outro_closure_events.csv")
    save_csv(region_transitions, "va_region_transitions.csv")
    save_csv(profiles, "song_emotional_strategy_profiles.csv")
    save_csv(portraits, "song_emotional_portraits_zh.csv")
    save_csv(summary, "emotional_organization_corpus_summary.csv")
    make_figures(function_events, profiles, region_transitions)
    write_report(sections, function_events, bridge_events, outro_events, profiles, region_transitions)

    output_files = [
        "emotional_function_events.csv", "prechorus_buildup_events.csv", "chorus_release_events.csv",
        "bridge_deviation_and_return_events.csv",
        "outro_closure_events.csv", "va_region_transitions.csv",
        "song_emotional_strategy_profiles.csv", "song_emotional_portraits_zh.csv",
        "emotional_organization_corpus_summary.csv",
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass",
        "songs": int(sections["song_id"].nunique()),
        "sections": int(len(sections)),
        "stable_change_threshold": STABLE_THRESHOLD,
        "strong_change_threshold": STRONG_THRESHOLD,
        "analysis_role": "descriptive emotional-organization layer; no new inferential tests or fixed song classes",
        "tables": output_files,
        "figures": [
            "15_buildup_and_release_function_counts",
            "16_continuous_emotional_strategy_space",
            "17_va_region_transition_matrix",
        ],
        "report": "reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md",
        "scientific_boundary": "VA outcomes support functional interpretation but not causal claims about harmony, rhythm, timbre, production, or lyrics",
    }
    (BASE / "emotional_organization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
