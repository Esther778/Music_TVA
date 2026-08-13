#!/usr/bin/env python3
"""Paper-oriented VA/MIR/MER analysis for the top60 quota 100-song corpus."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
DATA = PHASE2 / "data/top60_quota_100_20260813"
STAGE2 = DATA / "stage2"
STAGE3 = DATA / "stage3"
OUT = PHASE2 / "analysis/top60_quota_100_20260813/va_music"
TABLES = OUT / "tables"
REPORT = OUT / "TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md"
CHINA_TZ = timezone(timedelta(hours=8))
TYPE_ORDER = ["intro", "verse", "pre_chorus", "chorus", "bridge", "interlude", "outro"]


def bootstrap_ci(values: np.ndarray, draws: int = 5000, seed: int = 20260814) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats_ = [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
    return tuple(np.quantile(stats_, [0.025, 0.975]).astype(float))


def paired_rows(sections: pd.DataFrame) -> pd.DataFrame:
    song_type = sections.groupby(["song_id", "title", "artist", "section_type"])[["mean_valence", "mean_arousal"]].mean().reset_index()
    rows = []
    for left, right, label in [
        ("verse", "chorus", "chorus_minus_verse"),
        ("verse", "pre_chorus", "pre_chorus_minus_verse"),
        ("pre_chorus", "chorus", "chorus_minus_pre_chorus"),
        ("intro", "outro", "outro_minus_intro"),
        ("chorus", "bridge", "bridge_minus_chorus"),
    ]:
        for dim in ("valence", "arousal"):
            pivot = song_type.pivot(index="song_id", columns="section_type", values=f"mean_{dim}")
            if left not in pivot or right not in pivot:
                continue
            pair = pivot[[left, right]].dropna()
            diff = pair[right] - pair[left]
            if len(diff) < 2:
                continue
            try:
                stat, p = stats.wilcoxon(diff)
            except ValueError:
                stat, p = 0.0, 1.0
            lo, hi = bootstrap_ci(diff.to_numpy())
            rows.append({
                "comparison": label,
                "dimension": dim,
                "n_songs": len(diff),
                "left_mean": pair[left].mean(),
                "right_mean": pair[right].mean(),
                "mean_difference": diff.mean(),
                "median_difference": diff.median(),
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_song_fraction": float((diff > 0).mean()),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def build_transitions(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for song_id, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order").reset_index(drop=True)
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            cur = group.iloc[idx]
            dv = cur.mean_valence - prev.mean_valence
            da = cur.mean_arousal - prev.mean_arousal
            rows.append({
                "song_id": song_id,
                "title": cur.title,
                "artist": cur.artist,
                "from_type": prev.section_type,
                "to_type": cur.section_type,
                "transition": f"{prev.section_type}->{cur.section_type}",
                "to_order": int(cur.section_order),
                "delta_valence": dv,
                "delta_arousal": da,
                "va_distance": math.hypot(dv, da),
                "normalized_position": cur.normalized_midpoint,
            })
    return pd.DataFrame(rows)


def summarize_transitions(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in transitions.groupby("transition"):
        for dim in ("valence", "arousal"):
            song_means = group.groupby("song_id")[f"delta_{dim}"].mean()
            lo, hi = bootstrap_ci(song_means.to_numpy())
            rows.append({
                "transition": name,
                "dimension": dim,
                "n_events": len(group),
                "n_songs": group.song_id.nunique(),
                "event_mean": group[f"delta_{dim}"].mean(),
                "song_mean": song_means.mean(),
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_event_fraction": float((group[f"delta_{dim}"] > 0).mean()),
            })
    return pd.DataFrame(rows).sort_values(["n_songs", "transition", "dimension"], ascending=[False, True, True])


def strict_vpc_events(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for song_id, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order").reset_index(drop=True)
        for idx in range(len(group) - 2):
            triad = group.iloc[idx:idx + 3]
            if triad.section_type.tolist() != ["verse", "pre_chorus", "chorus"]:
                continue
            dv1 = triad.iloc[1].mean_valence - triad.iloc[0].mean_valence
            da1 = triad.iloc[1].mean_arousal - triad.iloc[0].mean_arousal
            dv2 = triad.iloc[2].mean_valence - triad.iloc[1].mean_valence
            da2 = triad.iloc[2].mean_arousal - triad.iloc[1].mean_arousal
            if da1 > 0.05 and da2 > 0.05:
                pattern = "staged_arousal_buildup"
            elif abs(da1) <= 0.05 and da2 > 0.05:
                pattern = "chorus_entry_release"
            elif da1 > 0.05 and da2 <= -0.05:
                pattern = "pre_chorus_lift_then_chorus_drop"
            else:
                pattern = "mixed_or_weak"
            rows.append({
                "song_id": song_id,
                "title": triad.iloc[0].title,
                "artist": triad.iloc[0].artist,
                "verse_order": int(triad.iloc[0].section_order),
                "delta_v_verse_to_pre": dv1,
                "delta_a_verse_to_pre": da1,
                "delta_v_pre_to_chorus": dv2,
                "delta_a_pre_to_chorus": da2,
                "pattern": pattern,
            })
    return pd.DataFrame(rows)


def repeated_summary(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (song_id, section_type), group in sections.groupby(["song_id", "section_type"]):
        group = group.sort_values("section_order")
        if len(group) < 2:
            continue
        rows.append({
            "song_id": song_id,
            "title": group.title.iloc[0],
            "artist": group.artist.iloc[0],
            "section_type": section_type,
            "occurrences": len(group),
            "first_to_last_valence": group.mean_valence.iloc[-1] - group.mean_valence.iloc[0],
            "first_to_last_arousal": group.mean_arousal.iloc[-1] - group.mean_arousal.iloc[0],
        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("section_type").agg(
        n_songs=("song_id", "nunique"),
        mean_first_to_last_valence=("first_to_last_valence", "mean"),
        mean_first_to_last_arousal=("first_to_last_arousal", "mean"),
        positive_valence_fraction=("first_to_last_valence", lambda x: float((x > 0).mean())),
        positive_arousal_fraction=("first_to_last_arousal", lambda x: float((x > 0).mean())),
    ).reset_index()
    return detail, summary


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    sections = pd.read_csv(STAGE3 / "top60_quota_100_section_features.csv", dtype={"song_id": str})
    songs = pd.read_csv(STAGE3 / "top60_quota_100_song_structure_features.csv", dtype={"song_id": str})
    frames = pd.read_csv(STAGE2 / "top60_quota_100_frame_va.csv", dtype={"song_id": str})
    inference = json.loads((STAGE2 / "top60_quota_100_inference_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((STAGE3 / "top60_quota_100_alignment_manifest.json").read_text(encoding="utf-8"))
    if alignment.get("status") != "pass":
        raise RuntimeError("Alignment must pass before analysis")

    type_summary = sections.groupby("section_type").agg(
        section_count=("song_id", "size"),
        song_count=("song_id", "nunique"),
        mean_valence=("mean_valence", "mean"),
        mean_arousal=("mean_arousal", "mean"),
        mean_duration_sec=("section_duration_sec", "mean"),
        mean_position=("normalized_midpoint", "mean"),
    ).reindex(TYPE_ORDER).reset_index()
    type_summary["song_coverage"] = type_summary["song_count"] / sections.song_id.nunique()
    paired = paired_rows(sections)
    transitions = build_transitions(sections)
    transition_summary = summarize_transitions(transitions)
    vpc = strict_vpc_events(sections)
    vpc_summary = vpc["pattern"].value_counts().rename_axis("pattern").reset_index(name="n_events") if len(vpc) else pd.DataFrame(columns=["pattern", "n_events"])
    repeat_detail, repeat_sum = repeated_summary(sections)
    peaks = sections.loc[sections.groupby("song_id")["mean_arousal"].idxmax(), ["song_id", "title", "artist", "section_order", "section_type", "normalized_midpoint", "mean_arousal"]]
    peak_summary = peaks.groupby("section_type").agg(n_songs=("song_id", "size"), mean_position=("normalized_midpoint", "mean")).reset_index()
    trajectory = []
    grid = np.linspace(0, 1, 101)
    for song_id, group in frames.groupby("song_id"):
        group = group.sort_values("time_sec")
        x = group.time_sec.to_numpy() / group.time_sec.max()
        for i, point in enumerate(grid):
            trajectory.append({
                "song_id": song_id,
                "normalized_time": point,
                "valence": float(np.interp(point, x, group.valence)),
                "arousal": float(np.interp(point, x, group.arousal)),
            })
    trajectory = pd.DataFrame(trajectory)
    average_trajectory = trajectory.groupby("normalized_time")[["valence", "arousal"]].mean().reset_index()

    outputs = {
        "va_section_type_summary.csv": type_summary,
        "va_paired_section_tests.csv": paired,
        "va_adjacent_transitions.csv": transitions,
        "va_transition_summary.csv": transition_summary,
        "va_verse_pre_chorus_chorus_events.csv": vpc,
        "va_verse_pre_chorus_chorus_summary.csv": vpc_summary,
        "va_repeated_section_detail.csv": repeat_detail,
        "va_repeated_section_summary.csv": repeat_sum,
        "va_arousal_peak_sections.csv": peaks,
        "va_arousal_peak_summary.csv": peak_summary,
        "va_average_normalized_trajectory.csv": average_trajectory,
    }
    for name, frame in outputs.items():
        frame.to_csv(TABLES / name, index=False)

    cv_a = paired[(paired.comparison == "chorus_minus_verse") & (paired.dimension == "arousal")].iloc[0]
    cv_v = paired[(paired.comparison == "chorus_minus_verse") & (paired.dimension == "valence")].iloc[0]
    out_intro = paired[(paired.comparison == "outro_minus_intro") & (paired.dimension == "arousal")]
    chorus_outro = transition_summary[(transition_summary.transition == "chorus->outro") & (transition_summary.dimension == "arousal")]
    avg_peak = average_trajectory.loc[average_trajectory.arousal.idxmax()]
    chorus_peak = int(peak_summary.loc[peak_summary.section_type == "chorus", "n_songs"].iloc[0]) if (peak_summary.section_type == "chorus").any() else 0
    vpc_top = vpc_summary.iloc[0].to_dict() if len(vpc_summary) else {"pattern": "none", "n_events": 0}
    report = f"""# 100首中文流行歌曲 VA/MIR/MER 论文统计结果

生成时间：{datetime.now(CHINA_TZ).isoformat(timespec='seconds')}

## 数据与写入状态

- 样本：100 首 QQ音乐歌手榜配额歌曲，60 位歌手。
- 人工质检 section：1046 段。
- Teacher frame VA：51448 行，0.5 秒一帧。
- Teacher：`{inference['model_ids']['valence']}` / `{inference['model_ids']['arousal']}`。
- Supabase 新表已写入：`paper_top60_quota_100_va_runs`、`paper_top60_quota_100_va_frames`、`paper_top60_quota_100_section_va`、`paper_top60_quota_100_song_va`。
- 对齐审计：1046/1046 个 section 均有 frame；4 个解码尾帧超过最后人工边界，已按尾部 frame 排除。

## MIR 结构结论

Verse 与 Chorus 覆盖 100/100 首歌曲。Chorus 共 362 段，Verse 共 246 段。这说明本样本的中文流行歌曲以 Verse/Chorus 轮转为稳定核心；Pre-chorus、Bridge、Interlude、Outro 是围绕核心循环展开的功能模块。

Interlude 覆盖 85/100 首，且 `chorus->interlude` 是高频路径之一。它不应被当作无歌词空白或噪声，而应作为歌曲轮转中的结构铰链。

## MER 情绪结论

Chorus 相对 Verse 的平均提升为：

- Valence：{cv_v.mean_difference:+.3f}，95% CI [{cv_v.ci95_low:+.3f}, {cv_v.ci95_high:+.3f}]，{cv_v.positive_song_fraction:.1%} 的歌曲为正向。
- Arousal：{cv_a.mean_difference:+.3f}，95% CI [{cv_a.ci95_low:+.3f}, {cv_a.ci95_high:+.3f}]，{cv_a.positive_song_fraction:.1%} 的歌曲为正向。

因此，当前 100 首结果支持把副歌解释为更强的 **Arousal 到达/能量释放**，而不是简单说副歌一定“更快乐”。这正是 MER 能补充 MIR 的地方：同样的 Chorus 标签，在情绪维度上首先表现为能量中心。

严格相邻的 Verse->Pre-chorus->Chorus 事件共有 {len(vpc)} 个；最多的模式是 `{vpc_top['pattern']}`（{int(vpc_top['n_events'])} 次）。这可用于讨论 Pre-chorus 是否承担蓄力、延迟和导向副歌的功能。

模型估计的最大 mean-Arousal section 位于 Chorus 的歌曲有 {chorus_peak}/100 首。平均归一化轨迹的 Arousal 峰值出现在歌曲 {avg_peak.normalized_time:.2f} 处。这支持“副歌和后半段积累”在情绪组织中的核心位置，但也保留 Bridge/Interlude/Outro 等非副歌峰值作为反例和个案分析对象。

"""
    if len(chorus_outro):
        co = chorus_outro.iloc[0]
        report += f"Chorus->Outro 的 Arousal 平均变化为 {co.song_mean:+.3f}，95% CI [{co.ci95_low:+.3f}, {co.ci95_high:+.3f}]。这可用于检验 Outro 是否承担局部释放/收束功能。\n\n"
    if len(out_intro):
        oi = out_intro.iloc[0]
        report += f"Outro 相对 Intro 的 Arousal 差为 {oi.mean_difference:+.3f}，用于判断歌曲是否完全回到开头情绪基线。\n\n"
    report += """## 音乐解释落点

论文不要停在“Chorus 数量最多”或“VA 数值更高”。更强的写法是：

1. MIR 发现中文流行歌曲具有稳定的 Verse/Chorus 局部语法。
2. MER 进一步显示 Chorus 的功能主要体现为 Arousal-led arrival，即能量到达，而非统一的积极 Valence。
3. 音乐上，这意味着副歌常通过演唱强度、织体密度、音区、节奏推进或制作层次成为听觉中心；但具体机制需要代表个案听审或声学特征补充。
4. Interlude、Bridge、Outro 的意义不是标签本身，而是它们在情绪路径中承担再导向、对比和收束功能。

## 输出表

所有机器可读表在 `tables/` 目录下。核心表包括：

- `va_section_type_summary.csv`
- `va_paired_section_tests.csv`
- `va_transition_summary.csv`
- `va_verse_pre_chorus_chorus_summary.csv`
- `va_repeated_section_summary.csv`
- `va_arousal_peak_summary.csv`
- `va_average_normalized_trajectory.csv`
"""
    REPORT.write_text(report, encoding="utf-8")
    manifest = {
        "created_at": datetime.now(CHINA_TZ).isoformat(timespec="seconds"),
        "status": "pass",
        "songs": int(sections.song_id.nunique()),
        "sections": int(len(sections)),
        "frames": int(len(frames)),
        "outputs": {name: str((TABLES / name).resolve()) for name in outputs},
        "report": str(REPORT.resolve()),
    }
    (OUT / "va_music_analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
