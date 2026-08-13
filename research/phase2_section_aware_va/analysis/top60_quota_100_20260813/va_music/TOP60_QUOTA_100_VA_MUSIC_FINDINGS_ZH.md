# 100首中文流行歌曲 VA/MIR/MER 论文统计结果

生成时间：2026-08-14T00:14:01+08:00

## 数据与写入状态

- 样本：100 首 QQ音乐歌手榜配额歌曲，60 位歌手。
- 人工质检 section：1046 段。
- Teacher frame VA：51448 行，0.5 秒一帧。
- Teacher：`pmemo_final_clean_valence_teacher_v1` / `pmemo_final_clean_arousal_teacher_v1`。
- Supabase 新表已写入：`paper_top60_quota_100_va_runs`、`paper_top60_quota_100_va_frames`、`paper_top60_quota_100_section_va`、`paper_top60_quota_100_song_va`。
- 对齐审计：1046/1046 个 section 均有 frame；4 个解码尾帧超过最后人工边界，已按尾部 frame 排除。

## MIR 结构结论

Verse 与 Chorus 覆盖 100/100 首歌曲。Chorus 共 362 段，Verse 共 246 段。这说明本样本的中文流行歌曲以 Verse/Chorus 轮转为稳定核心；Pre-chorus、Bridge、Interlude、Outro 是围绕核心循环展开的功能模块。

Interlude 覆盖 85/100 首，且 `chorus->interlude` 是高频路径之一。它不应被当作无歌词空白或噪声，而应作为歌曲轮转中的结构铰链。

## MER 情绪结论

Chorus 相对 Verse 的平均提升为：

- Valence：+0.119，95% CI [+0.100, +0.138]，89.0% 的歌曲为正向。
- Arousal：+0.279，95% CI [+0.251, +0.308]，99.0% 的歌曲为正向。

因此，当前 100 首结果支持把副歌解释为更强的 **Arousal 到达/能量释放**，而不是简单说副歌一定“更快乐”。这正是 MER 能补充 MIR 的地方：同样的 Chorus 标签，在情绪维度上首先表现为能量中心。

严格相邻的 Verse->Pre-chorus->Chorus 事件共有 67 个；最多的模式是 `staged_arousal_buildup`（46 次）。这可用于讨论 Pre-chorus 是否承担蓄力、延迟和导向副歌的功能。

模型估计的最大 mean-Arousal section 位于 Chorus 的歌曲有 53/100 首。平均归一化轨迹的 Arousal 峰值出现在歌曲 0.78 处。这支持“副歌和后半段积累”在情绪组织中的核心位置，但也保留 Bridge/Interlude/Outro 等非副歌峰值作为反例和个案分析对象。

Chorus->Outro 的 Arousal 平均变化为 -0.188，95% CI [-0.230, -0.147]。这可用于检验 Outro 是否承担局部释放/收束功能。

Outro 相对 Intro 的 Arousal 差为 +0.186，用于判断歌曲是否完全回到开头情绪基线。

## 音乐解释落点

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
