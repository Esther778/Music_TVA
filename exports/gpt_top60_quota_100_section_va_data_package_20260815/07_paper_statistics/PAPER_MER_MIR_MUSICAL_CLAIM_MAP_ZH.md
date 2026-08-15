# 100首中文流行歌曲统计结果的 MER/MIR/音乐结论映射

本文件用于论文写作：把当前 100 首已质检语料的统计结果，从 section 结构统计推进到 MIR 与 MER 研究问题，并最终落到音乐形式与听觉功能解释上。

## 一、论证边界

当前已完成的是 **人工质检 section 边界与 section type 的结构统计**，数据来源为 Supabase `human_annotations_readable` 冻结快照。它可以直接支持 MIR 层面的歌曲结构、段落类型、段落转移与形式语法结论。

当前尚未完成的是新版 100 首语料的 frame-level VA 推断与 section 对齐。因此，MER 层面现在只能提出研究假设和分析路径，不能把旧 97/99 首 pseudo-VA 结果直接写成本次 100 首样本的最终情绪结论。

论文写作中应明确区分：

- 已验证结论：来自 100 首人工质检 section 的结构统计。
- 待验证结论：需要新版 100 首 VA 曲线、section-level VA 汇总和 transition-level VA 差异后才能成立。
- 音乐解释：从结构与情绪统计共同推导出的形式功能解释，仍需代表个案听审支撑。

## 二、当前统计能直接支持的 MIR 结论

### 1. 中文流行歌曲存在稳定的核心段落语法

统计结果：

- 100 首歌曲全部包含 Chorus。
- 100 首歌曲全部包含 Verse。
- Chorus 共 362 段，占所有 section 的 34.61%。
- Verse 共 246 段，占所有 section 的 23.52%。
- 每首歌平均 10.46 个 section，中位数 10 个 section。

MIR 结论：

本样本中的中文流行歌曲不是任意拼接的音频序列，而是围绕 Verse/Chorus 形成稳定的段落语法。Verse 和 Chorus 应被视为结构建模中的核心类别，而 Intro、Outro、Pre-chorus、Bridge、Interlude 是围绕核心循环展开的功能模块。

音乐结论：

中文流行歌曲的基本组织不是单一线性展开，而是通过 Verse/Chorus 的反复轮转建立可识别的歌曲身份。副歌的高频出现说明它承担记忆点、情绪集中和形式回归的核心作用。

论文可写句：

> In the 100-song corpus, verse and chorus appeared in every song, while chorus accounted for the largest proportion of annotated sections. This suggests that contemporary Chinese pop songs in the sample are organized around a stable verse/chorus-based formal grammar rather than arbitrary segmentation.

### 2. 歌曲结构稳定性主要存在于局部转移，而不是完整模板

统计结果：

- 高频相邻转移包括 `verse->chorus` 116 次、`chorus->interlude` 111 次、`intro->verse` 92 次、`chorus->outro` 91 次。
- `verse->pre_chorus` 与 `pre_chorus->chorus` 各自也是高频路径。

MIR 结论：

建模时不应只预测单个 section type，也不应假设每首歌共享同一完整结构模板。更合理的 MIR 建模对象是局部段落语法，例如 Verse->Chorus、Verse->Pre-chorus->Chorus、Chorus->Interlude->Verse、Chorus->Outro。

音乐结论：

歌曲形式的稳定性来自局部功能关系：开头进入 verse，verse 导向 chorus，chorus 后进入间奏、回到 verse 或走向 outro。也就是说，中文流行歌曲共享的是“局部语法”和“功能路径”，而不是一张固定的完整曲式图。

论文可写句：

> The most frequent transitions indicate that formal regularity is concentrated at the local-transition level. This supports a MIR strategy that models adjacent functional relations, rather than treating whole-song form as a single fixed template.

### 3. Interlude 在中文流行歌曲中是高频结构节点

统计结果：

- Interlude 共 121 段，覆盖 85 首歌。
- `chorus->interlude` 出现 111 次。
- `interlude->verse` 出现 68 次。

MIR 结论：

Interlude 不能作为边缘类别或简单噪声处理。它在结构序列中承担稳定的连接功能，特别是在 chorus 之后重新导向下一轮 verse/chorus 循环。

音乐结论：

间奏在本样本中不是单纯“无歌词空白”，而是流行歌曲循环结构中的重要铰链。它可能承担器乐过门、能量维持、情绪缓冲或下一轮段落启动前的再导向功能。

论文可写句：

> The high coverage of interludes and the frequent chorus-to-interlude and interlude-to-verse transitions suggest that interludes function as formal hinges in Chinese pop songs, rather than merely as non-vocal filler.

### 4. Bridge 和 Pre-chorus 是可选但有功能意义的模块

统计结果：

- Pre-chorus 共 77 段，覆盖 38 首歌。
- Bridge 共 45 段，覆盖 43 首歌。

MIR 结论：

Pre-chorus 和 Bridge 的低于 50% 覆盖率说明它们不是所有歌曲的必要组成部分，但一旦出现，就应作为独立功能类别建模，而不是并入 verse 或 chorus。

音乐结论：

Pre-chorus 更可能承担副歌前的蓄力、延迟或导向功能；Bridge 更可能承担后半段对比、偏离或重组功能。二者都是对 Verse/Chorus 核心循环的扩展。

论文可写句：

> Pre-chorus and bridge are optional modules in the corpus, but their recurrent presence indicates that they should be modeled as functional expansions of the verse/chorus core rather than collapsed into neighboring sections.

### 5. 歌手排名层级没有显著改变基本结构语法

统计结果：

- Section type 与排名层的关联很弱：Cramer's V = 0.055，χ²(12)=6.28，p=0.901。
- 歌曲层结构指标的 Kruskal-Wallis 检验未发现 p<.05 的排名层差异。

MIR 结论：

在当前样本中，前 10、11-30、31-60 名歌手的歌曲共享相近的宏观段落结构。排名层级不是解释 section 结构差异的主要变量。

音乐结论：

中文流行歌曲的 Verse/Chorus 语法具有跨歌手热度层级的稳定性。也就是说，头部歌手与中高排名歌手之间的差异，可能更多体现在旋律、制作、演唱、音色、歌词或情绪轨迹，而不是是否采用基本 Verse/Chorus 框架。

论文可写句：

> Rank-band comparisons showed no statistically significant differences in the main structural indicators, suggesting that the verse/chorus-based formal grammar is stable across popularity strata in the selected QQ Music singer-ranking corpus.

## 三、需要 VA 后才能支持的 MER 结论

### 1. Chorus 是否主要是 Arousal 到达，而非 Valence 正向化

需要统计：

- 每个 section 的 mean_valence、mean_arousal。
- Chorus 与 Verse 的配对或混合效应比较。
- `verse->chorus` 与 `pre_chorus->chorus` 的 delta_valence、delta_arousal。

MER 结论目标：

如果 Chorus 的 Arousal 提升稳定大于 Valence 提升，可以将副歌解释为能量到达、强度释放或情绪公开化，而不是简单写作“更开心”。

音乐落点：

副歌的核心作用可能是增强演唱强度、织体密度、音区高度、节奏推进或制作层级，从而制造听觉到达感。VA 只能说明情绪输出结果，具体音乐机制需要个案听审或声学特征支持。

### 2. Pre-chorus 是否形成 Verse 到 Chorus 的阶段性蓄力

需要统计：

- 严格相邻 `verse->pre_chorus->chorus` 三段事件。
- Verse 到 Pre-chorus、Pre-chorus 到 Chorus 的 arousal 变化。
- 连续上升、主要在副歌入口释放、预副歌后回落等模式分类。

MER 结论目标：

如果多数事件表现为 arousal 连续上升或副歌入口集中释放，可以将 Pre-chorus 解释为情绪目的性路径中的蓄力模块。

音乐落点：

Pre-chorus 不只是位置标签，而是延迟副歌到达、增强期待、改变和声/节奏/织体指向性的功能区域。

### 3. Interlude 是否是情绪缓冲或轮转铰链

需要统计：

- `chorus->interlude` 与 `interlude->verse` 的 VA 变化。
- Interlude 的平均位置、平均 VA、进入/退出变化方向。
- Interlude 内部是否维持高 arousal 或明显降温。

MER 结论目标：

如果 Interlude 在 Chorus 之后保持或重新分配 arousal，再导向 Verse，则它可被解释为情绪轮转铰链，而不是简单休止段。

音乐落点：

间奏可能通过器乐化、音色替换、节奏保持或短暂卸力，让歌曲从副歌高点过渡到下一轮叙事段落。

### 4. Outro 是否形成局部释放但不完全回到起点

需要统计：

- `chorus->outro` 的 delta_valence/delta_arousal。
- Intro 与 Outro 的 VA 距离。
- 歌曲起点到终点的 net_valence_change、net_arousal_change。

MER 结论目标：

如果 Chorus 到 Outro 普遍下降，但 Outro 仍高于 Intro，则可以写成“局部释放”和“情绪残余”并存。

音乐落点：

中文流行歌曲的结尾可能完成形式闭合，但不必回到开头的情绪基线。尾奏、淡出、重复副歌尾句、终止式延长都可能造成这种残余。

### 5. 完整歌曲是否是循环中的后程累积，而非单线性上升

需要统计：

- 归一化 0-100% 时间的平均 VA 曲线。
- 每首歌 VA path length 与 start-end displacement 的 path directness。
- 最大 arousal 所在 section type 与 normalized position。
- 首次/末次 Chorus、首次/末次 Verse 的 VA 变化。

MER 结论目标：

如果平均轨迹表现后半段抬升但路径直接度较低，可将歌曲情绪组织解释为“重复-偏离-回归中的后程累积”，而不是单一 crescendo。

音乐落点：

流行歌曲的情绪推进来自多轮段落循环中的累积强化，而非从头到尾持续爬坡。相同 section label 的重复也可能因编曲、演唱和制作层变化而具有不同情绪强度。

## 四、论文主线建议

建议把论文主线写成三层：

1. MIR 层：人工质检 section 建立了中文流行歌曲的结构语法。核心发现是 Verse/Chorus 全覆盖、Chorus 最大占比、Interlude 高频连接、Pre-chorus/Bridge 可选扩展、排名层结构差异不显著。
2. MER 层：在结构语法基础上，VA 模型不再只预测整首歌平均情绪，而是把情绪变化对齐到 section 功能。这样可以回答“哪个段落承担到达、蓄力、释放、偏离、回归”的问题。
3. 音乐层：最终解释对象不是数字本身，而是中文流行歌曲如何通过段落轮转组织听觉体验。统计结果服务于音乐解释，音乐解释再通过代表个案听审约束。

一句话总论点：

> 本研究将人工质检的流行歌曲段落结构与连续 Valence-Arousal 预测对齐，说明中文流行歌曲的情绪组织并非简单的整曲平均情绪，而是嵌入 Verse/Chorus 轮转、Pre-chorus 蓄力、Interlude 再导向、Bridge 对比和 Outro 收束等形式功能之中。

## 五、当前可以写入论文的保守结论

当前 100 首结构统计已经可以写：

1. 中文流行歌曲样本具有稳定的 Verse/Chorus 核心语法。
2. Chorus 是最主要的结构中心，覆盖所有歌曲且数量最多。
3. Verse 与 Chorus 构成跨歌手排名层级稳定存在的基本形式框架。
4. Interlude 在中文流行歌曲中高度常见，应作为结构连接功能处理。
5. Pre-chorus 与 Bridge 是可选扩展模块，不应被机械并入相邻类别。
6. 歌曲结构差异没有显著随歌手排名层级变化，说明样本结构规律具有一定跨热度稳定性。

当前不能直接写：

1. Chorus 一定具有更高 Arousal。
2. Pre-chorus 一定形成情绪蓄力。
3. Outro 一定形成情绪释放。
4. 中文流行歌曲的平均 VA 曲线在后半段达到峰值。
5. 某类 section 在 Valence 或 Arousal 上显著高于另一类 section。

这些需要下一步新版 100 首 VA 推断与 section 对齐完成后再写。

## 六、下一步统计分析清单

为了让论文从 MIR 走到 MER 并落到音乐，应继续生成以下结果：

1. 100 首 frame-level VA 预测。
2. frame-level VA 与人工质检 section 边界对齐。
3. section-level mean/max/slope Valence 与 Arousal。
4. Verse vs Chorus、Pre-chorus vs Chorus、Intro vs Outro 的配对统计。
5. `verse->pre_chorus->chorus`、`chorus->interlude->verse`、`chorus->outro` 的 transition-level VA 统计。
6. 首次与末次 Verse/Chorus 的重复段落变化。
7. 平均归一化 VA 曲线与每首歌的情绪路径指标。
8. 代表个案清单：典型趋势、弱趋势、反例各选 2-3 首。

完成这些以后，论文才能从“结构统计”升级为“结构感知的音乐情感组织研究”。
