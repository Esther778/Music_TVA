# 基于冻结连续 VA Teacher 的 Section-Aware 情感组织分析：来自 97 首中文流行歌曲便利样本的探索性证据

**英文题目：** Section-Aware Emotional Organization Analysis Using Frozen Continuous Valence-Arousal Teachers: Exploratory Evidence from a Convenience Sample of 97 Chinese Popular Songs
**作者：** 【投稿前补充】
**单位：** 【投稿前补充】
**通讯作者：** 【投稿前补充】
**稿件版本：** v0.1 中文完整初稿
**撰写日期：** 2026-07-26
**当前科研进度：** VA Teacher 已冻结并完成最终测试；99 首完整歌曲已完成推理与 section 对齐；97 首中文歌曲已完成主分析；15 首歌曲已完成单标注者盲法趋势验证。

> **编辑说明：** 文中的 `【投稿前待完善 P0-xx】` 和 `【投稿前待完善 P1-xx】` 是研究缺口标记，不属于最终论文正文。正式投稿前应完成相应工作，或将其改写为明确的局限性陈述。

---

## 摘要

现有连续音乐情感识别研究多以短音频片段的 Valence-Arousal（VA）回归为目标，较少进一步回答完整流行歌曲如何借助段落顺序、重复与对比组织情感体验。本文提出一种基于冻结 VA Teacher 的 section-aware 完整歌曲情感结构分析框架。第一阶段分别训练 Valence 与 Arousal 两个纯音频 global-local BiGRU Teacher，以 MERT-v1-95M 中间层表征和 PMEmo OpenSMILE 动态特征为输入，在不读取最终测试标签的条件下冻结模型预测。115 个 PMEmo 最终测试片段的完整片段尺度结果显示，Valence 的 RMSE、MAE、PCC 和 CCC 分别为 0.1863、0.1529、0.8029 和 0.8002，Arousal 分别为 0.1886、0.1469、0.8574 和 0.8548。第二阶段采用 30 s 窗口、15 s 步长和 Hann overlap-add 将 Teacher 部署到完整歌曲，获得 0.5 s 分辨率的 pseudo-VA，并与人工质检的 section 边界对齐。

主分析包含 97 首中文流行歌曲、75 位歌手和 1,076 个 section。统计推断以歌曲为独立单位，并结合歌曲级配对比较、bootstrap 置信区间、重复段落对照和二维 VA 转换分析。结果表明，Chorus 相对 Verse 的歌曲级平均 Valence 和 Arousal 分别提高 0.101 和 0.266，显示副歌差异主要由唤醒度主导；72 个严格 Verse-Pre-chorus-Chorus 序列中有 47 个呈现分阶段 Arousal 蓄积；368 次副歌入口并非单一“变得更快乐”，而表现为积极释放、强度释放、延续、反向降温及高唤醒负向释放等多种过程。后续 Chorus 虽整体增强，但其增幅未显著超过后续 Verse，说明该现象更接近全曲后程累积，而非副歌独有强化。Interlude 常承担轮转再导向，Bridge 的作用主要表现为上下文偏离及可能回归，Outro 则呈现局部卸力但不必回到开场情绪位置。15 首歌曲的模型盲人工验证对 Arousal 主导的段落趋势提供了中等支持，但尚不能替代连续 VA ground truth 或独立标注者一致性。

本文的主要贡献不是提出固定的情感弧线类别，而是建立“数值证据—情感过程—音乐形式功能”的可审计解释链，将连续 VA 预测转化为对流行歌曲蓄力、释放、重复、偏离、回归与收束的结构分析。鉴于分析歌曲属于便利样本，且完整歌曲 VA 为 pseudo-label，本文结论应理解为本数据集中的探索性结构倾向。

**关键词：** 音乐情感识别；Valence-Arousal；MERT；Section-aware；中文流行歌曲；情感组织；形式功能；Pseudo-label

---

## 1 引言

音乐情感识别（Music Emotion Recognition, MER）通常将情绪表示为离散类别或连续维度。Valence-Arousal 二维模型分别刻画情绪的正负倾向与激活强度，为描述随时间变化的音乐情感提供了连续坐标系[1-3]。DEAM、PMEmo 等数据集推动了动态 VA 回归研究，使模型能够以固定时间步输出连续情感曲线[4-5]。然而，回归误差较低并不自动意味着模型已经解释了完整歌曲如何组织情绪。

完整流行歌曲由 Intro、Verse、Pre-chorus、Chorus、Bridge、Interlude 和 Outro 等段落构成。段落的功能不仅取决于局部声响，还取决于其在整首歌曲中的位置、前后关系和重复语境。Verse-Chorus 轮转可以建立和集中情绪，Pre-chorus 可以延长到达预期，Bridge 可以打破既有循环，Outro 可以在高潮之后卸力或保留余韵[8-11]。如果只计算全曲平均 VA，或只评价孤立片段，便难以捕捉这些由形式顺序产生的情感过程。

本文在既有连续 VA 建模工作的基础上，将研究问题从“哪个预测头的 RMSE 更低”推进到“完整歌曲如何在 section 序列中组织情感体验”。研究包含两个阶段。第一阶段训练并冻结 Valence 与 Arousal Teacher，为新音频生成连续 pseudo-VA；第二阶段将 pseudo-VA 与人工质检 section 对齐，分析段落位置、转换、重复和全曲进程。Teacher 是测量工具，section-aware 结构分析才是论文主线。

本文回答以下研究问题：

- **RQ1：** 冻结纯音频 Teacher 能否为新音频提供具有可报告一致性的连续 VA 估计？
- **RQ2：** Verse、Pre-chorus、Chorus、Bridge、Interlude、Intro 与 Outro 是否呈现不同的 VA 位置和转换特征？
- **RQ3：** 副歌入口是否普遍表现为情感抬升；若存在，主要由 Valence、Arousal 还是二维共同变化驱动？
- **RQ4：** 重复 section 的增强是段落类型特有现象，还是全曲后程累积的一部分？
- **RQ5：** 段落级 VA 变化如何进一步解释为蓄力、释放、偏离、回归与收束等情感过程及流行音乐形式功能？

本文的主要贡献如下：

1. 建立从冻结纯音频 VA Teacher、完整歌曲重叠窗口推理到 section 对齐的可复现链路。
2. 以歌曲而非 section 行作为统计独立单位，降低伪重复和同曲帧泄漏造成的乐观偏差。
3. 将 Chorus lift 拆分为多种二维释放过程，并区分能量高点、Valence 高点与最大结构转折。
4. 通过重复段落对照表明，后续 Chorus 增强主要属于全曲后程累积，而非副歌独有放大。
5. 提出“数据—情感—音乐功能”三层解释框架，在保留音乐学意义的同时限制从相关结果直接跳到作曲机制的推断。

---

## 2 相关工作

### 2.1 连续音乐情感识别

Russell 的情感环形模型以 Valence 和 Arousal 描述情绪空间[1]。Eerola 和 Vuoskoski 进一步比较了音乐情绪研究中的离散与维度模型，说明连续维度适合描述音乐中渐变和混合的情感状态[2]。DEAM 提供大规模音乐片段及 2 Hz 动态 VA 标注，已成为连续 MER 的常用基准[4]。PMEmo 则面向流行音乐，提供动态情感标注、音频特征和生理信号，为流行情境下的 MER 研究提供了补充[5]。

连续 VA 预测的评价不能只依赖单一 RMSE。RMSE 和 MAE 衡量绝对误差，Pearson 相关系数（PCC）衡量曲线共变趋势，一致性相关系数（CCC）同时考虑相关性、均值和方差偏差[12]。此外，动态音乐标注还受到听者反应延迟、标签平滑、歌曲间差异和时间对齐的影响。本文因此分别报告 Valence/Arousal 的 RMSE、MAE、PCC 与 CCC，并明确区分帧级和完整片段聚合结果。

### 2.2 音频表征与时序建模

OpenL3 通过视听自监督学习获得通用音频嵌入[6]，但其表征并非专为音乐结构和情感设计。MERT 使用大规模音乐自监督训练，并结合声学与音乐教师信号，在多个音乐理解任务中表现出较强的迁移能力[7]。本文早期实验曾比较 OpenL3、MERT、Log-Mel 和手工特征；最终 Teacher 采用 MERT 中间层表征与 OpenSMILE 动态特征融合，并使用 global-local BiGRU 分别建模总体情感位置和局部时间偏移。

### 2.3 流行音乐段落与形式功能

流行音乐形式研究强调，Verse、Pre-chorus 与 Chorus 的关系不能仅由标签名称决定，而应结合局部顺序、重复和目标指向理解。Nobile 将 Verse-Pre-chorus-Chorus 描述为 initiation-buildup-arrival 的目的性过程[8]；Temperley、de Clercq 等研究讨论了流行音乐段落、轮转形式及标签歧义[9-10]；Osborn 对终末高潮形式的研究说明，高潮可能被推迟到歌曲后部，且不必等同于常规副歌[11]。Harmonix Set 等资源进一步推动了功能段落标注研究[13]。

现有 MER 与流行音乐形式研究之间仍存在断层：前者擅长预测连续情感，后者擅长解释段落功能，但很少在完整歌曲尺度上将两者结合。本文不根据 section 标签预设情感功能，而是先计算 VA 变化，再讨论这些变化是否与已有形式理论相容。

---

## 3 数据与研究设计

### 3.1 总体设计

研究流程如图 1 所示，分为 Teacher 建模和 section-aware 分析两个阶段。

```text
阶段一：PMEmo 音频与动态 VA
    -> MERT + OpenSMILE 特征
    -> 独立 Valence/Arousal global-local BiGRU
    -> 冻结 Teacher 与最终测试

阶段二：完整中文流行歌曲 MP3
    -> 重叠窗口连续 pseudo-VA
    -> 人工质检 section 边界对齐
    -> section/transition/song 三层特征
    -> 数值证据—情感过程—音乐形式功能解释
```

**【投稿前待完善 P1-00：框架图】** 将上述文本流程绘制为正式图 1，标明训练数据、冻结节点、盲法节点、推理窗口和三层分析输出。

### 3.2 Teacher 数据

DEAM 用于早期模型开发与严格 song-level split 诊断。可用数据包含 1,802 首音频，按歌曲划分为 1,441/180/181 首训练、验证和测试。动态标注的前 15 s 被排除，后续标签以 0.5 s 为时间步，并统一转换到 `[-1,1]` 范围。

最终 Teacher 使用 PMEmo。项目中 767 个可用片段被预先划分为 652 个开发片段和 115 个最终测试片段。最终测试包含 5,390 个 0.5 s 帧。模型选择、损失权重、训练轮数与部署参数均在开发数据上确定；最终预测文件在读取测试标签前冻结并记录哈希。

### 3.3 完整歌曲语料

数据库审计得到 99 首已质检歌曲、1,097 个连续 section 和 99 个本地 MP3。所有歌曲均通过 section 顺序、边界连续性、音频时长和本地文件一致性检查。为限定本文对中文流行音乐的研究范围，`ANGEL` 和 `Easy On Me` 的推理产物被保留，但不进入主统计。最终主分析集包含：

- 97 首中文流行歌曲；
- 75 位歌手；
- 1,076 个 section；
- 48,804 个进入主分析的 0.5 s pseudo-VA 帧。

表 1 给出主要 section 覆盖。

| Section 类型 | Section 数 | 歌曲覆盖数 |
| --- | ---: | ---: |
| Verse | 262 | 97 |
| Chorus | 371 | 96 |
| Pre-chorus | 90 | 46 |
| Bridge | 35 | 32 |
| Interlude | 127 | 86 |
| Intro | 94 | 92 |
| Outro | 97 | 94 |

**【投稿前待完善 P0-01：样本设计】** 现有 97 首歌曲由研究者在项目推进中逐步收集，属于便利样本，不具备预冻结抽样框、分层配额和随机种子。正文所有总体性表述必须限定为“在本数据集中”。较强投稿版本建议新增独立的排名分层外部样本；若不扩样，应在标题、摘要、方法和局限性中始终保持探索性定位。

### 3.4 Section 标注与名称归一化

Section 边界和类型在进入分析前完成人工质检。边界要求第一段从 0 s 开始、相邻段落连续且无实质重叠、末段终点与解码音频时长一致。分析同时保留原始 section 名称和归一化类型，避免把清洗后的标签误作原始标注。相邻同名段仅在形式序列描述中折叠；VA 聚合和转换分析仍保留原始连续区间。

### 3.5 人工趋势验证集

从 99 首歌曲中按时长和 section 数量分层冻结 15 首，每位歌手最多一首，共 179 个 section。Teacher 在不知道 section 类型和人工答案的条件下完成推理与对齐。标注者听取独立裁剪的 section 音频，填写 Valence/Arousal 的 low-mid-high 等级、相对上一段的 up-down-stable 变化，以及局部峰值候选。页面不展示模型结果。

为降低缺少跨歌曲参照造成的等级漂移，标注者在第二轮使用 30 个与 Teacher 无关的参考片段重新校准。该设计只能评价序数趋势和同一标注者的复核稳定性，不能产生连续 VA ground truth。

**【投稿前待完善 P0-02：独立标注者】** 当前 15 首验证仅有一位标注者。同一标注者延迟复核不能替代 inter-rater reliability。投稿前应由至少一位独立标注者在不查看现有答案和模型输出的条件下完成同一冻结样本。

---

## 4 方法

### 4.1 独立 Valence/Arousal Teacher

最终 Teacher 使用相同骨干但分别训练。对长度为 `T` 的音频序列，每个时间步输入由 MERT-v1-95M hidden-6 的 768 维表征和 PMEmo 风格 OpenSMILE 的 260 维动态特征拼接而成。投影层将融合特征映射到 112 维，随后进入双向 GRU 局部分支与全局汇聚分支：

\[
\hat{y}_t = g(\mathbf{x}_{1:T}) + \lambda d_t(\mathbf{x}_{1:T}),
\]

其中，`g` 预测片段总体情感位置，`d_t` 预测围绕总体位置的局部偏移，`λ` 控制动态幅度。Valence 和 Arousal 分别使用 `λ=0.55` 和 `λ=0.70`，以适应两维度不同的局部可预测性。

共同参数为：BiGRU hidden size 64、双向、global hidden size 96、dropout 0.25、batch size 24、AdamW 学习率 `4×10^-4`、weight decay `5×10^-4`、gradient clipping 1.0。开发集内部另保留 15% song-level validation。Valence 训练 9 epoch，Arousal 训练 7 epoch。

损失函数由帧级均方误差、片段均值误差、CCC 约束、时间平滑和软教师约束组成：

\[
\mathcal{L}=\mathcal{L}_{MSE}+\alpha\mathcal{L}_{clip}
+\beta(1-CCC)+\gamma\mathcal{L}_{smooth}+\eta\mathcal{L}_{soft}.
\]

这些权重仅在开发数据上确定。最终测试标签未参与模型选择、线性校准、平滑窗口选择或集成权重搜索。

### 4.2 完整歌曲重叠窗口推理

由于 Teacher 在短片段上训练，直接输入数分钟序列可能造成长度分布外推。本文将完整歌曲划分为 30 s 窗口，相邻窗口间隔 15 s，并对重叠预测进行 Hann 加权：

\[
\hat{y}(t)=\frac{\sum_k w_k(t)\hat{y}_k(t)}{\sum_k w_k(t)}.
\]

该策略保留局部上下文并减少窗口接缝。最终输出的时间间隔为 0.5 s。99 首歌曲共生成 49,763 帧，所有歌曲通过有限值、范围、时间轴、曲线方差和哈希重构审计，未发现曲线塌缩。

### 4.3 Frame-to-section 对齐

设第 `j` 个 section 的边界为 `[s_j,e_j)`，时间中心为 `t_i` 的 pseudo-VA 帧仅在 `s_j≤t_i<e_j` 时分配到该 section；最后一段允许闭区间终点，以覆盖精确落在音频末端的帧。全部 49,763 帧均且仅被分配一次，1,097 个 section 均至少包含一个帧。

每个 section 提取 duration、V/A 的均值、标准差、极值、入口、出口、斜率、内部峰值、歌曲归一化位置，以及与前后 section 的二维差异。每首歌曲进一步提取动态范围、路径长度、最大转换、三类高点位置、重复段落变化、Intro-Outro 距离和 V/A 耦合等结构指标。

### 4.4 Section 转换与功能事件

对相邻 section `i→i+1`，定义：

\[
\Delta V_i=V_{i+1}-V_i,\qquad
\Delta A_i=A_{i+1}-A_i,
\]

\[
D_i=\sqrt{(\Delta V_i)^2+(\Delta A_i)^2}.
\]

主描述采用 `±0.05` 作为上升、下降和近似稳定的透明阈值，并以 `±0.03`、`±0.07` 做敏感性检查。该阈值用于描述局部方向，不代表心理学类别边界。

本文识别以下非互斥功能事件：

- Verse-Pre-chorus-Chorus 的分阶段蓄力、突然释放、高位延续与反向结果；
- Chorus 入口的积极释放、强度释放、延续、混合、反向降温、高唤醒负向释放和舒缓释放；
- Bridge 相对前序 Chorus 中心的偏离，以及后续 Chorus 的可能回归；
- Outro 的局部卸力、高位余韵、情绪回归或迁移式结束。

这些名称是由 VA 变化规则生成的情感过程描述，不是对歌曲进行互斥分类。

### 4.5 统计分析

歌曲是推断的独立单位。不同 section 类型的比较先在每首歌内聚合，再执行歌曲级配对分析，以避免把同一歌曲的多个 section 误当作独立样本。本文报告样本量、均值、中位数、bootstrap 95% 置信区间、正向覆盖比例和效应量；关键配对比较使用 Wilcoxon signed-rank test，并对同一分析族执行 Holm 校正。Bootstrap 以歌曲为重采样单位，共 5,000 次。

重复 Chorus 的增强采用 difference-in-differences 思路：先计算每首歌最后一次与第一次 Chorus 的变化，再减去同曲最后一次与第一次 Verse 的变化。该对照用于区分 Chorus 特有强化与一般后程累积。

本文不提出固定“情感弧线类型”。KMeans 与 Ward 聚类仅作为方法审计：不同 `k` 和子样本下的稳定性不足，因此不进入实质性音乐结论。

### 4.6 数据—情感—音乐功能解释框架

所有主要结论按三层推进：

1. **数据层：** 报告 VA 位置、差值、方向、距离、时间和覆盖比例。
2. **情感层：** 将数值变化描述为蓄力、释放、偏离、回归、累积或收束。
3. **音乐功能层：** 结合 section 顺序和重复语境，讨论建立、过渡、到达、轮转、对比和结束功能。

本文不从 VA 结果直接推断和声、调式、旋律、节奏、响度、音色、配器或歌词机制。

---

## 5 Teacher 实验与验证

### 5.1 模型选择过程

早期 OpenL3+MLP 在 frame-level random split 下曾获得 Test RMSE 0.1203、Pearson `r_V=0.878` 和 `r_A=0.909`；项目内同协议三随机种子复现甚至达到 RMSE 0.0984、Mean CCC 0.9206。然而，同一歌曲的不同帧可同时进入训练和测试，导致歌曲身份及制作特征泄漏。改用 song-level split 后，OpenL3+MLP 的 Test RMSE 上升至 0.2083。

MERT+BiLSTM 在 1,802 首 DEAM 的 song-level split 上获得 Test RMSE 0.2025、Mean CCC 0.5383。后续 global-local、特征融合和分维训练表明，Arousal 明显比 Valence 更容易由纯音频动态预测，且歌曲间总体位置比歌曲内细微走势更稳定。因此最终系统使用 PMEmo 和两个独立 Teacher，而不再强迫 V/A 共享同一输出头。

上述历史实验用于说明协议修正和模型主线调整，不与最终 PMEmo 指标作不加条件的横向排名。

### 5.2 最终 Teacher 性能

表 2 给出 115 个冻结 PMEmo 最终测试片段的结果。

| 评价尺度 | 维度 | RMSE | MAE | PCC | CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| 完整片段 | Valence | 0.1863 | 0.1529 | 0.8029 | 0.8002 |
| 完整片段 | Arousal | 0.1886 | 0.1469 | 0.8574 | 0.8548 |
| 0.5 s 帧级 | Valence | 0.1944 | 0.1584 | 0.8106 | 0.8016 |
| 0.5 s 帧级 | Arousal | 0.2004 | 0.1575 | 0.8487 | 0.8392 |

完整片段均值基线的 Valence/Arousal RMSE 分别为 0.2978 和 0.3543，说明 Teacher 明显优于常数预测。PCC 与 CCC 表明模型能够捕捉较稳定的片段间排序和共变趋势，但 RMSE 仍显示连续数值存在不可忽略的误差。本文因此将完整歌曲输出称为 pseudo-VA，而不视为真实人类连续标注。

开发集与最终测试 ID 无交集，但审计发现 3 组跨集合 MP3 内容完全相同。剔除重复内容后的 112 首敏感性结果变化较小；由于该分析发生在正式测试后，本文仍报告原 115 首结果，并将内容重复列为效度威胁。

### 5.3 人工趋势验证

同一标注者在 30 个参考片段校准后的复核中，Valence、Arousal、相邻变化和局部峰值的一致率分别为 87.15%、88.83%、100% 和 99.44%，quadratic weighted kappa 分别为 0.858、0.877、1.000 和 0.987。该结果只反映同一标注者的延迟稳定性。

Teacher 与人工序数等级的结果如下：

| 指标 | Valence | Arousal |
| --- | ---: | ---: |
| Pooled section Spearman | 0.614 | 0.576 |
| Within-song macro Spearman | 0.437 | 0.525 |
| Within-song centered Spearman | 0.509 | 0.562 |
| Between-song mean Spearman | 0.742 | 0.614 |

在人工标注为非稳定的相邻变化中，Teacher 的方向一致率为 `88/129=68.2%`。模型最高 Arousal section 与人工局部峰值精确命中 `8/15`，允许相邻 section 时命中 `12/15`。人工 Chorus 相对 Verse 的 V/A 提升均在 15 首中的 13 首为正；模型与人工 Chorus lift 的歌曲间 Spearman 在 Valence/Arousal 上分别为 0.228 和 0.586。

这些结果对 Arousal 主导的结构趋势提供中等支持，对 Valence 段落差异的支持较弱。由于人工标签是序数等级，本文不计算人工 RMSE、MAE 或 CCC。

---

## 6 Section-Aware 分析结果

### 6.1 Section 的 VA 分布

![不同 section 类型的 VA 分布](../../research/phase2_section_aware_va/analysis/full_corpus_97/figures/01_section_type_va_distributions.svg)

**图 2.** 不同 section 类型的 pseudo-Valence 与 pseudo-Arousal 分布。图用于描述分布，统计推断采用歌曲级聚合。

Chorus 相对 Verse 的歌曲级平均 Valence 差为 `+0.101`，95% bootstrap CI 为 `[0.083,0.119]`，90.6% 的可比较歌曲为正；Arousal 差为 `+0.266`，95% CI 为 `[0.237,0.297]`，96/96 首均为正。直接 Verse→Chorus 转换的 Valence/Arousal 平均变化为 `+0.054/+0.139`，说明副歌差异并非完全由歌曲内 section 采样数量造成。

数值上，Chorus 与 Verse 的最稳定区别是 Arousal，而非统一的 Valence 正向化。情感上，这更接近能量提高、激活增强和情绪集中。音乐功能上，Chorus 在本样本中经常承担到达、放大或公开化已有情绪的作用，而不能简单等同于“更快乐”。

### 6.2 Verse-Pre-chorus-Chorus 的蓄力与到达

72 个严格相邻 Verse-Pre-chorus-Chorus 事件中，47 个表现为 Verse→Pre-chorus 与 Pre-chorus→Chorus 的连续 Arousal 上升，15 个在 Pre-chorus 阶段近似稳定而主要升幅发生在 Chorus 入口，4 个在 Pre-chorus 已达到较高位置并在 Chorus 维持，5 个为弱或混合方向，1 个在 Pre-chorus 上升后进入 Chorus 回落。

![蓄力与副歌释放功能计数](../../research/phase2_section_aware_va/analysis/full_corpus_97/figures/15_buildup_and_release_function_counts.svg)

**图 3.** Verse-Pre-chorus-Chorus 蓄力过程与 Chorus 入口二维释放方式。

该结果与 initiation-buildup-arrival 理论相容，但并非对理论标签的机械验证。数据层只显示 Arousal 的分阶段变化；情感层可以描述为蓄力与到达；音乐功能层才进一步解释为 Verse 建立、Pre-chorus 延长目标指向、Chorus 实现到达。当前数据不能确定这些结果是由和声终止、织体增厚、旋律上行还是制作响度造成。

### 6.3 Chorus 入口的多维释放

368 次 Chorus 入口被描述为：积极释放 111 次、Arousal 主导的强度释放 89 次、延续型 71 次、混合或 Valence 主导 50 次、反向降温 30 次、高唤醒负向释放 14 次、舒缓释放 3 次。

这说明“副歌抬升”不是单一现象。Valence 与 Arousal 同时上升时，副歌可能表现为积极释放；Valence 近似稳定而 Arousal 上升时，更接近强度集中；Arousal 上升而 Valence 下降时，可能是偏痛苦、紧张或冲突性的高强度释放。反向和延续型事件构成必要反例，防止把副歌功能写成普遍定律。

### 6.4 重复 section 与后程累积

第一次到最后一次 Chorus 的平均 Valence/Arousal 变化为 `+0.087/+0.209`。若只看这一结果，容易认为 Final Chorus 存在独有强化。为控制全曲时间位置，在 89 首同时具有重复 Verse 和 Chorus 的歌曲中，本文计算 Chorus 变化减 Verse 变化。difference-in-differences 结果为 Valence `-0.025`、Arousal `+0.015`，Holm 校正后 `p=0.437/0.934`。

因此，后续 Chorus 的确常比首次 Chorus 更强，但其增强没有超过同期 Verse 的一般变化。数据支持“歌曲后半程普遍累积”，不支持“副歌具有独占的重复强化机制”。从音乐功能看，重复 section 同时维持段落身份并在新语境中获得不同表达重量，可能承担确认、强化、重释或耗散，而不是简单复制。

### 6.5 Interlude 与 Bridge 的上下文功能

Interlude 覆盖 86/97 首。Chorus→Interlude 出现在 84 首，Interlude→Verse 出现在 73 首；进入 Interlude 的平均 Arousal 变化仅为 `-0.005`，随后 Interlude→Verse 的平均变化为 `-0.041`。这提示 Interlude 经常不立即清空副歌能量，而是作为轮转铰链重新分配能量并导向下一 Verse。

Bridge 共 35 个，来自 32 首歌曲。31 个事件同时具有可比较的前序 Chorus 中心和后续 Chorus，其中 16 个后续 Chorus 比 Bridge 更接近此前 Chorus 中心。Bridge 的绝对 VA 方向并不统一，其更稳定的意义是相对既有循环的上下文偏离，以及在部分歌曲中的偏离—回归。

![常见段落转换的情感功能](../../research/phase2_section_aware_va/analysis/full_corpus_97/figures/13_musicological_transition_functions.svg)

**图 4.** 常见 section 转换及其可能的情感过程与形式功能。

### 6.6 高点、转折与 Outro 收束

最大平均 Arousal section 位于 Chorus 的歌曲为 53/97；全部歌曲中，54.6% 的 Arousal 高点位于最后四分之一；在 53 首 Chorus 高点歌曲中，62.3% 的高点是最后一次 Chorus。然而，最大 VA 转换最常落到 Outro，达到 36/97 首。这说明“最高能量位置”和“最大结构转折”是不同概念。

Chorus→Outro 的平均 Valence/Arousal 变化为 `-0.112/-0.183`。在具有可比 Intro 和 Outro 的歌曲中，只有 21.1% 的 Outro 回到距 Intro 不超过 0.10 的 VA 区域；79.4% 的歌曲结束 Arousal 仍高于开始。Outro 因而常承担局部卸力或收束，但形式闭合不要求情绪回到开场原点。结尾还可以表现为高位余韵、部分回归或向新情绪位置迁移。

### 6.7 整首歌曲的连续组织

![整首歌曲平均情感进程与形式语法](../../research/phase2_section_aware_va/analysis/full_corpus_97/figures/14_formal_grammar_and_average_arc.svg)

**图 5.** 归一化歌曲位置上的平均 pseudo-VA 进程及常见局部形式语法。平均曲线不代表任何单曲模板。

97 首歌曲的 VA 路径直接度均值为 0.220、中位数为 0.180，说明多数歌曲不是从起点直接移动到终点，而经历多轮起伏。74.2% 的歌曲中，Arousal range 至少为 Valence range 的 1.25 倍；只有 11.3% 的歌曲将一半以上 VA 路径集中在两个最大转换。这些结果表明，本样本中的情感组织通常由多次段落轮转构成，并主要依靠 Arousal 调节强弱，而非持续改变情绪正负方向。

歌曲内部 section 的 V/A Spearman 中位数为 0.736，但 5/97 首为负，说明两个维度多数同向却并非必然同步。23.6% 的相邻 section 跨越以零点划分的 VA 区域，但负 Valence、高 Arousal 区域仅有 15/1,076 个 section，表明象限结果对 Teacher 零点校准高度敏感。本文因此以连续差值为主要证据，象限只作辅助描述。

---

## 7 讨论：从数值变化到流行音乐功能

### 7.1 Chorus 更稳定地集中强度，而不是统一改变情绪极性

Chorus 的 Arousal lift 明显大于 Valence lift，且人工验证对 Arousal 变化的支持强于 Valence。这一结果提示，流行歌曲副歌的共同作用更接近提高表达强度、集中主题和形成到达，而不是把所有情绪推向积极。悲伤、冲突或紧张内容同样可以通过高 Arousal 获得副歌释放。该解释能同时容纳积极释放、强度释放和高唤醒负向释放，也能解释为何单一“副歌更快乐”的假设过于狭窄。

### 7.2 Pre-chorus 组织“如何到达”，而不仅是位于 Verse 与 Chorus 之间

严格三段序列中，多数歌曲分两步提高 Arousal，但另一些歌曲把主要升幅保留到副歌入口，或提前在 Pre-chorus 达到高位。Pre-chorus 的功能因此体现在到达路径的时间安排：它可以逐级蓄力、延迟释放，也可以制造高位悬置。该结果支持目的性形式解释，但不能证明具体音乐材料采用了哪种张力机制。

### 7.3 重复的核心是可识别性与语境变化并存

重复 Chorus 的绝对增强容易被误读为副歌特有机制。Verse 对照显示，后程增强是多类段落共享的时间效应。音乐上，这意味着重复 section 的身份可以保持稳定，而其情感重量随演唱、制作、前序材料和全曲位置改变。当前 VA 数据能够确认情感结果发生变化，却不能区分变化来自配器增加、音区提高、动态压缩或演唱方式。

### 7.4 Bridge、Interlude 与 Outro 应由上下文而非固定高低定义

Bridge 没有统一 VA 方向，却可以通过偏离 Verse/Chorus 中心改变全曲关系；Interlude 常处于副歌后、下一 Verse 前，承担能量重新分配；Outro 即使降低 Arousal，也不必回到 Intro。三类段落的共同点是，其功能更依赖“从哪里来、到哪里去”，而不是孤立均值。这也说明 section-aware 分析相对于单片段 MER 的必要性。

### 7.5 不建立固定情感弧线类型的原因

本文不把 97 首歌曲硬分为若干固定“情感弧线类型”。这不是声称情感弧线不存在，而是因为当前聚类审计的最佳 silhouette 为 0.152、子样本稳定性 ARI 为 0.424，未达到预设接受条件。较稳妥的表示是保留连续指标，例如后程斜率、Arousal 主导比、路径集中度、重复变化和收束量。未来在更大、抽样更规范的语料上可重新检验是否存在稳定类别。

### 7.6 对 MER 研究的启示

Teacher 的片段级相关性和一致性达到可用水平，但完整歌曲分析最有价值的输出并非单个 VA 数值，而是相邻段落的相对方向、重复差异和时间位置。对于 section-aware 应用，模型评价应同时关注：

- 绝对误差是否可接受；
- 曲线趋势是否与人工感知一致；
- 段落间方向能否稳定复现；
- 结果是否在不同歌曲、标注者与特征提取域中保持稳健。

---

## 8 有效性威胁与局限性

### 8.1 Teacher 的测试边界

最终 Teacher 的正式指标来自 PMEmo 短标注片段，其标注有效跨度中位数约 20.5 s，而第二阶段对象是多分钟完整歌曲。30/15 s 重叠窗口缓解了序列长度外推，却不能替代完整歌曲连续人工 ground truth。

**【投稿前待完善 P0-03：完整歌曲外部效度】** 当前没有新的完整歌曲连续 VA 标注可用于独立 RMSE/CCC 评价。若无法新增连续标注，应将 15 首 section 趋势验证定位为补充证据，并在结论中明确避免“任意完整歌曲达到相同精度”的表述。

### 8.2 特征提取域偏移

PMEmo 正式测试使用其发布的 OpenSMILE 动态特征；新 MP3 使用本地流程重提取。抽查特征的相关性约为 0.773-0.881，但并非 bit-equivalent，尤其可能影响 Arousal 数值。

**【投稿前待完善 P0-04：特征域审计】** 最理想做法是恢复与 PMEmo 官方配置完全一致的 OpenSMILE 提取链，并对官方/本地特征及预测差异进行系统等价性分析。若无法完全复现，应把第二阶段描述为部署域适配后的 pseudo-VA 分析。

### 8.3 样本代表性

97 首歌曲不是概率抽样，无法代表全部中文流行歌曲。样本包含 75 位歌手且单一歌手占比不高，可以降低个别歌手支配，但不能解决年代、风格、平台、制作传统和选择偏差。

### 8.4 人工验证限制

15 首歌曲只有一位标注者，且 low-mid-high 是序数等级。参考片段改善了同一标注者的标尺稳定性，但不能证明群体感知一致。人工结果还可能受到连续聆听疲劳、section 边界上下文不足和局部静音的影响。

### 8.5 Pseudo-label 与解释边界

第二阶段所有 VA 都是 Teacher 估计值。多个均值差异可能部分继承 Teacher 对 section 声学特征的系统偏置。特别是 96/96 首 Chorus Arousal 高于 Verse 的结果过于整齐，必须与人工反例和模型偏差同时讨论。

### 8.6 音乐机制不可识别

本文拥有 section、时间和 pseudo-VA，但没有系统测量和声、调式、旋律、节奏、响度、音色、配器、制作动态或歌词。因此可以讨论“情感集中”“蓄力”“偏离”“收束”等宏观功能，不能断言这些功能由某一具体作曲或制作手段造成。

**【投稿前待完善 P1-01：典型与反例听审】** 建议按预冻结数值规则选择连续蓄力、反向副歌、Bridge 偏离-回归、强 Outro 卸力和 V/A 分离等典型与反例，各进行模型盲听觉复核。必要时只补充可直接测量的 RMS、onset、tempo、spectral 或 chroma 描述。

### 8.7 多重分析与探索性定位

尽管主要比较使用 Holm 校正和 bootstrap，研究问题仍由同一语料反复探索。本文应定位为探索性研究；未来外部样本应预注册主要假设、效应方向和排除规则，用于确认性复现。

---

## 9 结论

本文提出并实现了一条从连续 VA Teacher 到完整歌曲 section-aware 情感结构分析的可复现路径。两个独立 global-local BiGRU Teacher 在冻结 PMEmo 测试中取得约 0.80 以上的 PCC/CCC，并被用于生成 0.5 s 分辨率的完整歌曲 pseudo-VA。通过与人工质检 section 对齐，本文不再将歌曲视为同质音频，而是将其表示为具有顺序、重复和上下文的情感过程。

在 97 首中文流行歌曲便利样本中，Chorus 相对 Verse 的差异主要由 Arousal 驱动；Pre-chorus 常组织分阶段蓄力或延迟释放；Chorus 入口包含多种二维释放方式；后续 Chorus 增强更多反映全曲后程累积，而非副歌独有机制；Interlude、Bridge 和 Outro 的意义则主要来自其在轮转、偏离-回归与收束中的上下文位置。完整歌曲的情感推进通常由多次建立、集中、回落和再启动构成，且 Arousal 比 Valence 更常承担段落强弱组织。

本文的核心方法论结论是：连续 VA 数值不能直接等同于音乐功能。只有依次经过数值证据、情感过程和形式语境，才能形成可复核且不过度解释的音乐学结论。当前结果足以支持 section-aware 情感组织的探索性框架，但样本设计、独立标注者、完整歌曲 ground truth 和特征域一致性仍决定最终论文能够主张到何种强度。

---

## 数据与代码可用性声明

分析代码、冻结配置、哈希清单、中间表格和可视化保存在本项目的可复现目录中。受音乐版权和数据库访问权限限制，完整歌曲 MP3 不随论文公开。投稿时应提供去标识化的 section 特征表、统计脚本和不含受版权音频的复现说明。

**【投稿前待完善】** 补充正式匿名仓库地址、版本标签、许可证和数据申请方式。

## 伦理声明

本文分析研究者合法访问的音乐音频及其计算特征，不涉及可识别个人敏感信息。人工验证的参与人数、知情同意和机构伦理审查适用性应按实际情况补充。

**【投稿前待完善】** 填写人工标注者的知情同意、补偿、数据保存方式及伦理审查结论。

## 利益冲突声明

作者声明不存在利益冲突。【投稿前由作者确认】

## 资助声明

【投稿前补充资助项目；如无资助，写“本研究未获得专项资助”。】

## 作者贡献声明

【投稿前按 CRediT taxonomy 补充 Conceptualization、Methodology、Software、Validation、Formal analysis、Writing 等贡献。】

---

## 参考文献

[1] Russell J A. A circumplex model of affect[J]. Journal of Personality and Social Psychology, 1980, 39(6): 1161-1178. https://doi.org/10.1037/h0077714

[2] Eerola T, Vuoskoski J K. A comparison of the discrete and dimensional models of emotion in music[J]. Psychology of Music, 2011, 39(1): 18-49. https://doi.org/10.1177/0305735610362821

[3] Yang Y H, Chen H H. Machine recognition of music emotion: A review[J]. ACM Transactions on Intelligent Systems and Technology, 2012, 3(3): 40. https://doi.org/10.1145/2168752.2168754

[4] Aljanaki A, Yang Y H, Soleymani M. Developing a benchmark for emotional analysis of music[J]. PLOS ONE, 2017, 12(3): e0173392. https://doi.org/10.1371/journal.pone.0173392

[5] Zhang K, Zhang H, Li S, et al. The PMEmo dataset for music emotion recognition[C]//Proceedings of the 2018 ACM International Conference on Multimedia Retrieval. 2018: 135-142. https://doi.org/10.1145/3206025.3206037

[6] Cramer J, Wu H H, Salamon J, et al. Look, listen, and learn more: Design choices for deep audio embeddings[C]//2019 IEEE International Conference on Acoustics, Speech and Signal Processing. 2019: 3852-3856. https://doi.org/10.1109/ICASSP.2019.8682475

[7] Li Y, Yuan R, Zhang G, et al. MERT: Acoustic music understanding model with large-scale self-supervised training[C]//International Conference on Learning Representations. 2024. https://arxiv.org/abs/2306.00107

[8] Nobile D. Teleology in verse-prechorus-chorus form, 1965-2020[J]. Music Theory Online, 2022, 28(3). https://www.mtosmt.org/issues/mto.22.28.3/mto.22.28.3.nobile.html

[9] Temperley D. Form[M]//The Musical Language of Rock. Oxford: Oxford University Press, 2018. https://doi.org/10.1093/oso/9780190653774.003.0008

[10] de Clercq T. Embracing ambiguity in the analysis of form in pop/rock music[J]. Music Theory Online, 2017, 23(3). https://www.mtosmt.org/issues/mto.17.23.3/mto.17.23.3.de_clercq.php

[11] Osborn B. Subverting the verse-chorus paradigm: Terminally climactic forms in recent rock music[J]. Music Theory Spectrum, 2013, 35(1): 23-47. https://doi.org/10.1525/mts.2013.35.1.23

[12] Lin L I K. A concordance correlation coefficient to evaluate reproducibility[J]. Biometrics, 1989, 45(1): 255-268. https://doi.org/10.2307/2532051

[13] Nieto O, McCallum M C, Davies M E P, et al. The Harmonix Set: Beats, downbeats, and functional segment annotations of western popular music[C]//Proceedings of the 20th International Society for Music Information Retrieval Conference. 2019: 565-572. https://archives.ismir.net/ismir2019/paper/000068.pdf

---

## 附录 A 投稿前待完善清单

| 编号 | 优先级 | 待完善事项 | 最低可接受处理 | 更强处理 |
| --- | --- | --- | --- | --- |
| P0-01 | P0 | 97 首便利样本 | 全文限制为探索性便利样本 | 新增预冻结排名分层外部样本 |
| P0-02 | P0 | 单标注者 | 明确只报告同标注者稳定性 | 增加独立标注者并报告 inter-rater kappa |
| P0-03 | P0 | 无完整歌曲连续 VA ground truth | 不宣称完整歌曲同精度 | 新建小规模完整歌曲连续 VA 验证集 |
| P0-04 | P0 | OpenSMILE 特征域不完全等价 | 披露部署域偏移 | 复现官方配置并做等价性审计 |
| P1-00 | P1 | 缺正式框架图 | 使用文本流程 | 绘制投稿级系统框架图 |
| P1-01 | P1 | 缺典型与反例听审 | 只保留宏观功能解释 | 冻结个案并补盲听及可测声学特征 |
| P1-02 | P1 | 匿名复现仓库未建 | 提供本地产物索引 | 建匿名仓库、版本标签和数据字典 |

## 附录 B 主要可复现产物

```text
Teacher 审计：
research/phase1f_external_va_datasets/pmemo/final_clean_teacher/
research/phase1f_external_va_datasets/pmemo/final_clean_teacher/FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md

完整歌曲推理与样本审计：
research/phase2_section_aware_va/docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md

人工验证：
research/phase2_section_aware_va/docs/BLIND_VALIDATION_HUMAN_RESULTS_30REF_REVIEW_20260726.md

97 首分析：
research/phase2_section_aware_va/analysis/full_corpus_97/
research/phase2_section_aware_va/analysis/full_corpus_97/reports/EXPLORATORY_ANALYSIS_REPORT_ZH.md
research/phase2_section_aware_va/analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md
research/phase2_section_aware_va/analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md
```
