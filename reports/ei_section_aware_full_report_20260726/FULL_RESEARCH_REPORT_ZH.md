# 基于冻结 VA Teacher 的 Section-Aware 中文流行歌曲情感组织研究报告

**报告版本：** v1.0（科研全链路整合稿）
**报告时间：** 2026-07-26
**当前进度：** 第一阶段 VA Teacher 已冻结并完成一次最终测试；第二阶段已完成 99 首完整歌曲推理、97 首中文歌曲主分析、15 首单标注者趋势验证，以及数据—情感—音乐功能解释。
**报告定位：** 用于 EI 论文设计、方法复核、结果撰写和后续补强，不等同于已完成投稿稿件。

---

## 完成状态说明

本文使用以下标记，避免把尚未完成的部分写成既定事实：

- **【已完成】**：已有冻结数据、代码、模型、指标或报告可复核。
- **【待完善-P0】**：对论文可信度影响较大，正式投稿前应优先完成。
- **【待完善-P1】**：不阻断初稿，但会限制结论强度。
- **【后续扩展】**：不属于当前最短投稿路径。

最重要的未完成事项不是继续增加统计检验，而是：

1. **【待完善-P0-01：样本设计】** 现有 97 首歌曲是研究者逐步收集形成的便利样本，没有预先冻结的抽样框、随机种子、分层规则或代表性设计。
2. **【待完善-P0-02：独立标注者】** 15 首 section 人工验证只有一位标注者；同一标注者的延迟复核不能替代 inter-rater reliability。
3. **【待完善-P0-03：完整歌曲外部效度】** Teacher 的正式指标来自 PMEmo 短标注片段，尚无多分钟完整歌曲连续 VA ground truth 测试。
4. **【待完善-P0-04：原始 MP3 特征域】** 任意 MP3 重提取的 OpenSMILE 特征与 PMEmo 官方特征高度相关但不完全等价。
5. **【待完善-P1-01：音乐机制验证】** 当前只有 section、时间和 pseudo-VA，不能确定情绪变化由响度、节奏、音色、配器、旋律或和声中的哪一项造成。

---

## 摘要

本研究面向完整中文流行歌曲的情感结构分析，提出一条由“连续 VA 估计—section 对齐—情感过程识别—流行音乐形式功能解释”组成的 section-aware 研究路径。第一阶段训练两个相互独立的纯音频 Valence 和 Arousal Teacher。最终模型以 MERT-v1-95M hidden-6 表征和 PMEmo OpenSMILE 动态特征为输入，采用 global-local BiGRU，将歌曲全局情感位置与局部时间变化分开建模。模型在 652 个 PMEmo 开发片段上训练，并在冻结预测后一次性读取 115 个最终测试片段标签。完整片段尺度上，Valence 的 RMSE/MAE/PCC/CCC 为 `0.1863/0.1529/0.8029/0.8002`，Arousal 为 `0.1886/0.1469/0.8574/0.8548`。

第二阶段将冻结 Teacher 以 30 秒窗口、15 秒步长和 Hann overlap-add 部署到 99 首质检歌曲，输出 49,763 个 0.5 秒 pseudo-VA 帧，并与 1,097 个连续 section 完整对齐。排除两首外语歌曲后，中文主分析集包含 97 首、75 位歌手、1,076 个 section。研究以歌曲为统计独立单位，同时构建 section 功能事件、二维 VA 转换、重复段落变化和整首歌曲连续组织策略。

主要结果显示：Chorus 相对 Verse 的差异以 Arousal 为主，歌曲级平均 Valence/Arousal 差为 `+0.101/+0.266`；72 个严格 Verse→Pre-chorus→Chorus 事件中，47 个呈现分阶段 Arousal 蓄积；368 次 Chorus 入口包含积极释放、强度释放、延续、负向高唤醒释放和反向降温等不同过程；后续 Chorus 虽然通常增强，但其增幅不显著超过后续 Verse，因而更适合解释为全曲后程累积，而非副歌独有强化；Interlude 常承担轮转再导向，Bridge 的意义主要在于上下文偏离与可能回归，Outro 则表现为局部卸力、余韵或迁移式结束。

本研究的核心解释原则是：**数据只提供 VA 位置和变化证据；情感层描述蓄力、释放、偏离、回归、累积和收束；音乐功能层再讨论 Verse、Pre-chorus、Chorus、Bridge、Interlude 与 Outro 在全曲形式进程中的可能作用。** 三层之间逐层推导，不直接从统计差异跳到作曲机制。

由于 97 首歌曲属于便利样本、完整歌曲 VA 为 pseudo-label、人工验证只有一位标注者，当前结论应表述为“本样本中的探索性结构倾向”，不能写成中文流行歌曲的普遍规律。

**关键词：** 音乐情感识别；Valence-Arousal；MERT；Section-aware；中文流行歌曲；情感组织；形式功能；pseudo-VA

---

## 1. 研究背景与问题提出

### 1.1 从连续 VA 预测到完整歌曲情感结构

Valence-Arousal 二维模型分别描述情绪正负倾向与唤醒强度。传统音乐情感识别研究通常关注音频片段的 VA 回归误差，但完整歌曲并不是同质片段的简单延长。流行歌曲通过 Verse、Pre-chorus、Chorus、Bridge、Interlude、Intro 和 Outro 等功能段落安排重复、对比、积累、释放和收束。

上一篇已发表研究使用 OpenL3 表征和 DEAM 数据训练 MLP、BiLSTM、Transformer，其中 OpenL3+MLP 在原 frame-level random split 协议中取得较高指标。本项目不修改上一篇论文，而是从其连续 VA 能力继续推进，同时修正“同一歌曲不同帧进入不同集合”造成的泛化评估偏乐观问题。

本研究的第二阶段不把“再训练一个 section 模型提高 VA 准确率”作为主线，而把 VA Teacher 作为测量工具，研究完整歌曲如何在 section 序列中组织情感体验。

### 1.2 核心研究问题

1. 能否训练一个对新音频可执行、以 0.5 秒输出连续 VA 的纯音频 Teacher？
2. Teacher 的性能在严格冻结测试中达到什么水平，哪些指标仍未达到项目目标？
3. 完整歌曲中的 Verse、Pre-chorus、Chorus、Bridge、Interlude、Intro 和 Outro 是否呈现不同的 pseudo-VA 位置与转换特征？
4. 这些数据变化可能对应怎样的情感过程，例如建立、蓄积、释放、偏离、回归和收束？
5. 情感过程如何进一步联系到流行歌曲的段落形式功能？
6. 哪些解释是数据直接支持的，哪些只是可能的音乐学解释？

### 1.3 核心分析框架：数据—情感—音乐功能

本报告规定所有主要结论都经过三层转换：

| 层级 | 回答的问题 | 可使用证据 | 禁止越界 |
| --- | --- | --- | --- |
| 数据层 | 数值发生了什么变化？ | V/A 均值、差值、范围、斜率、距离、时间位置、覆盖比例 | 不能直接命名为具体听觉体验 |
| 情感层 | 这些变化可能形成什么情感过程？ | Arousal 蓄积、Valence 转向、二维移动、偏离/回归、释放/残留 | 不能直接指定和声、配器或节奏原因 |
| 音乐功能层 | 该过程可能服务于什么段落功能？ | section 类型、相邻顺序、重复位置、全曲语境 | 不能把相关关系写成作曲因果 |

例如：

> 数据层：Pre-chorus 相对 Verse 的 Arousal 增加，Chorus 再次增加。
> 情感层：形成分阶段的能量蓄积与到达。
> 音乐功能层：在部分歌曲中，Verse—Pre-chorus—Chorus 可能承担建立—蓄力—到达功能。

而不能直接写成：

> “Pre-chorus 因属和弦和配器叠加产生张力。”

因为当前数据没有和声、调式、响度、配器或音色变量。

---

## 2. 整体研究设计

### 2.1 两阶段结构

**第一阶段：VA Teacher 建模**

- 输入：纯音频特征。
- 输出：每 0.5 秒一个 Valence 或 Arousal 值。
- 目标：获得可冻结、可审计、可迁移到新 MP3 的 VA 测量工具。

**第二阶段：Section-aware 完整歌曲分析**

- 输入：完整歌曲音频、冻结 Teacher、质检 section 边界与类型。
- 输出：frame-level pseudo-VA、section-level 特征、song-level 情感组织指标。
- 目标：分析情绪推进、段落对比、重复变化、高点位置和收束方式。

### 2.2 研究对象边界

- 当前论文只研究音乐音频，不研究歌词。
- 当前不再进行 human/AIGC 比较。
- DEAM 和 PMEmo 只用于第一阶段 Teacher 研发与评价。
- 97 首中文流行歌曲用于第二阶段结构探索，不作为 Teacher 训练标签。
- 第二阶段 VA 统一称为 **pseudo-VA**，不称为 ground truth。

---

## 3. 第一阶段：VA Teacher 的探索与主线调整

### 3.1 OpenL3+DEAM：高指标与评估泄漏问题

已发表研究的 OpenL3+MLP 在原协议中报告 Test RMSE `0.1203`、Pearson r(V) `0.878`、r(A) `0.909`。项目内三种子 OpenL3+MLP 复现实验在同一 frame-level random split 下达到 Test RMSE `0.0984`、Mean CCC `0.9206`。

但是，该划分按帧随机抽取，因此同一歌曲的不同时间片可能同时进入训练、验证和测试。模型可能借助歌曲音色、制作和整体风格的相似性识别同一歌曲，不能将该指标解释为对全新歌曲的严格泛化。

同一 OpenL3+MLP 在 song-level split 下，三种子集成 Test RMSE 上升至 `0.2083`。这一差异促使项目把 song-level split 设为后续基本规则。

**研究意义：** 这不是上一篇论文“无效”，而是说明原结果回答的是原 frame-level 协议下的性能；当前研究问题要求更严格的跨歌曲泛化，因此必须使用新的评价协议。

### 3.2 MERT+DEAM：音乐预训练表征与严格跨歌曲基线

项目使用 MERT-v1-95M 替代 OpenL3 作为主音乐表征。MERT 是面向音乐理解任务的自监督声学模型，其预训练同时考虑声学与音乐音高/调性信息，适合用作 MIR 音频表征。

DEAM 共审计 1,802 首可用音频与动态标签，标签从 15 秒附近开始，以 0.5 秒为时间步。数据按歌曲划分为 1,441/180/181 首 train/validation/test。MLP、BiLSTM、Transformer 均被保留比较，验证集 RMSE 最优的是 MERT+BiLSTM baseline：

| 指标 | 结果 |
| --- | ---: |
| Validation RMSE | 0.219934 |
| Test RMSE | 0.202452 |
| Test Mean CCC | 0.538263 |
| Test Pearson V | 0.511522 |
| Test Pearson A | 0.641069 |

MERT+BiLSTM 比同一 song split 的 OpenL3+MLP RMSE 更低，但 Valence 和动态一致性仍不足。

### 3.3 从联合 V/A 到全局—局部与分维建模

进一步实验发现：

- Arousal 通常比 Valence 更容易从纯音频动态中预测。
- 歌曲间平均情感预测明显强于歌曲内部细微变化。
- 仅在预测头后继续加深网络，不能稳定解决 Valence 泛化问题。

因此项目逐步引入：MERT 中间层、OpenSMILE 动态特征、global-local 分解、CCC 与 smoothness 约束、分维损失和 Valence/Arousal 独立模型。

DEAM 五折 global-local temporal Valence 模型达到 Frame RMSE `0.1854`；cross-fitted OOF stack 达到 RMSE `0.1830`、MAE `0.1399`，但 PCC/CCC 仍为 `0.6738/0.6318`，within-song PCC 仅 `0.1335`。这说明降低总体误差不等于已经稳定捕捉歌曲内部走向。

### 3.4 引入 PMEmo 的原因

PMEmo 面向流行音乐情感识别，公开版本包含 794 首歌曲及情感注释、预提取音频特征和生理信号。本项目最终可用协议包含 767 个片段，其中 652 个作为开发集，115 个冻结为最终测试。

选择 PMEmo 的主要原因是：

1. 任务领域更接近流行音乐；
2. 提供动态情感标注与官方音频特征；
3. 适合把 MERT 高层音乐表征与 OpenSMILE 局部动态描述融合；
4. 可为 Valence 和 Arousal 分别设计损失与局部变化尺度。

但 PMEmo 的标注有效跨度中位数仍约 20.5 秒，因此它改善了领域接近性，却没有直接解决“多分钟完整歌曲 ground truth”缺失。

---

## 4. 最终冻结 VA Teacher

### 4.1 输入与结构

最终使用两个独立、纯音频 global-local BiGRU：

```text
Valence Teacher = pmemo_final_clean_valence_teacher_v1
Arousal Teacher = pmemo_final_clean_arousal_teacher_v1
MERT-v1-95M hidden-6 = 768 维
PMEmo OpenSMILE dynamic = 260 维
输出范围 = [-1, 1]
输出时间步 = 0.5 秒
```

核心结构思想：

```text
音频序列
├── 全局分支：预测歌曲/片段的总体情感位置
└── 局部分支：预测围绕全局位置的时间变化

最终预测 = global component + local_scale × local deviation
```

这一分解与第二阶段研究目标一致：既保留歌曲间情感差异，也保留 section 内外的相对推进。

### 4.2 共同训练参数

| 参数 | 数值 |
| --- | ---: |
| Projection dimension | 112 |
| BiGRU hidden dimension | 64，双向 |
| Global hidden dimension | 96 |
| Dropout | 0.25 |
| Batch size | 24 |
| Optimizer | AdamW |
| Learning rate | 4e-4 |
| Weight decay | 5e-4 |
| Gradient clipping | 1.0 |
| Inner validation | 15% song-level split |

Valence 使用 `local_scale=0.55`，最终训练 9 epoch；Arousal 使用 `local_scale=0.70`，最终训练 7 epoch。两者均组合 clip-mean、CCC、temporal smoothness 和 soft-teacher loss，但权重分别设置，以适应两个维度不同的可预测性。

### 4.3 数据划分与冻结规则

- 开发数据：652 个 PMEmo 片段。
- 最终测试：115 个冻结 PMEmo 片段，共 5,390 个 0.5 秒帧。
- 预测文件在读取最终标签前冻结并记录哈希。
- 最终标签只读取一次，用于正式指标计算。
- 最终测试后没有调参、校准、平滑选择、模型替换或权重搜索。
- 开发和最终测试的 ID 无交集。

**已知问题：** 存在 3 组跨开发/最终测试的 MP3 内容完全相同。剔除后的 112 首敏感性结果变化较小，但 115 首测试仍不能称为严格的“音频内容完全独立测试”。

### 4.4 最终测试指标

#### 完整片段聚合（主要 Teacher 性能）

| 维度 | RMSE | MAE | PCC | CCC | 项目最低线状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Valence | 0.186298 | 0.152855 | 0.802852 | 0.800168 | PCC/CCC 通过；RMSE/MAE 略未通过 |
| Arousal | 0.188573 | 0.146912 | 0.857441 | 0.854803 | MAE/PCC/CCC 通过；RMSE 略未通过 |

#### 0.5 秒帧级（动态结果）

| 维度 | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.194416 | 0.158386 | 0.810613 | 0.801561 |
| Arousal | 0.200411 | 0.157457 | 0.848749 | 0.839168 |

训练集均值基线在 115 首完整片段上的 Valence/Arousal RMSE 为 `0.2978/0.3543`，说明冻结 Teacher 明显优于常数预测。

项目内部最低线 `RMSE≤0.18、MAE≤0.15、PCC≥0.80、CCC≥0.80` 是项目目标，不应写成所有 MER 数据集通用的权威标准。最终 Teacher 的相关和一致性达到可用水平，但误差没有全面通过内部最低线。

### 4.5 Teacher 可用性结论

从结构看，模型不依赖歌曲 ID、歌词、section 或历史 VA 标签，可直接接收新音频，因此具备任意新 MP3 推理的技术条件。

但科学上应写为：

> 冻结的纯音频 Teacher 可以为新 MP3 输出 0.5 秒连续 pseudo-VA；其正式指标来自未参与拟合的 PMEmo 短片段。多分钟完整歌曲输出尚未获得连续人工 VA ground truth 验证。

不能写为：

> “模型已经证明能以相同精度预测任何完整歌曲的真实 VA。”

---

## 5. 第二阶段数据与完整歌曲部署

### 5.1 质检歌曲与 section 数据

阶段一审计时，数据库中有 99 首 reviewed 歌曲、1,097 个连续 section、99 个本地 MP3。所有歌曲满足：

- section order 唯一且连续；
- 第一段从 0 秒开始；
- 相邻段落无超过 0.05 秒的 gap/overlap；
- section duration 与 end-start 一致；
- 最后一段结束时间与歌曲时长误差不超过 0.5 秒。

两首外语歌曲 `ANGEL` 和 `Easy On Me` 保留推理产物但不进入中文主分析，最终得到 97 首、75 位歌手、1,076 个 section。

### 5.2 样本来源与代表性

**【待完善-P0-01】现有 97 首不能称为科学随机样本。**

实际情况是：歌曲由研究者在项目推进中逐步寻找和录入，虽然包含不同歌手和常见中文流行歌曲，但没有事前冻结：

- 明确的总体抽样框；
- 榜单日期与版本；
- 分层变量；
- 随机种子；
- 排除与替补规则；
- 年代、风格、性别、厂牌或平台分布配额。

因此应将其称为 **convenience sample / 便利样本**，不能在论文中写成“随机抽取 97 首中文流行歌曲”。

当前样本内部：75 位歌手；单一歌手最多 5 首，占 5.2%；62 位歌手仅贡献 1 首。这能降低单一歌手支配，但不能自动证明总体代表性。

推荐补强方案是新增 30 首预冻结排名分层样本：冻结带日期的 QQ 音乐歌手榜，在 1–20、21–50、51–100 三层各随机抽 5 位歌手，再分别抽 3、2、1 首符合条件的歌曲。所有排除和补抽必须提前写明并冻结。该扩展用于外部效度，不用于重新训练 Teacher。

### 5.3 完整歌曲推理策略

直接把多分钟序列一次性输入短片段训练的 Teacher 会产生长度分布外推风险，因此采用：

```text
窗口长度 = 30 秒
步长 = 15 秒
重叠融合 = Hann overlap-add
输出 hop = 0.5 秒
```

99 首歌曲共生成 49,763 帧，84 首新推理，15 首通过哈希复用。总运行时间 3,043.35 秒；99/99 通过，0 失败，0 曲线塌缩警告。

### 5.4 Section 对齐

- 49,763/49,763 帧成功对齐；
- 1,097/1,097 section 均有帧；
- 0 未分配帧；
- 0 重叠帧；
- 内部区间采用 `[start,end)`；
- 最终 section 采用 `[start,end]`。

Teacher 推理和边界对齐阶段没有读取 section 类型、人工 VA、变化方向或高潮答案。类型在预测冻结后才揭示。

### 5.5 原始 MP3 特征域问题

**【待完善-P0-04】** 正式 PMEmo 指标使用官方 OpenSMILE 特征；任意 MP3 推理使用本地重提取特征。抽查显示标准化特征相关约 `0.773–0.881`，但不是 bit-equivalent，单曲预测差异在 Arousal 上可较明显。

论文必须把这一路径称为“部署域适配后的 pseudo-VA 推理”，不能暗示第二阶段完整歌曲和 PMEmo 正式测试完全同分布。

---

## 6. 人工 section 趋势验证

### 6.1 盲法设计

在 99 首中冻结 15 首样本，覆盖不同时长和 section 数量，且包含 15 位不同歌手。Teacher 先生成 7,375 个帧并与 179 个 section 对齐，随后才揭示 section 类型和人工答案。

标注者在页面中听取裁剪后的 section 音频，填写：

- Valence：low/mid/high；
- Arousal：low/mid/high；
- 相对上一段：up/down/stable；
- 局部情绪峰值候选。

模型结果在标注时隐藏。30 个参考片段按 V/A 等级提供跨歌曲参照，并且参考片段选择未使用 Teacher 预测。

### 6.2 同标注者复核稳定性

| 字段 | 完全一致率 | Weighted Kappa |
| --- | ---: | ---: |
| Valence 等级 | 87.15% | 0.858 |
| Arousal 等级 | 88.83% | 0.877 |
| 相邻段变化 | 100.00% | 1.000 |
| 局部峰值 | 99.44% | 0.987 |

这说明同一标注者休息后对相对结构判断较稳定，但不能作为不同标注者之间的一致性证据。

### 6.3 Teacher 与人工等级一致性

| 指标 | Valence | Arousal |
| --- | ---: | ---: |
| Pooled section Spearman | 0.614 | 0.576 |
| Within-song macro Spearman | 0.437 | 0.525 |
| Within-song centered Spearman | 0.509 | 0.562 |
| Between-song mean Spearman | 0.742 | 0.614 |

非 stable 相邻变化方向准确率为 `88/129=68.2%`。Teacher 最高 Arousal section 对人工局部峰值精确命中 `8/15`，相邻命中 `12/15`。

人工 Chorus 相对 Verse 的 V/A 提升在 15 首中均有 13 首为正；模型与人工 Chorus lift 的歌曲间 Spearman 为 Valence `0.228`、Arousal `0.586`。因此人工验证对 Arousal-led Chorus lift 提供中等趋势支持，对 Valence 只提供较弱支持。

### 6.4 验证边界

**【待完善-P0-02】** 当前只有一位标注者。下一步应邀请至少一位独立标注者，在不查看现有答案和模型输出的情况下标注同一冻结 15 首，并计算 inter-rater weighted Kappa、变化方向一致率和峰值命中。

不能在当前数据上计算人工 RMSE、MAE 或 CCC，因为 low/mid/high 是序数等级，不是连续 ground truth。

---

## 7. Section-aware 分析方法

### 7.1 三个分析层级

1. **Section 层：** 各类型的 VA 位置、范围、内部变化和歌曲内相对位置。
2. **转换层：** 相邻 section 的 ΔV、ΔA、VA 距离和二维方向。
3. **歌曲层：** 起止状态、路径长度、动态范围、高点位置、重复一致性、收束强度和 V/A 耦合。

### 7.2 统计原则

- 歌曲是推断的独立单位，不把 1,076 个 section 当作独立歌曲样本。
- 使用歌曲级配对差、Wilcoxon、5,000 次 bootstrap 置信区间和 Holm 校正。
- 原始 pseudo-VA 为主分析，时长加权和歌曲内标准化为敏感性检查。
- 方向阈值使用 0.03/0.05/0.07 复核；音乐功能描述固定使用 `±0.05`。
- 不为了得到整齐结论删除反例。
- 不将最大 Arousal 自动命名为真实高潮。

### 7.3 情感组织指标

每首歌曲生成连续指标，而非互斥类别：

- 基础 VA 位置；
- Valence/Arousal 动态范围；
- 平均与最大 section 对比；
- 全曲和后半段 V/A 斜率；
- Verse/Chorus 首末变化；
- 副歌入口方向一致性；
- 重复核心段 VA 距离；
- Bridge 回归增益；
- Intro—Outro 距离；
- 最大 Arousal、最大 Valence、最大转换位置；
- 高点到结尾的收束量；
- section 内 V/A Spearman；
- 最大两个转换占全路径比例。

---

## 8. 结果：从数据到情感，再到音乐功能

### 8.1 Section 覆盖

| Section | 数量 | 歌曲覆盖 |
| --- | ---: | ---: |
| Verse | 262 | 97 |
| Chorus | 371 | 96 |
| Pre-chorus | 90 | 46 |
| Bridge | 35 | 32 |
| Interlude | 127 | 86 |
| Intro | 94 | 92 |
| Outro | 97 | 94 |

Verse/Chorus 证据覆盖最充分；Bridge 和 Pre-chorus 的结论覆盖范围较小。

### 8.2 Verse 与 Chorus：基础建立和情感集中

**数据层**

| 比较 | n | Valence 差 | Arousal 差 |
| --- | ---: | ---: | ---: |
| Chorus − Verse | 96 首 | +0.101 | +0.266 |
| 直接 Verse→Chorus | 56 首 | +0.054 | +0.139 |

Chorus−Verse 的 95% bootstrap CI 为 Valence `[0.083,0.119]`、Arousal `[0.237,0.297]`；正向歌曲覆盖分别为 90.6% 和 100%。

**情感层**

Arousal 差明显大于 Valence 差，说明最稳定的变化不是统一“更快乐”，而是更高的能量、激活程度和情绪集中。368 次 Chorus 入口中：

- 积极释放：111 次；
- Valence 近似稳定、Arousal 上升的强度释放：89 次；
- 两维近似稳定的延续型：71 次；
- 混合或 Valence 主导：50 次；
- 反向降温：30 次；
- Arousal 上升而 Valence 下降的高强度负向释放：14 次；
- Valence 上升而 Arousal 下降的舒缓释放：3 次。

**音乐功能层**

在本样本中，Verse 可以作为情绪底色与轮转起点；Chorus 更常承担主题集中、强度放大或到达/释放功能。不同 Chorus 可以是积极释放、痛苦释放、强度释放或延续，不能把“副歌”与“快乐”直接等同。

**证据边界**

Teacher 在 96/96 首上都预测 Chorus Arousal 高于 Verse，整齐程度较高，可能包含模型结构偏差。15 首人工验证中只有 13/15 同方向，因此论文应同时报告人工反例。

### 8.3 Verse—Pre-chorus—Chorus：蓄力方式

**数据层**

72 个严格相邻事件按 `±0.05` Arousal 阈值分为：

| 模式 | 事件数 | 数据规则 |
| --- | ---: | --- |
| 连续蓄力 | 47 | Verse→Pre 与 Pre→Chorus 均上升 |
| 突然释放 | 15 | Verse→Pre 近似稳定，Chorus 入口上升 |
| 弱或混合 | 5 | 不满足主要方向规则 |
| 延迟释放/高位延续 | 4 | Pre 已上升，Chorus 入口近似稳定 |
| 虚假蓄力 | 1 | Pre 上升，Chorus 入口回落 |

**情感层**

这些模式区分了分阶段增强、把主要变化保留到副歌入口、预副歌先到高位、以及蓄力后不继续释放等不同情感体验。

**音乐功能层**

结果与流行音乐理论中的 initiation—buildup—arrival 框架相容：Verse 建立，Pre-chorus 蓄力，Chorus 到达。但当前证据只说明宏观情感结果，不证明该目的性来自和声终止、旋律碎片化、织体增厚或制作层次。

### 8.4 重复：再现、强化与重释

**数据层**

第一次到最后一次 Chorus 的平均变化为 Valence `+0.087`、Arousal `+0.209`。但在 89 首同时重复 Verse 和 Chorus 的歌曲中，Chorus 相对 Verse 的额外重复变化只有 Valence `-0.025`、Arousal `+0.015`，Holm p=`0.437/0.934`。

**情感层**

后续副歌更强并不是副歌独有现象；后续 Verse、Pre-chorus 和 Interlude 也常上升。重复段落可能保持 Valence 底色并提高 Arousal，也可能双维升级、减弱或改变 Valence 方向。

**音乐功能层**

重复不是简单复制。流行歌曲可以用相同 section 身份提供可识别性，同时让其在新的全曲语境中承担情绪确认、强化、重释、升级或耗散。当前更稳妥的总解释是“轮转中的后程累积”，而不是“最后副歌具有独占强化机制”。

### 8.5 Interlude：能量重新分配与轮转再导向

**数据层**

Interlude 覆盖 86/97 首，平均归一化位置约 0.515。Chorus→Interlude 出现在 84 首，Interlude→Verse 出现在 73 首；进入 Interlude 时平均 ΔA 仅 `-0.005`，随后 Interlude→Verse 平均 ΔA 为 `-0.041`。

**情感层**

Interlude 常没有立即大幅卸力，而是维持或重新分配副歌后的能量，再进入较低强度的 Verse。

**音乐功能层**

在本样本中，Interlude 更适合解释为 Verse/Chorus 轮转之间的铰链或再导向区域，而非统一的“休息段”。是否通过器乐独奏、音色替换、和声延长或其他方式实现，需要听觉和底层特征验证。

### 8.6 Bridge：偏离、连接和回归

**数据层**

共有 35 个 Bridge，涉及 32 首歌曲。31 个事件同时具有可比较的前序 Chorus 中心和后续 Chorus，其中 16/31（51.6%）的下一次 Chorus 比 Bridge 更接近此前 Chorus 中心。

**情感层**

Bridge 可以继续提高 Arousal、暂时降温、改变 Valence 方向、偏离此前 Verse/Chorus 情绪区域，或只产生弱变化。部分歌曲出现“偏离—回归”。

**音乐功能层**

Bridge 的功能不在固定的 VA 高低，而在打破或悬置既有轮转、制造上下文对比，并可能把歌曲重新导向 Final Chorus。由于仅有 32 首含 Bridge，该结论必须限定覆盖范围。

### 8.7 Outro：卸力、余韵与情绪迁移

**数据层**

Chorus→Outro 平均 ΔV/ΔA 为 `-0.112/-0.183`。在具有可比 Intro—Outro 的歌曲中，仅 21.1% 的 Outro 回到距 Intro 不超过 0.10 的 VA 区域。整首歌曲结束 Arousal 高于开始的比例为 79.4%。

**情感层**

Outro 常形成局部卸力，但并不总是回到开场情绪。结尾可能是快速释放、渐进消散、高位余韵、情绪回归、正向收束、消极沉降或迁移到新的情绪位置。

**音乐功能层**

Outro 可以完成形式收束，同时保留高潮后的情绪残余。形式闭合不等于 VA 回到原点。

### 8.8 高点与高潮候选

**数据层**

- 最大 section 平均 Arousal 位于 Chorus：53/97 首；
- 全部歌曲中 54.6% 的最大 Arousal 位于最后四分之一；
- 在 Arousal 高点位于 Chorus 的 53 首中，62.3% 是最后一次 Chorus；
- 最大 VA 转换最常落到 Outro：36/97 首。

**情感层**

高能量位置偏向歌曲后部和 Chorus，但最大情感转折经常是进入 Outro 的下降。能量高点和转折高点是不同概念。

**音乐功能层**

后置 Chorus 经常承担能量汇聚，但 Bridge、Interlude 或 Outro 也可能形成模型高点。论文应分开报告能量高潮候选、Valence 高点和结构转折，不制造单一加权高潮指标。

### 8.9 整首歌曲的连续情感组织

**数据层**

- 平均 VA 路径直接度为 0.220，中位数 0.180；
- 74.2% 的歌曲 Arousal range 至少是 Valence range 的 1.25 倍；
- 11.3% 的歌曲有至少一半 VA 路径集中在两个最大 section 转换；
- section 内 V/A Spearman 中位数为 0.736，5/97 首为负；
- 23.6% 的相邻 section 跨越以零点划分的 VA 区域。

**情感层**

多数歌曲主要通过 Arousal 组织段落强弱；V/A 多数同向，但少量歌曲出现明显分离。多数歌曲的变化分布在多个边界，而非只依赖一两个突变。

**音乐功能层**

完整歌曲表现为多轮建立、集中、释放、回落和重新启动，并在后半段形成不同程度的累积与收束。这里描述的是连续组织机制，不把歌曲硬分成固定“情感弧线类型”。此前聚类实验只作为方法审计保留，不作为音乐学结论。

### 8.10 VA 象限的辅助解释边界

负 Valence、高 Arousal 只有 15/1,076 个 section。绝对象限高度依赖 Teacher 的零点校准，因此 quadrant 只能用于帮助理解二维方向，不能据此断言中文流行歌曲缺少“愤怒、焦虑或痛苦”等情绪，也不能把象限词汇当成精确离散标签。

---

## 9. 逐歌情感画像

项目已为 97 首歌生成规则驱动的中文画像，每条包含：

1. 开场与全曲基础 VA 区域；
2. Verse—Pre-chorus—Chorus 的主要蓄力方式；
3. Chorus 入口的释放方式；
4. Verse/Chorus 重复中的确认、增强、重释或耗散；
5. Bridge 的偏离或回归；
6. Outro 的收束方式；
7. Arousal 高点和最大转换位置；
8. 可复核数值依据。

画像使用“模型估计显示”“可能表现为”等限制语，不虚构歌词叙事，不推断和声、调式、节奏、音色或配器。

**【待完善-P1-02：个案听审】** 应按预先生成的候选规则，从连续蓄力、反向副歌、Bridge 偏离—回归、强 Outro 卸力、V/A 分离等情形各选 2–3 首，进行模型盲听审。个案既要包含典型例，也要包含反例，不能只挑符合结论的歌曲。

---

## 10. 主要研究结论

### 结论一：Chorus 的稳定属性是高唤醒集中，而非统一正向化

- **数据：** Chorus−Verse 的 Arousal 差 `+0.266`，大于 Valence `+0.101`；副歌入口存在七种二维变化。
- **情感：** Chorus 更常放大、集中或释放已有情绪，不一定改变情绪正负性质。
- **音乐功能：** Chorus 在流行歌曲轮转中经常承担情绪到达和主题集中。
- **边界：** Teacher 的 96/96 Arousal 正向结果可能偏整齐，需第二标注者和外部样本复核。

### 结论二：Pre-chorus 不是单纯中间段，而是多种蓄力路径

- **数据：** 47/72 连续蓄力，15/72 把主要释放留到 Chorus。
- **情感：** 存在平滑蓄力、突然释放、高位延续和少量反向结果。
- **音乐功能：** Pre-chorus 可能负责延长目标指向、强化到达预期或延迟释放。
- **边界：** 不能从 VA 判断具体的和声、旋律和制作机制。

### 结论三：重复 section 的意义是身份保持与语境重释并存

- **数据：** 后续 Chorus 上升，但没有超过 Verse 的独有重复增幅。
- **情感：** 重复可能确认、增强、减弱或重释相同的情绪底色。
- **音乐功能：** 流行歌曲通过熟悉段落身份维持可识别性，同时通过全曲位置改变其表达重量。
- **边界：** 尚无配器、音色或演唱强度数据解释变化来源。

### 结论四：Bridge 与 Interlude 主要体现上下文功能

- **数据：** Interlude 常连接 Chorus 与下一 Verse；约半数可比较 Bridge 后 Chorus 向先前 Chorus 区域回归。
- **情感：** Interlude 重新分配能量，Bridge 产生偏离、降温、强化或悬置。
- **音乐功能：** Interlude 是轮转铰链；Bridge 是核心循环外的对比与再导向空间。
- **边界：** Bridge 样本仅 32 首，解释应视为探索性。

### 结论五：Outro 的形式闭合与情绪回归并不等价

- **数据：** Chorus→Outro 明显下降，但只有 21.1% 回到接近 Intro 的 VA 区域。
- **情感：** 结束可以卸力、消散、保留余韵或迁移到新位置。
- **音乐功能：** Outro 完成收束，但不必恢复开场情绪原点。
- **边界：** 尾奏中的淡出、静音和尾音可能影响均值，需个案复核。

### 结论六：完整歌曲主要通过多段轮转和 Arousal 变化组织体验

- **数据：** 74.2% 的歌曲由 Arousal 主导动态范围，平均路径直接度较低。
- **情感：** 情绪推进包含多次蓄积、释放、回落和再启动。
- **音乐功能：** 完整流行歌曲的情感组织应放在 section 顺序和重复语境中理解，而非只看全曲均值。
- **边界：** 当前只描述本便利样本，不建立固定曲式或情感弧线分类。

---

## 11. 研究贡献

### 11.1 方法贡献

1. 建立从纯音频连续 VA Teacher 到完整歌曲 section-aware 分析的可复现链路。
2. 使用冻结预测、后揭示 section 类型和盲人工复核，降低第二阶段调参泄漏。
3. 将歌曲作为独立单位，避免把多个 section 当作独立样本造成伪重复。
4. 提出“数据—情感—音乐功能”三层解释规范，限制跨层越界。

### 11.2 分析贡献

1. 将 Chorus lift 拆分为不同二维释放方式，而非单一平均抬升。
2. 将 Pre-chorus 拆分为连续蓄力、突然释放、高位延续和反向路径。
3. 将重复解释为情绪确认、强化、重释和耗散的连续变化。
4. 把 Bridge、Interlude 和 Outro 放回完整轮转与收束语境分析。
5. 为每首歌曲生成可追溯的连续情感组织画像。

### 11.3 工程与开放性贡献

- Teacher、配置、预测和关键输入均有哈希或冻结清单；
- 99 首完整歌曲推理和 section 对齐经过独立审计；
- 统计、音乐学解释和逐歌画像均有可执行脚本；
- 原始数据不因分析而修改。

---

## 12. 局限性与待完善事项

| 编号 | 优先级 | 当前问题 | 对结论的影响 | 建议补强 | 完成状态 |
| --- | --- | --- | --- | --- | --- |
| P0-01 | P0 | 97 首为便利样本，无预冻结抽样设计 | 不能外推中文流行歌曲总体 | 增加 30 首排名分层冻结样本，或明确只做便利样本探索 | 待完善 |
| P0-02 | P0 | 15 首只有一位标注者 | 无法报告 inter-rater reliability | 增加独立第二标注者 | 待完善 |
| P0-03 | P0 | Teacher 正式测试为短标注片段 | 完整歌曲同等精度未证明 | 寻找/构建多分钟动态 VA 外部测试 | 待完善 |
| P0-04 | P0 | 原始 MP3 OpenSMILE 与官方特征不完全等价 | 第二阶段存在特征域偏移 | 固定提取版本，扩大官方/重提取一致性审计 | 待完善 |
| P0-05 | P0 | 开发/最终测试有 3 个完全相同 MP3 | 测试不是严格内容独立 | 正文披露 112 首敏感性；未来新数据按音频哈希去重后划分 | 已审计，无法追溯修复 |
| P1-01 | P1 | 无响度、节奏、音色、和声、旋律与配器变量 | 不能解释情绪变化机制 | 对代表个案补充可测音频特征 | 待完善 |
| P1-02 | P1 | Section 类型与边界缺少独立第二人可靠性 | 形式标签仍可能有感知歧义 | 抽取 15–20 首做独立 section 复核 | 待完善 |
| P1-03 | P1 | Bridge 仅覆盖 32 首 | Bridge 结论精度和覆盖有限 | 分层扩样时提高 Bridge 覆盖但不可结果导向抽样 | 待完善 |
| P1-04 | P1 | 年代、流派、性别、版本、发行来源元数据不完整 | 无法控制构成差异 | 建立冻结 metadata dictionary | 待完善 |
| P1-05 | P1 | 象限依赖 Teacher 零点校准 | 离散情绪词可能误导 | 以连续 VA 为主，象限仅作辅助 | 已限制解释 |
| P1-06 | P1 | 同标注者参考校准非随机实验 | 不能把指标改善归因于 30 个参考 | 保留 12/30 参考敏感性，不做因果表述 | 已限制解释 |
| EXT-01 | 后续 | 未分析歌词 | 无法研究词曲情感一致性 | 另立多模态研究，不混入当前论文 | 后续扩展 |
| EXT-02 | 后续 | 不做 human/AIGC 比较 | 当前结论只针对人工创作样本 | 未来需独立预注册语料设计 | 后续扩展 |

### 12.1 对当前 EI 论文可行性的判断

当前工程链路和探索结果足以形成一篇以 section-aware 情感组织为主线的 EI 论文初稿，但要达到较稳妥的投稿质量，至少应完成：

1. 第二独立标注者；
2. 便利样本问题的明确披露，最好补充一个预冻结分层外部样本；
3. 6–10 首典型与反例的模型盲听审；
4. 正文中完整披露 Teacher 的短片段与特征域限制。

不建议在投稿前继续大量搜索更复杂 Teacher 参数，因为最终测试已读取，继续围绕其结果优化会破坏前瞻性。后续模型改进必须使用新的开发/测试设计。

---

## 13. 下一步最短执行路径

### 阶段 A：证据补强

1. 冻结第二标注者说明和表格；完成 15 首独立标注。
2. 计算 inter-rater weighted Kappa、方向一致率和峰值相邻命中。
3. 决定是否执行 30 首排名分层外部扩样；若不执行，正式接受便利样本论文定位。

### 阶段 B：音乐个案分析

1. 按冻结候选规则选择典型与反例。
2. 绘制 VA 曲线、section 边界和功能事件。
3. 进行不看模型结论的听觉记录。
4. 仅在可直接测量时补充 RMS、onset、tempo、spectral 或 chroma 描述。

### 阶段 C：论文写作

1. 方法部分优先写冻结、盲法和证据边界。
2. 结果采用“数据—情感—音乐功能—反例—边界”固定段落结构。
3. Teacher 指标只作为测量工具可靠性，不把模型精度作为论文唯一贡献。
4. 讨论部分重点解释 Arousal-led Chorus、重复语境、偏离—回归和收束。

---

## 14. 论文建议结构

1. Introduction
2. Related Work
   - Continuous music emotion recognition
   - Pretrained music audio representations
   - Popular-music section/form analysis
3. Data and Ethics
   - PMEmo Teacher data
   - 97-song convenience corpus
   - Section annotation and human validation
4. Method
   - Frozen global-local VA Teachers
   - Complete-song windowed inference
   - Frame-to-section alignment
   - Data—emotion—music-function framework
5. Teacher Evaluation
6. Human Trend Validation
7. Section-Level Results
8. Emotional Organization and Popular-Music Functions
9. Case Studies and Counterexamples
10. Limitations
11. Conclusion

---

## 15. 允许与禁止的论文表述

### 可以写

- “在本 97 首中文流行歌曲便利样本中观察到……”
- “冻结 Teacher 的 pseudo-Arousal 显示……”
- “结果提示 Chorus 更稳定地承担高唤醒到达功能……”
- “Bridge 可能形成偏离—回归，但覆盖有限……”
- “该音乐功能解释仍需底层音频特征与听觉个案验证……”

### 不应写

- “证明了中文流行歌曲普遍遵循该规律。”
- “模型输出就是真实人类情感。”
- “Chorus 一定比 Verse 更快乐。”
- “Bridge 必然是高潮或必然降温。”
- “Arousal 上升证明和声张力增加。”
- “模型已以相同指标适用于任何完整歌曲。”
- “97 首歌曲是科学随机抽样。”

---

## 16. 可复现产物索引

### Teacher

```text
research/phase1f_external_va_datasets/pmemo/final_clean_teacher/
research/phase1f_external_va_datasets/pmemo/final_clean_teacher/FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md
```

### 第二阶段冻结与验证

```text
research/phase2_section_aware_va/data/blind_validation/
research/phase2_section_aware_va/data/full_corpus/
research/phase2_section_aware_va/docs/BLIND_VALIDATION_HUMAN_RESULTS_30REF_REVIEW_20260726.md
research/phase2_section_aware_va/docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md
```

### 97 首分析

```text
research/phase2_section_aware_va/analysis/full_corpus_97/
research/phase2_section_aware_va/analysis/full_corpus_97/reports/EXPLORATORY_ANALYSIS_REPORT_ZH.md
research/phase2_section_aware_va/analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md
research/phase2_section_aware_va/analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md
research/phase2_section_aware_va/analysis/full_corpus_97/tables/song_emotional_strategy_profiles.csv
research/phase2_section_aware_va/analysis/full_corpus_97/tables/song_emotional_portraits_zh.csv
```

### 主要复现命令

```bash
MPLBACKEND=Agg .venv/bin/python \
  research/phase2_section_aware_va/scripts/run_full_corpus_exploratory_analysis.py

MPLBACKEND=Agg .venv/bin/python \
  research/phase2_section_aware_va/scripts/analyze_musicological_implications.py

MPLBACKEND=Agg .venv/bin/python \
  research/phase2_section_aware_va/scripts/analyze_emotional_organization_strategies.py
```

---

## 17. 参考文献与理论来源

1. Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLOS ONE, 12*(3), e0173392. https://doi.org/10.1371/journal.pone.0173392
2. Zhang, K., Zhang, H., Li, S., Yang, C., & Sun, L. (2018). The PMEmo Dataset for Music Emotion Recognition. *Proceedings of ICMR 2018*, 135–142. https://www.next.zju.edu.cn/cn/archive/pmemo/
3. Li, Y. et al. (2024). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *ICLR 2024*. https://arxiv.org/abs/2306.00107
4. Nobile, D. (2022). Teleology in Verse–Prechorus–Chorus Form, 1965–2020. *Music Theory Online, 28*(3). https://www.mtosmt.org/issues/mto.22.28.3/mto.22.28.3.nobile.html
5. Temperley, D. (2018). Form. In *The Musical Language of Rock*. https://doi.org/10.1093/oso/9780190653774.003.0008
6. de Clercq, T. (2017). Embracing Ambiguity in the Analysis of Form in Pop/Rock Music. *Music Theory Online, 23*(3). https://www.mtosmt.org/issues/mto.17.23.3/mto.17.23.3.de_clercq.php
7. Osborn, B. (2013). Subverting the Verse–Chorus Paradigm: Terminally Climactic Forms in Recent Rock Music. *Music Theory Spectrum, 35*(1), 23–47. https://doi.org/10.1525/mts.2013.35.1.23
8. Nieto, O. et al. (2019). The Harmonix Set: Beats, Downbeats, and Functional Segment Annotations of Western Popular Music. *ISMIR 2019*. https://archives.ismir.net/ismir2019/paper/000068.pdf

---

## 18. 总结

本项目已经从“训练一个 VA 回归模型”推进到“利用冻结音频 Teacher 描述完整歌曲的 section-aware 情感组织”。最有价值的结果不是某一项显著性，而是建立了一条可审计的解释链：

> **数值证据说明 VA 如何变化；情感层说明变化如何形成蓄力、释放、偏离、回归、累积与收束；音乐功能层再说明这些过程可能怎样服务于流行歌曲的 Verse/Chorus 轮转和完整形式。**

当前证据足以支持一篇探索性 EI 论文初稿，但论文质量将主要取决于能否诚实处理便利样本、单标注者、短片段 Teacher 和特征域偏移，而不是继续扩大统计检验数量。下一步应优先补第二标注者和样本设计，再完成典型与反例的音乐个案分析。
