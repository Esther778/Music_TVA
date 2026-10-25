# Arousal 预测模型阶段汇报

报告撰写时间：2026-06-24 20:30:48 CST

当前科研进度：

```text
第一阶段 VA Teacher 模型仍在优化中。
Arousal 维度已经达到当前项目设定的最低合格线，并且已经具备完整歌曲预测链路。
Valence 维度仍然是当前 VA Teacher 的主要瓶颈，后续优化重点应转向 Valence。
第二阶段 section-aware 完整歌曲情绪结构分析尚不应大规模启动，需等待 VA Teacher 尤其是 Valence 进一步稳定。
```

## 1. 当前研究目标

本项目的总目标不是只在某一个数据集上获得较高指标，而是训练一个可以用于任意歌曲 Valence-Arousal 预测的 Teacher 模型。后续 section-aware 研究将基于 Teacher 输出的完整歌曲 VA 曲线，结合人工质检 section 标签，分析完整歌曲中的情绪推进、高潮组织和段落对比。

因此，第一阶段 Teacher 模型需要同时满足两个条件：

```text
1. 指标达到最低可用标准；
2. 具备任意完整歌曲 MP3 的推理能力。
```

本报告只总结 Arousal 维度的当前阶段结果。

## 2. 数据集与特征基础

### 2.1 PMEmo 数据集

本阶段引入 PMEmo，是因为它提供了比当前 DEAM 严格完整歌曲子集更大的动态 VA 标注规模。当前本地 PMEmo 动态标注使用情况如下：

```text
dynamic songs = 767
dynamic frames = 36434
label step = 0.5 s
PMEmo 原始标签范围 = [0, 1]
当前统一标签范围 = [-1, 1]
```

标签统一公式为：

```text
canonical_value = 2 * pmemo_value - 1
```

之所以统一到 `[-1, 1]`，是为了和 DEAM 动态 VA 标注尺度保持一致，避免不同数据集的 RMSE / MAE 数值不可比。

### 2.2 OpenSMILE 特征

PMEmo 官方提供了预计算的 OpenSMILE dynamic acoustic features。我们本地核查后确认：

```text
feature_dim = 260
基础特征 = 65 个 ComParE_2016 low-level descriptors
delta 特征 = 65 个 ComParE_2016 low-level descriptor deltas
聚合方式 = mean / std
```

为了让模型能够处理任意完整歌曲，而不是只能读取 PMEmo 官方 CSV，本项目补充实现了 PMEmo-style OpenSMILE 特征复现：

```text
feature_set = ComParE_2016
feature_level = LowLevelDescriptors + LowLevelDescriptors_Deltas
window = trailing 1.0 s
hop = 0.5 s
output_dim = 260
```

在一首 PMEmo 音频上抽查时，该复现特征与 PMEmo 官方 `dynamic_features.csv` 的展开相关约为 `0.996`。因此论文中应表述为：

```text
PMEmo-style OpenSMILE feature reproduction
```

不应表述为完全 bit-identical 的官方特征复现。

## 3. 初始 VA 联合预测模型

最初我们训练的是联合 VA 预测模型，即一个模型同时输出 Valence 和 Arousal。

模型包括：

```text
PMEmo OpenSMILE + MLP
PMEmo OpenSMILE + BiLSTM
```

数据划分为 song-level split：

| Split | Songs | Frames |
| --- | ---: | ---: |
| Train | 614 | 29184 |
| Validation | 77 | 3912 |
| Test | 76 | 3338 |

联合预测中表现最好的模型是 OpenSMILE + BiLSTM：

```text
input_dim = 260
hidden_dim = 128
layers = 2
dropout = 0.25
optimizer = AdamW
learning_rate = 8e-4
weight_decay = 1e-4
selection rule = validation mean VA RMSE
```

测试集结果如下：

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.271199 | 0.212034 | 0.571292 | 0.546814 |
| Arousal | 0.186934 | 0.141942 | 0.837598 | 0.830559 |
| Mean | 0.232909 | 0.176988 | 0.704445 | 0.688687 |

从该结果可以看出，联合 VA 模型并不是整体完全失败，而是两个维度表现明显不均衡：

```text
Arousal 已经接近合格；
Valence 明显偏弱，并拉低了 mean VA 指标。
```

## 4. 为什么改为 V / A 分开训练

将 Valence 和 Arousal 分开训练，不是为了人为提高某个指标，而是因为联合模型暴露出明确的维度差异。

联合 BiLSTM 中：

```text
Arousal PCC/CCC = 0.837598 / 0.830559
Valence PCC/CCC = 0.571292 / 0.546814
```

这说明模型对 Arousal 的动态变化趋势和数值一致性已经学习得较好，而 Valence 仍较弱。

从音乐情绪识别角度看，这一现象是合理的：

1. Arousal 与音频能量和运动强度关系更直接，例如 loudness、RMS energy、onset strength、spectral flux、节奏密度、速度等。
2. Valence 受到更多复杂因素影响，例如调式、和声色彩、旋律走向、歌词语义、文化经验和听众个人偏好。
3. 因此，用同一个输出头、同一个损失和同一个选模规则同时优化 V 和 A，可能导致 Arousal 没有被充分优化，同时 Valence 也没有真正解决。

因此，本阶段策略调整为：

```text
先把 Arousal 独立训练到可用水平；
再集中解决 Valence。
```

具体技术变化为：

```text
原任务 = joint Valence + Arousal prediction
新任务 = Arousal-only prediction
原选模规则 = validation mean VA RMSE
新选模规则 = validation Arousal RMSE
```

该策略也符合后续 section-aware 分析需求，因为完整歌曲中的高潮位置、强弱推进和段落对比高度依赖 Arousal。

## 5. Arousal 单模型训练结果

Arousal-only 训练脚本为：

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_opensmile_arousal_trials.py
```

最佳单模型为：

```text
trial_id = a_only_h96_l1_do015_mse
model = 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 59
```

测试集结果：

| Metric | Value | Minimum pass | Gap |
| --- | ---: | ---: | ---: |
| RMSE | 0.180960 | ≤0.180000 | +0.000960 |
| MAE | 0.140092 | ≤0.150000 | -0.009908 |
| PCC | 0.852634 | ≥0.800000 | -0.052634 |
| CCC | 0.848122 | ≥0.800000 | -0.048122 |

该单模型已经通过 MAE、PCC、CCC 三项指标，但 RMSE 仍差 `0.000960`，因此不能严格称为全面达标。

## 6. 最终 Arousal Ensemble 模型

为了降低单模型随机波动并提升泛化表现，我们进一步使用验证集选择 ensemble 权重。

选模原则：

```text
ensemble 权重只根据 validation RMSE 选择；
test set 只用于最终报告；
不使用 test set 调参。
```

最终模型：

```text
model_id = pmemo_opensmile_arousal_ensemble_v1
```

成员与权重：

| Trial | Weight |
| --- | ---: |
| a_only_h128_l2_do025_mse | 0.15 |
| a_only_h160_l2_do030_mse_ccc | 0.30 |
| a_only_h192_l2_do035_mse_ccc_smooth | 0.20 |
| a_only_h96_l1_do015_mse | 0.35 |

最终 Arousal ensemble 测试集结果如下：

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.176695 | 0.180000 | -0.003305 | 0.150000 | +0.026695 | 0.100000 | +0.076695 | Pass minimum |
| MAE | 0.135837 | 0.150000 | -0.014163 | 0.120000 | +0.015837 | 0.080000 | +0.055837 | Pass minimum |
| PCC | 0.859718 | 0.800000 | -0.059718 | 0.850000 | -0.009718 | 0.900000 | +0.040282 | Pass good |
| CCC | 0.848452 | 0.800000 | -0.048452 | 0.850000 | +0.001548 | 0.900000 | +0.051548 | Pass minimum, near good |

结论：

```text
该 Arousal ensemble 已达到当前最低合格线；
PCC 达到良好水平；
CCC 非常接近良好水平；
RMSE 和 MAE 仍未达到良好水平；
整体定位为“合格到接近良好”，不能称为优秀。
```

## 7. 合格 / 良好 / 优秀标准的依据

当前项目采用以下阈值：

| Level | RMSE | MAE | Pearson r | CCC |
| --- | ---: | ---: | ---: | ---: |
| Minimum pass | ≤0.18 | ≤0.15 | ≥0.80 | ≥0.80 |
| Good | ≤0.15 | ≤0.12 | ≥0.85 | ≥0.85 |
| Excellent | ≤0.10 | ≤0.08 | ≥0.90 | ≥0.90 |

这些数值不是随意制定，也不是声称为所有 MER 研究的统一标准，而是结合统计学解释、计算机建模评价习惯和 MER/DMER 领域结果后设定的项目标准。

### 7.1 Pearson r 的依据

在统计学和应用建模领域，相关系数通常用于衡量两个连续变量变化趋势的一致性。Schober 等关于相关系数解释的综述将较高相关区间解释为较强或很强的关联。对于本项目而言，后续 section-aware 分析依赖的是情绪曲线趋势和段落变化，因此 Pearson r 不能只达到中等水平。

因此：

```text
Pearson r ≥ 0.80：可认为趋势一致性已经较强，可作为最低可用线；
Pearson r ≥ 0.85：更稳健，作为良好线；
Pearson r ≥ 0.90：非常强，作为优秀线。
```

### 7.2 CCC 的依据

CCC（Concordance Correlation Coefficient）由 Lin 提出，用于衡量预测值和真实值之间的一致性。它比 Pearson r 更严格，因为它不仅考察趋势相关，还惩罚均值偏差和方差偏差。

在本项目中，CCC 尤其重要。原因是后续 section-level 分析不仅看曲线涨跌，还要计算 section contrast、climax strength、chorus lift 等结构特征。如果模型虽然趋势相关但整体尺度偏差明显，就会扭曲段落对比强度。

因此：

```text
CCC ≥ 0.80：作为可用 Teacher 的最低一致性要求；
CCC ≥ 0.85：接近稳定可靠，作为良好线；
CCC ≥ 0.90：高度一致，作为优秀线。
```

### 7.3 RMSE / MAE 的依据

本项目统一使用 `[-1, 1]` VA 标尺，总跨度为 2。

在该标尺下：

```text
RMSE = 0.18 约等于全量程的 9%
MAE = 0.15 约等于全量程的 7.5%
RMSE = 0.15 约等于全量程的 7.5%
RMSE = 0.10 约等于全量程的 5%
```

因此，将 RMSE ≤0.18、MAE ≤0.15 作为最低可用标准是合理的：它要求点位误差不能过大，但也承认动态音乐情绪标注本身存在主观性、延迟和标注噪声。

### 7.4 MER / DMER 领域依据

动态音乐情绪识别本身是难任务。早期 DEAM/MediaEval 相关工作中，Arousal 的 RMSE 大约在 0.20 左右并不罕见；近年来的 DMER 前沿工作开始引入更强的时序建模、个性化机制、局部-全局特征和预训练音频表示。PDMER/DSAML 一类工作也强调短片段训练与完整歌曲测试之间的泛化问题。

因此，本项目的标准可以理解为：

```text
0.18：比许多早期或普通动态 MER baseline 更严格，适合作为内部 Teacher 的最低线；
0.15：代表更稳健的动态预测能力，适合作为良好线；
0.10：在严格 song-level / complete-song 场景中属于很强水平，应谨慎称为优秀。
```

需要注意的是，一些论文报告的极低 RMSE 可能来自不同标签尺度、不同数据划分、短片段测试、个性化设置、历史情绪标签输入或数据泄漏风险。不能不检查实验协议就直接横向比较。

## 8. 完整歌曲预测能力

当前 Arousal ensemble 已经具备完整歌曲推理链路：

```text
完整 MP3
-> OpenSMILE ComParE_2016 LLD + deltas
-> trailing 1s mean/std 聚合
-> 260维 PMEmo-style feature
-> Arousal ensemble
-> 每0.5秒一帧 Arousal 曲线
```

单曲预测脚本：

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal_opensmile_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions/jiangnan_arousal_opensmile_ensemble.csv"
```

批量预测脚本：

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal_opensmile_ensemble.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions_batch
```

已验证：

```text
单曲测试：songs/江南-林俊杰.mp3，输出 534 帧
批量 smoke test：2/2 首成功
```

## 9. 当前模型的论文定位

当前 Arousal 模型适合被描述为：

```text
一个达到最低可用标准的 Arousal Teacher component，
可用于后续完整歌曲 section-aware 情绪结构分析中的 Arousal pseudo-label 生成。
```

不应夸大为：

```text
SOTA 动态音乐情绪识别模型；
已经完整解决 VA 预测；
可以替代真实人工情绪标注；
对所有音乐类型都同样可靠。
```

更稳妥的论文表述是：

```text
We first trained an Arousal-specific Teacher model on PMEmo dynamic annotations
using PMEmo-style OpenSMILE acoustic features. The validation-selected ensemble
achieved minimum acceptable performance on the held-out test set and was then
used as the Arousal pseudo-label generator for complete-song analysis.
```

## 10. 局限性

当前模型仍有以下限制：

1. 训练与测试基于 PMEmo，主要是 chorus/片段级动态标注，尽管推理链路可以处理完整歌曲，但训练分布与完整歌曲仍有差异。
2. OpenSMILE 特征是 PMEmo-style 高相关复现，不是官方特征的 bit-identical 复制。
3. 当前结果尚未做多随机种子平均。
4. 当前只解决了 Arousal，Valence 仍未达标。
5. 模型对接近 PMEmo 分布的流行音乐最可靠，对极端风格、纯器乐长曲、现场录音、实验音乐等外推场景应谨慎。

## 11. 后续建议

短期建议：

```text
暂时冻结 pmemo_opensmile_arousal_ensemble_v1 作为当前 Arousal Teacher 候选；
后续主要资源转向 Valence 优化。
```

正式论文前建议补充：

```text
1. Arousal ensemble 做 3-seed 或 bootstrap 稳定性验证；
2. 如有条件，增加少量完整歌曲 Arousal 人工趋势验证；
3. 在论文中明确 pseudo-VA 是模型估计，不是真实 ground truth；
4. Arousal 主要用于高潮、能量推进、段落强弱对比等结构特征。
```

## 12. 参考依据

1. PMEmo dataset and released dynamic annotations/features: https://github.com/HuiZhangDB/PMEmo
2. openSMILE Python package and supported feature sets: https://audeering.github.io/opensmile-python/
3. PDMER / Personalized Dynamic Music Emotion Recognition: https://arxiv.org/abs/2412.19200
4. Lin, L. I-K. Concordance correlation coefficient, Biometrics, 1989. DOI: https://doi.org/10.2307/2532051
5. Schober, P., Boer, C., Schwarte, L. A. Correlation coefficients: appropriate use and interpretation, Anesthesia & Analgesia, 2018. DOI: https://doi.org/10.1213/ANE.0000000000002864
