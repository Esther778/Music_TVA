# 最终 VA Teacher 严谨性审计与模型卡

撰写时间：2026-07-15 22:50 CST
当前科研进度：第一阶段 VA Teacher 已冻结并完成一次最终测试；尚未进入
section-aware 第二阶段。

## 一、结论

当前可执行模型为两个独立的纯音频回归模型：

```text
Valence = pmemo_final_clean_valence_teacher_v1
Arousal = pmemo_final_clean_arousal_teacher_v1
输入 = MERT-v1-95M hidden-6（768维）+ PMEmo OpenSMILE dynamic（260维）
结构 = global-local BiGRU
输出 = [-1, 1]，每 0.5 秒一个 VA 点
训练数据 = 652 个 PMEmo 开发音频片段
最终测试 = 115 个 PMEmo 音频片段、5,390 帧
```

冻结预测文件在读取最终标签前已经生成，模型、配置和预测哈希均可复核；
最终指标已由独立审计脚本重新计算，和原结果逐项完全一致。最终测试后没有
训练、校准、平滑、集成权重选择或模型替换。

但本次审计发现三个必须披露的限制：

1. 115 个 `musicId` 与 652 个开发 `musicId` 没有交集，但有 3 组跨集合的
   MP3 内容完全相同，因此该测试不是严格的音频内容独立测试。
2. PMEmo 标注片段的中位有效跨度是 20.5 秒，不是多分钟完整歌曲；该测试
   不能单独证明完整歌曲全过程的泛化性能。
3. 原始 MP3 的 OpenSMILE 重提取与 PMEmo 官方特征高度相关但不完全等价；
   正式指标对应官方特征输入。任意 MP3 推理目前属于可运行但仍有特征域偏移
   的部署路径。

因此，论文中可以称其为“冻结的纯音频 VA Teacher，在未参与拟合的 PMEmo
片段上完成前瞻式测试”，不能称为“已在任意完整歌曲上证明达到同等指标”。

## 二、正式最终测试结果

### 完整片段聚合（主要结果）

每个 PMEmo 片段先对全部 0.5 秒预测取均值，再按 115 个片段计算指标。

| 维度 | RMSE | MAE | PCC | CCC | 最低线状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Valence | 0.186298 | 0.152855 | 0.802852 | 0.800168 | PCC/CCC 通过；RMSE 差 0.006298，MAE 差 0.002855 |
| Arousal | 0.188573 | 0.146912 | 0.857441 | 0.854803 | MAE/PCC/CCC 通过；RMSE 差 0.008573 |

### 0.5 秒帧级（次要结果）

| 维度 | RMSE | MAE | PCC | CCC | 最低线状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Valence | 0.194416 | 0.158386 | 0.810613 | 0.801561 | PCC/CCC 通过；RMSE/MAE 未通过 |
| Arousal | 0.200411 | 0.157457 | 0.848749 | 0.839168 | PCC/CCC 通过；RMSE/MAE 未通过 |

内部统一参照线为：RMSE `<=0.18`、MAE `<=0.15`、PCC `>=0.80`、
CCC `>=0.80`。这些是项目目标线，不应写成公认的跨数据集统一标准。

## 三、去重敏感性分析

跨开发集/最终测试的完全相同音频为：

| 最终 ID | 开发 ID | 歌曲 |
| ---: | ---: | --- |
| 90 | 92 | Make Me... - Britney Spears |
| 316 | 317 | Be Our Guest - Ewan McGregor |
| 759 | 758 | Pen Pineapple Apple Pen - Piko-Taro |

仅用于判断重复内容是否主导结果，事后剔除 3 首后得到 112 首敏感性分析：

| 维度 | 尺度 | RMSE | MAE | PCC | CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| Valence | 完整片段 | 0.185085 | 0.152035 | 0.804813 | 0.801790 |
| Arousal | 完整片段 | 0.190463 | 0.148743 | 0.855610 | 0.852849 |
| Valence | 0.5 秒帧 | 0.196156 | 0.160195 | 0.803645 | 0.794733 |
| Arousal | 0.5 秒帧 | 0.201893 | 0.158455 | 0.845732 | 0.835851 |

变化不大，说明三个重复音频没有决定整体结论；但 112 首结果是看到标签后的
敏感性分析，不是新的盲测，不能替代 115 首正式结果。

## 四、基线与数据核查

训练集均值基线在 115 首完整片段上的结果：

| 维度 | RMSE | MAE |
| --- | ---: | ---: |
| Valence | 0.297750 | 0.246363 |
| Arousal | 0.354260 | 0.288093 |

冻结模型显著优于均值基线。标签均在 `[-1, 1]`；开发集和最终测试均从
15.5 秒开始，时间步长严格为 0.5 秒。最终测试集在锁定前使用平均 V、平均 A
和帧数进行分层抽样。这是可接受的预锁定平衡抽样，但论文应披露“标签被算法
用于划分”，不能声称最终测试标签从未被任何程序访问。

## 五、模型与训练参数

共同参数：

```text
projection_dim = 112
BiGRU hidden_dim = 64（双向）
global_hidden_dim = 96
dropout = 0.25
batch_size = 24
AdamW learning_rate = 4e-4
weight_decay = 5e-4
gradient_clip = 1.0
inner validation = 15% song-level split
```

Valence：`local_scale=0.55`，clip-mean loss `0.65`，CCC loss `0.05`，
smoothness loss `0.005`，soft-teacher loss `0.15`，extreme weight `1.15`，
最终拟合 9 epoch。

Arousal：`local_scale=0.70`，clip-mean loss `0.50`，CCC loss `0.03`，
smoothness loss `0.008`，soft-teacher loss `0.15`，extreme weight `1.00`，
最终拟合 7 epoch。

## 六、开发 OOF 的边界

新的 clean model 在每个外层 fold 内使用 song-level inner validation 选择
epoch，再在完整 outer-train 上重拟合并评价 held fold。该拆分和重拟合机制是
严格的。

但是，损失中权重为 `0.15` 的 soft teacher 来自此前长期开发搜索形成的 OOF
预测。此前模型及组合策略反复使用过这 652 首开发数据做选择，因此 clean OOF
不是“所有模型选择完全嵌套”的无偏性能估计，只能作为开发证据。论文主要模型
性能应优先报告已冻结的 115 首最终测试，同时披露内容重复问题。

## 七、任意 MP3 与完整歌曲适用性

从模型结构看，checkpoint 不依赖歌曲 ID、歌词、section 或历史 VA 标签，输入
完全来自音频，因此具备迁移到新 MP3 的必要条件。MERT 可分块提取，BiGRU
也能接收变长序列。

但当前不能把“技术上可推理”写成“完整歌曲指标已验证”：

- 训练和最终测试主要是短片段；多分钟序列属于长度分布外推。
- 最终测试使用 PMEmo 官方 OpenSMILE 特征。三首重提取抽查中，标准化特征
  与官方特征的相关为 `0.773-0.881`；由此造成的单曲预测平均绝对差异约为
  Valence `0.001-0.055`、Arousal `0.014-0.120`。
- 因此，在批量处理 100 首质检歌曲前，应先固定并验证统一的原始 MP3 特征
  提取入口；这不是继续调最终测试，而是部署一致性验证。

最严谨的当前表述是：

> 该模型是面向新音频的纯音频 VA Teacher，可输出 0.5 秒连续 VA；其冻结
> PMEmo 最终片段测试结果如上。模型尚缺少带真实动态 VA 标注的多分钟整歌
> 外部验证，因此对任意完整歌曲的输出应作为 pseudo-VA，而非 ground truth。

## 八、不可再做的操作

- 不得根据 115 首或 112 首结果调参数、改结构、选平滑、做校准或换模型。
- 不得继续反复把这 115 首称为“未见测试集”。
- 如未来训练新模型，115 首只能作为历史外部参照，不能恢复成严格盲测。
- section-aware 和 100 首歌曲数据库写入仍需用户另行确认。

## 九、可复核产物

```text
审计脚本 = scripts/audit_final_teacher_rigor.py
审计 JSON = outputs/final_clean_teacher_v2/FINAL_TEST_RIGOR_AUDIT.json
预测 SHA-256 = f2c3b7ae36e9e12ae15d15c9bce6cda532bb09c9af8b8793ea2901937a3b0b9c
Valence checkpoint SHA-256 = 6cddf9ce3f12f341dac5d53b99c108675fdac9c8c17e323164f77388d88918ae
Arousal checkpoint SHA-256 = aba66d037ec8f4c30917a67dbb9d97f0dfc2fc781fa4191d4202812051c368bb
config SHA-256 = cb32a52d4b7c90f3b41aca17f82ab66cde8d33634497d6862bf0b6ab2b8288a1
```
