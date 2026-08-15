# 当前音乐情绪组织结论记录

最后更新：2026-08-15

用途：这个文档专门记录 section-aware VA / MIR / MER 分析目前已经形成的论文结论。以后发现新结论、需要修正旧结论，统一往这里追加。新增内容必须同时满足：

- 有统计证据支撑。
- 有音乐学解释价值。
- 不超出样本和模型的证据边界。

## 证据边界

- 旧版对照数据：`exports/gpt_section_va_data_package_20260802` 中的 97 首中文主分析语料。
- 当前新版数据：`exports/gpt_top60_quota_100_section_va_data_package_20260815` 中的 QQ 音乐歌手榜 top60 配额 100 首语料。
- 当前 teacher 输出：100 首歌、1046 个经人工质检的 section、51448 行 frame-level VA。
- 模型边界：VA 数值是 teacher 模型生成的 pseudo-label，适合做音乐情绪结构分析，但不能写成直接的人类情绪真值。
- 抽样边界：100 首是榜单配额样本，不是严格随机样本。论文中应写“在本样本中支持”或“提示中文流行歌曲可能存在……机制”，不要写成对所有中文流行歌曲的总体证明。

## 总论：情绪组织核心是能量戏剧，而不是单纯快乐-悲伤戏剧

当前数据支持的最大结论是：这些中文流行歌曲的情绪组织核心是 **arousal-led energy drama**，也就是由激活度驱动的能量戏剧，而不是单纯的 happy-sad / positive-negative 戏剧。

副歌最稳定的音乐情绪功能不是单纯“变得更积极”，而是把音乐推向更高的激活水平。Valence 通常也会上升，但 Arousal 对大尺度段落张力的解释更稳定。

统计证据：

- 97 首：Chorus - Verse 的 Arousal 为 `+0.266`，Valence 为 `+0.101`。
- 100 首：Chorus - Verse 的 Arousal 为 `+0.279`，95% CI `[+0.251, +0.308]`，p 约 `4.1e-18`，`99.0%` 歌曲为正向。
- 100 首：Chorus - Verse 的 Valence 为 `+0.119`，95% CI `[+0.100, +0.138]`，p 约 `5.9e-16`，`89.0%` 歌曲为正向。

推荐写法：

> 在本样本中，副歌入口最稳定的情绪功能是 arousal-led arrival：副歌通常同时提高 Valence 与 Arousal，但其更核心、更稳定的形式功能是激活度强化。

## 结论 1：Verse 是低激活叙事基底

Verse 很可能是整首歌真正的低能量叙事基底。很多歌不是从 Intro 开始一路爬升，而是 Intro 建立声音场景后，Verse 先收束到较低激活的叙事空间。

统计证据：

- 100 首：Intro->Verse 的 Arousal 变化为 `-0.058`，95% CI `[-0.084, -0.032]`。
- 100 首：Intro->Verse 的 Valence 变化为 `-0.004`，95% CI `[-0.021, +0.012]`。

解释边界：

这个结论主要是 Arousal 层面的，不能写成“Verse 让情绪整体变差”或“Verse 让歌曲更悲伤”。更准确的是：Verse 降低激活度，形成叙事基底。

推荐写法：

> Verse 不是线性爬坡的第一阶，而是 Intro 之后形成的低激活叙事基底。

## 结论 2：Pre-chorus 非必需，但功能稳定

Pre-chorus 不是中文流行歌曲的必备结构，但一旦出现，它的情绪功能非常稳定：承担蓄势、准备、导向 Chorus 的作用。

统计证据：

- 100 首：Pre-chorus 出现在 `38/100` 首中。
- Verse->Pre-chorus Arousal `+0.107`，95% CI `[+0.074, +0.138]`。
- Pre-chorus->Chorus Arousal `+0.154`，95% CI `[+0.126, +0.187]`。
- Paired Pre-chorus - Verse Arousal `+0.135`，95% CI `[+0.096, +0.173]`。
- Paired Chorus - Pre-chorus Arousal `+0.200`，95% CI `[+0.162, +0.243]`。

音乐学解释：

Pre-chorus 可以被解释为“可选但高度稳定的张力准备段”。它不是形式上的必要段落，却是功能上稳定的蓄势区。

## 结论 3：进入副歌存在两条主要路径

新版 100 首数据新增/强化了一个结论：进入 Chorus 不只有 Pre-chorus 蓄势后一种路径，也存在大量 Verse 直接进入 Chorus 的能量抬升。

统计证据：

- Verse->Chorus：116 次，Arousal 平均 `+0.155`，`94.0%` 为正向。
- Pre-chorus->Chorus：76 次，Arousal 平均约 `+0.154`，`94.7%` 为正向。

音乐学解释：

中文流行歌曲可以通过两种方式实现副歌到达：

- 直接路径：Verse->Chorus。
- 分阶段路径：Verse->Pre-chorus->Chorus。

两者都服务于同一个 arousal-led arrival 功能。

## 结论 4：不是单向爬升，而是局部循环 + 全局累积

整首歌不是一次性的 Verse->Chorus 爬坡，而是反复旋转的情绪循环。更准确的模型是：局部形式循环控制每一轮内部的起伏，全局累积让后续轮次整体更强。

统计证据：

- 97 首：78 种 raw section sequence，53 种 collapsed sequence。
- 100 首：76 种 raw section sequence，50 种 collapsed sequence。
- 100 首 repeated section 首末 Arousal 变化：
  - Chorus `+0.225`，`90.0%` 正向。
  - Verse `+0.165`，`82.4%` 正向。
  - Pre-chorus `+0.156`，`91.9%` 正向。
  - Interlude `+0.260`，`93.9%` 正向。

音乐学解释：

歌词可能沿着线性叙事向前推进，但音乐形式常通过 Verse-Chorus-Verse-Chorus 的反复轮转回到相似位置。重要的是，每次回到相似形式位置时，情绪并不是复刻，而是在新的全局进程中被重新加权。

推荐写法：

> 中文流行歌曲的音乐情绪组织更接近“局部循环 + 全局累积”的双层结构，而不是单条线性情绪曲线。

## 结论 5：重复 section 不是情绪复刻，而是重新强化

同样的 section 类型重复出现时，并没有简单复刻前一次的情绪状态，而是在整首歌推进中被重新强化。

统计证据：

- Chorus first-to-last Arousal：`+0.225`，`90.0%` 正向。
- Verse first-to-last Arousal：`+0.165`，`82.4%` 正向。
- Pre-chorus first-to-last Arousal：`+0.156`，`91.9%` 正向。
- Interlude first-to-last Arousal：`+0.260`，`93.9%` 正向。

音乐学解释：

情绪累积发生在整个形式循环系统中，而不是副歌孤立强化。

## 结论 6：高潮偏向副歌和后部，但不是副歌垄断

“高潮偏后置”仍然成立，但需要收窄表述。新版数据支持“Chorus 是最常见的最高 Arousal 位置”，但不支持“所有高潮都由 Chorus 承担”。

统计证据：

- 97 首：最高 mean-Arousal section 出现在 Chorus 的歌曲为 `53/97`。
- 100 首：最高 mean-Arousal section 出现在 Chorus 的歌曲为 `53/100`。
- 100 首非 Chorus 最高 Arousal 分布：
  - Interlude `19/100`。
  - Bridge `13/100`。
  - Outro `10/100`。
  - Intro `4/100`。
  - Verse `1/100`。
- 100 首：`53/100` 的歌曲在最后四分之一达到最高 mean-Arousal section。

推荐写法：

> 高潮具有副歌偏向和后部偏向，但不是副歌垄断；Interlude、Bridge、Outro 也会承担 peak arousal。

## 结论 7：Bridge 是高张力的情境偏离区，而不是固定方向转折段

Bridge 的作用不是固定地升高或降低情绪，而是脱离此前反复核心循环，制造上下文差异。新版数据还提示 Bridge 经常处于较高张力区域。

统计证据：

- 100 首：Bridge 出现在 `43/100` 首中。
- Paired Bridge - Chorus Arousal `+0.088`，95% CI `[+0.040, +0.140]`。
- Chorus->Bridge Arousal 的 CI 跨 0，约 `[-0.008, +0.070]`。
- Bridge->Chorus Arousal 的 CI 也跨 0，约 `[-0.003, +0.052]`。

音乐学解释：

Bridge 可以写成“高张力的情境偏离区”，但不能写成稳定的上升段或下降段。它的方向依赖上下文。

## 结论 8：Interlude 是 rotation hinge，不是简单休息段

Interlude 不能简单等同于副歌之后降低能量的休息区。它更像一轮核心循环之后连接下一轮 Verse / Chorus 的枢纽。

统计证据：

- 100 首：Interlude 出现在 `85/100` 首中。
- Chorus->Interlude Arousal `+0.010`，95% CI `[-0.011, +0.032]`，并没有稳定下降。
- Interlude->Verse Arousal `-0.048`，95% CI `[-0.080, -0.017]`。
- Repeated Interlude first-to-last Arousal `+0.260`，`93.9%` 正向。
- 最高 mean-Arousal section 出现在 Interlude 的歌曲为 `19/100`。

音乐学解释：

Interlude 往往承担 rotation hinge：它可以延续或转移 Chorus 后的能量，并把系统带回下一轮 Verse 的低激活叙事区。

## 结论 9：Outro 是 closure release，不是回到开头

Outro 的功能不是简单回到 Intro，而是从此前高能量状态释放和收束。Closure 本身是重要的情绪事件。

统计证据：

- 100 首：Chorus->Outro Arousal `-0.188`，95% CI `[-0.230, -0.147]`。
- 100 首：Chorus->Outro Valence `-0.132`，95% CI `[-0.159, -0.105]`。
- 100 首：Outro - Intro Arousal `+0.186`，95% CI `[+0.131, +0.243]`。
- 100 首：Outro - Intro Valence `+0.036`，CI 跨 0。

音乐学解释：

Outro 通常从高点释放，但并不回到开头状态。歌曲最剧烈的情绪变化不仅可能发生在进入高潮，也可能发生在退出高潮。

## 结论 10：Valence 和 Arousal 多数同向，但不能等价

Valence 和 Arousal 在多数歌曲中同向变化，但不能合并成一个情绪轴。

统计证据：

- 97 首：song-level section-wise Spearman 中位数 `rho = 0.736`，5 首为负相关。
- 100 首：song-level section-wise Spearman 中位数 `rho = 0.751`，3 首为负相关。

音乐学解释：

Arousal 更适合解释形式张力和段落功能；Valence 更适合解释情绪色彩、积极/消极倾向或情感质地。

## 当前最强论文命题

> 在这组 100 首中文流行歌曲榜单配额样本中，音乐情绪组织主要表现为 arousal-led energy drama。Verse 提供低激活叙事基底；可选的 Pre-chorus 承担稳定蓄势；Chorus 是主要能量到达点；Interlude 是形式轮转枢纽；Bridge 形成高张力的上下文偏离；Outro 则从此前高能量状态释放。重复 section 并非情绪复刻，而是在多次形式轮转中被重新加权，由此形成“局部循环 + 全局累积”的双层结构。

## 需要避免或限定的说法

- 避免：“中文流行歌曲都遵循同一种情绪模板。”
  - 改成：“本样本显示出若干共享的局部情绪组织机制，但整首歌路径高度异质。”
- 避免：“副歌总是高潮。”
  - 改成：“副歌是最常见的 peak arousal 位置，但高潮也分布在 Interlude、Bridge、Outro 等 section 中。”
- 避免：“Bridge 一定上升/一定下降。”
  - 改成：“Bridge 是方向异质的上下文偏离区。”
- 避免：“Interlude 是休息段。”
  - 改成：“Interlude 更常承担 rotation hinge，可保持、转移或重新导向能量。”
- 避免：“Outro 回到开头。”
  - 改成：“Outro 通常从高能量状态释放，并且平均 Arousal 仍高于 Intro。”

## 后续发现记录区

以后新增结论统一追加在这里，格式如下：

```text
日期：
发现：
统计证据：
音乐学解释：
状态：保留 / 修正 / 推翻 / 证据不足
```

