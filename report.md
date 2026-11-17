# Qwen 系列模型实验结果分析

基础模型评测分析日期：2026-11-16。重复反馈诊断补充日期：2026-11-17。

## 结论

在当前实验配置下，Qwen3.5-27B 的步骤完成检测最好，Qwen3-VL-30B 的错误召回最高，Qwen3.5-9B 的纠错 F1 最高。三个模型均存在明显不足，尚未同时实现准确的完成检测、高错误召回和低误报。

- **Qwen3.5-27B**：完成检测领先、误报相对较少，但错误召回仅约 9%，主动纠错不足。
- **Qwen3-VL-30B**：检出约 68% 的错误，但误报和重复提醒最多，纠错 F1 低于 9B。
- **Qwen3.5-9B**：纠错 F1 为 0.19，三者最高；完成检测较弱，并受到完成回复格式失败的影响。

## 数据范围与实验配置

本分析读取本项目 `outputs/` 下三组官方评测报告、汇总预测和逐次推理日志。三组预测均覆盖相同的 **39 个视频、562 条指令和 397 条人工错误反馈**，汇总预测与各自单视频预测文件一致。

主要配置一致：

| 配置项 | 设置 |
| --- | --- |
| 推理协议 | per-recipe-step，每步使用给定的人工指令 |
| 查询间隔 | 5 秒 |
| 判断顺序 | 先检查完成，明确未完成时再检查错误 |
| 候选采样率 | 1 FPS |
| 单次输入帧数上限 | 32 |
| 图像最长边 | 448 像素 |
| 生成长度上限 | 256 tokens |
| 错误提示词 | detailed |
| 思考模式 | 关闭（enable_thinking=false） |
| Ego-CoMist 微调 | 未使用 |
| 视频截断 / dry-run | 未启用 |

输入采用带时间标记的历史多图，而非原生视频编码。这些结果属于当前实现的基础模型实验，不应作为论文训练模型数值的严格复现。9B 配置中的输出目录名称与现存目录不同，且记录了 `overwrite=true`；本分析以现存预测和日志为准。

## 官方指标对比

官方报告逐视频输出累计指标，下表取各报告的**最后一个累计区块**，不对中间区块求平均。Precision、Recall 和 F1 保留报告中的舍入精度。

| 指标 | Qwen3.5-9B | Qwen3.5-27B | Qwen3-VL-30B |
| --- | ---: | ---: | ---: |
| 步骤完成 IC-Acc ↑ | 11.4% | **26.1%** | 18.6% |
| 错误检测 Precision ↑ | 12% | **19%** | 10% |
| 错误检测 Recall ↑ | 41% | 9% | **68%** |
| 错误检测 F1 ↑ | **0.19** | 0.12 | 0.17 |
| 错误检测 Accuracy ↑ | 0.12 | **0.20** | 0.12 |
| 反馈 BERTScore ↑ | 0.336 | **0.342** | 0.304 |
| 反馈 ROUGE-L ↑ | 0.191 | **0.205** | 0.179 |
| 正确检出错误 TP | 163 | 36 | **269** |
| 误报 FP ↓ | 1,191 | **149** | 2,420 |
| TN（官方定义） | 38 | 95 | 73 |
| 漏检 FN ↓ | 234 | 361 | **128** |

### 指标口径

- **IC-Acc** 仅在没有人工错误的步骤上计算。预测段必须只有指令和一次 Success，且完成时间与标注终点的绝对差小于 15 秒。因此，额外纠错会使该步骤无法命中完成指标，IC-Acc 不是所有步骤的简单完成分类准确率。
- **错误检测 TP** 根据预测反馈与人工错误反馈的时间差是否小于 15 秒判定，不要求纠错语义正确；文本质量另用 BERTScore 和 ROUGE-L 衡量。
- 同一个人工错误可收到多次预测反馈，但 TP 只计一次；窗口外反馈仍会产生 FP。因此，生成反馈条数不等于 TP 与 FP 之和。
- BERTScore 和 ROUGE-L 只对时间匹配成功的反馈计算，不能代表全部输出的文本质量；27B 的这两个分数优势也不代表它能覆盖更多错误。
- 官方 TN 按部分无错误步骤计数，TP/FN 按错误事件计数，FP 按误报反馈计数；Accuracy 不宜解释为统一采样单位上的常规分类准确率。

## 各模型表现

### Qwen3.5-27B：完成检测最好，主动纠错偏保守

IC-Acc 达到 26.1%，比 9B 高 **14.7 个百分点**，比 30B 高 **7.5 个百分点**。完成判断没有解析失败。

其错误检测只命中 397 个错误中的 36 个，漏掉 361 个，漏检约 91%。虽然 Precision 是三者最高，仍只有约 19%，纠错可靠性仍然有限。

综合来看，27B 适合作为后续步骤完成检测的基线；当前数据不足以支持其作为主动纠错的最佳方案。

### Qwen3-VL-30B：召回最高，误报与重复提醒突出

30B 检出 269 个错误，Recall 约 68%。相比 9B，它多检出 **106 个错误**，同时多出 **1,229 次误报**，使 F1 降至 0.17。

它共生成 3,462 条纠错反馈，约为 27B 的 12 倍。日志中存在大量完全相同的重复提醒，例如关于反复调整火力、确保鸡肉和蔬菜充分煮熟的同一回复出现 122 次。即使提示中提供最近 60 秒反馈，模型仍未稳定遵守防重复要求。

它输出了最多的 Success，但 IC-Acc 低于 27B。完成声明数量不能直接代表完成时机准确，也需要考虑额外纠错对 IC-Acc 的影响。

### Qwen3.5-9B：纠错 F1 最高，完成输出存在格式问题

9B 的错误检测 F1 为 0.19，略高于 30B；匹配反馈的文本分数也高于 30B。但其 Precision 仅约 12%，仍有大量误报。

完成判断中有 **208 次只回复 YES、没有解释**，占完成调用的 **3.14%**，被严格解析器拒绝。当前代码在这种情况下不记录 Success，也不在该查询时刻继续检查错误，之后仍会继续查询。

因此，9B 的低 IC-Acc 同时受到判断能力与输出格式遵循问题影响。不能直接把这些 YES 改为成功事件并宣称性能提升，需先检查判断本身是否正确，再通过独立实验评估格式处理的影响。

## 输出行为与耗时

下表由逐次推理日志和预测文件统计，Success/Feedback 为实际序列化事件数量，不代表正确事件数量。

| 日志统计 | Qwen3.5-9B | Qwen3.5-27B | Qwen3-VL-30B |
| --- | ---: | ---: | ---: |
| 完成判断调用数 | 6,626 | 5,860 | 5,416 |
| 错误判断调用数 | 6,331 | 5,635 | 5,078 |
| 生成 Success 数量 | 87 | 225 | 311 |
| 生成 Feedback 数量 | 1,735 | 288 | 3,462 |
| 完成判断解析失败数 | 208 | **0** | 27 |
| 错误判断解析失败数 | 0 | 0 | 0 |
| 完成调用平均耗时 | 2.32 秒 | 2.14 秒 | **1.80 秒** |
| 错误调用平均耗时 | **1.45 秒** | 2.46 秒 | 2.01 秒 |
| 日志调用累计耗时 | 6.82 小时 | 7.33 小时 | 5.55 小时 |

调用耗时包含输入处理和生成，不是纯模型解码速度。累计值也不包含完整实验的所有开销。三个模型提前结束步骤的情况不同，调用负载和输出长度不同，且未核实硬件条件完全一致，因此不能据此得出严格的速度排名或实时性结论。

## 后续实验建议

1. **拆解完成检测失败原因**：区分提前完成、延迟完成、没有成功事件、额外纠错和格式失败，明确 27B 的优势来源以及 9B 的主要瓶颈。
2. **按内容分析误报**：检查是否将尚未完成、正常准备动作或无法从画面确认的细节误判为错误，重点分析 30B。
3. **量化重复反馈**：统计每个步骤的反馈密度、完全重复率和语义重复率。任何去重策略应单独报告为新实验，保留原始结果。
4. **按视频配对比较**：检查整体差距是否由少数长视频和高频重复反馈主导，并计算置信区间。当前没有显著性分析，不能将 9B 与 30B 的 F1 差距直接视为稳定优势。
5. **基线分工**：完成检测优先使用 27B；纠错综合基线保留 9B；高召回方案保留 30B，重点研究误报与重复提醒。

## 数据与代码来源

- [Qwen3.5-9B 官方报告](outputs/qwen3.5-9b/official-report.txt)
- [Qwen3.5-27B 官方报告](outputs/qwen3.5-27b/official-report.txt)
- [Qwen3-VL-30B 官方报告](outputs/qwen3-vl-30b/official-report.txt)
- 各模型目录中的 `config.json`、`predictions.json`、`<video_id>.prediction.json` 和 `<video_id>.jsonl`
- [官方指标实现](vendor/eval.py)
- [推理实现](run_inference.py)
- [实验协议与复现限制](README.md)

本报告基于已有结果和日志进行分析，未重新运行模型推理或官方 GPU 评分。

## 补充实验：Qwen3.5-9B 重复反馈与指令遵循诊断（2026-11-17）

### 目的、数据来源与边界

本节检查：在历史反馈已经进入提示词的条件下，9B 是否仍对同一次错误再次触发提醒，以及生成的反馈是否与历史逐字相同。采用纯文本观察替代视频，先排除视觉理解的干扰。这是局部诊断，不是完整视频实验或官方基准评测，不能将其通过率与前文的 Precision、Recall、F1 比较。

结果来自用户在服务器执行 `test_9b_following.py` 后提供的终端日志；本次整理没有重新执行真实模型推理。已将两份原始粘贴内容原样归档，包含完整渲染输入、输出及运行警告：

- [实验一：YES/NO 与 A/B/C 对照完整日志](reports/feedback_repeat_2026-11-17/decision_abc.log)
- [实验二：真实反馈生成与重复比较完整日志](reports/feedback_repeat_2026-11-17/repeat_original.log)

归档保留用户粘贴时的文本，包括部分词间空格缺失；不将其视为服务器原始字节级日志。以下反馈原文按日志中的 `RESPONSE`/`second_raw` 展示。模型按用户运行上下文标识为 Qwen3.5-9B；本次日志没有完整记录模型路径、权重修订、Transformers 版本、精度和硬件，因此不能仅凭这些结果比较不同模型规模。

### 实验一：简短规则下的 YES/NO 判断

设置：纯文本、单条 user 消息、`enable_thinking=False`、`do_sample=False`，不携带跨调用 KV 状态。该轮脚本生成上限为 64 tokens，只要求输出 YES/NO，不生成具体纠正反馈。渲染模板中出现空 `<think>...</think>`，与显式关闭 thinking 的调用设置相符，本身不代表启用了思考。

共同规则原文：

```text
You are checking cooking mistakes.
Output exactly YES or NO, with no explanation.
Output YES only if the current observation describes a NEW mistake.
Output NO if there is no mistake, or if the SAME mistake was already reported.
A mistake remaining uncorrected is not a new mistake.
```

每次按 `Previous feedback: <JSON 数组>` 和 `Current observation: <观察>` 输入。七项结果如下，所有回答均符合 YES/NO 格式。

| 条件 | 预期 | 实际 | 判定 |
| --- | --- | --- | --- |
| 正确添加盐，无错误 | NO | NO | 通过 |
| 要求两勺盐，仅加一勺后停止；无历史反馈 | YES | YES | 通过 |
| 历史直接描述同一次少加盐错误，观察明确没有新动作 | NO | NO | 通过 |
| A：历史只描述错误 | NO | NO | 通过 |
| B：历史只给纠正建议 | NO | YES | 失败 |
| C：历史同时描述错误和纠正建议 | NO | NO | 通过 |
| 已纠正盐量，但新出现错加醋的问题 | YES | YES | 通过 |

基础四项的完整输入内容：

| 条件 | Previous feedback | Current observation |
| --- | --- | --- |
| 无错误 | `[]` | The required amount of salt was added correctly. |
| 新错误 | `[]` | The recipe requires two teaspoons of salt, but the person finished after adding only one. |
| 同一次错误，文字直接对应 | `["You added only one teaspoon of salt instead of two."]` | This is the SAME previously reported incident. The person added only one teaspoon instead of two. Nothing new happened. |
| 新出现其他错误 | `["Add one more teaspoon of salt."]` | The person has now corrected the salt amount. Then they added vinegar, although the recipe explicitly requires lemon juice. |

A/B/C 三组除历史反馈外保持完全一致，共同观察原文：

```text
This is the SAME incident. The bowl still contains just one teaspoon of salt. No additional action occurred.
```

三组历史反馈分别为单元素数组：

```text
A: You added only one teaspoon instead of the required two.
B: Add one more teaspoon of salt to reach two teaspoons.
C: You added only one teaspoon instead of two. Add one more teaspoon.
```

结果为 6/7 通过。该组对照支持：在这个案例和简短规则下，9B 能处理直接描述错误的历史，但将纠正建议映射回已报告错误时出现失败。它不能证明模型普遍不具备语义匹配能力。特别是，此轮 B 的 YES 只表示重复触发检测，不能证明会重复输出相同反馈文字，因为此轮根本没有要求生成反馈。

此前第一次五项预试验同样在“同一次错误，换一种说法”项失败，结果为 4/5；其观察、历史与 A/B/C 中 B 相同，属于相同输入的复现，不是新增独立样本，也不计入本节 6/7 的分母。

### 实验二：论文详细提示词下的真实反馈重复

设置：使用项目 `prompt_for('mistake', ..., 'detailed')` 构建的论文详细错误检测提示词，替换当前指令和历史反馈占位符，再追加 `Current observation` 文字观察。没有视频、没有额外 assistant 历史消息。`enable_thinking=False`、`do_sample=False`、`max_new_tokens=256`；输出要求为 `NO` 或 `YES. <feedback>`。

每个场景先以空历史生成首轮反馈；去掉响应的 YES 前缀，将反馈正文原样填入下一轮历史数组。随后做两次相互独立的调用，两者均只使用首轮反馈作为历史，彼此不累积输出：

1. 完全相同观察：指令和观察文字不变，仅将空历史替换为首轮反馈。
2. 几乎相同观察：在第一种条件末尾追加 `No additional action occurred; the state is unchanged.`。

下一轮没有新错误，预期都为 NO。各场景完整指令、观察及首轮输出如下。

**盐量不足**

```text
Instruction: Add two teaspoons of salt to the bowl.
Observation: The person added one teaspoon of salt to the bowl, put the spoon away, and stopped adding salt.
First response: YES. You only added one teaspoon of salt, but the instruction requires two teaspoons. You need to add one more teaspoon to complete the step correctly.
```

**油量不足**

```text
Instruction: Put three teaspoons of sesame oil in the bowl.
Observation: The person added two teaspoons of sesame oil to the bowl, put the spoon away, and stopped adding oil.
First response: YES. You only added two teaspoons of sesame oil, but the instruction requires three teaspoons. Please add one more teaspoon to complete the step correctly.
```

**温度错误**

```text
Instruction: Set the oven to 180 degrees Celsius.
Observation: The person set the oven to 220 degrees Celsius, confirmed the setting, and walked away.
First response: YES. You set the oven to 220 degrees Celsius, but the recipe requires 180 degrees Celsius. This temperature error will likely cause your food to cook too quickly and burn before it is properly done.
```

首轮三项均输出 YES 并包含反馈。下一轮全部结果如下；“逐字相同”比较去掉 YES 前缀后的反馈正文，不比较整份提示词。

| 场景 | 下一轮观察 | 预期 | 实际 | 再次触发 | 与首轮反馈逐字相同 | 文本相似度 |
| --- | --- | --- | --- | --- | --- | --- |
| 盐量不足 | 完全相同 | NO | YES＋首轮原文 | 是 | 是 | 1.000 |
| 盐量不足 | 几乎相同 | NO | YES＋首轮原文 | 是 | 是 | 1.000 |
| 油量不足 | 完全相同 | NO | YES＋首轮原文 | 是 | 是 | 1.000 |
| 油量不足 | 几乎相同 | NO | YES＋首轮原文 | 是 | 是 | 1.000 |
| 温度错误 | 完全相同 | NO | NO | 否 | 不适用 | 不适用 |
| 温度错误 | 几乎相同 | NO | NO | 否 | 不适用 | 不适用 |

四次重复响应的完整文本分别就是上述盐量和油量的首轮响应；两次温度响应均完整输出 `NO`，无反馈正文。重复项的 `exact_same_text` 和 `same_ignoring_case_whitespace` 都为 true，`text_similarity` 都为 1.0。相似度来自字符序列比较，不是语义评测。

在这六个后续调用中，4/6 再次触发且逐字重复，2/6 正确保持静默。按场景看是两个场景的两种观察均失败、一个场景的两种观察均通过。六次调用共享三个场景及各自首轮反馈，不能当作六个独立错误事件；4/6 仅是本次诊断结果，不是实际视频任务的总体重复率。

### 综合分析

**已被直接支持的结论：**

1. 在该运行设置下，纯文本输入即可复现逐字重复。视频、长视频上下文和异步缓存竞争不是这次复现的必要条件；这不排除它们影响完整视频实验。
2. 反馈正文已进入渲染输入中的历史数组，因此不能用“历史根本没传进去”解释此次输出。当前日志仍不是完整 token 张量或模型内部注意力证据，不能据此声称已证明模型内部如何处理历史。
3. 模型并非完全不使用去重规则：温度场景能正确输出 NO，但盐和油场景未能遵守。样本不足以断言所有计量错误都比温度错误难。
4. 实验一提出的“建议型反馈难匹配错误”只能解释局部现象：实验二中盐和油的历史均包含错误事实与建议，仍逐字重复，因此单纯把反馈改成“事实＋建议”不能视为已验证的解决方案。

**仍需验证的解释：**详细提示词同时要求“不重复已检测错误”和“观察到错误就回答 YES”，但没有明确规定“错误依然存在、已经提醒、没有新错误”时的决策优先级。模型可能优先响应可见错误而忽略历史排除条件。该解释与结果相容，尚未通过提示词消融证实。实验一与实验二还同时改变了提示词长度、措辞、输出任务和具体历史，因此不能直接把两轮差异归因于某一个因素。

当前未比较其他模型规模、thinking 开关或多种措辞的大样本结果，不能判定主要原因是 9B 参数量不足、模型通用能力不足、提示词长度，或生成模式。确定性解码下再次运行同一输入不是独立统计证据。

### 待运行对照：追加明确的去重决策规则

测试脚本现已增加 `original` 与 `explicit_dedup` 两种后续调用。两者共用原提示词生成的同一条首轮反馈和同一观察，仅后者在整份输入末尾追加：

```text
Before answering, compare the current mistake with the previous feedback.
If the same mistake was already reported and no new mistake occurred,
answer NO, even if the mistake remains uncorrected.
Answer YES only for a new, previously unreported mistake.
```

三场景 × 两种观察 × 两种策略，共 12 次后续比较，加上 3 次首轮生成。脚本保留原始输出，分别汇总重复触发、逐字重复和解析失败。运行方式：

```bash
CUDA_VISIBLE_DEVICES=2 python test_9b_following.py \
  --model-path /实际模型绝对路径/Qwen3.5-9B \
  --mode repeat
```

**这组追加规则的真实模型结果尚未提供，不能报告去重率提升。** 已完成的是本地模拟响应验证：确认配对输入只差追加规则、历史反馈共享、重复输出与 NO 可分别记录。追加规则会改变论文提示词，应作为独立对照，不改写原基线结果。若新规则有效，还需增加“已纠正后再次犯错”“出现不同新错误”等负向控制，排除模型仅仅更倾向于输出 NO。

原始测试日志包含缺少 `causal_conv1d`、`flash-linear-attention` 时回退到参考 PyTorch 实现的警告，其文字明确标注正确但更慢。日志没有提供证据表明这些警告导致本次重复反馈，也没有开展优化内核与参考实现的一致性对照。
