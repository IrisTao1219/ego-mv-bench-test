# Qwen 系列模型实验结果分析

## 2026-11-19：新增评测指标汇总（custom_metrics v1）

以下结果直接提取自各实验 `report.txt` 末尾的 Supplementary metrics JSON，未重新运行模型或评测。实验名称使用文件夹名称，视频覆盖不足 39/39 的实验在名称后标注“（未完成）”，状态以本节源报告为准；仅统计已提交视频，不把缺失视频计为失败。per-step 与 full-recipe 分表展示，各表内按模型前缀集中排列，视频列保留各实验覆盖范围，不能跨覆盖范围直接排名，也不能把 full-recipe 与 per-step 当作相同协议比较。比例统一换算为百分数并保留两位小数；时间单位为秒。原始精度见源报告。

### 官方评测指标

取各实验报告最后一组累计输出，检测窗口均为 15 秒。优先使用与下方补充指标相同的 `report.txt` 快照；仅有 `official-report.txt` 的实验也列入本区块。IC-Acc 使用百分数，其余指标保留官方输出的小数精度；TP/FP/TN/FN 均为错误检测计数，与补充指标的完成检测计数不同。未完成标记按该份报告的覆盖数确定，不代表当前推理进度。

#### per-step

| 实验 | 视频 | IC-Acc | 错误 TP | 错误 FP | 错误 TN | 错误 FN | Mistake Acc | Precision | Recall | F1 | BERT | Rouge-L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-per-step | 39/39 | 18.6% | 269 | 2420 | 73 | 128 | 0.12 | 0.10 | 0.68 | 0.17 | 0.304 | 0.179 |
| qwen3-vl-30b-per-step-dedup（未完成） | 20/39 | 19.9% | 110 | 728 | 37 | 41 | 0.16 | 0.13 | 0.73 | 0.22 | 0.295 | 0.172 |
| qwen3-vl-30b-per-step-dedup-manual | 39/39 | 18.2% | 305 | 2481 | 64 | 92 | 0.13 | 0.11 | 0.77 | 0.19 | 0.295 | 0.167 |
| qwen3.5-27b-per-step | 39/39 | 26.1% | 36 | 149 | 95 | 361 | 0.20 | 0.19 | 0.09 | 0.12 | 0.342 | 0.205 |
| qwen3.5-27b-per-step-dedup-manual（未完成） | 20/39 | 24.8% | 2 | 1 | 53 | 149 | 0.27 | 0.67 | 0.01 | 0.03 | 0.410 | 0.337 |
| qwen3.5-9b-per-step | 39/39 | 11.4% | 163 | 1191 | 38 | 234 | 0.12 | 0.12 | 0.41 | 0.19 | 0.336 | 0.191 |
| qwen3.5-9b-per-step-dedup | 39/39 | 5.3% | 171 | 631 | 18 | 226 | 0.18 | 0.21 | 0.43 | 0.29 | 0.336 | 0.197 |

#### full-recipe

| 实验 | 视频 | IC-Acc | 错误 TP | 错误 FP | 错误 TN | 错误 FN | Mistake Acc | Precision | Recall | F1 | BERT | Rouge-L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-full-recipe（未完成） | 16/39 | 12.2% | 35 | 967 | 22 | 67 | 0.05 | 0.03 | 0.34 | 0.06 | 0.300 | 0.165 |
| qwen3.5-27b-full-recipe（未完成） | 7/39 | 13.5% | 8 | 271 | 23 | 28 | 0.09 | 0.03 | 0.22 | 0.05 | 0.366 | 0.200 |

补充来源（仅官方指标）：[qwen3.5-9b-per-step-dedup](outputs/qwen3.5-9b-per-step-dedup/official-report.txt)。其余源报告链接见本节末尾。

### 覆盖范围与完成检测

完成检测目标为全部参考步骤（包括含错误步骤）。按步骤序号对齐，并采用官方指令匹配规则；每个步骤最多贡献一个 TP。指令匹配且 Success 与参考步骤结束时间的绝对差严格小于 15 秒时记为 TP；其他 Success 为 FP，未命中的参考步骤为 FN。该指标衡量步骤结束检测，不代表发生错误的操作已经正确完成。

#### per-step

| 实验 | 视频 | 参考步骤 | 无错误步骤 | TP | FP | FN | Precision@15s | Recall@15s | F1@15s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-per-step | 39/39 | 562 | 264 | 167 | 144 | 395 | 53.70% | 29.72% | 38.26% |
| qwen3-vl-30b-per-step-dedup（未完成） | 20/39 | 254 | 141 | 74 | 45 | 180 | 62.18% | 29.13% | 39.68% |
| qwen3-vl-30b-per-step-dedup-manual | 39/39 | 562 | 264 | 159 | 104 | 403 | 60.46% | 28.29% | 38.55% |
| qwen3.5-27b-per-step | 39/39 | 562 | 264 | 137 | 88 | 425 | 60.89% | 24.38% | 34.82% |
| qwen3.5-27b-per-step-dedup-manual（未完成） | 20/39 | 254 | 141 | 58 | 41 | 196 | 58.59% | 22.83% | 32.86% |
| qwen3.5-9b-per-step | 39/39 | 562 | 264 | 55 | 32 | 507 | 63.22% | 9.79% | 16.95% |

#### full-recipe

| 实验 | 视频 | 参考步骤 | 无错误步骤 | TP | FP | FN | Precision@15s | Recall@15s | F1@15s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-full-recipe（未完成） | 16/39 | 177 | 98 | 34 | 51 | 143 | 40.00% | 19.21% | 25.95% |
| qwen3.5-27b-full-recipe（未完成） | 7/39 | 83 | 52 | 17 | 37 | 66 | 31.48% | 20.48% | 24.82% |

### 时间命中率与完成时间误差

命中率 @5/15/30 秒均以全部参考步骤为分母，无 Success 或指令不匹配视为未命中，边界采用严格小于。Hit@15s 与完成检测 Recall@15s 数值相同。时间误差仅统计指令匹配且有 Success 的步骤，每步取第一次 Success，包含窗口外的预测；有符号误差＝预测时间−参考结束时间，负值表示提前、正值表示延迟。因此时间误差较小不代表漏检更少，需同时看样本数和召回率。

#### per-step

| 实验 | Hit@5s | Hit@15s | Hit@30s | 时间误差样本数 | 平均有符号误差(s) | 平均绝对误差(s) | 绝对误差中位数(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-per-step | 12.99% | 29.72% | 43.42% | 311 | -23.15 | 25.73 | 12.00 |
| qwen3-vl-30b-per-step-dedup（未完成） | 12.60% | 29.13% | 38.58% | 119 | -12.83 | 17.47 | 10.00 |
| qwen3-vl-30b-per-step-dedup-manual | 12.99% | 28.29% | 38.26% | 263 | -17.11 | 21.63 | 10.00 |
| qwen3.5-27b-per-step | 16.01% | 24.38% | 30.78% | 225 | -21.25 | 28.07 | 8.97 |
| qwen3.5-27b-per-step-dedup-manual（未完成） | 16.14% | 22.83% | 31.50% | 99 | -13.98 | 23.03 | 10.90 |
| qwen3.5-9b-per-step | 5.87% | 9.79% | 12.63% | 87 | -17.84 | 21.06 | 8.00 |

#### full-recipe

| 实验 | Hit@5s | Hit@15s | Hit@30s | 时间误差样本数 | 平均有符号误差(s) | 平均绝对误差(s) | 绝对误差中位数(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-full-recipe（未完成） | 9.04% | 19.21% | 27.12% | 85 | 44.38 | 56.21 | 25.50 |
| qwen3.5-27b-full-recipe（未完成） | 9.64% | 20.48% | 34.94% | 54 | 35.63 | 47.84 | 27.14 |

### 无错误步骤误报率与重复反馈

无错误步骤误报率＝包含至少一条 Feedback 的无错误参考步骤数÷无错误参考步骤数（即使指令不匹配也计入）。逐字重复仅在同一视频、同一预测步骤内比较去掉首尾空白的 Feedback 正文，每个正文第一次之后的出现均算重复；重复率＝重复条数÷全部 Feedback 条数，不代表语义重复率。

#### per-step

| 实验 | 误报步骤/无错误步骤 | 步骤误报率 | Feedback 总数 | 重复条数 | 重复率 |
| --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-per-step | 150/264 | 56.82% | 3462 | 2905 | 83.91% |
| qwen3-vl-30b-per-step-dedup（未完成） | 95/141 | 67.38% | 1062 | 614 | 57.82% |
| qwen3-vl-30b-per-step-dedup-manual | 176/264 | 66.67% | 3563 | 2358 | 66.18% |
| qwen3.5-27b-per-step | 8/264 | 3.03% | 288 | 239 | 82.99% |
| qwen3.5-27b-per-step-dedup-manual（未完成） | 0/141 | 0.00% | 4 | 0 | 0.00% |
| qwen3.5-9b-per-step | 90/264 | 34.09% | 1735 | 1426 | 82.19% |

#### full-recipe

| 实验 | 误报步骤/无错误步骤 | 步骤误报率 | Feedback 总数 | 重复条数 | 重复率 |
| --- | --- | --- | --- | --- | --- |
| qwen3-vl-30b-full-recipe（未完成） | 21/98 | 21.43% | 1102 | 1034 | 93.83% |
| qwen3.5-27b-full-recipe（未完成） | 9/52 | 17.31% | 306 | 274 | 89.54% |

### 结果解读与来源

- 完整覆盖的 `qwen3-vl-30b-per-step` → `qwen3-vl-30b-per-step-dedup-manual`：完成 Precision 从 53.70% 升至 60.46%，Recall 从 29.72% 降至 28.29%，F1 从 38.26% 微升至 38.55%；无错误步骤误报率从 56.82% 升至 66.67%。因此完成 Precision 提升不能解释为完成覆盖能力全面提升。
- 同一对照中，重复率从 83.91% 降至 66.18%，但 Feedback 总数从 3462 增至 3563：逐字重复减少与反馈总量下降并不是同一件事。
- `qwen3.5-27b-per-step-dedup-manual`（未完成） 的重复率为 0%，但仅有 4 条 Feedback，且只覆盖 20 个视频，不能仅凭零重复判断纠错能力优秀。
- 当前 `qwen3.5-9b-per-step-dedup` 有官方报告，但未找到包含 Supplementary metrics 的报告，因此本节不补造该实验的新增指标。
- 实现与完整计算口径见 [custom_metrics.py](custom_metrics.py)，评测入口为 [evaluate_official.py](evaluate_official.py)。所有报告的 `metric_version=1`、`scope=submitted_videos_only`；未定义的比例或空时间样本使用 null，表中显示为“—”。

- [qwen3-vl-30b-per-step](outputs/qwen3-vl-30b-per-step/report.txt)（39/39 视频）
- [qwen3-vl-30b-per-step-dedup（未完成）](outputs/qwen3-vl-30b-per-step-dedup/report.txt)（20/39 视频）
- [qwen3-vl-30b-per-step-dedup-manual](outputs/qwen3-vl-30b-per-step-dedup-manual/report.txt)（39/39 视频）
- [qwen3-vl-30b-full-recipe（未完成）](outputs/qwen3-vl-30b-full-recipe/report.txt)（16/39 视频）
- [qwen3.5-27b-per-step](outputs/qwen3.5-27b-per-step/report.txt)（39/39 视频）
- [qwen3.5-27b-per-step-dedup-manual（未完成）](outputs/qwen3.5-27b-per-step-dedup-manual/report.txt)（20/39 视频）
- [qwen3.5-27b-full-recipe（未完成）](outputs/qwen3.5-27b-full-recipe/report.txt)（7/39 视频）
- [qwen3.5-9b-per-step](outputs/qwen3.5-9b-per-step/report.txt)（39/39 视频）

---

2026-11-18 代码接入更新：根据四组文本诊断，将真实 per-step/full-recipe 输入改为单条 user 消息；详细错误提示词中的整步反馈数组和明确规则保留，完成检测不再接收 assistant 反馈历史。配置新增消息结构及反馈注入策略字段。诊断脚本现只保留单条 user 的原提示词/明确规则两组比较；simple 模板仍无历史数组，不额外改写原文。启动脚本使用新的 `-single-user` 目录并取消旧目录续跑，避免合并不同条件结果。31 项测试通过；尚无此项修改后的真实视频结果，下文已有视频分析对应修改前的消息结构。

## 2026-11-18：9B 明确规则 × assistant 历史四组对照结果

数据来源：[完整结构化输出](outputs/following-test/9b-factorial.json)。状态为 complete，含 39 次生成记录和 36 项比较；四组汇总已与逐项结果重新计算核对一致。模型路径记录为 `../../Data/Hf_model/Qwen3.5-9B`，thinking 关闭，确定性解码，生成上限 256 tokens。每组 9 项比较：三个场景各有两个同事件重复条件（共 6 项，预期 NO），以及一个纠正后新一次执行再次犯错条件（共 3 项，预期 YES）。沿用同一场景的首轮反馈，各条件不互相累积。全部响应解析成功。

| 提示规则 | 消息结构 | 同事件逐字重复（越少越好） | 新一次错误检出（越多越好） | 总通过数 |
| --- | --- | --- | --- | --- |
| 原提示词 | 单条 user | 4/6 | 3/3 | 5/9 |
| 原提示词 | 增加 assistant 历史 | 4/6 | 3/3 | 5/9 |
| 明确规则 | 单条 user | 0/6 | 2/3 | 8/9 |
| 明确规则 | 增加 assistant 历史 | 0/6 | 0/3 | 6/9 |

逐项看，同事件盐量和油量在原提示词的两种消息结构下均逐字重复，温度均正确回答 NO；明确规则使三场景的同事件条件全部回答 NO。新一次执行再次犯错的盐量、油量、温度，在原提示词两组均检出；明确规则＋单条 user 检出盐和油、漏掉温度；明确规则＋assistant 历史三个场景均回答 NO，全部漏检。

**结论：**在本组小规模文本样本中，明确规则有效减少重复，但不能只看零重复：它也会压制新发生的同类错误。新增 assistant 历史单独使用没有改善，叠加明确规则后没有额外去重收益，却使新事件检出从 2/3 降至 0/3。四组中按本测试总通过数计，明确规则＋单条 user 最好（8/9），并非两项叠加最好；样本仅三个场景，不能据此做总体性能或统计显著性结论。

控制项特别说明已纠正、正在新一次独立执行，故即使反馈正文与历史相同，仍应允许提醒。本实验测的是事件去重与新事件保留之间的取舍；纯粹“同一步永不重复同一句话”的硬过滤也可能拦截这种新事件，需按产品目标确定策略。由于新实例沿用同一指令且没有 event_id、resolved 等结构化状态，这组失败支持继续澄清事件作用域，不能证明具体内部机制。结果也未检验不同错误类型的新事件、视频输入或完成检测。

建议先在后续独立实验中去掉额外 assistant 历史，仅保留提示词内的历史数组＋明确规则，并明确“同一次事件”与“纠正后再次发生”的区别，再复测控制项。不要将此次 9B 文本结果直接解释成 30B 视频结果的原因，也不要覆盖已有正式运行。当前仅更新分析，未修改正式推理输入结构。

## 2026-11-18 最新更新：共同完成四视频的配对比较

本次重新读取 `outputs/qwen3-vl-30b-per-step` 和 `outputs/qwen3-vl-30b-per-step-explicit-dedup`。新组汇总已包含 GX010021、GX010023、GX010029、GX010030 四个完整视频；仅比较这四个共同视频，第五个未进入汇总的视频不计入。逐字重复口径仍为同视频、同 step_id 内已接受 Feedback 正文的严格字符串重复，不归一化、不作语义判断。四视频单文件预测与各组汇总条目均一致。下面两视频分析保留为此前阶段快照，以本节四视频结果为最新结论。

| 视频 | 反馈数 原→新 | 精确重复数 原→新 | 重复占比 原→新 | Success 数 原→新 |
| --- | --- | --- | --- | --- |
| GX010021 | 44→25 | 33→8 | 75.0%→32.0% | 5→6 |
| GX010023 | 7→18 | 1→4 | 14.3%→22.2% | 7→5 |
| GX010029 | 26→17 | 18→4 | 69.2%→23.5% | 8→10 |
| GX010030 | 23→21 | 17→5 | 73.9%→23.8% | 7→7 |
| 合计 | 100→81 | 69→21 | 69.0%→25.9% | 27→28 |

| 其他统计 | 原组 | 新组 |
| --- | ---: | ---: |
| 完成检测调用 | 292 | 308 |
| 错误检测调用 | 265 | 280 |
| 按步骤去除逐字重复后的反馈数 | 31 | 60 |
| 解析失败 | 0 | 0 |
| 触及生成上限 | 0 | 0 |

**主要发现：**三个视频的重复条数减少，一个视频增加；新增两视频均延续减少趋势。总体重复条数减少 48 条（69.6%），重复占比下降 43.1 个百分点，反馈总数只减少 19%。原因在算术上很清楚：重复少了 48 条，同时按步骤计数的不同反馈正文多了 29 条（31→60），两者相抵净少 19 条。这里“不同正文”不代表不同错误或正确反馈，不能据此声称召回或精度提高。

**新增视频的变化位置：**

- GX010029：step 3 的重复从 9→2，step 8 从 9→2，正好构成总重复减少的 14 条。step 8 完成时间却从 315.12 秒延至 325.12 秒，说明该处重复减少不是仅靠提早结束步骤实现的。
- GX010030：step 1 重复从 15→2，step 8 从 2→1，但 step 4 新增 2 条重复，合计由 17 降到 5。step 1 原组在 150.955 秒输出 Success，新组直到该步骤查询结束也没有 Success；该步反馈仍从 16 降到 4。step 4 完成时间从 255.815 延至 275.815 秒，反馈从 0 增至 5，抵消了部分总量收益。

进一步按 `(step_id, timestamp)` 对齐两组错误检测：共有 251 个检测时刻，原组输出 88 条反馈，新组 71 条。另有原组独有的 14 次错误检测、输出 12 条反馈，以及新组独有的 29 次错误检测、输出 10 条反馈。即在相同时刻，新组总体也少输出反馈，变化并非完全由查询次数差异造成。此对齐不保证输入完全一致：历史消息和此前模型决策已经可能不同。

**对因果判断的修正：**四视频原组均未触及 256-token 上限（各视频最大输出分别为 102、84、93、119 tokens），不能用生成截断解释本次差异。按当前模型不支持 thinking 的前提，也不将 off/default 的参数写法差异视作已成立的解释。最明确、仍未隔离的输入变化是新组额外加入 assistant 历史消息，并向完成检测提供反馈历史；原组只在错误提示词数组内提供历史。追加明确规则与历史消息变化同时发生，不能将全部改善归因于规则本身。

新组仍无官方评测报告，本次不报告新 IC-Acc、错误检测 Precision/Recall/F1 或文本指标。Success 数量 27→28 不是完成准确率。下一步应继续比较全量配对结果，并在同一代码和历史策略下只切换 paper/explicit，以隔离追加规则；若还要隔离 assistant 历史作用，需另做消息结构对照。当前结论是：四视频层面精确重复减少的趋势得到更多样本支持，但重复没有消失，检测质量是否改善仍未知。

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

### 实验三：追加明确的去重决策规则

测试脚本现已增加 `original` 与 `explicit_dedup` 两种后续调用。两者共用原提示词生成的同一条首轮反馈和同一观察，仅后者在整份输入末尾追加：

```text
Before answering, compare the current mistake with the previous feedback.
If the same mistake was already reported and no new mistake occurred,
answer NO, even if the mistake remains uncorrected.
Answer YES only for a new, previously unreported mistake.
```

三场景 × 两种观察 × 两种策略，共 12 次后续比较，加上 3 次首轮生成。脚本保留原始输出，分别汇总重复触发、逐字重复和解析失败。运行方式：

```bash
CUDA_VISIBLE_DEVICES=2 python -m tests.test_9b_following \
  --model-path /实际模型绝对路径/Qwen3.5-9B \
  --mode repeat
```

用户随后提供了本组真实模型运行日志，已原样归档为 [追加规则对照完整日志](reports/feedback_repeat_2026-11-17/repeat_explicit_dedup.log)。以下从日志末尾的 12 项 JSON 结果逐项核对，并与策略汇总一致；本地未重新运行模型。

| 场景 | 观察 | 原提示词 original | 追加规则 explicit_dedup |
| --- | --- | --- | --- |
| 盐量不足 | 完全相同 | YES，逐字重复首轮反馈 | NO |
| 盐量不足 | 几乎相同 | YES，逐字重复首轮反馈 | NO |
| 油量不足 | 完全相同 | YES，逐字重复首轮反馈 | NO |
| 油量不足 | 几乎相同 | YES，逐字重复首轮反馈 | NO |
| 温度错误 | 完全相同 | NO | NO |
| 温度错误 | 几乎相同 | NO | NO |

| 统计项 | original | explicit_dedup |
| --- | --- | --- |
| 后续调用数 | 6 | 6 |
| 有效解析数 | 6 | 6 |
| 重复触发数 | 4 | 0 |
| 逐字重复数 | 4 | 0 |
| 符合预期的 NO | 2 | 6 |
| 解析失败数 | 0 | 0 |

本轮首轮输出及 original 分支的盐、油重复正文与实验二一致，完整文本见前节和原始日志；explicit_dedup 六次响应均为完整的 `NO`。在本组固定案例中，后续调用重复触发比例由 4/6 降为 0/6，四个失败条件全部转为预期响应。这是三个场景的配对诊断，不是独立大样本统计或完整视频重复率。

**分析更新：**固定模型、解码设置、首轮历史和当前观察，只追加明确的去重决策规则，就能改变这四个失败条件的输出。该结果支持“原提示词下的去重行为可通过更明确的规则改善”，不支持直接归因为参数量不足或完全不具备指令遵循能力。但本实验同时改变了规则内容、输入长度和末尾强调位置，无法单独确认是消除了逻辑歧义、增强了优先级，还是末尾位置效应；也不能证明模型内部的具体推理过程。

**仍待验证：**本轮所有后续调用都预期 NO，没有检验追加规则是否导致一律沉默。需增加“历史不变但出现不同新错误”“已纠正后再次发生同类错误”“无历史且有新错误”等预期 YES 的控制，再在视频输入下检查重复率和错误召回。当前只能报告该小组文本案例中的改善，不能报告完整任务的检测性能提升。追加规则改变了论文提示词，应作为独立对照，不改写原基线结果。

原始测试日志包含缺少 `causal_conv1d`、`flash-linear-attention` 时回退到参考 PyTorch 实现的警告，其文字明确标注正确但更慢。日志没有提供证据表明这些警告导致本次重复反馈，也没有开展优化内核与参考实现的一致性对照。

### 实验代码接入（2026-11-17）

根据上述配对结果，已将同一条明确去重规则默认接入 `run_inference.py` 和 `run_full_recipe.py` 的错误检测提示词末尾，覆盖 detailed/simple 两种模板。完成检测提示词不变。`prompts.py` 保留论文原文；新运行配置标记 `prompt_version=arxiv_2606.09547v2_appendix_D_explicit_dedup_v1`、`paper_prompt_verbatim=false`，并保存完整 `mistake_dedup_rule`。该接入仅增加模型输入规则，没有新增输出去重过滤，也没有改变反馈缓存作用域。

`test_9b_following.py` 显式使用原模板构建对照输入，再按分支追加同一份规则，保证 original 分支不受正式实验默认值变化影响、explicit_dedup 分支不会重复追加。未重跑正式 GPU 视频实验或更新前文官方评测指标。新运行应使用独立输出目录。

后续将该实验选择显式化：两个推理入口新增 `--feedback-dedup-rule paper|explicit`（默认 explicit），配置随选项记录实际提示词版本、`paper_prompt_verbatim` 和规则正文；paper 模式规则正文为 null。该选项只控制追加规则，不恢复旧版本的缓存策略或其他实现细节。启动脚本六组命令全部显式选择 explicit，并将输出和评测路径统一为 `-explicit-dedup` 后缀；两条已有续跑配置改为全量新实验，以匹配完整覆盖评测要求。已通过 27 项单元测试、Bash 语法检查，以及六组推理/评测路径配对检查。
