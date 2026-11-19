# ProAssist 与 Ego-MC-Bench 论文实验设置核对

核对日期：2026-11-16；本地实现状态更新：2026-11-19。目标论文：Streaming Interventions，arXiv:2606.09547v2，表 3。

## 当前结论

2026-11-19 提示更新：默认改为 `--mistake-prompt simple`，基于附录 D 完成检测与简洁错误检测，压缩为单路流式适配（YES 完成、无 YES 前缀的纠错、其余静默）。详细旧规则仍可显式选择。纯文本对照已证明最简 YES/WAIT 提示可改善格式遵循，但此新三分支提示尚未经真实模型验证；不可把两者结果混用。

`run_proassist.py` 当前使用单路自主发言：开始或步骤切换时提供指令，其余时刻仅持续输入新帧；原生静默机制和官方进度摘要保留。用户指定的映射规则为：以独立单词 YES 开头的非空正文（忽略大小写）存为固定 `Success: Step complete.`（共用基线事件转换），其余非空正文全存为 Feedback，原始正文保留日志。已移除两阶段轮询及双缓存。该系统提示与事件映射没有表 3 原始实现依据，不能作为严格复现结果。

## 来源区分

- 目标论文正文第 5 节：https://arxiv.org/html/2606.09547v2#S5
- 目标论文附录 D/E/F/G：https://arxiv.org/html/2606.09547v2
- 目标论文评测代码：https://github.com/Qualcomm-AI-research/qualcomm_interactive_cooking_eval
- 目标论文项目主页：https://apratimbh.github.io/livecookv2/
- 项目主页 Model Code 链接：https://github.com/Qualcomm-AI-research/Ego-Counterfactual-Mistakes-Finetune
- ProAssist 自身实现：https://github.com/pro-assist/ProAssist
- ProAssist 自身演示：https://github.com/pro-assist/ProAssist/blob/main/notebooks/streaming_inference.ipynb

ProAssist 自身论文、训练配置、演示 notebook 的设置，不自动等同于目标论文表 3 使用的设置。附录 D 的 YES/NO 提示是 turn-based 基线设置；附录 E 的辅助 LLM 是旁白模型设置，都不能无依据移植到 ProAssist。

补充核查：作者 Model Code 仓库的 README、`src/model.py`、`src/eval.py` 和 `configs/experiment/eval.yaml` 已检查。公开模型类是 Qwen3.5 + action head；eval 配置也选择 Qwen3.5，不能直接拿它替换 ProAssist 接口。已读文件未提供 ProAssist 的 checkpoint、角色编排或自然语言事件映射。不能由此断言整个仓库不存在其他 ProAssist 文件，尚未完成整个仓库的源码搜索。

项目主页当前展示的指标与本项目指定的 arXiv v2 已有差异；后续应固定目标版本及代码 revision，不能把新发布代码默认当作 v2 实验原始版本。

## 当前实现逐项审计

| 项目 | 公开依据与当前实现 | 状态 |
|---|---|---|
| 基准和指标 | Ego-MC-Bench；现有入口复用官方 evaluator | 已有依据，仍需完整数据覆盖及版本记录 |
| 单步起点 | 正文规定从 ground-truth Instruction 起点启动每步视频 | 已有依据 |
| 单步终止/超时 | 当前在模型 Success 或下一条人工 Instruction 边界切换 | 具体终止规则需要作者代码确认 |
| 完整食谱 | 正文规定视频连续；当前仅按有效 Success 切换指令 | 连续视频有依据；注入/切换细节待确认 |
| 输入标签 | 当前不提供参考 Feedback、Success 或用户对话 | 用户对话是否输入的表 3 实际设置待确认 |
| 权重 | 支持 I1/I5/I10，实际用户日志每帧有 5 个 image token | 表 3 采用哪种 checkpoint 尚未确认 |
| 任务提示 | 复用共享错误定义，另加自主发言、YES 完成与否则静默的本地规则 | 与本地 Qwen 一致；未确认表 3 的 ProAssist 采用此提示 |
| 当前步骤角色 | 单路缓存；system 在开始时注入规则，user 仅在开始或切换时提供步骤，后续帧不追加问题 | 共享任务规则，但角色分工为本地适配；无表 3 对应依据 |
| 输出映射 | 用户指定：开头独立 YES（忽略大小写）→Success，其他非空正文→Feedback，空正文无事件 | 与本地 Qwen 一致；未确认作者语义映射 |
| 采样频率 | 当前默认 1 FPS（用户指定）；ProAssist 官方演示为 2 FPS | 表 3 数值尚未确认 |
| 静默阈值 | 当前默认 0.3，参考 ProAssist 演示 | 表 3 数值及选择方法尚未确认 |
| 上下文 | 当前默认 4096、调用官方 summarize_and_drop；逐帧累积 KV cache，无均匀 32 帧截取 | 表 3 策略尚未确认 |
| 生成预算 | 当前默认 0（按静默规则或 EOS 停止），摘要触发预留至少 128 tokens | 表 3 预算尚未确认；单次生成仍可能超过上下文余量 |
| 图像处理 | 当前使用官方视觉编码器在线处理 RGB 帧 | 表 3 是否使用同样预处理/预提特征待确认 |
| 注意力实现 | 当前为本地兼容性默认 SDPA | 是否与原始实现一致待确认 |
| 软件环境 | 用户服务器 torch 2.11.0+cu128；官方 ProAssist 原始清单 torch 2.3.1 | 明确存在版本差异，需记录和验证 |

## 已收到的真实日志

52 帧，步骤起点为 4.418 秒和 17 秒。模型在两步第一帧分别生成后续操作话术，且均未遵循本地前缀要求，accepted 均为空。输出的两条事件是输入指令。该日志仅证明加载/流式控制循环可运行，不能证明协议或评测有效。

## 严格对齐的缺口

需要目标论文作者的 ProAssist 推理配置/适配代码，或能够逐项说明以下信息的原始材料：checkpoint 标识与 revision；系统提示和角色编排；输入对话范围；采样/预处理；发言阈值；生成与缓存设置；自由文本到完成/纠错事件的映射；两种协议的具体步骤调度。

在这些缺口补齐前，不应把另一个自拟提示词、正则或辅助裁判替换进去，并声称已与论文所有设置一致。现有代码暂保留为问题复现材料，不继续推荐全量跑分。
