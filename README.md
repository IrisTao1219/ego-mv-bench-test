# Ego-MC-Bench 论文复现

本项目使用本地 Qwen3-VL-2B-Instruct，复现论文 *Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?* 中的 per-recipe-step 推理与官方评测流程。当前实现是遵循论文流程的基础模型基线，具体差异见下文。

Python 3.10+，建议 Linux NVIDIA CUDA 服务器。模型默认路径：`/datasets/Hf_model/Qwen3-VL-2B-Instruct`。

## 安装

先按服务器 CUDA 环境安装兼容的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
```

依赖参考[模型官方 Transformers 示例](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。官方评测所需依赖见下方“官方指标”。

## 数据准备

```text
/datasets/ego-mc-bench/
  annotations/annotations.json
  videos/GX010021.MP4
  videos/...
```

下方命令使用相对路径 `../datasets/ego-mc-bench` 和 `../datasets/Hf_model/Qwen3-VL-2B-Instruct`，请按实际存放位置调整。标注用于枚举视频、获取当前人工 Instruction 及步骤边界；不把人工 Feedback、Success 或未来指令放入模型提示词。

## 论文流程

依据 https://arxiv.org/html/2606.09547v2 的第 5 节、附录 D/F。

### 已实现

- per-recipe-step：给定单条人工 Instruction，从该指令时间开始独立处理；不输入用户评论、人工 Feedback、Success 或未来指令。
- 相对每步起点每 5 秒查询。先判断 completion 并要求理由，仅在明确 false 时进一步检查 mistake；坏格式记日志，不当作 false。
- 模型确认成功即结束该步。下一步从其人工起点重新开始，步间清空记忆。
- 最近 60 秒模型反馈作为防重复上下文。不额外硬过滤输出，以免改变官方计分。
- 原始回复、两个阶段、实际帧时间、提示词、耗时及配置均保存。
- 官方评测代码保留在 vendor/，BSD-3-Clause-Clear 许可见 vendor/LICENSE。来自本机 QualcommInteractiveCookingEvaluator 原始项目，未改其指标实现。evaluate_official.py 仅将标注加载替换成本地 JSON。

### 实现选择与复现范围

这是遵循论文流程的基础模型基线，不是论文数值的严格复现。附录 HTML 未显示提示词框中的原文，因此当前提示词是根据附录描述重写，并非作者逐字提示词。每步视频前缀按 1 FPS 候选帧取样，最多 32 帧均匀下采样，最长边 448；这些是本实现的可调预算，不是论文确认的配置。输入为带时间标记的多图，不是原生视频编码。未使用 Ego-CoMist+ 微调和 action head，不能对标论文表 3 的训练模型。

每步终点采用下一条 Instruction 的起点，末步采用视频终点。该边界是本实现选择，未使用隐藏 Success 时刻提前截断。首个查询在步起点 +5 秒；只有短于 5 秒的步骤可能没有查询。保存这些选择，比较实验时保持一致。没有实现 full-recipe 模式，不应把按时间自动切换指令称为 full-recipe 复现。

## 运行

先验证前 60 秒解码：

```bash
python run_inference.py --data-root ../datasets/ego-mc-bench \
  --output-dir outputs/decode-test --max-seconds 60 --dry-run
```

单视频推理（测试时可添加 --max-seconds 60）：

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-2B-Instruct \
  --device-map cuda:0 --output-dir outputs/qwen3vl-one --limit 1
```

全量将 --limit 设为 0，并使用新的输出目录。显存不足可降低 --max-frames，但这改变实验配置，需重新记录。max-seconds 截断运行只做排错，不用于正式计分。

## 官方指标

```bash
python -m pip install bert-score rouge-score datasets tqdm
CUDA_VISIBLE_DEVICES=0 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3vl-one/predictions.json \
  --report outputs/qwen3vl-one/official-report.txt
```

官方指标使用 CUDA RoBERTa-large BERTScore，首次需要下载其权重。报告包括 IC-Acc、Mistake Precision/Recall/F1、BERTScore、ROUGE-L。官方代码条件为绝对时间差 <15 秒，对应论文所说的 30 秒窗口。predictions 中的 Instruct: 是给定指令，不是模型生成；使用该前缀是因为官方解析用它移除指令前缀。

包装器使用官方 run() 原样计算，保留原实现的重复反馈匹配、尾步处理及无零分母保护等行为；不宣称修复或重定义指标。只提交少量/截断预测可能导致零分母异常或结果失真。正式结果必须使用所有视频完整预测，核对视频覆盖、解析失败日志和异常。当前官方数据卡视频数和标注行数存在差异，以下载后实际数据为准。


## 输出

- `predictions.json`：所有已完成视频的预测，字段为 `video_id`、`pred_texts`、`pred_timestamps`。
- `<video_id>.prediction.json`：推理过程中保存的单视频结果，中断时可检查。
- `<video_id>.jsonl`：每次调用的实际帧时间、提示词、原始回复、两个判断阶段、解析错误与推理耗时。
- `<video_id>.model_inputs.jsonl`：逐次记录模型输入消息（保持文本、图片顺序），含视频 ID、步骤、视频时间和判断阶段。图片路径相对于输出目录；空跑时也保存准备好的输入。
- `model_input_frames/<video_id>/*.png`：实际输入的缩放后帧，采用无损 PNG，同一帧复用。
- `config.json`：运行参数与实验配置。

每次运行请使用新的 `--output-dir`，避免覆盖已有实验。可用 `--video-ids GX010021 GX010023` 选择视频（数量仍受 `--limit` 控制，选择多个时设 `--limit 0`）。dry-run 仅用于读取验证，不用于正式计分。

## 代码逻辑测试

在项目根目录运行：

```bash
python -m unittest test_inference
```

`test_inference.py` 使用构造的数据检查以下逻辑，无需视频、模型权重或 GPU，也无需安装模型推理依赖：

- 从人工标注中只提取 Instruction，并检查生成的提示词不包含测试样例中的人工纠错答案。
- 严格解析模型回复：判断字段必须为 JSON 布尔值，必须提供非空理由；判断有错误时必须提供纠正建议。格式不合规时应抛出异常。

正常情况下会显示 `Ran 2 tests` 和 `OK`。这些测试不运行真实模型、不计算论文指标；视频读取、实际推理与官方评测仍需按上述流程分别验证。

## 运行状态与限制

本实现使用带时间标记的历史图片模拟流式输入，不输入音频、不使用原生视频编码，也不保证实时速度。时间戳是视频观察时刻，推理延迟单独保存。

本地尚未运行真实模型权重；需要在服务器完成短时推理验证后再跑全量。数据集仅限研究用途，请遵守其 `LICENSE.pdf`。
