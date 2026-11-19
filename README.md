# Ego-MC-Bench 论文复现

> 2026-11-18 消息结构恢复：per-step/full-recipe 推理恢复当前步骤的 assistant 反馈历史，供完成检测和错误检测共同使用；有历史时消息顺序为 user（当前指令）→ assistant（最近 60 秒内按正文严格去重的反馈，不附加时间戳）→ user（视频帧与检测提示词），无历史时仅发送检测 user 消息。步骤切换时清空历史。detailed 错误提示词继续包含反馈数组及明确去重规则。配置记录消息结构和反馈输入策略。

> 2026-11-17 提示词更新：`run_inference.py`（per-recipe-step）和 `run_full_recipe.py` 现在默认在错误检测提示词末尾追加明确去重规则：同一错误已经提醒且没有新错误时，即使尚未纠正也输出 NO；只有新的、尚未报告的错误才输出 YES。完成检测不变，`prompts.py` 仍保留论文原文。新配置记录 `prompt_version=arxiv_2606.09547v2_appendix_D_explicit_dedup_v1`、`paper_prompt_verbatim=false` 和完整 `mistake_dedup_rule`。下文关于逐字使用论文模板的描述对应此前基线。请使用新输出目录运行并与旧结果分开评测；当前证据来自小规模纯文本对照，尚未验证完整视频上的错误召回。诊断脚本的 `original` 分支仍使用原模板，`explicit_dedup` 分支仍只追加一次规则。

本项目使用本地 Qwen3-VL-2B-Instruct，复现论文 *Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?* 中的 per-recipe-step 推理与官方评测流程。当前实现是遵循论文流程的基础模型基线，具体差异见下文。

Python 3.10+，建议 Linux NVIDIA CUDA 服务器。模型默认路径：`/datasets/Hf_model/Qwen3-VL-2B-Instruct`。

### 独立控制 assistant 历史与去重提示规则

`run_inference.py` 和 `run_full_recipe.py` 共享两个独立参数，默认保持 assistant 历史开启、明确去重规则开启：

| 参数 | 行为 |
| --- | --- |
| `--assistant-feedback true` | 将近期反馈作为 assistant 历史提供给完成和错误检测（默认） |
| `--assistant-feedback false` | 仅发送一条 user 消息；detailed 错误提示词仍包含反馈数组，完成提示词不含反馈 |
| `--feedback-dedup-rule explicit` | 追加 `prompts.py` 中的 `DEDUP_RULE`（默认） |
| `--feedback-dedup-rule paper` | 不追加规则；保留原模板中已有的防重复要求 |

例如，只在 user 中提供反馈、同时追加明确去重规则：

```bash
python run_inference.py --data-root ../datasets/ego-mc-bench \
  --model-path /path/to/model --output-dir outputs/user-only-explicit \
  --assistant-feedback false --feedback-dedup-rule explicit
```

四种组合均可用于对照。配置记录参数、消息结构及提示规则全文；真实输入继续写入 `*.model_inputs.jsonl`。无反馈历史时，即使开启 assistant 开关也只发一条 user。simple 模板不包含反馈数组；上述开关不改变最近 60 秒的历史缓存策略，也不硬过滤输出。切换实验条件时使用独立输出目录。

两个推理入口均支持 `--feedback-dedup-rule explicit|paper`，默认 `explicit`。`explicit` 在错误检测提示词末尾追加明确去重决策规则；`paper` 使用原论文错误检测模板。两者共享当前步骤的反馈历史策略和输入打包方式，因此 `paper` 只恢复提示词模板，不是恢复历史版本的完整实验实现。配置自动记录所选参数、提示词版本、原文标识和追加规则（paper 时为 null）。完成检测不受该开关影响。

`scripts/run_qwen_experiments.sh` 的六组命令现均显式指定 `--feedback-dedup-rule explicit`，输出使用 `-explicit-dedup` 后缀，评测路径与推理目录一致。原有两条局部续跑命令已改为全视频运行，以匹配 `--require-full-coverage`。下文旧目录示例对应此前实验；新脚本路径以当前脚本为准。要做 paper 对照，应同时将参数改为 `paper`、推理与评测目录改为独立的 `-paper` 目录。

## 安装

服务器配置（用户于 2026-11-16 确认）：

- GPU：NVIDIA A800 80GB PCIe。
- PyTorch：`2.11.0+cu128`。
- torchvision：`0.26.0+cu128`。
- PyTorch CUDA runtime：`12.8`（不代表系统 CUDA Toolkit 或驱动版本）。

后续实验环境以此为兼容性基准；ProAssist 与这套版本组合尚待真实推理验证。不要直接用 ProAssist 官方旧依赖清单覆盖已工作的环境。

依赖固定使用 CUDA 12.8 版 PyTorch 2.11.0 和 torchvision 0.26.0，适用于本项目的 Linux NVIDIA 服务器环境。执行：

```bash
python -m pip install -r requirements.txt
```

依赖参考[模型官方 Transformers 示例](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。官方评测所需依赖见下方“官方指标”。

模型通过 `AutoModelForImageTextToText` 根据本地 `config.json` 自动选择架构，支持 Qwen3-VL 和 Qwen3.5 多模态模型；目录名称不决定模型类型。Qwen3.5 需要支持该架构的 Transformers，本项目依赖为 `transformers>=5.5.0,<6`。PyTorch 包通过官方 CUDA 12.8 索引安装，并使用 `+cu128` 版本后缀锁定 CUDA 构建。

如果旧环境装过 CUDA 13 版 PyTorch，请先卸载旧的 torch、torchvision、torchaudio 和 CUDA 13 运行时包，再安装本文件，避免残留库冲突。本项目不需要 torchaudio；如其他任务需要，请安装匹配的 `torchaudio==2.11.0+cu128`。

`--device-map` 使用进程内可见 GPU 的编号。例如 `CUDA_VISIBLE_DEVICES=1` 只暴露物理 GPU 1，此时应传 `--device-map cuda:0`。多卡自动分配可使用 `--device-map auto`，仍受 `CUDA_VISIBLE_DEVICES` 限制。

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
- 相对每步起点每 5 秒查询，视频按 1 FPS 取样，每次最多输入 32 帧。先判断 completion 并要求理由，仅在明确 `NO` 时进一步检查 mistake；坏格式记日志，不当作 false。
- 模型确认成功即结束该步。下一步从其人工起点重新开始，步间清空记忆。
- 当前步骤内最近 60 秒（含边界）的纠错反馈作为防重复上下文，按正文严格去重；相同正文保留最近一次，其时间用于刷新有效期；同时作为 assistant 历史提供给完成和错误判断，步骤切换时清空。不额外硬过滤输出，以免改变官方计分。
- 原始回复、两个阶段、实际帧时间、提示词、耗时及配置均保存。
- 官方评测代码保留在 vendor/，BSD-3-Clause-Clear 许可见 vendor/LICENSE。来自本机 QualcommInteractiveCookingEvaluator 原始项目，未改其指标实现。evaluate_official.py 仅将标注加载替换成本地 JSON。

### 实现选择与复现范围

这是遵循论文流程的基础模型基线，不是论文数值的严格复现。提示词已按本地 PDF 附录 D（第 15-17 页）原文对齐，保留原文笔误，仅归并排版换行。完成检测要求 `YES/NO`，`YES` 附完成理由；错误检测要求 `YES/NO`，`YES` 附纠正反馈，不追加自拟 JSON 契约或额外生成用户话术的提示要求。完成理由仅保留在日志中，预测的成功事件固定写为 `Success: Step complete.`；该固定话术是本地展示/序列化选择，论文未规定其映射。错误检测的 `YES` 后文本直接写入 `Feedback:`，不额外重写或语义过滤，以保留官方文本评测的原始输出。若模型仍生成第三人称分析，应作为模型未遵循提示词的输出保留。`prompts.py` 保存原文模板，`PAPER_PROMPTS_2606.09547v2.md` 保存全部提示词与出处。默认 `--mistake-prompt detailed` 对应 Qwen3-VL/Qwen3.5/Gemini；`simple` 对应附录 D 的较早模型基线。帧上的时间标签是输入打包选择，不属于论文提示词原文。每步视频前缀默认按 1 FPS 取样，超过 32 帧时均匀下采样为最多 32 帧，图像保持宽高比，最长边不超过 448，不放大小图；这些是本地输入设置，尚未确认与论文表 2 的轮询基线一致。输入为带时间标记的多图，不是原生视频编码。未使用 Ego-CoMist+ 微调和 action head，不能对标论文表 3 的训练模型。

每步终点采用下一条 Instruction 的起点，末步采用视频终点。该边界是本实现选择，未使用隐藏 Success 时刻提前截断。首个查询在步起点 +5 秒；长度不超过 5 秒的步骤可能没有查询。保存这些选择，比较实验时保持一致。上述 `run_inference.py` 仍仅运行 per-recipe-step；独立的 Full recipes 实验见下文，不应把按时间自动切换指令称为 full-recipe 复现。

## 统一输入与输出预算

两个 Qwen 入口默认 `--interval 5 --sample-fps 1 --max-side 448 --max-frames 32 --max-new-tokens 0`。`--max-frames 32` 限制每次输入帧数；`--max-new-tokens 0` 表示不设置生成 token 人为上限；生成保留模型 EOS 停止规则，同时清除模型配置继承的长度、最短长度和时间预算。Transformers 的必填 `max_length` 使用系统最大整数哨兵，并非预分配输出缓冲区。模型原生上下文容量与显存仍有限；超出时不静默截断。ProAssist 同样取消固定输出上限，默认采用连续帧输入与 4096-token 进度摘要；其视觉处理器和内部 token 表示仍由该模型决定。

原问答提示词、YES/NO 解析、thinking 开关、事件格式保留；纠错历史改为当前步骤内完整保留。查询间隔恢复为论文基础模型的 5 秒协议；1 FPS 和最多 32 帧是恢复后的本地采样预算，不代表已确认论文轮询基线也使用同样设置。旧 Qwen 预算可显式使用 `--interval 5 --sample-fps 1 --max-frames 32 --max-new-tokens 256`，Qwen 图像处理已恢复为保持宽高比、最长边不超过 448。

## 六组 Qwen 完整实验脚本

`scripts/run_qwen_experiments.sh` 包含六条实验命令：Qwen3-VL-30B、Qwen3.5-27B 和 Qwen3.5-9B 各跑 per-recipe-step 和 full-recipe，每条通过 `&&` 在推理成功后运行官方评测。Qwen3.5-27B 和 9B 均关闭 thinking。9B 使用 GPU 2、模型路径 `../../Data/Hf_model/Qwen3.5-9B`，输出目录为 `outputs/qwen3.5-9b-new` 和 `outputs/qwen3.5-9b-full-recipe-new`。所有参数、路径和输出目录直接写在命令中，可按服务器实际情况修改。

在项目根目录执行：

```bash
bash scripts/run_qwen_experiments.sh
```

30B/27B 默认使用 GPU 1，数据路径为 `../datasets/ego-mc-bench` 和 `../datasets/Hf_model/`。原四组输出目录分别为 `outputs/qwen3-vl-30b-per-step`、`outputs/qwen3-vl-30b-full-recipe`、`outputs/qwen3.5-27b-per-step`、`outputs/qwen3.5-27b-full-recipe`；运行前这些目录应不存在。

## 运行

先验证前 60 秒解码：

```bash
python run_inference.py --data-root ../datasets/ego-mc-bench \
  --output-dir outputs/decode-test --max-seconds 60 --dry-run
```

单视频推理（测试时可添加 --max-seconds 60）：

```bash
CUDA_VISIBLE_DEVICES=1 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --device-map cuda:0 --output-dir outputs/qwen3-vl-30b --limit

CUDA_VISIBLE_DEVICES=1 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --device-map cuda:0 --output-dir outputs/qwen3.5-27b --limit
```

全量将 --limit 设为 0，并使用新的输出目录。显存不足可降低 --max-frames，但这改变实验配置，需重新记录。max-seconds 截断运行只做排错，不用于正式计分。

## 官方指标

### 中断后按视频续跑

`run_inference.py` 和 `run_full_recipe.py` 均支持通过 `--continue` 直接复用原输出目录：

```bash
python run_inference.py --data-root ../datasets/ego-mc-bench \
  --output-dir outputs/run1 --continue --limit 0
```

请保持模型和采样等参数与原实验一致。`--continue` 允许目录已存在，按所选视频的标注顺序找到最后开始的视频（包括只有日志、尚未写入汇总的中断视频），从头覆盖其日志、预测和输入帧，然后继续后续视频；`predictions.json` 保留此前视频的结果。空目录会从头开始。`--limit` 仍限制所选视频范围，全量续跑请设为 `0`。

使用 `--start-video-id` 从指定视频开头重新运行，包含该视频，并按 `annotations.json` 中的顺序继续。添加 `--limit 0` 跑完剩余所有视频；不能与 `--video-ids` 同时使用。模型、采样、思考模式等参数应与原实验一致。

```bash
CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --device-map cuda:0 --output-dir outputs/qwen3.5-27b-continued \
  --start-video-id GX010037 --limit 0
```

请使用新的输出目录保留旧结果。新目录的 `predictions.json` 只包含本次视频；正式评测前需与旧目录中已完成且位于起始视频之前的结果合并，相同视频采用本次完整重跑结果。不要用同一输出目录加 `--overwrite` 来保留总预测：它会重写 `predictions.json`。

```bash
python -m pip install -r requirements-eval.txt
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b/predictions.json \
  --report outputs/qwen3.5-27b/official-report.txt
```

评分代码已合并在本仓库 `vendor/`，无需相邻的 `qualcomm_interactive_cooking_eval` 目录。`vendor/data.py`、`vendor/eval.py`、`vendor/utils.py` 和许可与源目录一致。

先检查预测格式和视频覆盖（无需加载模型、CUDA 或评分依赖）：

```bash
python evaluate_official.py --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3vl-one/predictions.json --check-only
```

正式评分可加 `--require-full-coverage`，缺少视频时会直接报错。该选项仅检查视频 ID 覆盖，仍需自行确认没有使用截断或 dry-run 预测。报告路径必须尚不存在，避免覆盖已有报告。

官方指标使用 CUDA RoBERTa-large BERTScore，首次需要下载其权重。报告包括 IC-Acc、Mistake Precision/Recall/F1、BERTScore、ROUGE-L。官方代码条件为绝对时间差 <15 秒，对应论文所说的 30 秒窗口。predictions 中的 Instruct: 是给定指令，不是模型生成；使用该前缀是因为官方解析用它移除指令前缀。

包装器使用官方 run() 原样计算，保留原实现的重复反馈匹配、尾步处理及无零分母保护等行为；不宣称修复或重定义指标。只提交少量/截断预测可能导致零分母异常或结果失真。正式结果必须使用所有视频完整预测，核对视频覆盖、解析失败日志和异常。当前官方数据卡视频数和标注行数存在差异，以下载后实际数据为准。


## 输出

- `predictions.json`：所有已完成视频的预测，字段为 `video_id`、`pred_texts`、`pred_timestamps`。
- `<video_id>.prediction.json`：推理过程中保存的单视频结果，中断时可检查。
- `<video_id>.jsonl`：每次调用的实际帧时间、提示词、原始回复、两个判断阶段、解析错误与推理耗时。
- `<video_id>.model_inputs.jsonl`：逐次记录模型输入消息（保持文本、图片顺序），含视频 ID、步骤、视频时间和判断阶段。图片路径相对于输出目录；空跑时也保存准备好的输入。
- `model_input_frames/<video_id>/*.png`：实际输入的缩放后帧，采用无损 PNG，同一帧复用。
- `config.json`：运行参数与实验配置。

默认要求 `--output-dir` 不存在。添加 `--overwrite` 可复用已有目录并覆盖本次生成的配置、预测、日志和输入帧；不会清空目录，未涉及的视频文件会保留。示例：`python run_inference.py --data-root /path/to/data --output-dir outputs/run1 --overwrite`。可用 `--video-ids GX010021 GX010023` 选择视频（数量仍受 `--limit` 控制，选择多个时设 `--limit 0`）。dry-run 仅用于读取验证，不用于正式计分。

## 代码逻辑测试

在项目根目录运行：

```bash
python -m unittest tests.test_inference
```

`test_inference.py` 使用构造的数据检查以下逻辑，无需视频、模型权重或 GPU，也无需安装模型推理依赖：

- 从人工标注中只提取 Instruction，并检查生成的提示词不包含测试样例中的人工纠错答案。
- 解析论文的 `YES/NO` 回复：肯定的完成判断必须提供理由，肯定的错误判断必须提供反馈；否定判断允许只回复 `NO`。拒绝 JSON、缺少决策和缺少肯定说明的回复。
- 验证完成理由不会进入成功话术，错误反馈原样保留。
- 验证详细/简洁模板选择、近期反馈注入、原文占位符替换和非法阶段处理。

正常情况下会显示测试数量和 `OK`。这些测试不运行真实模型、不计算论文指标；视频读取、实际推理与官方评测仍需按上述流程分别验证。

## 运行状态与限制

本实现使用带时间标记的历史图片模拟流式输入，不输入音频、不使用原生视频编码，也不保证实时速度。时间戳是视频观察时刻，推理延迟单独保存。

本地尚未运行真实模型权重；需要在服务器完成短时推理验证后再跑全量。数据集仅限研究用途，请遵守其 `LICENSE.pdf`。

## 本次提示词对齐

来源：本地 `2606.09547v2.pdf` 及整理文件。配置中 `prompt_version` 标记原文版本，`response_format` 为 `YES_NO`，`paper_prompt_verbatim=true`；`success_message_policy` 记录完成理由仅写日志、成功事件使用固定确认话术。修改 `--interval` 偏离论文 5 秒间隔时会在 `interval_matches_paper` 中记录。旧运行日志和预测不会自动转换；请使用新的输出目录重新推理。没有执行 GPU 推理或声称复现论文数值。


## 论文协议复核

依据[正文第 5 节](https://arxiv.org/html/2606.09547v2#S5)和[附录 D](https://arxiv.org/html/2606.09547v2#A4)：每 5 秒先检查完成；未完成才检查错误；详细错误提示使用最近 60 秒模型反馈防重复。完成检测明确要求肯定时解释依据，错误检测明确要求肯定时直接向用户提供简短反馈。这两类文本含义不同，不能把完成解释当作纠错反馈。此前追加的 JSON 输出契约已移除。论文未公开具体解析与 Success 序列化实现，本文实现选择不应称为作者原始实现。

### Qwen3.5 重跑与输出诊断

通过 `--thinking` 选择聊天模板的思考模式：

- `--thinking off`（默认）：传入 `enable_thinking=False`，延续此前关闭思考的设置。
- `--thinking on`：传入 `enable_thinking=True`，用于支持该开关的模型（如 Qwen3.5）。
- `--thinking default`：不传 `enable_thinking`，使用本地模型聊天模板的默认行为。Qwen3.5 官方模板默认开启思考；Qwen3-VL-Instruct 不使用此开关，建议选择 `default` 以避免不支持参数的警告。

例如，在上方推理命令末尾添加 `--thinking on` 即可测试开启思考的 Qwen3.5。这个开关通过 `processor.apply_chat_template()` 传入，实际行为取决于本地聊天模板；不能为不支持它的模型开启思考。论文未披露思考模式设置，所选模式属于本地实验配置。

论文提示词和严格解析规则保持不变。开启思考可能先生成推理内容而非 YES/NO，或耗尽 `--max-new-tokens` 预算，因此可能出现解析失败；当前代码不会自动剥离思考内容。`config.json` 记录 `thinking`、`enable_thinking`（`default` 时为 null）、实际传入的 `chat_template_kwargs`、模型/processor 类型、Transformers 和 PyTorch 版本。每条推理日志增加 `generated_tokens` 和 `hit_token_limit`（达到生成上限且末尾不是 EOS），用于区分输出截断与格式错误。

建议先使用新目录执行 `--max-seconds 30` 短跑，确认 `raw_response` 以 YES/NO 开头且 `parse_error` 为 null，再去掉时间限制完成整段视频。旧输出目录需要 `--overwrite`，会覆盖该次运行文件。Qwen3-VL-30B 的乱码原因尚未定位，关闭思考不能视为对其乱码的修复；先短跑检查并保留新增环境信息。


## 独立 Full recipes 实验

入口为 `run_full_recipe.py`，不修改 `run_inference.py` 的行为。复用原入口的样本加载、提示词替换、YES/NO 解析、事件序列化和 JSON 保存函数。模型加载与生成参数保持一致。

视频从 0 秒开始，每 5 秒先判断当前步骤是否完成；有效 `YES` 记录 `Success:` 并在同一时刻注入下一条 `Instruct:`，下一次查询仍按统一的 5 秒时钟进行。`NO` 才检查错误。完成回复格式错误时保留当前步骤，并跳过该轮错误检测。不会在人工标注边界强制推进，也不设置强制步骤超时；直到所有步骤完成或视频结束。未完成步骤不会补造成功事件。

推理仅提取按标注顺序排列的 Instruction 文本，不读取人工指令时间戳、Feedback 或 Success。模型只看到当前指令，不预先看到后续指令。第一条指令在 0 秒注入；后续指令在模型判断前一步完成时注入。这些注入时刻属于本地协议，并非人工真实起点。

视觉上下文从当前步骤起点到当前时刻的已观察前缀按 1 FPS 取样，超过 32 帧时均匀下采样，默认保持宽高比且最长边不超过 448；步骤起点由模型上一次成功事件确定，切换时重置视觉前缀。帧时间标签继续保留。当前步骤内最近 60 秒的纠错反馈按正文严格去重后，作为 assistant 历史供两个阶段使用，并用于 detailed 错误提示词内的反馈数组；步骤切换时清空。这些提示词和反馈历史策略与官方微调模型不同，不能声称完整复现作者实现。

```bash
# 仅检查视频解码和输入构造，不加载模型；不会模拟完成或生成有效预测。
python run_full_recipe.py --data-root ../datasets/ego-mc-bench \
  --output-dir outputs/full-recipe-decode-test --max-seconds 60 --dry-run

# 全量实验：使用独立输出目录，默认关闭思考，可按需使用 --thinking on。
CUDA_VISIBLE_DEVICES=2 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --device-map cuda:0 --output-dir outputs/qwen3-vl-30b-full-recipe --limit 0

CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-full-recipe/predictions.json \
  --report outputs/qwen3.5-9b-full-recipe/official-report.txt \
  --require-full-coverage
```

输出格式与原实验一致：`config.json` 标记 `protocol=full_recipe` 和本地协议选择，`*.jsonl` 保存原始响应与当前 `step_id`，`*.model_inputs.jsonl` 和 `model_input_frames/` 保存模型输入，`predictions.json` 可交给现有官方评测入口。`--dry-run` 的空预测不可用于评测；`--max-seconds` 截断仅用于排错。输出目录默认必须不存在。中断后可在原命令添加 `--continue --limit 0`，覆盖重跑最后开始的视频并继续，同时保留此前视频的汇总预测；模型和采样参数应与原实验一致。

验证：`python -m unittest tests.test_full_recipe tests.test_inference tests.test_evaluate_official`。新增测试用模拟视频和模型运行完整控制循环，检查模型驱动切换、连续时钟与上下文、标注时间戳不影响结果、滞留与格式错误行为，以及官方输出格式兼容性；不代表已运行真实 GPU 推理或复现论文数值。

## ProAssist 自主发言实验

**协议核对状态：当前为单路自主发言实验，不是目标论文表 3 的严格复现。** 不再逐帧发出完成/错误问题。按用户指定的本地规则，非空输出以完整单词 YES 开头（忽略大小写）映射为 Success，其余发言全部映射为 Feedback；这仅是序列化规则，不保证事件语义正确。系统提示及事件映射仍需真实模型验证。详见 [PROASSIST_PROTOCOL_AUDIT.md](PROASSIST_PROTOCOL_AUDIT.md)。

独立入口 `run_proassist.py` 使用 ProAssist 连续流式历史机制：默认 `--sample-fps 1 --context-length 4096 --max-side 448 --max-new-tokens 0`，每 1 秒输入一张新帧，累积视觉与生成对话的 KV cache，不做均匀 32 帧截取，也不重放历史帧。接近 token 预算时调用官方 `manage_kv_cache` 的 `summarize_and_drop` 进度摘要机制；保留至少 128 tokens 的摘要触发余量。`--context-length 0` 可显式关闭摘要，但不再是默认设置。生成仍按静默规则或 EOS 停止，阈值默认 0.3 是本地默认值，不代表所有数据集最优值。

数据选择、输出参数、图像缩放、生成预算参数与 JSON 保存复用 `run_inference.py`，完整食谱指令提取复用 `run_full_recipe.py`。图像仍保持宽高比、不放大、最长边 448，再交给官方视觉处理器；这项本地预处理及 ProAssist 上的提示应用方式尚未与表 3 对齐。单步协议切换时重置缓存；完整食谱协议只追加下一条指令，持续保留历史并按需摘要。支持 `--overwrite` 和 `--continue`。ProAssist 不提供 `--max-frames`，以免将均匀帧预算误用于流式历史。

请使用 **独立 Python 3.10 或 3.11 环境**。本项目提供 `requirements-proassist.txt` 推理候选依赖：保留用户已工作的 PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128，采用官方代码对应的 Transformers 4.43.1、PEFT 0.11.1；该新旧组合尚未在服务器上跑通真实模型，不是验证过的锁文件。不要安装本项目的 Qwen `requirements.txt`，也不要安装 ProAssist 官方整套旧版训练/演示依赖，后者会要求 torch 2.3.1，并包含锁定旧 torch 的 vLLM。

```bash
conda create -n proassist python=3.10 -y
conda activate proassist
# 在本项目根目录执行；先检查依赖解析，再安装。
python -m pip install --dry-run -r requirements-proassist.txt
python -m pip install -r requirements-proassist.txt
python -m pip check
```

入口通过 `--proassist-root /path/to/ProAssist` 指定服务器已有的官方源码目录，无需执行官方的整套依赖安装。安装后先运行 `python run_proassist.py --help` 和下方 30 秒短跑；成功解析依赖并不代表模型运行兼容。推理候选清单暂不包括 DeepSpeed、vLLM、Gradio 等训练/服务组件；若所用官方源码版本的间接导入确实需要额外包，应按具体 traceback 处理，避免直接补装官方全部依赖。

需要三个本地模型目录：ProAssist 适配器、Llama-3.1-8B-Instruct 基础模型、**与适配器 `config.json` 中 `vision_pretrained` 对应的视觉模型**（包含 processor）。之前只下载适配器和 Llama 的说明不完整，还需要准备视觉模型。可在服务器上查看对应名称：

```bash
python -c 'import json; print(json.load(open("/path/to/ProAssist-Model-L4096-I1/config.json"))["vision_pretrained"])'
```

入口默认强制 Hugging Face 离线加载。它通过临时符号链接及配置副本重定向基础模型、视觉模型路径，不修改下载好的权重或配置，不自动下载缺少的模型。默认使用 SDPA；若使用 Flash Attention，可传 `--attn-implementation flash_attention_2` 并自行准备相应依赖。

```bash
# 不加载模型，检查数据与逐帧解码；输出为空预测，不可用于评测。
python run_proassist.py --data-root ../datasets/ego-mc-bench \
  --output-dir outputs/proassist-decode-test --max-seconds 30 --dry-run

# 首先短跑：将三个模型路径和源码路径替换成服务器实际路径。
CUDA_VISIBLE_DEVICES=1 python run_proassist.py \
  --data-root ../datasets/ego-mc-bench \
  --proassist-root ../ProAssist \
  --model-path ../models/ProAssist-Model-L4096-I5 \
  --base-model-path ../models/Llama-3.1-8B-Instruct \
  --vision-model-path ../models/siglip-so400m-patch14-384 \
  --output-dir outputs/proassist-i5-smoke --max-seconds 30
```

检查 `*.jsonl` 中的 `stage`、`prompt`、`input_messages`、`raw_response`、`parsed`、`parse_error`、`hit_token_limit`；`raw_response` 与 Qwen 一样是可解析正文，`decoded_response` 额外保留含特殊标记的解码原文，不再使用 `text` 字段。确认加载和格式正常后，使用**新的输出目录**，去掉 `--max-seconds 30` 并加 `--limit 0` 运行全部视频。`--device-index` 指 CUDA_VISIBLE_DEVICES 后的可见设备编号，默认 0。输出目录必须不存在，避免覆盖已有实验。

- 默认 `--protocol per_recipe_step`：每步从人工 Instruction 时间重置视频上下文；完成后转下一步的人工 Instruction 起点。若始终未完成，在下一条人工 Instruction 边界重置，保留该步未完成结果。模型不读取参考 Feedback、Success 或用户对话。
- `--protocol full_recipe`：从 0 秒连续播放，仅在模型输出有效 Success 时注入下一条指令，不读取人工时间戳；步骤切换保留缓存并按需摘要。每帧的查询提示包含当前指令；只在模型有效 Success 后切换到下一条指令。
- 单路自主发言：开始时发送 system 规则及 user 当前步骤；此后仅输入新帧，不再追加逐帧文字问题。模型通过原生发言/静默阈值决定何时输出。单步骤切换重置缓存，完整食谱切换只追加新指令并保留缓存。摘要续接使用相同系统规则与当前步骤知识。
- 非空正文以独立单词 `YES` 开头（忽略大小写）时，调用与 Qwen 共用的事件转换保存为 `Success: Step complete.`，原始 YES 及说明只保留日志；其余任何非空发言（包括 No、Not yet、普通提醒）保存为 `Feedback: <原始正文>`；空正文不生成事件。Yes 单独出现也接受，Yesterday 不算 YES；不要求解释或指定前缀、不做语义猜测或硬去重。原始正文及含特殊标记的解码结果仍写入日志。
- `--mistake-prompt simple` 现在是默认值：依据论文附录 D 的完成检测和简洁错误检测提示，改写为短版单路流式规则（完成以 YES 开头并解释依据；错误直接纠正且不以 YES 开头；其余静默）。这不是论文 ProAssist 原始提示的逐字复现。`--mistake-prompt detailed` 保留此前完整规则用于对照；不再调用对应的轮询问答段落。系统提示明确要求每帧首先根据当前及历史视频判断整步完成，即使无错误也必须检查完成；完成可见时以 YES 开头给出证据。仅未完成时检查错误并反馈；二者均不满足才静默。禁止用指令已给出或部分进度推断完成，禁止输出 NO、鼓励、旁白或复述指令，纠错不得以 YES 开头。历史发言通过原生 KV cache 和官方摘要保留，不再显式重复注入 assistant 反馈数组；移除 `--assistant-feedback`、`--feedback-dedup-rule` 参数。
- 日志的 `stage` 固定为 `stream`；后续纯画面帧的 `prompt` 为空、`input_messages=[]` 是正常增量输入。日志保留静默概率、阈值、生成前后及摘要管理后的缓存长度。`config.json` 记录单路模式、完整 system 及映射规则，明确 `paper_exact_reproduction=false`。

`predictions.json` 与现有官方评测入口兼容。全量推理结束后，可在原有**评测环境**执行：

```bash
python evaluate_official.py --data-root ../datasets/ego-mc-bench \
  --predictions outputs/proassist-i1/predictions.json --check-only --require-full-coverage

CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/proassist-i1/predictions.json \
  --report outputs/proassist-i1/official-report.txt --require-full-coverage
```

本地验证：`python -m unittest tests.test_proassist tests.test_full_recipe tests.test_inference tests.test_evaluate_official`。测试覆盖实际控制循环的静默/纠错/成功、两种步骤协议、无参考反馈泄漏、格式错误、官方输出格式，以及模拟官方模型接口的路径重定向和 KV cache 传递。尚未在真实 ProAssist 权重/GPU 上运行推理；服务器短跑是必要的后续验证。

## 纯文本去重规则对照

```bash
CUDA_VISIBLE_DEVICES=2 python -m tests.test_9b_following \
  --model-path /实际模型路径/Qwen3.5-9B \
  --mode repeat \
  --output outputs/following-test/9b-repeat.json
```

只使用单条 user 消息，比较原提示词与追加明确去重规则。每个场景共享首轮反馈，保留历史数组；覆盖盐量、油量、温度三个场景，每个场景包含相同观察、几乎相同观察及新一次执行再次犯错的控制，正常共 21 次生成。分别汇总逐字重复和新事件检出，JSON 保存完整输入、响应、结果及汇总。decision 模式保留。
