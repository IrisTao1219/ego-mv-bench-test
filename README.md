# Ego-MC-Bench 本地模型推理 MVP

使用本地 Qwen3-VL-2B-Instruct，读取 MP4，按时间抽取历史帧，输出带时间戳的烹饪指导。Python 3.10+，建议 Linux NVIDIA CUDA 服务器。模型默认路径：`/datasets/Hf_model/Qwen3-VL-2B-Instruct`。

## 安装

先按服务器 CUDA 环境安装兼容的 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
```

依赖参考模型官方 Transformers 示例：
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

## 数据

```text
/datasets/ego-mc-bench/
  annotations/annotations.json
  videos/GX010021.MP4
  videos/...
```

标注仅用于枚举视频 ID，不把标准答案或未来指令放入模型提示词。

## 先验证视频读取（不加载模型）

```bash
python run_inference.py --data-root /datasets/ego-mc-bench \
  --output-dir outputs/decode-test --max-seconds 20 --dry-run
```

## 跑一个视频的前 60 秒

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --data-root /datasets/ego-mc-bench \
  --model-path /datasets/Hf_model/Qwen3-VL-2B-Instruct \
  --output-dir outputs/qwen3vl-smoke --limit 1 --max-seconds 60
```

## 跑完整数据集

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --data-root /datasets/ego-mc-bench \
  --output-dir outputs/qwen3vl-full --limit 0
```

每次运行请使用新的 output-dir，避免覆盖已有实验。可用 `--video-ids GX010021 GX010023` 选择视频（数量仍受 limit 控制，选多个时设 `--limit 0`）。显存不足时先减小 `--frames 2 --max-side 336`。

## 输出

- `predictions.json`：所有已完成视频的预测，字段为 video_id、pred_texts、pred_timestamps。
- `<video_id>.prediction.json`：每次调用后保存的单视频结果，中断时可检查。
- `<video_id>.jsonl`：每次调用的帧时间、原始回复、解析错误、推理耗时。
- `config.json`：运行参数。

默认每 5 秒调用一次，最近 10 秒取最多 4 帧，最长边 448。输出类型是 Instruction、Feedback、Success、None；None 不写入预测文件。格式解析失败会记日志并跳过，模型加载和视频解码错误则停止运行。

## 协议与限制

这是基于多张历史图片的模拟流式基线；不输入音频、不使用原生视频时间编码、不保证实时速度。时间戳是视频观察时刻，推理延迟单独保存。只保留最近 12 条模型事件作为文本记忆。无标准菜谱，模型自行决定指令，因此不等同于论文协议。

此 MVP 实现视频推理及预测导出，不计算官方指标。Qualcomm 现有评测要求第一条为指令，且按指令分段；本程序忠实保存模型实际输出，不强制伪造首条指令。直接运行官方评测前，应检查首条是否为 Instruction，并明确指令对齐协议。dry-run 生成空预测，仅用于读取验证。

本地尚未运行真实模型权重；需要在服务器完成上述 smoke test 后再跑全量。数据集仅限研究用途，请遵守其 LICENSE.pdf。
