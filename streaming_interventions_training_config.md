# Streaming Interventions 训练配置

整理日期：2026-11-20。以下区分论文设置、当前公开代码设置和单卡建议；这是配置说明文件，不是可直接运行的完整 Hydra 配置。

## 论文训练设置

来源：https://arxiv.org/html/2606.09547v2#A7

| 项目 | 设置 |
|---|---|
| 微调模型 | Qwen3.5-2B、Qwen3.5-0.8B、Qwen3-VL-2B-Instruct |
| 微调方式 | 语言骨干 LoRA |
| LoRA rank | 64 |
| 训练长度 | 约 100,000 iterations，根据验证损失选择 |
| 优化器 | AdamW |
| 初始学习率 | 2e-4 |
| 调度 | cosine annealing，最终降到 1e-6 |
| 总 batch size | 32 |
| GPU | 8 × NVIDIA H100 |
| 梯度累积 | 4 |
| Ego-CoMist+ 配比 | Ego-CoMist 60%，QICD 30%，Ego4D Goal-Step 10% |

模型额外增加 action head，在每个输入帧之后预测说话/保持沉默。只做普通视频问答 SFT 不等同于论文方法。论文未公开实际训练小时数，也没有清楚说明 iterations 是否全部指 optimizer updates。
vcf v
## 当前公开代码中的配置

官方仓库：https://github.com/Qualcomm-AI-research/Ego-Counterfactual-Mistakes-Finetune

以下值来自当前 main 分支，不保证和论文实验版本完全一致。

| 项目 | 当前代码值 |
|---|---|
| 基础模型 | Qwen/Qwen3.5-2B |
| LoRA rank / alpha / dropout | 32 / 16 / 0.05 |
| 视觉编码器训练 | 关闭 |
| 帧采样率 | 2 fps |
| 图像尺寸 | 224 × 224 |
| 每样本帧数上限 | 200；长窗口保留最近帧 |
| 文本长度上限 | 1024 |
| 每 epoch 训练采样数 | 80,000 |
| 数据划分配置 | 0.98 / 0.01 / 0.01 |
| 每卡 micro-batch | 1 |
| 梯度累积 | 4 |
| 分布式策略配置名 | ds-st3-bfloat16-no-opt-state-speed |
| 优化器配置名 | DeepSpeedCPUAdam |
| 学习率调度配置名 | CosineAnnealingLR |
| 梯度检查点 | 模型文件为 true，但 experiment=train 覆盖为 false |

LoRA 作用于语言模型注意力、MLP 和线性注意力的部分投影层；完整目标层名单和 action loss 权重请以模型配置源文件为准。

源文件：

- https://github.com/Qualcomm-AI-research/Ego-Counterfactual-Mistakes-Finetune/blob/main/configs/experiment/train.yaml
- https://github.com/Qualcomm-AI-research/Ego-Counterfactual-Mistakes-Finetune/blob/main/configs/model/qwen3p5_2b_lora.yaml
- https://github.com/Qualcomm-AI-research/Ego-Counterfactual-Mistakes-Finetune/blob/main/configs/data/default.yaml

注意：论文 rank=64，当前代码 rank=32。训练总步数、学习率及优化器其他参数还需要读取完整 trainer/optimizer/scheduler 配置后确认，不能假定当前入口完全复现论文。

## 单张 A800 的起步建议（非作者配置，未实测）

先确认显存是 40GB 还是 80GB。保留 Qwen3.5-2B、LoRA 和 action head，优先沿用官方数据加载与损失实现。

| 项目 | 建议起点 |
|---|---|
| GPU 数量 | 1 |
| 每卡 micro-batch | 1 |
| 梯度累积 | 32，使有效 batch size 为 32 |
| 梯度检查点 | 开启 |
| 精度 | BF16 |
| 图像尺寸 / 帧率 | 先保持 224 × 224 / 2 fps |
| 帧数上限 | 80GB 先尝试 200；显存不足时降低到 64–128 |
| 初次运行 | 几百次参数更新，测吞吐、显存、损失和 action head 输出 |
| 第一轮实验 | 1,000–3,000 次参数更新并做验证，再决定是否继续 |

降低帧数上限会缩短可见历史，可能影响计数和长动作识别，属于改变实验条件。不要仅修改一处帧数参数：数据集、collator 中相关上限需要一致检查。

不能直接沿用 8 卡时的梯度累积 4：单卡这样只有有效 batch size 4。单卡使用 DeepSpeed 的具体策略和启动参数也需要在完整仓库中验证，以上不是已经验证可运行的单卡启动方案。

## 数据准备及时间估计

Ego-CoMist 的标注对应 CaptainCook4D、Ego4D 和 Ego-Exo4D 视频，需准备相应视频/帧目录；Ego-CoMist+ 还涉及 QICD 与 Goal-Step 数据。仅下载标注并不等于训练数据已齐备。

之前聊天中的“单 A800 完整训练 1–2 个月”没有实测依据，不应视为确定预算。用实际参数更新耗时估计：

训练时间 = 剩余参数更新次数 × 平均每次参数更新耗时

平均耗时应包含梯度累积，避开初始化和预热阶段，并另计验证、保存和数据准备时间。不要将 dataloader batch 计数与参数更新次数混淆。
