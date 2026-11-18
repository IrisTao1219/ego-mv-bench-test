#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking default \
  --output-dir outputs/qwen3-vl-30b-per-step-explicit-dedup &&


CUDA_VISIBLE_DEVICES=2 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking off \
  --output-dir outputs/qwen3.5-27b-per-step-explicit-dedup &&


CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking default \
  --output-dir outputs/qwen3-vl-30b-full-recipe-explicit-dedup &&


CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking off \
  --output-dir outputs/qwen3.5-27b-full-recipe-explicit-dedup &&


CUDA_VISIBLE_DEVICES=2 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../../Data/Hf_model/Qwen3.5-9B \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking off \
  --output-dir outputs/qwen3.5-9b-per-step-explicit-dedup &&

CUDA_VISIBLE_DEVICES=2 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../../Data/Hf_model/Qwen3.5-9B \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule explicit \
  --thinking off \
  --output-dir outputs/qwen3.5-9b-full-recipe-explicit-dedup &&


python scripts/wait_for_gpu.py \
  --gpu 1 \
  --min-free-mib 71680 \
  -- python run_inference.py

