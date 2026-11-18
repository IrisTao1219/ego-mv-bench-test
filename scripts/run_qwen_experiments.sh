#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3-vl-30b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3-VL-30B-Instruct \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3-vl-30b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-27b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-27b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python run_inference.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../../Data/Hf_model/Qwen3.5-9B \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-9b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../../Data/Hf_model/Qwen3.5-9B \
  --device-map cuda:0 \
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
CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-9b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage

