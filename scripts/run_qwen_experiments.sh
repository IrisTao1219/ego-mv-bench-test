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
  --thinking default \
  --output-dir outputs/qwen3-vl-30b-per-step &&
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step/predictions.json \
  --report outputs/qwen3-vl-30b-per-step/official-report.txt \
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
  --thinking default \
  --output-dir outputs/qwen3-vl-30b-full-recipe-continue --start-video-id GX010050 &&
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-full-recipe/predictions.json \
  --report outputs/qwen3-vl-30b-full-recipe/official-report.txt \
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
  --thinking off \
  --output-dir outputs/qwen3.5-27b-per-step &&
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step/predictions.json \
  --report outputs/qwen3.5-27b-per-step/official-report.txt \
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
  --thinking off \
  --output-dir outputs/qwen3.5-27b-continue --overwrite --start-video-id GX010040 &&
CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-full-recipe/predictions.json \
  --report outputs/qwen3.5-27b-full-recipe/official-report.txt \
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
  --thinking off \
  --output-dir outputs/qwen3.5-9b-new &&
CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-new/predictions.json \
  --report outputs/qwen3.5-9b-new/official-report.txt \
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
  --thinking off \
  --output-dir outputs/qwen3.5-9b-full-recipe-new &&
CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-full-recipe-new/predictions.json \
  --report outputs/qwen3.5-9b-full-recipe-new/official-report.txt \
  --require-full-coverage

