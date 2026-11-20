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
  --output-dir outputs/qwen3-vl-30b-per-step-dedup-manual &&


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
  --output-dir outputs/qwen3.5-27b-per-step-dedup &&


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
  --output-dir outputs/qwen3-vl-30b-full-recipe-dedup &&


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
  --output-dir outputs/qwen3.5-27b-full-recipe-dedup &&


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
  --output-dir outputs/qwen3.5-9b-per-step-dedup-manual &&

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
  --feedback-dedup-rule paper \
  --assistant-feedback false \
  --thinking off \
  --output-dir outputs/qwen3.5-9b-full-recipe &&


python scripts/wait_for_gpu.py \
  --gpu 1 \
  --min-free-mib 21680 \
  --interval 5 \
  -- CUDA_VISIBLE_DEVICES=1 python run_full_recipe.py \
  --data-root ../datasets/ego-mc-bench \
  --model-path ../datasets/Hf_model/Qwen3.5-27B \
  --limit 0 \
  --interval 5 \
  --sample-fps 1 \
  --max-side 448 \
  --max-frames 32 \
  --max-new-tokens 0 \
  --mistake-prompt detailed \
  --feedback-dedup-rule paper \
  --assistant-feedback false \
  --thinking off \
  --output-dir outputs/qwen3.5-27b-full-recipe \
  --continue

python scripts/wait_for_gpu.py \
  --gpu 2 \
  --min-free-mib 21680 \
  --interval 5 \
  -- CUDA_VISIBLE_DEVICES=2 python run_proassist.py \
  --data-root ../datasets/ego-mc-bench \
  --proassist-root ../ProAssist \
  --model-path ../models/ProAssist-Model-L4096-I5 \
  --base-model-path ../models/Llama-3.1-8B-Instruct \
  --vision-model-path ../models/siglip-so400m-patch14-384 \
  --protocol per_recipe_step \
  --output-dir outputs/proassist-i5-smoke --overwirte \
  --limit 0
  

CUDA_VISIBLE_DEVICES=2 python run_proassist.py \
  --data-root ../datasets/ego-mc-bench   --proassist-root ../ProAssist \
  --model-path ../models/ProAssist-Model-L4096-I5   --base-model-path ../models/Llama-3.1-8B-Instruct \
  --vision-model-path ../models/siglip-so400m-patch14-384 \
  --protocol per_recipe_step   --output-dir outputs/proassist-i5   --limit 0 --paper
