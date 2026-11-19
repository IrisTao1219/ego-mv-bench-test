CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step/predictions.json \
  --report outputs/qwen3-vl-30b-per-step/report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step-dedup/predictions.json \
  --report outputs/qwen3-vl-30b-per-step-dedup/report.txt \

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step-dedup-manual/predictions.json \
  --report outputs/qwen3-vl-30b-per-step-dedup-manual/report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step/predictions.json \
  --report outputs/qwen3.5-27b-per-step/report.txt \
  --require-full-coverage
  
CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step-dedup-manual/predictions.json \
  --report outputs/qwen3.5-27b-per-step-dedup-manual/report.txt

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-per-step/predictions.json \
  --report outputs/qwen3.5-9b-per-step/report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-per-step-dedup-manual/predictions.json \
  --report outputs/qwen3.5-9b-per-step-dedup-manual/report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-full-recipe/predictions.json \
  --report outputs/qwen3-vl-30b-full-recipe/report.txt

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-full-recipe/predictions.json \
  --report outputs/qwen3.5-27b-full-recipe/report.txt

CUDA_VISIBLE_DEVICES=3 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/proassist-i5/predictions.json \
  --report outputs/proassist-i5/report.txt