CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3-vl-30b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-27b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-per-step-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-9b-per-step-explicit-dedup/official-report.txt \
  --require-full-coverage



CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3-vl-30b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=1 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-27b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-full-recipe-explicit-dedup/predictions.json \
  --report outputs/qwen3.5-9b-full-recipe-explicit-dedup/official-report.txt \
  --require-full-coverage
