CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3-vl-30b-per-step-dedup-manual/predictions.json \
  --report outputs/qwen3-vl-30b-per-step-dedup-manual/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-27b-per-step-dedup/predictions.json \
  --report outputs/qwen3.5-27b-per-step-dedup/official-report.txt \
  --require-full-coverage

CUDA_VISIBLE_DEVICES=2 python evaluate_official.py \
  --data-root ../datasets/ego-mc-bench \
  --predictions outputs/qwen3.5-9b-per-step-dedup/predictions.json \
  --report outputs/qwen3.5-9b-per-step-dedup/official-report.txt \
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
