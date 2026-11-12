"""Run the vendored Qualcomm evaluator against local annotations."""
import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch


def validate_predictions(rows, predictions):
    """Validate inference output before importing GPU metric dependencies."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Annotations must be a nonempty JSON list')
    ids = {r['video_id'] for r in rows}
    if not isinstance(predictions, list) or not predictions:
        raise ValueError('Predictions must be a nonempty JSON list')
    seen = set()
    for r in predictions:
        if not isinstance(r, dict) or not isinstance(r.get('video_id'), str):
            raise ValueError('Each prediction needs a string video_id')
        if r['video_id'] not in ids or r['video_id'] in seen:
            raise ValueError('Unknown or duplicate prediction video ID')
        seen.add(r['video_id'])
        texts, times = r.get('pred_texts'), r.get('pred_timestamps')
        if not isinstance(texts, list) or not isinstance(times, list) or not texts or len(texts) != len(times):
            raise ValueError('Predictions must be nonempty, aligned lists')
        if any(not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError('Timestamps must be finite nonnegative numbers')
        if times != sorted(times):
            raise ValueError('Predictions must be chronological')
        if any(not isinstance(t, str) or not t.startswith(('Instruct:', 'Feedback:', 'Success:'))
               or not t.partition(':')[2].strip() for t in texts):
            raise ValueError('Events need Instruct:, Feedback: or Success: prefixes and nonempty text')
        if not texts[0].startswith('Instruct:'):
            raise ValueError('Use run_inference.py outputs (first event must be Instruct:)')
    return seen, ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--check-only', action='store_true', help='Validate inputs without loading GPU metrics')
    p.add_argument('--require-full-coverage', action='store_true', help='Reject missing annotated videos')
    a = p.parse_args()
    if not a.check_only and a.report is None:
        p.error('--report is required unless --check-only is used')
    if a.report is not None and a.report.exists() and not a.check_only:
        p.error('Report already exists; choose a new --report path')
    try:
        rows = json.loads((a.data_root / 'annotations/annotations.json').read_text())
        predictions = json.loads(a.predictions.read_text())
        seen, ids = validate_predictions(rows, predictions)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    if a.require_full_coverage and seen != ids:
        p.error(f'Missing predictions for {len(ids - seen)} annotated videos')
    if a.check_only:
        print(f'Inputs valid: {len(seen)} / {len(ids)} annotated videos; metrics not computed.')
        return
    # Official run() scores only submitted videos. Never silently call this full-dataset coverage.
    print(f'Evaluating {len(seen)} / {len(ids)} annotated videos; use only complete runs for reporting.')
    sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
    import data
    import eval as official_eval
    class LocalRows:
        def to_list(self):
            return rows
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open('x') as report, redirect_stdout(report):
        print('Source: QualcommInteractiveCookingEvaluator, unmodified vendored evaluator')
        print(f'Submitted videos: {len(seen)} / {len(ids)}')
        with patch.object(data, 'load_dataset', return_value=LocalRows()):
            dataset = data.EgoMCBench(split='test')
        official_eval.run(dataset, predictions)
    print(f'Report: {a.report}')


if __name__ == '__main__':
    main()
