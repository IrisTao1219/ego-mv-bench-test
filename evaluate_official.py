"""Run the vendored Qualcomm evaluator against local annotations."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads((a.data_root / 'annotations/annotations.json').read_text())
    predictions = json.loads(a.predictions.read_text())
    ids = {r['video_id'] for r in rows}
    seen = set()
    for r in predictions:
        if r['video_id'] not in ids or r['video_id'] in seen:
            p.error('Unknown or duplicate prediction video ID')
        seen.add(r['video_id'])
        texts, times = r['pred_texts'], r['pred_timestamps']
        if not texts or len(texts) != len(times) or times != sorted(times):
            p.error('Predictions must be nonempty, aligned and chronological')
        if not texts[0].startswith('Instruct:'):
            p.error('Use run_inference.py outputs (first event must be Instruct:)')
    if not predictions:
        p.error('No predictions')
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
