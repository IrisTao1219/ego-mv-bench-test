"""Exercise the real full-recipe loop with scripted model outputs (no GPU)."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_full_recipe import main, instruction_texts
from evaluate_official import validate_predictions


class FakeTokens:
    shape = (1, 2)

    def __getitem__(self, key):
        return self

    def item(self):
        return 0


class FakeInputs(dict):
    input_ids = FakeTokens()

    def to(self, device):
        return self


class FullRecipeTests(unittest.TestCase):
    def run_recipe(self, replies, times=None, dry_run=False, interval=0.5):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'annotations').mkdir()
            (root / 'videos').mkdir()
            (root / 'videos/v.MP4').touch()
            sample = {'video_id': 'v', 'texts': ['FIRST', 'SECRET', 'SECOND'],
                      'text_types': ['Instruction', 'Feedback', 'Instruction'],
                      'text_timestamps': times or [99, 199, 299]}
            (root / 'annotations/annotations.json').write_text(json.dumps([sample]))
            cap = SimpleNamespace(
                get=lambda prop: 1 if prop == cv2.CAP_PROP_FPS else 21,
                isOpened=lambda: True, set=lambda *args: None,
                read=lambda: (True, np.zeros((32, 32, 3), dtype=np.uint8)),
                release=lambda: None)
            responses = iter(replies)
            processor = SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: FakeInputs(),
                batch_decode=lambda *args, **kwargs: [next(responses)])
            model = SimpleNamespace(device='cpu', config=SimpleNamespace(model_type='fake'),
                generation_config=SimpleNamespace(eos_token_id=0),
                generate=lambda **kwargs: FakeTokens())
            model.eval = lambda: model
            transformers = SimpleNamespace(__version__='fake',
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
                AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model))
            torch = SimpleNamespace(__version__='fake', inference_mode=contextlib.nullcontext)
            args = ['run_full_recipe.py', '--data-root', str(root), '--output-dir', str(root/'out'), '--interval', str(interval)]
            if dry_run:
                args.append('--dry-run')
            with patch.object(sys, 'argv', args), patch.dict(sys.modules, transformers=transformers, torch=torch), \
                 patch.object(cv2, 'VideoCapture', return_value=cap), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root/'out/predictions.json').read_text())
            logs = [json.loads(line) for line in (root/'out/v.jsonl').read_text().splitlines()]
            inputs = (root/'out/v.model_inputs.jsonl').read_text()
            if not dry_run:
                validate_predictions([sample], result)
            return result[0], logs, inputs

    def test_model_completion_drives_continuous_transitions(self):
        result, logs, inputs = self.run_recipe(['YES. Done first.', 'NO',
                                               'YES. Fix it.', 'YES. Done second.'])
        self.assertEqual(result['pred_timestamps'], [0, .5, .5, 1, 1.5])
        self.assertEqual(result['pred_texts'][2], 'Instruct: SECOND')
        self.assertEqual([(x['timestamp'], x['step_id'], x['stage']) for x in logs],
                         [(.5, 0, 'completion'), (1, 1, 'completion'),
                          (1, 1, 'mistake'), (1.5, 1, 'completion')])
        self.assertEqual(logs[1]['frame_timestamps'][0], 1)
        self.assertNotIn('SECRET', inputs)
        self.assertEqual(result, self.run_recipe(
            ['YES. Done first.', 'NO', 'YES. Fix it.', 'YES. Done second.'], [0, 1, 2])[0])

    def test_stalled_and_malformed_completion_never_force_advance(self):
        result, logs, _ = self.run_recipe(['invalid', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO'], interval=5)
        self.assertEqual(result['pred_texts'], ['Instruct: FIRST'])
        self.assertTrue(all(row['step_id'] == 0 for row in logs))
        self.assertEqual([row['stage'] for row in logs[:2]], ['completion', 'completion'])
        self.assertIsNotNone(logs[0]['parse_error'])

    def test_instruction_extraction_does_not_require_timestamps(self):
        self.assertEqual(instruction_texts({'texts': ['A', 'secret', 'B'],
            'text_types': ['Instruction', 'Success', 'Instruction']}), ['A', 'B'])

    def test_dry_run_has_no_invented_transitions_or_predictions(self):
        result, logs, _ = self.run_recipe([], dry_run=True)
        self.assertEqual(result['pred_texts'], [])
        self.assertTrue(all(row['step_id'] == 0 for row in logs))


if __name__ == '__main__':
    unittest.main()
