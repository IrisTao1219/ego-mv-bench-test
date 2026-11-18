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
    def run_recipe(self, replies, times=None, dry_run=False, interval=0.5, frame_count=21,
                   dedup_rule=None, per_step=False, continue_run=False):
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
                get=lambda prop: 1 if prop == cv2.CAP_PROP_FPS else frame_count,
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
            if continue_run:
                output = root / 'out'
                output.mkdir()
                kept = {'video_id': 'earlier', 'pred_texts': ['kept'], 'pred_timestamps': [0]}
                (output / 'predictions.json').write_text(json.dumps([kept]))
                (output / 'v.jsonl').write_text('interrupted log\n')
                args.append('--continue')
            if dry_run:
                args.append('--dry-run')
            if dedup_rule is not None:
                args.extend(['--feedback-dedup-rule', dedup_rule])
            with patch.object(sys, 'argv', args), patch.dict(sys.modules, transformers=transformers, torch=torch), \
                 patch.object(cv2, 'VideoCapture', return_value=cap), contextlib.redirect_stdout(io.StringIO()):
                if per_step:
                    from run_inference import main as per_step_main
                    per_step_main()
                else:
                    main()
            result = json.loads((root/'out/predictions.json').read_text())
            if continue_run:
                self.assertEqual(result[0], kept)
                result = result[1:]
            logs = [json.loads(line) for line in (root/'out/v.jsonl').read_text().splitlines()]
            inputs = (root/'out/v.model_inputs.jsonl').read_text()
            if dedup_rule is not None:
                config = json.loads((root/'out/config.json').read_text())
                self.assertEqual(config['feedback_dedup_rule'], dedup_rule)
                self.assertEqual(config['paper_prompt_verbatim'], dedup_rule == 'paper')
                from run_inference import DEDUP_RULE
                self.assertEqual(config['mistake_dedup_rule'],
                                 DEDUP_RULE if dedup_rule == 'explicit' else None)
                self.assertEqual(config['prompt_version'], 'arxiv_2606.09547v2_appendix_D_' +
                                 ('explicit_dedup_v1' if dedup_rule == 'explicit' else 'verbatim'))
            if not dry_run:
                validate_predictions([sample], result)
            return result[0], logs, inputs

    def test_continue_overwrites_partial_log_and_preserves_previous_predictions(self):
        result, logs, _ = self.run_recipe(
            ['YES. First done.', 'YES. Second done.'], continue_run=True)
        self.assertEqual(result['video_id'], 'v')
        self.assertEqual(len(logs), 2)

    def test_cli_dedup_policy_controls_both_protocols(self):
        from run_inference import DEDUP_RULE
        for per_step in (False, True):
            for policy in ('paper', 'explicit'):
                with self.subTest(per_step=per_step, policy=policy):
                    _, logs, inputs = self.run_recipe(
                        [], times=[0, 1, 2], dry_run=True, interval=1, frame_count=4,
                        dedup_rule=policy, per_step=per_step)
                    for row in logs:
                        self.assertEqual(DEDUP_RULE in row['prompt'],
                                         row['stage'] == 'mistake' and policy == 'explicit')
                    rows = [json.loads(line) for line in inputs.splitlines()]
                    self.assertEqual(len(rows), len(logs))
                    for row, log in zip(rows, logs):
                        self.assertEqual(row['messages'][-1]['content'][-1]['text'], log['prompt'])

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

    def test_feedback_expires_after_sixty_seconds_and_resets_on_transition(self):
        _, _, inputs = self.run_recipe(
            ['NO', 'YES. Add more salt.', 'NO', 'NO', 'YES. First done.', 'YES. Second done.'],
            interval=65, frame_count=301)
        rows = [json.loads(line) for line in inputs.splitlines()]
        self.assertEqual([m['role'] for m in rows[0]['messages']], ['user'])
        later = next(row for row in rows if row['timestamp'] == 130 and row['stage'] == 'mistake')
        self.assertNotIn('Add more salt.', json.dumps(later['messages']))
        completion = next(row for row in rows if row['timestamp'] == 130 and row['stage'] == 'completion')
        self.assertEqual([m['role'] for m in completion['messages']], ['user'])
        self.assertNotIn('Add more salt.', json.dumps(completion['messages']))
        next_step = next(row for row in rows if row['timestamp'] == 260)
        self.assertEqual(len(next_step['messages']), 1)
        self.assertNotIn('Add more salt.', json.dumps(next_step))

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
