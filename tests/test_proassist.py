"""Protocol tests using scripted frame-triggered speech; no GPU/model downloads."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate_official import validate_predictions
from run_proassist import ProAssistBackend, parse_event, run_video


class ProAssistTests(unittest.TestCase):
    def run_sample(self, protocol='full_recipe', replies=(), times=None, dry=False):
        import cv2
        import numpy as np
        sample = dict(video_id='v', texts=['FIRST', 'SECRET', 'SECOND'],
                      text_types=['Instruction', 'Feedback', 'Instruction'],
                      text_timestamps=times or [1, 2, 4])
        cap = SimpleNamespace(isOpened=lambda: True,
            get=lambda prop: 2 if prop == cv2.CAP_PROP_FPS else 12,
            set=lambda *args: None,
            read=lambda: (True, np.zeros((8, 8, 3), dtype=np.uint8)))
        class Backend:
            def __init__(self):
                self.messages = []
                self.replies = iter(replies)
            def reset(self, instruction):
                self.messages.append(('reset', instruction))
            def set_instruction(self, instruction):
                self.messages.append(('next', instruction))
            def step(self, rgb, timestamp):
                text = next(self.replies, '')
                return dict(raw_response=text, text=text)
        backend = None if dry else Backend()
        args = SimpleNamespace(protocol=protocol, sample_fps=2, max_seconds=0, dry_run=dry)
        log = io.StringIO()
        result = run_video(sample, args, backend, cap, log)
        if not dry:
            validate_predictions([sample], [result])
        return result, [json.loads(x) for x in log.getvalue().splitlines()], backend

    def test_full_recipe_frame_clock_and_model_transitions(self):
        replies = ['', 'Feedback: Add more salt.', 'Success: Done.', '', 'Success: Done.']
        result, logs, backend = self.run_sample(replies=replies)
        self.assertEqual(result['pred_timestamps'], [0, .5, 1, 1, 2])
        self.assertEqual(result['pred_texts'][3], 'Instruct: SECOND')
        self.assertEqual(backend.messages, [('reset', 'FIRST'), ('reset', 'SECOND')])
        self.assertEqual(len(logs), 5)
        self.assertEqual(result, self.run_sample(replies=replies, times=[90, 100, 200])[0])
        self.assertNotIn('SECRET', json.dumps(logs))

    def test_malformed_speech_never_advances(self):
        result, logs, backend = self.run_sample(replies=['Great job!', 'Success:',
            'Success: done Feedback: more', 'Feedback: Fix it.'])
        self.assertEqual(result['pred_texts'], ['Instruct: FIRST', 'Feedback: Fix it.'])
        self.assertTrue(all(row['step_id'] == 0 for row in logs))
        self.assertIsNotNone(logs[0]['parse_error'])
        self.assertEqual(len(backend.messages), 1)

    def test_per_step_resets_at_next_instruction_without_success(self):
        result, logs, backend = self.run_sample(protocol='per_recipe_step')
        self.assertEqual(result['pred_timestamps'], [1, 4])
        self.assertEqual(backend.messages, [('reset', 'FIRST'), ('reset', 'SECOND')])
        self.assertEqual(logs[0]['timestamp'], 1)
        self.assertEqual(next(x for x in logs if x['step_id'] == 1)['timestamp'], 4)

    def test_per_step_completion_restarts_at_reference_instruction(self):
        result, logs, backend = self.run_sample(protocol='per_recipe_step',
                                               replies=['Success: Done.', 'Success: Done.'])
        self.assertEqual(result['pred_timestamps'], [1, 1, 4, 4])
        self.assertEqual(len(logs), 2)

    def test_dry_run_does_not_fabricate_events(self):
        result, logs, _ = self.run_sample(dry=True)
        self.assertEqual(result['pred_texts'], [])
        self.assertEqual(len(logs), 12)

    def test_strict_event_parser(self):
        self.assertIsNone(parse_event('  '))
        self.assertEqual(parse_event('Feedback: Add salt.'), 'Feedback: Add salt.')
        for text in ['NO', 'Success:', 'Success: done\nFeedback: fix', 'Instruct: next']:
            with self.assertRaises(ValueError):
                parse_event(text)

    def test_backend_checkpoint_paths_and_stream_cache(self):
        import torch
        import numpy as np
        calls = []
        buffer = torch.tensor([[5, 6]])
        config = SimpleNamespace(num_tokens_per_img=1, use_binary_decision_head=False,
                                 eos_token_id=6)
        def generate(embeds, past, **kwargs):
            calls.append(('generate', past, kwargs))
            buffer[:] = torch.tensor([[5, 6]])
            return buffer, 'cache'
        model = SimpleNamespace(config=config, device='cpu', dtype=torch.float32,
            eval=lambda: None, joint_embed=lambda **kw: kw,
            fast_greedy_generate=generate)
        def add_last(inputs, last):
            calls.append(('last', None if last is None else last.clone()))
            return inputs
        processor = SimpleNamespace(
            get_input_sequence=lambda n, messages, first: (torch.tensor([[1]]), str(messages)),
            add_last_assistant_message=add_last,
            cleanup_text=lambda raw: ('Feedback: Fix it.', 'assistant'))
        def manage(frame, past, last):
            self.assertEqual(frame.model_inputs['images'].shape, (1, 3, 224, 224))
            # Simulate upstream summary overwriting the shared generation buffer.
            buffer.fill_(99)
            return 'managed-cache', last
        runner = SimpleNamespace(processor=processor, manage_kv_cache=manage)
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids.tolist()), img_sep_token_id=6)
        def build(path, rank):
            cfg = json.loads((Path(path) / 'config.json').read_text())
            self.assertIn('Meta-Llama-3', cfg['llm_pretrained'])
            self.assertIn('siglip', cfg['vision_pretrained'])
            self.assertTrue((Path(path) / 'adapter_model.safetensors').is_file())
            self.assertTrue(Path(cfg['llm_pretrained']).is_dir())
            return model, tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('adapter', 'base', 'encoder'):
                (root / name).mkdir()
            original = json.dumps(dict(vision_pretrained='google/siglip-test'))
            (root / 'adapter/config.json').write_text(original)
            (root / 'adapter/adapter_model.safetensors').touch()
            args = SimpleNamespace(proassist_root=None, model_path=root/'adapter',
                base_model_path=root/'base', vision_model_path=root/'encoder',
                attn_implementation='sdpa', device_index=0, sample_fps=2,
                context_length=4096, max_new_tokens=100, not_talk_threshold=.3)
            modules = {
                'mmassist': SimpleNamespace(),
                'mmassist.model': SimpleNamespace(build_from_checkpoint=build),
                'mmassist.eval': SimpleNamespace(),
                'mmassist.eval.runners': SimpleNamespace(),
                'mmassist.eval.runners.stream_inference': SimpleNamespace(
                    StreamInferenceRunner=SimpleNamespace(build=lambda **kw: runner))}
            with patch.dict('sys.modules', modules), patch.dict(os.environ):
                backend = ProAssistBackend(args)
                try:
                    backend.reset('FIRST')
                    first = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), 0)
                    backend.set_instruction('SECOND')
                    second = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), .5)
                    self.assertEqual(first['raw_response'], '[5, 6]')
                    self.assertFalse(first['hit_token_limit'])
                    self.assertEqual(second['input_messages'], [{'role': 'user', 'content': 'SECOND'}])
                    self.assertEqual(calls[3][1], 'managed-cache')
                    self.assertEqual(calls[2][1].tolist(), [[6]])
                    self.assertEqual((root/'adapter/config.json').read_text(), original)
                finally:
                    backend.close()


if __name__ == '__main__':
    unittest.main()
