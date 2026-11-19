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
from run_proassist import ProAssistBackend, parse_event, run_video, cache_tokens, system_rules


class ProAssistTests(unittest.TestCase):
    def run_sample(self, protocol='full_recipe', replies=(), times=None, dry=False, count=12):
        import cv2
        import numpy as np
        sample = dict(video_id='v', texts=['FIRST', 'SECRET', 'SECOND'],
                      text_types=['Instruction', 'Feedback', 'Instruction'],
                      text_timestamps=times or [1, 2, 4])
        cap = SimpleNamespace(isOpened=lambda: True,
            get=lambda prop: 2 if prop == cv2.CAP_PROP_FPS else count,
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
                return dict(responses=[dict(stage='stream', raw_response=text,
                    parse_error=None, accepted=parse_event(text))])
        backend = None if dry else Backend()
        args = SimpleNamespace(protocol=protocol, sample_fps=2, max_seconds=0, dry_run=dry)
        log = io.StringIO()
        result = run_video(sample, args, backend, cap, log)
        if not dry:
            validate_predictions([sample], [result])
        return result, [json.loads(x) for x in log.getvalue().splitlines()], backend

    def test_full_recipe_frame_clock_and_model_transitions(self):
        replies = ['', 'Add more salt.', 'Yes. Done.', '', 'yes. Done.']
        result, logs, backend = self.run_sample(replies=replies)
        self.assertEqual(result['pred_timestamps'], [0, .5, 1, 1, 2])
        self.assertEqual(result['pred_texts'][3], 'Instruct: SECOND')
        self.assertEqual(backend.messages, [('reset', 'FIRST'), ('next', 'SECOND')])
        self.assertEqual(len(logs), 5)
        self.assertEqual(result, self.run_sample(replies=replies, times=[90, 100, 200])[0])
        self.assertNotIn('SECRET', json.dumps(logs))

    def test_all_non_yes_speech_is_feedback_without_advancing(self):
        result, logs, backend = self.run_sample(replies=['Great job!', 'No', 'Not yet', 'Fix it.'])
        self.assertEqual(result['pred_texts'], ['Instruct: FIRST', 'Feedback: Great job!',
                                              'Feedback: No', 'Feedback: Not yet', 'Feedback: Fix it.'])
        self.assertTrue(all(row['step_id'] == 0 for row in logs))
        self.assertTrue(all(row['parse_error'] is None for row in logs))
        self.assertEqual(backend.messages, [('reset', 'FIRST')])

    def test_per_step_resets_at_next_instruction_without_success(self):
        result, logs, backend = self.run_sample(protocol='per_recipe_step')
        self.assertEqual(result['pred_timestamps'], [1, 4])
        self.assertEqual(backend.messages, [('reset', 'FIRST'), ('reset', 'SECOND')])
        self.assertEqual(logs[0]['timestamp'], 1)
        self.assertEqual(next(x for x in logs if x['step_id'] == 1)['timestamp'], 4)

    def test_per_step_completion_restarts_at_reference_instruction(self):
        result, logs, backend = self.run_sample(protocol='per_recipe_step',
                                               replies=['Yes. Done.', 'YES. Done.'])
        self.assertEqual(result['pred_timestamps'], [1, 1, 4, 4])
        self.assertEqual(len(logs), 2)

    def test_dry_run_does_not_fabricate_events(self):
        result, logs, _ = self.run_sample(dry=True)
        self.assertEqual(result['pred_texts'], [])
        self.assertEqual(len(logs), 12)

    def test_long_stream_processes_each_frame_once_without_reset(self):
        _, logs, backend = self.run_sample(count=160)
        self.assertEqual([r['frame_index'] for r in logs], list(range(160)))
        self.assertEqual(backend.messages, [('reset', 'FIRST')])

    def test_native_speech_event_mapping(self):
        for text in ('', '  ', '\n'):
            self.assertIsNone(parse_event(text))
        for text in ('Yes', 'yes. Done.', 'YES, ready.', 'yEs: finished'):
            self.assertEqual(parse_event(text), 'Success: Step complete.')
        for text in ('No', 'Not yet', 'Yesterday', 'Add salt.', 'I said yes earlier.'):
            self.assertEqual(parse_event(text), 'Feedback: ' + text)
        self.assertEqual(parse_event('  Add salt.\nUse a teaspoon.  '),
                         'Feedback: Add salt.\nUse a teaspoon.')

    def test_cache_size_and_shared_system_rules(self):
        import torch
        self.assertEqual(cache_tokens(None), 0)
        self.assertEqual(cache_tokens(((torch.zeros(1, 2, 19, 3), None),)), 19)
        self.assertEqual(cache_tokens(SimpleNamespace(get_seq_length=lambda: 27)), 27)
    def test_backend_checkpoint_paths_and_stream_cache(self):
        import torch
        import numpy as np
        calls = []
        buffer = torch.tensor([[5, 6]])
        config = SimpleNamespace(num_tokens_per_img=1, use_binary_decision_head=False,
                                 eos_token_id=6, w2t_target_id=7)
        def generate(embeds, past, **kwargs):
            model.last_not_talk_prob = torch.tensor(.25)
            calls.append(('generate', past, kwargs))
            buffer[:] = torch.tensor([[5, 6]])
            return buffer, 'cache-' + str(sum(c[0] == 'generate' for c in calls))
        model = SimpleNamespace(config=config, device='cpu', dtype=torch.float32,
            eval=lambda: None, joint_embed=lambda **kw: kw,
            forward=lambda **kw: SimpleNamespace(past_key_values='prefill-cache'),
            fast_greedy_generate=generate)
        def add_last(inputs, last):
            calls.append(('last', None if last is None else last.clone()))
            return inputs
        cleaned_replies = iter(['Fix it.', 'yes. Done.', ''])
        processor = SimpleNamespace(
            get_input_sequence=lambda n, messages, first: (torch.tensor([[1]]), str(messages)),
            add_last_assistant_message=add_last,
            cleanup_text=lambda raw: (next(cleaned_replies), 'assistant'))
        def manage(frame, past, last):
            self.assertEqual(frame.model_inputs['images'].shape, (1, 3, 8, 8))
            # Simulate upstream summary overwriting the shared generation buffer.
            buffer.fill_(99)
            return 'managed-' + past, last
        runner = SimpleNamespace(processor=processor, manage_kv_cache=manage)
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids.tolist()), img_sep_token_id=6)
        def build(path, rank):
            cfg = json.loads((Path(path) / 'config.json').read_text())
            self.assertIn('Meta-Llama-3', cfg['llm_pretrained'])
            self.assertIn('siglip', cfg['vision_pretrained'])
            self.assertEqual(cfg['exceed_context_handling'], 'summarize_and_drop')
            self.assertTrue((Path(path) / 'adapter_model.safetensors').is_file())
            self.assertTrue(Path(cfg['llm_pretrained']).is_dir())
            return model, tokenizer
        def build_runner(**kwargs):
            self.assertEqual(kwargs['fps'], 2)
            self.assertEqual(kwargs['eval_max_seq_len'], 4096)
            self.assertEqual(kwargs['reserved_seq_len'], 132)
            return runner
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
                context_length=4096, max_new_tokens=100, not_talk_threshold=.3,
                max_side=448, mistake_prompt='detailed', feedback_dedup_rule='explicit',
                assistant_feedback=True)
            modules = {
                'mmassist': SimpleNamespace(),
                'mmassist.model': SimpleNamespace(build_from_checkpoint=build),
                'mmassist.eval': SimpleNamespace(),
                'mmassist.eval.runners': SimpleNamespace(),
                'mmassist.eval.runners.stream_inference': SimpleNamespace(
                    StreamInferenceRunner=SimpleNamespace(build=build_runner))}
            with patch.dict('sys.modules', modules), patch.dict(os.environ):
                backend = ProAssistBackend(args)
                try:
                    backend.reset('FIRST')
                    first = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), 0)['responses'][0]
                    second = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), .5)['responses'][0]
                    backend.set_instruction('SECOND')
                    third = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), 1)['responses'][0]
                    self.assertEqual(first['decoded_response'], '[5, 6]')
                    self.assertEqual(first['accepted'], 'Feedback: Fix it.')
                    self.assertEqual(second['accepted'], 'Success: Step complete.')
                    self.assertIsNone(third['accepted'])
                    self.assertEqual(first['input_messages'], [
                        {'role': 'system', 'content': system_rules(args)},
                        {'role': 'user', 'content': 'Current recipe step: FIRST'}])
                    self.assertEqual(second['input_messages'], [])
                    self.assertEqual(second['prompt'], '')
                    self.assertEqual(third['input_messages'], [
                        {'role': 'user', 'content': 'Current recipe step: SECOND'}])
                    self.assertEqual(first['not_talk_probability'], .25)
                    self.assertEqual(first['kv_cache_tokens_before'], 0)
                    generated = [c for c in calls if c[0] == 'generate']
                    self.assertEqual([c[1] for c in generated],
                                     [None, 'managed-cache-1', 'managed-cache-2'])
                    self.assertEqual(backend.past, 'managed-cache-3')
                    self.assertTrue(all(r['stage'] == 'stream' for r in (first, second, third)))
                    self.assertNotIn('answer ‘NO’', system_rules(args))
                    self.assertNotIn('Has the person already completed', system_rules(args))
                    self.assertEqual((root/'adapter/config.json').read_text(), original)
                finally:
                    backend.close()


if __name__ == '__main__':
    unittest.main()
