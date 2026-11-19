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
from run_proassist import ProAssistBackend, parse_event, run_video, cache_tokens, check_proassist_response, system_rules
from run_inference import check_response, prediction_event, prompt_for


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
                # Script speech at the task level; parsing is shared with Qwen.
                stage = 'mistake' if text.startswith('Feedback:') else 'completion'
                raw = ('YES. ' + text.split(':', 1)[1].strip()
                       if text.startswith(('Feedback:', 'Success:')) else text)
                field = 'mistake' if stage == 'mistake' else 'completed'
                parsed, error = None, None
                try:
                    parsed = check_response(raw, field) if raw.strip() else None
                except ValueError as exc:
                    error = str(exc)
                event = prediction_event(parsed, field) if parsed else None
                return dict(responses=[dict(stage=stage, raw_response=raw, parsed=parsed,
                                           parse_error=error, accepted=event)])
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
        self.assertEqual(backend.messages, [('reset', 'FIRST'), ('next', 'SECOND')])
        self.assertEqual(len(logs), 5)
        self.assertEqual(result, self.run_sample(replies=replies, times=[90, 100, 200])[0])
        self.assertNotIn('SECRET', json.dumps(logs))

    def test_malformed_speech_never_advances(self):
        result, logs, backend = self.run_sample(replies=['Great job!', 'Success:',
            'Perhaps it is done', 'Feedback: Fix it.'])
        self.assertEqual(result['pred_texts'], ['Instruct: FIRST', 'Feedback: Fix it.'])
        self.assertTrue(all(row['step_id'] == 0 for row in logs))
        self.assertIsNotNone(logs[0]['parse_error'])
        self.assertEqual(set(backend.messages), {('reset', 'FIRST')})

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

    def test_long_stream_processes_each_frame_once_without_reset(self):
        _, logs, backend = self.run_sample(count=160)
        self.assertEqual([r['frame_index'] for r in logs], list(range(160)))
        self.assertEqual(backend.messages, [('reset', 'FIRST')])

    def test_shared_yes_no_event_parser(self):
        self.assertIsNone(parse_event('  '))
        self.assertIsNone(parse_event('NO'))
        self.assertEqual(parse_event('YES. Add salt.', 'mistake'), 'Feedback: Add salt.')
        self.assertEqual(parse_event('YES. Both eggs are in the bowl.'),
                         'Success: Step complete.')
        for text in ['Great job!', 'YES', 'Feedback: Add salt.', 'Instruct: next']:
            with self.assertRaises(ValueError):
                parse_event(text)

    def test_proassist_case_tolerance_does_not_change_shared_parser(self):
        for negative in ('NO', 'No', 'no', 'nO, not yet.'):
            self.assertFalse(check_proassist_response(negative, 'completed')['completed'])
        for positive in ('YES. Done.', 'Yes, Done.', 'yes: Done.'):
            self.assertEqual(parse_event(positive), 'Success: Step complete.')
        self.assertEqual(parse_event('yes. Add salt.', 'mistake'), 'Feedback: Add salt.')
        for invalid in ('Not yet', 'Nobody', 'Yesterday', 'Yes'):
            with self.assertRaises(ValueError):
                parse_event(invalid)
        with self.assertRaises(ValueError):
            check_response('No', 'completed')

    def test_mistake_silence_prompt_has_no_conflicting_NO_rule(self):
        for variant in ('detailed', 'simple'):
            for dedup in (False, True):
                prompt = prompt_for('mistake', 'FIRST', 0, [], variant, dedup,
                                    silent_negative=True)
                self.assertIn('remain silent', prompt)
                self.assertIn('YES. <feedback>', prompt)
                self.assertNotIn('answer NO', prompt)
                self.assertNotIn('else ‘No’', prompt)
                self.assertIn('else ‘No’', prompt_for('mistake', 'FIRST', 0, [], variant, dedup))
        self.assertEqual(prompt_for('completion', 'FIRST', 0, [], silent_negative=True),
                         prompt_for('completion', 'FIRST', 0, []))
        self.assertIsNone(parse_event('', 'mistake'))

    def test_silence_and_bad_completion_only_prefill_mistake_stream(self):
        from unittest.mock import Mock, call
        for raw in ('', 'Great job!'):
            backend = ProAssistBackend.__new__(ProAssistBackend)
            backend.recent = [{'time': 0, 'feedback': 'old feedback'}]
            completion = dict(parsed=None, accepted=None, raw_response=raw)
            backend._run_stage = Mock(side_effect=[completion,
                dict(context_only=True, kv_cache_tokens_after_management=17)])
            self.assertEqual(backend.step('rgb', 61)['responses'], [completion])
            self.assertEqual(backend._run_stage.call_args_list, [
                call('rgb', 61, 'completion'),
                call('rgb', 61, 'mistake', context_only=True)])
            self.assertEqual(backend.recent, [])
            self.assertTrue(completion['mistake_context_only'])

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
        cleaned_replies = iter(['No', 'Yes. Fix it.', 'yes. Done.'])
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
                    first = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), 0)
                    backend.set_instruction('SECOND')
                    second = backend.step(np.zeros((8, 8, 3), dtype=np.uint8), .5)
                    stages = first['responses']
                    self.assertEqual([r['stage'] for r in stages], ['completion', 'mistake'])
                    self.assertEqual(stages[0]['decoded_response'], '[5, 6]')
                    self.assertEqual(stages[0]['raw_response'], 'No')
                    self.assertFalse(stages[0]['hit_token_limit'])
                    self.assertEqual(stages[1]['accepted'], 'Feedback: Fix it.')
                    self.assertEqual(second['responses'][0]['accepted'], 'Success: Step complete.')
                    for response, stage in zip(stages, ('completion', 'mistake')):
                        self.assertEqual([m['role'] for m in response['input_messages']], ['system', 'user'])
                        self.assertEqual(response['input_messages'][0]['content'], system_rules(args, stage))
                        self.assertIn('recipe_step: FIRST', response['prompt'])
                    self.assertNotIn('mistakes occur', system_rules(args, 'completion').lower())
                    self.assertNotIn('Has the person already completed', system_rules(args, 'mistake'))
                    self.assertEqual([m['role'] for m in second['responses'][0]['input_messages']], ['user'])
                    self.assertIn('recipe_step: SECOND', second['responses'][0]['prompt'])
                    self.assertIn('YES', system_rules(args, 'completion'))
                    self.assertIn('NO', system_rules(args, 'mistake'))
                    self.assertEqual(stages[0]['not_talk_probability'], .25)
                    self.assertEqual(stages[0]['kv_cache_tokens_before'], 0)
                    generated = [c for c in calls if c[0] == 'generate']
                    self.assertEqual([c[1] for c in generated], [None, None, 'managed-cache-1'])
                    self.assertIsNone([c for c in calls if c[0] == 'last'][1][1])
                    self.assertIsNot(backend.states['completion']['runner'],
                                     backend.states['mistake']['runner'])
                    self.assertEqual(backend.states['completion']['past'], 'managed-cache-3')
                    self.assertEqual(backend.states['mistake']['past'], 'managed-prefill-cache')
                    self.assertTrue(second['responses'][0]['mistake_context_only'])
                    self.assertEqual((root/'adapter/config.json').read_text(), original)
                finally:
                    backend.close()


if __name__ == '__main__':
    unittest.main()
