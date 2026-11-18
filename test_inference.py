import unittest
import json
import tempfile
from pathlib import Path
from run_inference import continuation_state
from run_inference import recipe_steps, check_response, prompt_for, prediction_event, inference_messages
from paper_prompts import COMPLETION_PROMPT, DETAILED_MISTAKE_PROMPT, SIMPLE_MISTAKE_PROMPT
from run_inference import DEDUP_RULE


class InferenceTests(unittest.TestCase):
    def test_continue_restarts_partial_video_and_keeps_completed_predictions(self):
        samples = [{'video_id': value} for value in ('a', 'b', 'c')]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(continuation_state(output, samples), (samples, []))
            previous = [{'video_id': 'a', 'pred_texts': ['kept']}]
            (output / 'predictions.json').write_text(json.dumps(previous))
            self.assertEqual(continuation_state(output, samples), (samples, []))
            (output / 'b.jsonl').touch()
            self.assertEqual(continuation_state(output, samples), (samples[1:], previous))
            previous.append({'video_id': 'b', 'pred_texts': ['rerun']})
            (output / 'predictions.json').write_text(json.dumps(previous))
            self.assertEqual(continuation_state(output, samples), (samples[1:], previous[:1]))

    def test_no_reference_feedback_in_prompt(self):
        sample = {'texts': ['Get a bowl', 'SECRET correction', 'Great', 'Crack eggs'],
                  'text_types': ['Instruction', 'Feedback', 'Success', 'Instruction'],
                  'text_timestamps': [17., 24., 34., 36.]}
        steps = recipe_steps(sample)
        self.assertEqual(steps, [{'instruction': 'Get a bowl', 'start': 17.},
                                 {'instruction': 'Crack eggs', 'start': 36.}])
        for stage in ('completion', 'mistake'):
            self.assertNotIn('SECRET', prompt_for(stage, steps[0]['instruction'], 22, []))

    def test_assistant_history_in_both_stages(self):
        history = [{'time': 1, 'feedback': 'Add salt.'},
                   {'time': 90, 'feedback': 'Lower heat.'}]
        for variant in ('detailed', 'simple'):
            for stage in ('completion', 'mistake'):
                prompt = prompt_for(stage, 'Cook', 200, history, variant)
                messages = inference_messages([], prompt, 'Cook', history)
                self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
                self.assertEqual(messages[0]['role'], 'user')
                self.assertEqual(messages[1]['content'],
                                 'Feedback: Add salt.\nFeedback: Lower heat.')
                self.assertEqual(messages[-1]['content'][-1]['text'], prompt)
                for entry in history:
                    self.assertIn(entry['feedback'], messages[1]['content'])
                    self.assertEqual(entry['feedback'] in prompt,
                                     stage == 'mistake' and variant == 'detailed')
        self.assertEqual(len(inference_messages([], 'Prompt')), 1)

    def test_paper_responses(self):
        for field in ('completed', 'mistake'):
            self.assertFalse(check_response('NO', field)[field])
            self.assertFalse(check_response('NO. Still working.', field)[field])
            self.assertTrue(check_response(' YES. The bowl contains three spoonfuls. ', field)[field])
        self.assertEqual(check_response('YES. Add one more spoonful.', 'mistake')['feedback'],
                         'Add one more spoonful.')
        self.assertEqual(check_response('YES\nThe oven is set.', 'completed')['reason'],
                         'The oven is set.')

    def test_malformed_response_is_not_negative(self):
        for field in ('completed', 'mistake'):
            for raw in ('YES', 'YES.', 'Yesterday it was done', 'Not yet',
                        '{"completed":false,"reason":"Still pouring"}',
                        '```text\nNO\n```', ''):
                with self.subTest(field=field, raw=raw), self.assertRaises(ValueError):
                    check_response(raw, field)

    def test_exact_prompt_substitution(self):
        self.assertEqual(prompt_for('completion', 'Get a bowl', 42, []),
                         COMPLETION_PROMPT.replace('[recipe_step]', 'Get a bowl'))
        recent = [{'time': 40, 'feedback': 'Add one more spoonful.'}]
        expected = DETAILED_MISTAKE_PROMPT.replace('[recipe_step]', 'Add oil').replace(
            '[prev_detected_mistakes]', '["Add one more spoonful."]')
        self.assertEqual(prompt_for('mistake', 'Add oil', 42, recent), expected + '\n\n' + DEDUP_RULE)
        self.assertEqual(prompt_for('mistake', 'Add oil', 42, recent, explicit_dedup=False), expected)
        self.assertEqual(prompt_for('mistake', 'Add oil', 42, recent, 'simple'),
                         SIMPLE_MISTAKE_PROMPT.replace('[recipe_step]', 'Add oil') + '\n\n' + DEDUP_RULE)
        self.assertNotIn('Video observation time', expected)

    def test_completion_reason_is_logged_not_displayed(self):
        parsed = check_response('YES. The bowl is visible in frame 12.', 'completed')
        self.assertEqual(parsed['reason'], 'The bowl is visible in frame 12.')
        self.assertEqual(prediction_event(parsed, 'completed'), 'Success: Step complete.')
        self.assertNotIn('feedback', parsed)
        self.assertIsNone(prediction_event(check_response('NO', 'completed'), 'completed'))

    def test_mistake_feedback_is_preserved(self):
        parsed = check_response('YES. Add one more spoonful.', 'mistake')
        self.assertNotIn('reason', parsed)
        self.assertEqual(prediction_event(parsed, 'mistake'), 'Feedback: Add one more spoonful.')
        self.assertIsNone(prediction_event(check_response('NO', 'mistake'), 'mistake'))

    def test_invalid_stage_and_variant(self):
        with self.assertRaises(ValueError):
            prompt_for('unknown', 'x', 0, [])
        with self.assertRaises(ValueError):
            prompt_for('mistake', 'x', 0, [], 'unknown')
        with self.assertRaises(ValueError):
            check_response('NO', 'unknown')


if __name__ == '__main__':
    unittest.main()
