import unittest
from run_inference import recipe_steps, check_response, prompt_for
from paper_prompts import COMPLETION_PROMPT, DETAILED_MISTAKE_PROMPT, SIMPLE_MISTAKE_PROMPT


class InferenceTests(unittest.TestCase):
    def test_no_reference_feedback_in_prompt(self):
        sample = {'texts': ['Get a bowl', 'SECRET correction', 'Great', 'Crack eggs'],
                  'text_types': ['Instruction', 'Feedback', 'Success', 'Instruction'],
                  'text_timestamps': [17., 24., 34., 36.]}
        steps = recipe_steps(sample)
        self.assertEqual(steps, [{'instruction': 'Get a bowl', 'start': 17.},
                                 {'instruction': 'Crack eggs', 'start': 36.}])
        for stage in ('completion', 'mistake'):
            self.assertNotIn('SECRET', prompt_for(stage, steps[0]['instruction'], 22, []))

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
        self.assertEqual(prompt_for('mistake', 'Add oil', 42, recent), expected)
        self.assertEqual(prompt_for('mistake', 'Add oil', 42, recent, 'simple'),
                         SIMPLE_MISTAKE_PROMPT.replace('[recipe_step]', 'Add oil'))
        self.assertNotIn('Video observation time', expected)

    def test_invalid_stage_and_variant(self):
        with self.assertRaises(ValueError):
            prompt_for('unknown', 'x', 0, [])
        with self.assertRaises(ValueError):
            prompt_for('mistake', 'x', 0, [], 'unknown')
        with self.assertRaises(ValueError):
            check_response('NO', 'unknown')


if __name__ == '__main__':
    unittest.main()
