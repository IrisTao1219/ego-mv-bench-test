import unittest
from run_inference import recipe_steps, check_response, prompt_for


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

    def test_strict_decision(self):
        self.assertFalse(check_response('{"completed":false,"reason":"Still pouring"}', 'completed')['completed'])
        for raw in ('{"completed":"false","reason":"x"}', '[]',
                    '{"completed":true,"reason":""}'):
            with self.assertRaises(ValueError):
                check_response(raw, 'completed')
        with self.assertRaises(ValueError):
            check_response('{"mistake":true,"reason":"Too much"}', 'mistake')


if __name__ == '__main__':
    unittest.main()
