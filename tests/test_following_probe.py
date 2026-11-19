"""Validate experimental controls without loading model dependencies."""
import contextlib
import io
import unittest

from prompts import DEDUP_RULE
from tests.test_9b_following import repeat_probe


class FollowingProbeTests(unittest.TestCase):
    def test_paired_inputs_and_new_incident_controls(self):
        calls = []

        def generate(label, payload):
            calls.append((label, payload))
            self.assertIsInstance(payload, str)
            text = payload
            if 'NEW, SEPARATE execution' in text:
                return 'YES. Add one more teaspoon.'
            if text.endswith(DEDUP_RULE):
                return 'NO'
            return 'YES. Add one more teaspoon.'

        with contextlib.redirect_stdout(io.StringIO()):
            rows = repeat_probe(generate)
        self.assertEqual(len(calls), 21)  # 3 first calls + 3 x 3 x 2 comparisons
        self.assertEqual(len(rows), 18)
        for offset in (0, 7, 14):
            for variant in range(3):
                start = offset + 1 + variant * 2
                original, explicit = [x[1] for x in calls[start:start + 2]]
                self.assertEqual(explicit, original + '\n\n' + DEDUP_RULE)
                self.assertIn('previously detected mistakes:["Add one more teaspoon."]', original)
        controls = [r for r in rows if r['expected_decision'] == 'YES']
        self.assertEqual(len(controls), 6)
        self.assertTrue(all(r['passed'] and not r['repeated_trigger'] for r in controls))
        self.assertTrue(all(r['exact_same_text'] for r in controls))

    def test_original_repeat_mode_still_has_two_policies(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = repeat_probe(lambda *_: 'YES. Add more salt.')
        self.assertEqual(len(rows), 18)
        self.assertEqual({r['prompt_policy'] for r in rows}, {'original', 'explicit_dedup'})


if __name__ == '__main__':
    unittest.main()
