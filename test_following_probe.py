"""Validate experimental controls without loading model dependencies."""
import contextlib
import io
import unittest

from test_9b_following import DEDUP_RULE, repeat_probe


class FollowingProbeTests(unittest.TestCase):
    def test_factorial_inputs_and_new_incident_controls(self):
        calls = []

        def generate(label, payload):
            calls.append((label, payload))
            text = payload[-1]['content'][-1]['text'] if isinstance(payload, list) else payload
            if 'NEW, SEPARATE execution' in text:
                return 'YES. Add one more teaspoon.'
            if text.endswith(DEDUP_RULE):
                return 'NO'
            return 'YES. Add one more teaspoon.'

        with contextlib.redirect_stdout(io.StringIO()):
            rows = repeat_probe(generate, factorial=True)
        self.assertEqual(len(calls), 39)  # 3 first calls + 3 x 3 x 4 comparisons
        self.assertEqual(len(rows), 36)
        for offset in (0, 13, 26):
            for variant in range(3):
                start = offset + 1 + variant * 4
                original, history, explicit, combined = [x[1] for x in calls[start:start + 4]]
                self.assertEqual([m['role'] for m in history], ['user', 'assistant', 'user'])
                self.assertEqual(history[1]['content'], 'Feedback: Add one more teaspoon.')
                self.assertEqual(history[-1]['content'][-1]['text'], original)
                self.assertEqual(explicit, original + '\n\n' + DEDUP_RULE)
                self.assertEqual(combined[:-1], history[:-1])
                self.assertEqual(combined[-1]['content'][-1]['text'], explicit)
        controls = [r for r in rows if r['expected_decision'] == 'YES']
        self.assertEqual(len(controls), 12)
        self.assertTrue(all(r['passed'] and not r['repeated_trigger'] for r in controls))
        self.assertTrue(all(r['exact_same_text'] for r in controls))

    def test_original_repeat_mode_still_has_two_policies(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = repeat_probe(lambda *_: 'YES. Add more salt.')
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['message_structure'] == 'single_user' for r in rows))


if __name__ == '__main__':
    unittest.main()
