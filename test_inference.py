import unittest
from run_inference import frame_indices, parse_answer, build_prompt, is_recent_duplicate


class ProtocolTests(unittest.TestCase):
    def test_prompt_does_not_offer_pipe_type(self):
        self.assertNotIn('Instruction|Feedback|Success|None', build_prompt(0, []))
        with self.assertRaises(ValueError):
            parse_answer('{"type":"Instruction|Feedback|Success|None","text":"x"}')

    def test_duplicate_window(self):
        history = [{'time': 0, 'message': 'Success: Done.'}]
        self.assertTrue(is_recent_duplicate('success:  DONE.', 5, history))
        self.assertFalse(is_recent_duplicate('Success: Done.', 31, history))
        self.assertFalse(is_recent_duplicate('Feedback: Stop.', 5, history))

    def test_causal_sampling(self):
        for current in (0, 1, 30, 301):
            indices = frame_indices(current, 30, 4, 10)
            self.assertLessEqual(len(indices), 4)
            self.assertEqual(indices[-1], current)
            self.assertTrue(all(0 <= i <= current for i in indices))

    def test_response_protocol(self):
        self.assertEqual(parse_answer('```json\n{"type":"Feedback","text":"Stop."}\n```'),
                         ('Feedback', 'Stop.'))
        self.assertEqual(parse_answer('{"type":"None","text":""}'), ('None', ''))
        for value in ('[]', '{"type":"Oops","text":"x"}', '{"type":"Feedback","text":""}'):
            with self.assertRaises(ValueError):
                parse_answer(value)


if __name__ == '__main__':
    unittest.main()
