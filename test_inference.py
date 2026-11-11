import unittest
from run_inference import frame_indices, parse_answer


class ProtocolTests(unittest.TestCase):
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
