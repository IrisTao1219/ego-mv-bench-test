import unittest
from evaluate_official import validate_predictions


class EvaluationInputTests(unittest.TestCase):
    def prediction(self):
        return {'video_id': 'video', 'pred_texts': ['Instruct: Add oil', 'Feedback: Add more oil'],
                'pred_timestamps': [0, 5]}

    def test_inference_format_and_coverage(self):
        seen, ids = validate_predictions([{'video_id': 'video'}, {'video_id': 'other'}],
                                         [self.prediction()])
        self.assertEqual(seen, {'video'})
        self.assertEqual(ids - seen, {'other'})

    def test_invalid_events(self):
        for field, value in [('pred_timestamps', [0, float('nan')]),
                             ('pred_timestamps', [0, float('inf')]),
                             ('pred_timestamps', [5, 0]),
                             ('pred_timestamps', [0, True]),
                             ('pred_timestamps', [0]),
                             ('pred_texts', ['Instruct: Add oil', 'Feedback:']),
                             ('pred_texts', ['Success: Done', 'Feedback: Add oil']),
                             ('video_id', 'unknown')]:
            with self.subTest(field=field, value=value):
                row = self.prediction()
                row[field] = value
                with self.assertRaises(ValueError):
                    validate_predictions([{'video_id': 'video'}], [row])

    def test_empty_and_duplicate_predictions(self):
        for predictions in ([], [self.prediction(), self.prediction()]):
            with self.assertRaises(ValueError):
                validate_predictions([{'video_id': 'video'}], predictions)


if __name__ == '__main__':
    unittest.main()
