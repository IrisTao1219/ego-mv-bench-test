import unittest

from custom_metrics import calculate_custom_metrics


class CustomMetricsTests(unittest.TestCase):
    def score(self, texts, times, mistake=False):
        dataset = [{'video_id': 'v', 'texts': [['Add oil', 'Done']],
                    'texts_timestamps': [[0, 20]], 'has_mistake': [mistake]}]
        predictions = [{'video_id': 'v', 'pred_texts': texts, 'pred_timestamps': times}]
        return calculate_custom_metrics(dataset, predictions, lambda a, b: a == b)

    def test_feedback_does_not_disqualify_relaxed_completion(self):
        m = self.score(['Instruct: Add oil', 'Feedback: More', 'Feedback: More',
                        'Success: Done'], [0, 5, 10, 21])
        self.assertEqual(m['relaxed_ic_accuracy_at_15s'], 1)
        self.assertEqual(m['mistake_free_step_false_alarm_rate'], 1)
        self.assertEqual(m['exact_repeat_feedback_rate'], .5)
        self.assertEqual(m['completion_mean_signed_error_seconds'], 1)

    def test_strict_window_and_duplicate_success(self):
        m = self.score(['Instruct: Add oil', 'Success: Early', 'Success: Done',
                        'Success: Again'], [0, 5, 20, 21])
        self.assertEqual(m['completion_tp'], 1)
        self.assertEqual(m['completion_fp'], 2)
        self.assertEqual(m['completion_mean_absolute_error_seconds'], 15)
        boundary = self.score(['Instruct: Add oil', 'Success: Done'], [0, 5])
        self.assertEqual(boundary['completion_tp'], 0)
        self.assertEqual(boundary['completion_hit_rate_at_30s'], 1)

    def test_wrong_instruction_and_missing_success(self):
        for texts, times in [(['Instruct: Wrong', 'Success: Done'], [0, 20]),
                             (['Instruct: Add oil'], [0])]:
            m = self.score(texts, times)
            self.assertEqual(m['completion_fn'], 1)
            self.assertIsNone(m['completion_mean_absolute_error_seconds'])

    def test_mistake_step_in_completion_but_not_relaxed_denominator(self):
        m = self.score(['Instruct: Add oil', 'Success: Done'], [0, 20], mistake=True)
        self.assertEqual(m['completion_recall_at_15s'], 1)
        self.assertIsNone(m['relaxed_ic_accuracy_at_15s'])

    def test_coverage_extra_steps_and_repeat_scope(self):
        dataset = [{'video_id': v, 'texts': [['Add oil']],
                    'texts_timestamps': [[20]], 'has_mistake': [False]} for v in ['v', 'missing']]
        predictions = [{'video_id': 'v', 'pred_texts': [
            'Instruct: Add oil', 'Feedback: More', 'Success: Done',
            'Instruct: Extra', 'Feedback: More', 'Success: Done'],
            'pred_timestamps': [0, 10, 20, 30, 35, 40]}]
        m = calculate_custom_metrics(dataset, predictions, lambda a, b: a == b)
        self.assertEqual(m['reference_steps'], 1)
        self.assertEqual(m['reference_videos'], 2)
        self.assertEqual(m['completion_fp'], 1)
        self.assertEqual(m['exact_repeat_feedback_events'], 0)


if __name__ == '__main__':
    unittest.main()
