"""Supplementary metrics, version 1 (all rates are fractions, not percentages).

Uses the official preprocessed dataset, including its terminal-step exclusions.
Only submitted videos are scored. Steps align by ordinal position; instruction
matching is supplied by the entry point using the official ROUGE threshold.

Completion targets are ALL reference steps, including steps with mistakes. A
Success is a TP when its instruction matches and its distance from max(reference
step timestamps) is strictly below 15 seconds. At most one TP per step; all
other Success events are FP, and unmatched reference steps are FN. This measures
step-end detection, not whether a mistaken execution was actually correct.
Timing errors use the FIRST Success on each instruction-matched reference step,
including out-of-window events; signed error = prediction minus reference.
Hit rates @5/15/30 use all reference steps as denominator (missing = miss).

False-alarm rate is the fraction of mistake-free reference steps containing
Feedback (also counts wrong-instruction steps). Repeats are
exact, stripped Feedback bodies repeated within the same predicted step;
repeat rate divides extra occurrences by all Feedback events. Undefined ratios
and empty timing statistics are null. No semantic repetition metric is implied.
"""

from statistics import mean, median


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _segments(prediction):
    segments = []
    for text, timestamp in zip(prediction['pred_texts'], prediction['pred_timestamps']):
        if text.startswith('Instruct:'):
            segments.append([])
        segments[-1].append((text, timestamp))
    return segments


def calculate_custom_metrics(dataset, predictions, instruction_matches):
    """Return JSON-safe counts and metrics; dataset items follow vendor/data.py."""
    references = {}
    for index in range(len(dataset)):
        item = dataset[index]
        references[item['video_id']] = item
    steps = clean = alarms = successes = tp = feedbacks = repeats = 0
    errors = []
    hits = {window: 0 for window in (5, 15, 30)}
    for prediction in predictions:
        reference = references[prediction['video_id']]
        segments = _segments(prediction)
        steps += len(reference['texts'])
        for segment in segments:
            bodies = [text.partition(':')[2].strip() for text, _ in segment
                      if text.startswith('Feedback:')]
            feedbacks += len(bodies)
            repeats += len(bodies) - len(set(bodies))
            successes += sum(text.startswith('Success:') for text, _ in segment)
        for index, texts in enumerate(reference['texts']):
            segment = segments[index] if index < len(segments) else []
            is_clean = not reference['has_mistake'][index]
            clean += is_clean
            alarms += is_clean and any(t.startswith('Feedback:') for t, _ in segment)
            if not segment or not instruction_matches(texts[0], segment[0][0].partition(':')[2].strip()):
                continue
            end = max(reference['texts_timestamps'][index])
            times = [time for text, time in segment if text.startswith('Success:')]
            if times:
                errors.append(times[0] - end)
            for window in hits:
                hits[window] += any(abs(time - end) < window for time in times)
            matched = any(abs(time - end) < 15 for time in times)
            tp += matched
    fp, fn = successes - tp, steps - tp
    return {
        'metric_version': 1,
        'scope': 'submitted_videos_only',
        'submitted_videos': len(predictions), 'reference_videos': len(references),
        'reference_steps': steps, 'mistake_free_steps': clean,
        'completion_tp': tp, 'completion_fp': fp, 'completion_fn': fn,
        'completion_precision_at_15s': _ratio(tp, successes),
        'completion_recall_at_15s': _ratio(tp, steps),
        'completion_f1_at_15s': _ratio(2 * tp, 2 * tp + fp + fn),
        **{f'completion_hit_rate_at_{w}s': _ratio(n, steps) for w, n in hits.items()},
        'timing_matched_steps': len(errors),
        'completion_mean_signed_error_seconds': mean(errors) if errors else None,
        'completion_mean_absolute_error_seconds': mean(map(abs, errors)) if errors else None,
        'completion_median_absolute_error_seconds': median(map(abs, errors)) if errors else None,
        'mistake_free_steps_with_feedback': alarms,
        'mistake_free_step_false_alarm_rate': _ratio(alarms, clean),
        'feedback_events': feedbacks, 'exact_repeat_feedback_events': repeats,
        'exact_repeat_feedback_rate': _ratio(repeats, feedbacks),
    }
