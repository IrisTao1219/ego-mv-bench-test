"""Text-only instruction-following probe; run explicitly on a model server."""
import argparse
import json
from difflib import SequenceMatcher

from run_inference import prompt_for, check_response, DEDUP_RULE, inference_messages


RULE = """You are checking cooking mistakes.
Output exactly YES or NO, with no explanation.
Output YES only if the current observation describes a NEW mistake.
Output NO if there is no mistake, or if the SAME mistake was already reported.
A mistake remaining uncorrected is not a new mistake.
"""

# All three variants receive exactly the same observation and task rule.
OBSERVATION = (
    'This is the SAME incident. The bowl still contains just one teaspoon of salt. '
    'No additional action occurred.'
)


def cases():
    return [
        ('没有错误', [],
         'The required amount of salt was added correctly.', 'NO'),
        ('新错误', [],
         'The recipe requires two teaspoons of salt, but the person finished '
         'after adding only one.', 'YES'),
        ('同一次错误，文字直接对应',
         ['You added only one teaspoon of salt instead of two.'],
         'This is the SAME previously reported incident. The person added only '
         'one teaspoon instead of two. Nothing new happened.', 'NO'),
        ('A：只描述错误',
         ['You added only one teaspoon instead of the required two.'],
         OBSERVATION, 'NO'),
        ('B：只给纠正建议',
         ['Add one more teaspoon of salt to reach two teaspoons.'],
         OBSERVATION, 'NO'),
        ('C：错误描述＋纠正建议',
         ['You added only one teaspoon instead of two. Add one more teaspoon.'],
         OBSERVATION, 'NO'),
        ('已提醒盐的问题，但出现新错误',
         ['Add one more teaspoon of salt.'],
         'The person has now corrected the salt amount. Then they added vinegar, '
         'although the recipe explicitly requires lemon juice.', 'YES'),
    ]


def compare_feedback(first, second):
    """Text similarity is diagnostic only, not a semantic equivalence verdict."""
    normalize = lambda text: ' '.join(text.casefold().split())
    return dict(
        exact_same_text=first == second,
        same_ignoring_case_whitespace=normalize(first) == normalize(second),
        text_similarity=round(SequenceMatcher(None, normalize(first), normalize(second)).ratio(), 3),
    )


def repeat_probe(generate, factorial=False):
    scenarios = [
        ('盐量不足', 'Add two teaspoons of salt to the bowl.',
         'The person added one teaspoon of salt to the bowl, put the spoon away, '
         'and stopped adding salt.'),
        ('油量不足', 'Put three teaspoons of sesame oil in the bowl.',
         'The person added two teaspoons of sesame oil to the bowl, put the '
         'spoon away, and stopped adding oil.'),
        ('温度错误', 'Set the oven to 180 degrees Celsius.',
         'The person set the oven to 220 degrees Celsius, confirmed the setting, '
         'and walked away.'),
    ]
    results = []
    for name, instruction, observation in scenarios:
        # Each request is independent, just like the project's turn-based calls.
        # Textual observation substitutes for images in this diagnostic.
        def request(label, current_observation, history, explicit_dedup=False,
                    assistant_history=False):
            prompt = prompt_for('mistake', instruction, 0, history, explicit_dedup=False)
            text = prompt + '\n\nCurrent observation: ' + current_observation
            if explicit_dedup:
                text += '\n\n' + DEDUP_RULE
            if assistant_history:
                # Use exactly the production helper, including the initial
                # instruction user message and aggregated assistant feedback.
                return generate(label, inference_messages(instruction, [], text, history))
            return generate(label, text)

        first_raw = request(name + ' / 首轮', observation, [])
        try:
            first = check_response(first_raw, 'mistake')
        except ValueError as exc:
            results.append(dict(case=name, status='invalid_first_response',
                                first_raw=first_raw, parse_error=str(exc)))
            continue
        if not first['mistake']:
            results.append(dict(case=name, status='no_first_feedback', first_raw=first_raw))
            continue
        history = [{'time': 0, 'feedback': first['feedback']}]
        variants = [
            ('完全相同观察', observation, 'NO'),
            ('几乎相同观察', observation + ' No additional action occurred; the state is unchanged.', 'NO'),
        ]
        if factorial:
            variants.append(('纠正后新一次执行又犯错',
                'The previously reported incident was fully corrected. The person '
                'then started a NEW, SEPARATE execution of the same instruction. '
                'In this new execution: ' + observation, 'YES'))
        # Share the same original first response across all comparisons.
        # Within each observation pair, only the appended rule changes.
        comparisons = [(variant, current, expected, policy, enabled, use_history)
                       for variant, current, expected in variants
                       for policy, enabled in [('original', False), ('explicit_dedup', True)]
                       for use_history in ([False, True] if factorial else [False])]
        for variant, current, expected, policy, enabled, use_history in comparisons:
            structure = 'assistant_history' if use_history else 'single_user'
            raw = request(name + ' / ' + variant + ' / ' + policy + ' / ' + structure,
                          current, history, enabled, use_history)
            result = dict(case=name, variant=variant, prompt_policy=policy,
                          message_structure=structure, first_raw=first_raw,
                          history_feedback=first['feedback'], second_raw=raw,
                          expected_decision=expected)
            try:
                second = check_response(raw, 'mistake')
                result.update(status='ok', detected_mistake=second['mistake'],
                              passed=second['mistake'] == (expected == 'YES'),
                              repeated_trigger=expected == 'NO' and second['mistake'],
                              second_feedback=second['feedback'])
                if second['mistake']:
                    result.update(compare_feedback(first['feedback'], second['feedback']))
            except ValueError as exc:
                result.update(status='invalid_second_response', parse_error=str(exc))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print('\n--- REPEAT SUMMARY ---')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    groups = [(policy, structure) for policy in ('original', 'explicit_dedup')
              for structure in (['single_user', 'assistant_history'] if factorial else ['single_user'])]
    for policy, structure in groups:
        rows = [r for r in results if r.get('prompt_policy') == policy
                and r.get('message_structure') == structure]
        valid = [r for r in rows if r['status'] == 'ok']
        print(json.dumps(dict(
            prompt_policy=policy, message_structure=structure,
            comparisons=len(rows), valid_responses=len(valid),
            repeated_triggers=sum(r['repeated_trigger'] for r in valid),
            exact_text_repeats=sum(r.get('exact_same_text', False) for r in valid
                                   if r['expected_decision'] == 'NO'),
            repeat_test_count=sum(r['expected_decision'] == 'NO' for r in rows),
            new_incident_controls=sum(r['expected_decision'] == 'YES' for r in rows),
            new_incident_detected=sum(r['detected_mistake'] for r in valid
                                      if r['expected_decision'] == 'YES'),
            invalid_responses=len(rows) - len(valid),
        ), ensure_ascii=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device-map', default='cuda:0')
    parser.add_argument('--mode', choices=('decision', 'repeat', 'factorial'), default='decision',
                        help='decision: YES/NO; repeat: rule comparison; factorial: rule x assistant history with new-incident controls')
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype='auto', device_map=args.device_map,
        attn_implementation='sdpa', local_files_only=True,
    ).eval()

    def generate(name, prompt):
        messages = (prompt if isinstance(prompt, list) else
                    [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}])
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        print(f'\n--- {name}: INPUT ---\n{rendered}', flush=True)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', enable_thinking=False,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
        print('RESPONSE: ' + raw, flush=True)
        return raw

    if args.mode in ('repeat', 'factorial'):
        repeat_probe(generate, factorial=args.mode == 'factorial')
        return

    results = []
    for name, feedback, observation, expected in cases():
        prompt = (RULE + '\nPrevious feedback: ' + json.dumps(feedback)
                  + '\nCurrent observation: ' + observation)
        raw = generate(name, prompt)
        result = dict(case=name, expected=expected, actual=raw, passed=raw == expected)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    print('\n--- SUMMARY ---')
    for result in results:
        print(f"{result['case']}: expected={result['expected']} "
              f"actual={result['actual']!r} passed={result['passed']}")
    print(f"Passed: {sum(r['passed'] for r in results)}/{len(results)}")


if __name__ == '__main__':
    main()
