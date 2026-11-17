"""Text-only instruction-following probe; run explicitly on a model server."""
import argparse
import json


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device-map', default='cuda:0')
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype='auto', device_map=args.device_map,
        attn_implementation='sdpa', local_files_only=True,
    ).eval()

    results = []
    for name, feedback, observation, expected in cases():
        prompt = (RULE + '\nPrevious feedback: ' + json.dumps(feedback)
                  + '\nCurrent observation: ' + observation)
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        print(f'\n--- {name}: INPUT ---\n{rendered}', flush=True)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', enable_thinking=False,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0].strip()
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
