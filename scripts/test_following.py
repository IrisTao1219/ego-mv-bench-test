import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--device-map", default="cuda:0")
args = parser.parse_args()

processor = AutoProcessor.from_pretrained(
    args.model_path, local_files_only=True
)
model = AutoModelForImageTextToText.from_pretrained(
    args.model_path,
    dtype="auto",
    device_map=args.device_map,
    attn_implementation="sdpa",
    local_files_only=True,
).eval()

rule = """You are checking cooking mistakes.
Output exactly YES or NO, with no explanation.
Output YES only if the current observation describes a NEW mistake.
Output NO if there is no mistake, or if the SAME mistake was already reported.
A mistake remaining uncorrected is not a new mistake.
"""

cases = [
    (
        "没有错误",
        'Previous feedback: []\n'
        'Current observation: The required amount of salt was added correctly.',
        "NO",
    ),
    (
        "新错误",
        'Previous feedback: []\n'
        'Current observation: The recipe requires two teaspoons of salt, '
        'but the person finished after adding only one.',
        "YES",
    ),
    (
        "同一次错误，文字直接对应",
        'Previous feedback: ["You added only one teaspoon of salt instead of two."]\n'
        'Current observation: This is the SAME previously reported incident. '
        'The person added only one teaspoon instead of two. Nothing new happened.',
        "NO",
    ),
    (
        "同一次错误，换一种说法",
        'Previous feedback: ["Add one more teaspoon of salt to reach two teaspoons."]\n'
        'Current observation: This is the SAME incident. '
        'The bowl still contains just one teaspoon of salt. No additional action occurred.',
        "NO",
    ),
    (
        "已提醒盐的问题，但出现新错误",
        'Previous feedback: ["Add one more teaspoon of salt."]\n'
        'Current observation: The person has now corrected the salt amount. '
        'Then they added vinegar, although the recipe explicitly requires lemon juice.',
        "YES",
    ),
]

passed = 0
for name, observation, expected in cases:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": rule + "\n" + observation}],
    }]

    # 打印实际渲染的输入，确认模板没有漏掉历史反馈。
    rendered = processor.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    print(f"\n--- {name}: INPUT ---\n{rendered}")

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    raw = processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    ok = raw == expected
    passed += ok
    print(json.dumps({
        "case": name,
        "expected": expected,
        "actual": raw,
        "passed": ok,
    }, ensure_ascii=False))

print(f"\nPassed: {passed}/{len(cases)}")