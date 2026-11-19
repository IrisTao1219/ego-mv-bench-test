"""Text-only YES compliance probe. No frames, speech gate, or benchmark scoring."""
import argparse
import gc
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_proassist import ProAssistBackend, system_rules

CASES = [
    ("eggs_complete", "Crack two eggs into the bowl.",
     "Both eggs have been cracked and all their contents are in the bowl. The entire current step is confirmed complete.", True),
    ("broccoli_complete", "Cut the broccoli into florets.",
     "All the broccoli has been cut into florets. The entire current step is confirmed complete.", True),
    ("milk_complete", "Add half a cup of milk and whisk the mixture.",
     "Half a cup of milk has been added and the mixture has been fully whisked. The entire current step is confirmed complete.", True),
    ("eggs_partial", "Crack two eggs into the bowl.",
     "Only one egg has been cracked into the bowl. The second egg is still intact. The current step is not complete. No mistake is visible.", False),
    ("milk_partial", "Add half a cup of milk and whisk the mixture.",
     "Half a cup of milk has been added, but whisking has not started. The current step is not complete. No mistake is visible.", False),
]


def messages_for(instruction, state):
    return [
        {"role": "system", "content": system_rules(SimpleNamespace(mistake_prompt="detailed"))},
        {"role": "user", "content": "Current recipe step: " + instruction},
        {"role": "user", "content": "For this text-only diagnostic, treat the following state as established fact, replacing video observation: " + state + " Follow the system response rules for this state."},
    ]


def serialize(messages):
    # Same Llama 3 role delimiters as the streaming processor, with an explicit
    # assistant header to test response content independently of speech gating.
    return "<|begin_of_text|>" + "".join(
        "<|start_header_id|>" + m["role"] + "<|end_header_id|>\n\n" + m["content"].strip() + "<|eot_id|>"
        for m in messages) + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def decode(model, tokenizer, prompt, max_tokens):
    import torch
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    stop_ids.discard(None)
    past, tokens = None, []
    with torch.inference_mode():
        for _ in range(max_tokens):
            out = model(input_ids=ids, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            ids = out.logits[:, -1:].argmax(dim=-1)
            tokens.append(ids.item())
            if tokens[-1] in stop_ids:
                break
    return dict(raw_response=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                decoded_response=tokenizer.decode(tokens), generated_tokens=len(tokens),
                hit_token_limit=tokens[-1] not in stop_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["both", "base", "proassist"], default="both")
    p.add_argument("--base-model-path", type=Path, default=Path("../models/Llama-3.1-8B-Instruct"))
    p.add_argument("--model-path", type=Path, default=Path("../models/ProAssist-Model-L4096-I5"))
    p.add_argument("--vision-model-path", type=Path, default=Path("../models/siglip-so400m-patch14-384"))
    p.add_argument("--proassist-root", type=Path, default=Path("../ProAssist"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/proassist-yes-text-probe"))
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="Write exact prompts without loading models")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = [dict(case_id=k, expected_yes=y, messages=messages_for(i, s)) for k, i, s, y in CASES]
    for c in cases:
        c["input_str"] = serialize(c["messages"])
    (args.output_dir / "prompts.json").write_text(json.dumps(cases, indent=2))
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    if args.dry_run:
        print("Prompts written to", args.output_dir)
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    backends = ["base", "proassist"] if args.backend == "both" else [args.backend]
    summaries = {}
    for name in backends:
        backend = None
        if name == "base":
            tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(args.base_model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").to(f"cuda:{args.device_index}").eval()
        else:
            options = SimpleNamespace(**vars(args))
            options.attn_implementation = "sdpa"
            options.sample_fps = 1
            options.context_length = 4096
            options.not_talk_threshold = 0.3
            backend = ProAssistBackend(options)
            model, tokenizer = backend.model, backend.tokenizer
        rows = []
        with (args.output_dir / (name + ".jsonl")).open("w") as log:
            for case in cases:
                row = dict(case_id=case["case_id"], expected_yes=case["expected_yes"],
                           **decode(model, tokenizer, case["input_str"], args.max_new_tokens))
                row["leading_yes"] = bool(re.match(r"^YES\b", row["raw_response"], re.I))
                # Negative controls only measure false YES; non-YES does not prove silence compliance.
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(name, row["case_id"], repr(row["raw_response"]), flush=True)
        summaries[name] = dict(
            completed_cases=sum(r["expected_yes"] for r in rows),
            completed_leading_yes=sum(r["expected_yes"] and r["leading_yes"] for r in rows),
            incomplete_cases=sum(not r["expected_yes"] for r in rows),
            incomplete_false_yes=sum(not r["expected_yes"] and r["leading_yes"] for r in rows),
            truncated=sum(r["hit_token_limit"] for r in rows))
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
        if backend:
            backend.close()
        del model, tokenizer, backend
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
