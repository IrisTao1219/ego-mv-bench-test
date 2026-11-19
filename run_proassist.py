"""ProAssist frame-triggered interventions with the official mmassist backend.

The tagged response prompt and instruction scheduling are local adaptations,
not the unpublished Ego-MC-Bench authors' ProAssist evaluation implementation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

from run_inference import (load_samples, recipe_steps, save_json,
                           add_visual_arguments, add_data_arguments, resize_frame, continuation_state)
from run_full_recipe import instruction_texts


SYSTEM_PROMPT = (
    'You are a proactive cooking assistant watching a video stream. '
    'Follow only the current recipe step supplied by the user. '
    'Stay silent when no intervention is needed. When speaking, output exactly '
    'one message: "Feedback: <brief corrective feedback>" if a visible mistake '
    'directly interferes with the current instruction, or '
    '"Success: <brief confirmation>" if the entire current step is complete. '
    'Do not announce completion prematurely or invent the next recipe step. '
    'Do not repeat feedback for an unchanged mistake.'
)


def parse_event(text):
    """Unclassified speech is logged, never guessed to be feedback or success."""
    if not text.strip():
        return None
    match = re.fullmatch(r'(Feedback|Success):\s*([^\n]+)', text.strip())
    if not match or not match[2].strip():
        raise ValueError('Expected one nonempty Feedback: or Success: message')
    if re.search(r'\b(?:Feedback|Success|Instruct):', match[2]):
        raise ValueError('Multiple events in one response')
    return match[1] + ': ' + match[2].strip()


def generate_until_eos(model, embeds, past, threshold):
    """Greedy streaming decode with dynamic storage and upstream silence semantics.

    The silence decision applies only to the first token of a frame. Subsequent
    tokens run until EOS, retaining KV entries exactly as the bounded backend does.
    """
    import torch
    tokens = []
    while True:
        out = model.forward(inputs_embeds=embeds, past_key_values=past,
                            use_cache=True, return_dict=True)
        past = out.past_key_values
        scores = out.logits[:, -1:, :].clone()
        if not tokens:
            probability = out.w2t_probs[0, -1]
            model.last_not_talk_prob = probability
            scores[..., model.config.w2t_target_id] = (1e4 if probability > threshold else -1e4)
        token = scores.argmax(dim=-1)
        first = not tokens
        tokens.append(token)
        if token.item() == model.config.eos_token_id or (first and token.item() == model.config.w2t_target_id):
            return torch.cat(tokens, dim=1), past
        embeds = model.get_input_embeddings()(token)


class ProAssistBackend:
    """Reuse upstream formatting, greedy streaming generation and cache management."""
    def __init__(self, args):
        # All pretrained components must already exist on the execution machine.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        if args.proassist_root:
            sys.path.insert(0, str(args.proassist_root.resolve()))
        import torch
        from mmassist.model import build_from_checkpoint
        from mmassist.eval.runners.stream_inference import StreamInferenceRunner

        self.torch = torch
        self.args = args
        # Upstream reads llm_pretrained before applying overrides. A lightweight
        # checkpoint view redirects paths without editing the user's downloaded files.
        self.temp = tempfile.TemporaryDirectory(prefix='ego-proassist-')
        view = Path(self.temp.name)
        base_alias = view / 'Meta-Llama-3.1-8B-Instruct'
        base_alias.symlink_to(args.base_model_path.resolve(), target_is_directory=True)
        checkpoint = view / 'checkpoint'
        checkpoint.mkdir()
        for child in args.model_path.resolve().iterdir():
            if child.name != 'config.json':
                (checkpoint / child.name).symlink_to(child, target_is_directory=child.is_dir())
        cfg = json.loads((args.model_path / 'config.json').read_text())
        vision_name = 'vision-siglip' if 'siglip' in str(cfg.get('vision_pretrained', '')).lower() else 'vision'
        vision_alias = view / vision_name
        vision_alias.symlink_to(args.vision_model_path.resolve(), target_is_directory=True)
        cfg.update(llm_pretrained=str(base_alias),
                   vision_pretrained=str(vision_alias),
                   attn_implementation=args.attn_implementation,
                   exceed_context_handling='summarize_and_drop')
        (checkpoint / 'config.json').write_text(json.dumps(cfg))
        self.model, self.tokenizer = build_from_checkpoint(str(checkpoint), rank=args.device_index)
        self.model.eval()
        self.runner = StreamInferenceRunner.build(
            model=self.model, tokenizer=self.tokenizer, fps=args.sample_fps,
            eval_max_seq_len=args.context_length or sys.maxsize,
            reserved_seq_len=max(128, args.max_new_tokens + 32),
            not_talk_threshold=args.not_talk_threshold, eval_name='ego_mc_bench')
        self.metadata = dict(input_image_policy='aspect_ratio_thumbnail_then_model_vision_processor',
                             context_policy='upstream_summarize_and_drop' if args.context_length else 'complete_stream_cache',
                             output_length_policy='eos_only' if not args.max_new_tokens else 'token_cap',
                             num_tokens_per_img=self.model.config.num_tokens_per_img,
                             use_binary_decision_head=self.model.config.use_binary_decision_head,
                             torch_version=torch.__version__)

    def reset(self, instruction):
        self.past = self.last = None
        self.first = True
        self.pending = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': instruction}]
        self.runner.initial_sys_prompt = SYSTEM_PROMPT
        self.runner.knowledge = 'Task knowledge: Current recipe step: ' + instruction
        self.runner.init_key_values_to_keep = None
        self.model.last_not_talk_prob = None

    def set_instruction(self, instruction):
        self.pending = [{'role': 'user', 'content': instruction}]
        self.runner.knowledge = 'Task knowledge: Current recipe step: ' + instruction

    def step(self, rgb, timestamp):
        torch = self.torch
        import numpy as np
        rgb = np.asarray(resize_frame(rgb, getattr(self.args, 'max_side', 448)))
        messages = self.pending
        ids, input_str = self.runner.processor.get_input_sequence(1, messages, self.first)
        self.first, self.pending = False, []
        # The official vision encoder expects raw RGB values in 0..255, NCHW.
        images = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
        inputs = self.runner.processor.add_last_assistant_message(
            {'input_ids': ids, 'images': images}, self.last)
        inputs = {k: v.to(self.model.device, self.model.dtype) if v.is_floating_point()
                  else v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            embeds = self.model.joint_embed(**inputs)
            if self.args.max_new_tokens:
                output, self.past = self.model.fast_greedy_generate(
                    embeds, self.past, max_length=self.args.max_new_tokens,
                    not_talk_threshold=self.args.not_talk_threshold,
                    drop_generated_kv_cache=False)
            else:
                output, self.past = generate_until_eos(
                    self.model, embeds, self.past, self.args.not_talk_threshold)
            output = output.clone()  # Summary generation reuses the upstream output buffer.
            raw = self.tokenizer.decode(output[0])
            cleaned, role = self.runner.processor.cleanup_text(raw)
            self.last = (output[:, -1:].clone()
                         if self.tokenizer.img_sep_token_id is not None else None)
            # Upstream summary management needs only these two frame attributes.
            frame = SimpleNamespace(model_inputs=inputs, num_frames=1)
            if self.args.context_length:
                self.past, self.last = self.runner.manage_kv_cache(frame, self.past, self.last)
        return dict(raw_response=raw, text=cleaned, role=role,
                    generated_tokens=output.shape[1],
                    hit_token_limit=bool(self.args.max_new_tokens and output.shape[1] >= self.args.max_new_tokens
                                     and output[0, -1].item() != self.model.config.eos_token_id),
                    input_str=input_str, input_messages=messages)

    def close(self):
        self.temp.cleanup()


def run_video(sample, args, backend, cap, log):
    """Drive both protocols without exposing reference feedback/success to the model."""
    import cv2
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError(f"Invalid video: {sample['video_id']}")
    if args.sample_fps > fps:
        raise ValueError('sample-fps exceeds source video FPS')
    end = min(count / fps, args.max_seconds) if args.max_seconds else count / fps
    if args.protocol == 'per_recipe_step':
        steps = recipe_steps(sample)
    else:
        steps = [{'instruction': text, 'start': 0.0} for text in instruction_texts(sample)]
    if not steps:
        raise ValueError('No instructions')
    result = dict(video_id=sample['video_id'], pred_texts=[], pred_timestamps=[])

    def append(text, timestamp):
        if not args.dry_run:
            result['pred_texts'].append(text)
            result['pred_timestamps'].append(timestamp)

    step_id = 0
    start = steps[0]['start']
    if start >= end:
        raise ValueError('Video cutoff precedes first instruction')
    append('Instruct: ' + steps[0]['instruction'], start)
    if backend:
        backend.reset(steps[0]['instruction'])
    tick = 0
    while step_id < len(steps):
        requested = start + tick / args.sample_fps
        if requested >= end:
            break
        # Per-step boundary comes only from the next Instruction, never Success.
        if (args.protocol == 'per_recipe_step' and step_id + 1 < len(steps)
                and requested >= steps[step_id + 1]['start']):
            step_id += 1
            start, tick = steps[step_id]['start'], 0
            append('Instruct: ' + steps[step_id]['instruction'], start)
            if backend:
                backend.reset(steps[step_id]['instruction'])
            continue
        index = min(count - 1, math.floor(requested * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index}')
        # Never attach an event timestamp earlier than instruction injection.
        timestamp = requested
        response = (backend.step(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp)
                    if backend else dict(raw_response='', text='', input_messages=[]))
        event, error = None, None
        try:
            if response.get('role') not in (None, 'assistant'):
                raise ValueError('Generated role is not assistant')
            event = parse_event(response['text'])
        except ValueError as exc:
            error = str(exc)
        log.write(json.dumps(dict(step_id=step_id, timestamp=timestamp,
            frame_index=index, frame_timestamp=index / fps,
            instruction=steps[step_id]['instruction'], accepted=event,
            parse_error=error, **response), ensure_ascii=False) + '\n')
        log.flush()
        tick += 1
        if event:
            append(event, timestamp)
        if event and event.startswith('Success:'):
            step_id += 1
            if step_id == len(steps):
                break
            if args.protocol == 'per_recipe_step':
                start, tick = steps[step_id]['start'], 0
                if start >= end:
                    break
                append('Instruct: ' + steps[step_id]['instruction'], start)
                backend.reset(steps[step_id]['instruction'])
            else:
                append('Instruct: ' + steps[step_id]['instruction'], timestamp)
                backend.set_instruction(steps[step_id]['instruction'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(p)
    p.add_argument('--proassist-root', type=Path, help='Official ProAssist source checkout')
    p.add_argument('--model-path', type=Path, help='Local ProAssist I1/I5/I10 checkpoint')
    p.add_argument('--base-model-path', type=Path)
    p.add_argument('--vision-model-path', type=Path, help='Vision model matching checkpoint vision_pretrained')
    p.add_argument('--protocol', choices=('per_recipe_step', 'full_recipe'), default='per_recipe_step')
    add_visual_arguments(p, include_frame_limit=False)
    p.set_defaults(sample_fps=2)
    p.add_argument('--not-talk-threshold', type=float, default=0.3)
    p.add_argument('--context-length', type=int, default=4096, help='Streaming token context budget; 0 disables upstream progress summarization')
    p.add_argument('--device-index', type=int, default=0)
    p.add_argument('--attn-implementation', choices=('sdpa', 'eager', 'flash_attention_2'), default='sdpa')
    p.add_argument('--dry-run', action='store_true', help='Decode only; no model or valid predictions')
    args = p.parse_args()
    if (not all(math.isfinite(x) for x in (args.sample_fps, args.max_seconds, args.not_talk_threshold))
            or args.sample_fps <= 0 or args.max_seconds < 0
            or not 0 < args.not_talk_threshold < 1 or args.limit < 0 or args.device_index < 0
            or args.context_length < 0 or args.max_new_tokens < 0 or args.max_side < 32
            or (args.context_length and args.context_length < args.max_new_tokens + 256)):
        p.error('Invalid inference parameters')
    if not args.dry_run:
        for key in ('model_path', 'base_model_path', 'vision_model_path'):
            path = getattr(args, key)
            if path is None or not path.is_dir():
                p.error(f'--{key.replace("_", "-")} must be an existing local directory')
    samples = load_samples(args.data_root, args.video_ids, args.limit, args.start_video_id)
    results = []
    if args.continue_run:
        samples, results = continuation_state(args.output_dir, samples)
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite or args.continue_run)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(implementation='local_proassist_adapter', paper_exact_reproduction=False,
                  response_mapping='strict_tagged_speech_no_helper_llm', system_prompt=SYSTEM_PROMPT,
                  context_handling='upstream_summarize_and_drop' if args.context_length else 'complete_stream_cache',
                  context_frame_limit=None, historical_frame_policy='incremental_kv_with_generated_dialogue',
                  summary_reserved_tokens=max(128, args.max_new_tokens + 32), offline=True,
                  reference='https://github.com/pro-assist/ProAssist',
                  instruction_timestamps_used=args.protocol == 'per_recipe_step',
                  feedback_and_success_labels_used=False)
    save_json(args.output_dir / 'config.json', config)
    backend = None
    try:
        if not args.dry_run:
            backend = ProAssistBackend(args)
            config.update(backend.metadata)
            save_json(args.output_dir / 'config.json', config)
        import cv2
        for sample in samples:
            cap = cv2.VideoCapture(str(args.data_root / 'videos' / (sample['video_id'] + '.MP4')))
            try:
                with (args.output_dir / (sample['video_id'] + '.jsonl')).open('w') as log:
                    result = run_video(sample, args, backend, cap, log)
            finally:
                cap.release()
            save_json(args.output_dir / (sample['video_id'] + '.prediction.json'), result)
            results.append(result)
            save_json(args.output_dir / 'predictions.json', results)
            print(f"Finished {sample['video_id']}: {len(result['pred_texts'])} events", flush=True)
    finally:
        if backend:
            backend.close()


if __name__ == '__main__':
    main()
