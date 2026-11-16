"""Continuous full-recipe inference with model-driven instruction transitions.

Local implementation of the paper's full-recipe setting; no annotation timestamps
control the video clock or instruction transitions. Existing per-step entry unchanged.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

from run_inference import load_samples, save_json, prompt_for, check_response, prediction_event


def instruction_texts(sample):
    """Read ordered instruction text only; never read reference timestamps or feedback."""
    texts, kinds = sample['texts'], sample['text_types']
    if len(texts) != len(kinds):
        raise ValueError('Annotation text/type arrays have different lengths')
    instructions = [text for text, kind in zip(texts, kinds) if kind == 'Instruction']
    if any(not isinstance(text, str) or not text.strip() for text in instructions):
        raise ValueError('Instructions must be nonempty strings')
    return instructions


class FullRecipeState:
    def __init__(self, instructions):
        if not instructions:
            raise ValueError('No instructions')
        self.instructions = instructions
        self.step_id = 0

    @property
    def done(self):
        return self.step_id == len(self.instructions)

    @property
    def instruction(self):
        return self.instructions[self.step_id]

    def complete(self):
        if self.done:
            raise ValueError('Recipe already complete')
        self.step_id += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', default='/datasets/Hf_model/Qwen3-VL-2B-Instruct')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true',
                   help='Allow an existing output directory; replace files produced by this run')
    p.add_argument('--video-ids', nargs='+')
    p.add_argument('--start-video-id', help='Start at this video (inclusive), in annotation order; use --limit 0 for all remaining videos')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--max-seconds', type=float, default=0, help='Absolute video cutoff, 0 = all')
    p.add_argument('--interval', type=float, default=5, help='Paper: 5 seconds; changing this is a protocol deviation')
    p.add_argument('--mistake-prompt', choices=('detailed', 'simple'), default='detailed',
                   help='Appendix D: detailed for Qwen3-VL/Qwen3.5/Gemini; simple for older baselines')
    p.add_argument('--sample-fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=32, help='Uniformly sample the continuous observed video prefix')
    p.add_argument('--max-side', type=int, default=448)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--thinking', choices=('default', 'on', 'off'), default='off',
                   help='Chat-template thinking mode: default = omit the switch; on/off = explicit '
                        'enable_thinking (requires template support). Default: off, as before')
    p.add_argument('--device-map', default='cuda:0',
                   help='Process-visible device (e.g. cuda:0 with CUDA_VISIBLE_DEVICES=1), or auto')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    template_kwargs = ({} if a.thinking == 'default'
                       else {'enable_thinking': a.thinking == 'on'})
    if (not all(math.isfinite(v) for v in (a.interval, a.sample_fps, a.max_seconds))
            or min(a.interval, a.sample_fps) <= 0 or a.max_seconds < 0
            or a.max_frames < 1 or a.max_side < 32 or a.limit < 0 or a.max_new_tokens < 1):
        p.error('Invalid sampling parameters')
    samples = load_samples(a.data_root, a.video_ids, a.limit, a.start_video_id)
    for s in samples:
        if not instruction_texts(s):
            raise ValueError(f"No instructions: {s['video_id']}")
    import cv2
    from PIL import Image
    a.output_dir.mkdir(parents=True, exist_ok=a.overwrite)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(protocol='full_recipe', reference='arXiv:2606.09547v2',
                  prompt_version='arxiv_2606.09547v2_appendix_D_verbatim',
                  paper_prompt_verbatim=True,
                  success_message_policy='fixed_confirmation_reason_logged_only',
                  response_format='YES_NO', enable_thinking=template_kwargs.get('enable_thinking'),
                  chat_template_kwargs=template_kwargs, paper_interval_seconds=5,
                  interval_matches_paper=(a.interval == 5), trained_on_ego_comist=False,
                  implementation='local_full_recipe_state_machine',
                  instruction_timestamps_used=False, video_start_seconds=0,
                  transition_policy='model_success_only', context_policy='video_prefix_from_zero',
                  feedback_history_policy='last_60_seconds_across_steps',
                  dry_run_policy='no_model_no_transitions_no_predictions')
    save_json(a.output_dir / 'config.json', config)
    if not a.dry_run:
        import transformers
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Select the multimodal architecture from config.json, not the directory name.
        model = AutoModelForImageTextToText.from_pretrained(
            a.model_path, dtype='auto', device_map=a.device_map,
            attn_implementation='sdpa', local_files_only=True).eval()
        processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)
        config.update(transformers_version=transformers.__version__, torch_version=torch.__version__,
                      model_class=type(model).__name__, processor_class=type(processor).__name__,
                      model_type=model.config.model_type)
        save_json(a.output_dir / 'config.json', config)
        generation = copy.deepcopy(model.generation_config)
        generation.do_sample = False
        generation.temperature = generation.top_p = generation.top_k = None

    results = []
    for sample in samples:
        video_id = sample['video_id']
        cap = cv2.VideoCapture(str(a.data_root / 'videos' / f'{video_id}.MP4'))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f'Invalid video: {video_id}')
            duration = count / fps
            cutoff = min(duration, a.max_seconds) if a.max_seconds else duration
            state = FullRecipeState(instruction_texts(sample))
            result = {'video_id': video_id, 'pred_texts': [], 'pred_timestamps': []}
            if not a.dry_run:
                result['pred_texts'].append('Instruct: ' + state.instruction)
                result['pred_timestamps'].append(0.0)
            frame_dir = a.output_dir / 'model_input_frames' / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            with ((a.output_dir / f'{video_id}.jsonl').open('w') as log,
                  (a.output_dir / f'{video_id}.model_inputs.jsonl').open('w') as input_log):
                recent = []
                turn = 1
                while turn * a.interval < cutoff and not state.done:
                    step_id = state.step_id
                    step = {'instruction': state.instruction}
                    requested = turn * a.interval
                    current = min(count - 1, math.floor(requested * fps))
                    first = 0
                    stride = max(1, round(fps / a.sample_fps))
                    indices = list(range(first, current + 1, stride))
                    if not indices or indices[-1] != current:
                        indices.append(current)
                    if len(indices) > a.max_frames:
                        indices = [indices[round(i * (len(indices)-1) / (a.max_frames-1))]
                                   for i in range(a.max_frames)] if a.max_frames > 1 else [current]
                    content, timestamps, logged_content = [], [], []
                    for index in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f'Cannot decode {video_id}, frame {index}')
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image.thumbnail((a.max_side, a.max_side))
                        frame_path = frame_dir / f'{index:09d}.png'
                        if a.overwrite or not frame_path.exists():
                            image.save(frame_path)
                        logged_content += [
                            {'type': 'text', 'text': f'Frame {index/fps:.3f}s'},
                            {'type': 'image', 'image': str(frame_path.relative_to(a.output_dir)),
                             'width': image.width, 'height': image.height}]
                        timestamps.append(index / fps)
                        content += [{'type': 'text', 'text': f'Frame {index/fps:.3f}s'},
                                    {'type': 'image', 'image': image}]
                    recent = [r for r in recent if requested-r['time'] <= 60]
                    event = None
                    for stage, field in [('completion', 'completed'), ('mistake', 'mistake')]:
                        prompt = prompt_for(stage, step['instruction'], requested, recent, a.mistake_prompt)
                        messages = [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}]
                        input_log.write(json.dumps({
                            'video_id': video_id, 'step_id': step_id,
                            'timestamp': requested, 'stage': stage,
                            'messages': [{'role': 'user', 'content': logged_content + [
                                {'type': 'text', 'text': prompt}]}],
                        }, ensure_ascii=False) + '\n')
                        input_log.flush()
                        raw, error, parsed = '', None, None
                        generated_tokens, hit_token_limit = None, None
                        began = time.monotonic()
                        if not a.dry_run:
                            inputs = processor.apply_chat_template(
                                messages,
                                tokenize=True, add_generation_prompt=True, return_dict=True,
                                return_tensors='pt', **template_kwargs).to(model.device)
                            with torch.inference_mode():
                                ids = model.generate(**inputs, generation_config=generation,
                                                     do_sample=False, num_beams=1,
                                                     max_new_tokens=a.max_new_tokens)
                            output_ids = ids[:, inputs.input_ids.shape[1]:]
                            generated_tokens = output_ids.shape[1]
                            eos_ids = generation.eos_token_id
                            eos_ids = ([eos_ids] if isinstance(eos_ids, int) else eos_ids) or []
                            ended_with_eos = bool(generated_tokens and output_ids[0, -1].item() in eos_ids)
                            hit_token_limit = generated_tokens >= a.max_new_tokens and not ended_with_eos
                            raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                            try:
                                parsed = check_response(raw, field)
                            except (ValueError, TypeError, IndexError) as exc:
                                error = str(exc)
                            if parsed and parsed[field]:
                                event = prediction_event(parsed, field)
                        log.write(json.dumps({'step_id': step_id, 'instruction': step['instruction'],
                            'timestamp': requested, 'frame_timestamps': timestamps, 'stage': stage,
                            'prompt': prompt, 'raw_response': raw, 'parsed': parsed,
                            'parse_error': error, 'accepted': event,
                            'generated_tokens': generated_tokens, 'hit_token_limit': hit_token_limit,
                            'latency_seconds': time.monotonic()-began})+'\n')
                        log.flush()
                        # On malformed completion, do not silently treat it as false.
                        if stage == 'completion' and not a.dry_run and (parsed is None or parsed[field]):
                            break
                    if event:
                        result['pred_texts'].append(event)
                        result['pred_timestamps'].append(requested)
                        if event.startswith('Feedback:'):
                            recent.append({'time': requested, 'feedback': parsed['feedback'].strip()})
                    print(f'{video_id} step={step_id} {requested:.2f}s: {event or "None"}', flush=True)
                    save_json(a.output_dir / f'{video_id}.prediction.json', result)
                    if event and event.startswith('Success:'):
                        state.complete()
                        if not state.done:
                            result['pred_texts'].append('Instruct: ' + state.instruction)
                            result['pred_timestamps'].append(requested)
                        save_json(a.output_dir / f'{video_id}.prediction.json', result)
                    turn += 1
            results.append(result)
            save_json(a.output_dir / 'predictions.json', results)
        finally:
            cap.release()


if __name__ == '__main__':
    main()
