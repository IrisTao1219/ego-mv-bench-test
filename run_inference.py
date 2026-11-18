"""Per-recipe-step, completion-first, regular-interval inference."""
import argparse
import copy
import json
import math
import os
import re

from paper_prompts import COMPLETION_PROMPT, DETAILED_MISTAKE_PROMPT, SIMPLE_MISTAKE_PROMPT
from pathlib import Path
import time

def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def continuation_state(output_dir, samples):
    """Restart the last started video and retain predictions before it."""
    predictions_path = output_dir / 'predictions.json'
    previous = json.loads(predictions_path.read_text()) if predictions_path.is_file() else []
    previous_by_id = {result['video_id']: result for result in previous}
    started = [i for i, sample in enumerate(samples)
               if sample['video_id'] in previous_by_id
               or (output_dir / f"{sample['video_id']}.jsonl").is_file()
               or (output_dir / f"{sample['video_id']}.model_inputs.jsonl").is_file()
               or (output_dir / f"{sample['video_id']}.prediction.json").is_file()]
    if not started:
        return samples, previous
    restart = max(started)
    rerun_ids = {sample['video_id'] for sample in samples[restart:]}
    return samples[restart:], [result for result in previous if result['video_id'] not in rerun_ids]


def load_samples(root, video_ids, limit, start_video_id=None):
    samples = json.loads((root / 'annotations/annotations.json').read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError('Annotations must be a nonempty list')
    ids = [s['video_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate video IDs')
    if start_video_id is not None:
        if video_ids:
            raise ValueError('--start-video-id cannot be combined with --video-ids')
        if start_video_id not in ids:
            raise ValueError(f'Unknown start video ID: {start_video_id}')
        samples = samples[ids.index(start_video_id):]
    if video_ids:
        unknown = set(video_ids) - set(ids)
        if unknown:
            raise ValueError(f'Unknown video IDs: {sorted(unknown)}')
        samples = [s for s in samples if s['video_id'] in video_ids]
    samples = samples[:limit] if limit else samples
    for s in samples:
        if not isinstance(s['video_id'], str) or Path(s['video_id']).name != s['video_id']:
            raise ValueError('Invalid video ID')
        if not (root / 'videos' / (s['video_id'] + '.MP4')).is_file():
            raise FileNotFoundError(f"Missing video: {s['video_id']}")
    return samples


def recipe_steps(sample):
    texts, kinds, times = (sample[k] for k in ('texts', 'text_types', 'text_timestamps'))
    if not len(texts) == len(kinds) == len(times):
        raise ValueError('Annotation arrays have different lengths')
    if any(not math.isfinite(t) or t < 0 for t in times) or times != sorted(times):
        raise ValueError('Annotation timestamps must be finite and sorted')
    # Only instructions enter the inference protocol; labels are never prompts.
    return [{'instruction': text, 'start': t} for text, kind, t in zip(texts, kinds, times)
            if kind == 'Instruction']


def check_response(raw, field):
    """Parse Appendix D YES/NO responses; never infer NO from malformed output."""
    if field not in ('completed', 'mistake'):
        raise ValueError(f'Unknown decision field: {field}')
    match = re.match(r'^(YES|NO)\b[.\s:!,-]*(.*)$', raw.strip(), re.DOTALL)
    if not match:
        raise ValueError('Response must begin with YES or NO')
    positive = match.group(1) == 'YES'
    explanation = match.group(2).strip()
    if positive and not explanation:
        raise ValueError('YES requires a completion explanation or corrective feedback')
    if field == 'completed':
        return {field: positive, 'reason': explanation}
    return {field: positive, 'feedback': explanation if positive else ''}


DEDUP_RULE = """Before answering, compare the current mistake with the previous feedback.
If the same mistake was already reported and no new mistake occurred,
answer NO, even if the mistake remains uncorrected.
Answer YES only for a new, previously unreported mistake."""


def prompt_for(stage, instruction, now, recent, mistake_prompt='detailed', explicit_dedup=True):
    """Fill paper placeholders and append the local mistake deduplication rule."""
    if stage == 'completion':
        template = COMPLETION_PROMPT
    elif stage == 'mistake':
        if mistake_prompt not in ('detailed', 'simple'):
            raise ValueError(f'Unknown mistake prompt: {mistake_prompt}')
        template = (DETAILED_MISTAKE_PROMPT if mistake_prompt == 'detailed'
                    else SIMPLE_MISTAKE_PROMPT)
    else:
        raise ValueError(f'Unknown stage: {stage}')
    # Only model-generated feedback text belongs in this array, never reference labels.
    feedbacks = [entry['feedback'] for entry in recent]
    prompt = template.replace('[prev_detected_mistakes]', json.dumps(feedbacks, ensure_ascii=False)).replace(
        '[recipe_step]', instruction)
    if stage == 'mistake' and explicit_dedup:
        prompt += '\n\n' + DEDUP_RULE
    return prompt



def inference_messages(content, prompt):
    """Pack frames and task text into one user message."""
    return [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}]


def prediction_event(parsed, field):
    """Serialize decisions without presenting completion evidence as user speech.

    The paper specifies the prompts, but not this success-message mapping.
    Mistake feedback is preserved verbatim for official text metrics.
    """
    if not parsed[field]:
        return None
    if field == 'completed':
        return 'Success: Step complete.'
    return 'Feedback: ' + parsed['feedback']


def observed_frame_indices(start, requested, fps, count, sample_fps, max_frames=0):
    """Sample on a timestamp grid (also accurate for fractional source FPS)."""
    first = math.ceil(start * fps)
    current = min(count - 1, math.floor(requested * fps))
    indices = sorted({max(first, min(current, math.floor((start + tick / sample_fps) * fps)))
                      for tick in range(math.floor((requested - start) * sample_fps) + 1)
                      if start + tick / sample_fps <= requested})
    if first > current:
        return []
    if not indices or indices[-1] != current:
        indices.append(current)
    if max_frames and len(indices) > max_frames:
        indices = ([indices[round(i * (len(indices) - 1) / (max_frames - 1))]
                    for i in range(max_frames)] if max_frames > 1 else [current])
    return indices


def configure_generation(generation, token_limit):
    """Remove inherited length/time budgets; keep the model's EOS termination."""
    import sys
    generation.max_new_tokens = token_limit or None
    # Transformers requires an integer max_length. This sentinel disables its
    # default 20-token budget without allocating a buffer of this size.
    generation.max_length = sys.maxsize
    generation.max_time = None
    generation.min_length = 0
    generation.min_new_tokens = None
    generation.forced_eos_token_id = None


def generation_limits(token_limit):
    import sys
    return {'max_new_tokens': token_limit or None, 'max_length': sys.maxsize}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', default='/datasets/Hf_model/Qwen3-VL-2B-Instruct')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true',
                   help='Allow an existing output directory; replace files produced by this run')
    p.add_argument('--continue', dest='continue_run', action='store_true',
                   help='Reuse output directory; rerun the last started video and continue, preserving earlier predictions')
    p.add_argument('--video-ids', nargs='+')
    p.add_argument('--start-video-id', help='Start at this video (inclusive), in annotation order; use --limit 0 for all remaining videos')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--max-seconds', type=float, default=0, help='Absolute video cutoff, 0 = all')
    p.add_argument('--interval', type=float, default=5, help='Query cadence in seconds; paper turn-based baseline: 5 seconds')
    p.add_argument('--mistake-prompt', choices=('detailed', 'simple'), default='detailed',
                   help='Appendix D: detailed for Qwen3-VL/Qwen3.5/Gemini; simple for older baselines')
    p.add_argument('--feedback-dedup-rule', choices=('paper', 'explicit'), default='explicit',
                   help='paper: original mistake template; explicit: append the local dedup decision rule')
    p.add_argument('--sample-fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=32, help='0 = complete observed step prefix; positive = uniform frame cap')
    p.add_argument('--max-side', type=int, default=448, help='Maximum image side; preserve aspect ratio without upscaling')
    p.add_argument('--max-new-tokens', type=int, default=0, help='0 = EOS-only generation; positive = token cap')
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
            or a.max_frames < 0 or a.max_side < 32 or a.limit < 0 or a.max_new_tokens < 0):
        p.error('Invalid sampling parameters')
    samples = load_samples(a.data_root, a.video_ids, a.limit, a.start_video_id)
    results = []
    if a.continue_run:
        samples, results = continuation_state(a.output_dir, samples)
    for s in samples:
        if not recipe_steps(s):
            raise ValueError(f"No instructions: {s['video_id']}")
    import cv2
    from PIL import Image
    a.output_dir.mkdir(parents=True, exist_ok=a.overwrite or a.continue_run)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(feedback_history_policy='complete_current_step_emitted_feedback',
                  message_structure='single_user',
                  feedback_input_policy='detailed_mistake_prompt_array_only',
                  protocol='per_recipe_step', reference='arXiv:2606.09547v2',
                  prompt_version=('arxiv_2606.09547v2_appendix_D_explicit_dedup_v1'
                                  if a.feedback_dedup_rule == 'explicit' else 'arxiv_2606.09547v2_appendix_D_verbatim'),
                  paper_prompt_verbatim=a.feedback_dedup_rule == 'paper',
                  mistake_dedup_rule=DEDUP_RULE if a.feedback_dedup_rule == 'explicit' else None,
                  input_image_policy='aspect_ratio_thumbnail', output_length_policy='eos_only' if not a.max_new_tokens else 'token_cap',
                  context_frame_limit=a.max_frames or None,
                  success_message_policy='fixed_confirmation_reason_logged_only',
                  response_format='YES_NO', enable_thinking=template_kwargs.get('enable_thinking'),
                  chat_template_kwargs=template_kwargs, paper_interval_seconds=5,
                  interval_matches_paper=(a.interval == 5), trained_on_ego_comist=False)
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
        configure_generation(generation, a.max_new_tokens)

    for sample in samples:
        video_id = sample['video_id']
        cap = cv2.VideoCapture(str(a.data_root / 'videos' / f'{video_id}.MP4'))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f'Invalid video: {video_id}')
            duration = count / fps
            cutoff = min(duration, a.max_seconds) if a.max_seconds else duration
            steps = recipe_steps(sample)
            result = {'video_id': video_id, 'pred_texts': [], 'pred_timestamps': []}
            frame_dir = a.output_dir / 'model_input_frames' / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            with ((a.output_dir / f'{video_id}.jsonl').open('w') as log,
                  (a.output_dir / f'{video_id}.model_inputs.jsonl').open('w') as input_log):
                for step_id, step in enumerate(steps):
                    start = step['start']
                    end = min(steps[step_id + 1]['start'] if step_id + 1 < len(steps) else duration, cutoff)
                    if start >= cutoff:
                        break
                    # Injected instruction, not model output. Instruct: matches official parser.
                    if not a.dry_run:
                        result['pred_texts'].append('Instruct: ' + step['instruction'])
                        result['pred_timestamps'].append(start)
                    recent = []
                    turn = 1
                    # Stop at the next annotated instruction, not the hidden success label.
                    while start + turn * a.interval < end:
                        requested = start + turn * a.interval
                        indices = observed_frame_indices(start, requested, fps, count, a.sample_fps, a.max_frames)
                        content, timestamps, logged_content = [], [], []
                        for index in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                            ok, frame = cap.read()
                            if not ok:
                                raise RuntimeError(f'Cannot decode {video_id}, frame {index}')
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            image.thumbnail((a.max_side, a.max_side))
                            frame_path = frame_dir / f'{index:09d}.png'
                            if a.overwrite or a.continue_run or not frame_path.exists():
                                image.save(frame_path)
                            logged_content += [
                                {'type': 'text', 'text': f'Frame {index/fps:.3f}s'},
                                {'type': 'image', 'image': str(frame_path.relative_to(a.output_dir)),
                                 'width': image.width, 'height': image.height}]
                            timestamps.append(index / fps)
                            content += [{'type': 'text', 'text': f'Frame {index/fps:.3f}s'},
                                        {'type': 'image', 'image': image}]
                            event = None
                        for stage, field in [('completion', 'completed'), ('mistake', 'mistake')]:
                            prompt = prompt_for(stage, step['instruction'], requested, recent, a.mistake_prompt,
                                                explicit_dedup=a.feedback_dedup_rule == 'explicit')
                            messages = inference_messages(content, prompt)
                            input_log.write(json.dumps({
                                'video_id': video_id, 'step_id': step_id,
                                'timestamp': requested, 'stage': stage,
                                'messages': inference_messages(logged_content, prompt),
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
                                                         **generation_limits(a.max_new_tokens))
                                output_ids = ids[:, inputs.input_ids.shape[1]:]
                                generated_tokens = output_ids.shape[1]
                                eos_ids = generation.eos_token_id
                                eos_ids = ([eos_ids] if isinstance(eos_ids, int) else eos_ids) or []
                                ended_with_eos = bool(generated_tokens and output_ids[0, -1].item() in eos_ids)
                                hit_token_limit = bool(a.max_new_tokens and generated_tokens >= a.max_new_tokens and not ended_with_eos)
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
                            break
                        turn += 1
            results.append(result)
            save_json(a.output_dir / 'predictions.json', results)
        finally:
            cap.release()


if __name__ == '__main__':
    main()
