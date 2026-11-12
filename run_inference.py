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


def load_samples(root, video_ids, limit):
    samples = json.loads((root / 'annotations/annotations.json').read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError('Annotations must be a nonempty list')
    ids = [s['video_id'] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate video IDs')
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


def prompt_for(stage, instruction, now, recent, mistake_prompt='detailed'):
    """Fill paper placeholders only; now is logged separately, not added to the prompt."""
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
    return template.replace('[prev_detected_mistakes]', json.dumps(feedbacks, ensure_ascii=False)).replace(
        '[recipe_step]', instruction)



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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', default='/datasets/Hf_model/Qwen3-VL-2B-Instruct')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true',
                   help='Allow an existing output directory; replace files produced by this run')
    p.add_argument('--video-ids', nargs='+')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--max-seconds', type=float, default=0, help='Absolute video cutoff, 0 = all')
    p.add_argument('--interval', type=float, default=5, help='Paper: 5 seconds; changing this is a protocol deviation')
    p.add_argument('--mistake-prompt', choices=('detailed', 'simple'), default='detailed',
                   help='Appendix D: detailed for Qwen3-VL/Qwen3.5/Gemini; simple for older baselines')
    p.add_argument('--sample-fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=32, help='Uniformly sample the observed step prefix')
    p.add_argument('--max-side', type=int, default=448)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device-map', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if (not all(math.isfinite(v) for v in (a.interval, a.sample_fps, a.max_seconds))
            or min(a.interval, a.sample_fps) <= 0 or a.max_seconds < 0
            or a.max_frames < 1 or a.max_side < 32 or a.limit < 0 or a.max_new_tokens < 1):
        p.error('Invalid sampling parameters')
    samples = load_samples(a.data_root, a.video_ids, a.limit)
    for s in samples:
        if not recipe_steps(s):
            raise ValueError(f"No instructions: {s['video_id']}")
    import cv2
    from PIL import Image
    a.output_dir.mkdir(parents=True, exist_ok=a.overwrite)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(protocol='per_recipe_step', reference='arXiv:2606.09547v2',
                  prompt_version='arxiv_2606.09547v2_appendix_D_verbatim',
                  paper_prompt_verbatim=True,
                  success_message_policy='fixed_confirmation_reason_logged_only',
                  response_format='YES_NO', paper_interval_seconds=5,
                  interval_matches_paper=(a.interval == 5), trained_on_ego_comist=False)
    save_json(a.output_dir / 'config.json', config)
    if not a.dry_run:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model_path, dtype='auto', device_map=a.device_map,
            attn_implementation='sdpa', local_files_only=True).eval()
        processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)
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
                        current = min(count - 1, math.floor(requested * fps))
                        first = math.ceil(start * fps)
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
                            began = time.monotonic()
                            if not a.dry_run:
                                inputs = processor.apply_chat_template(
                                    messages,
                                    tokenize=True, add_generation_prompt=True, return_dict=True,
                                    return_tensors='pt').to(model.device)
                                with torch.inference_mode():
                                    ids = model.generate(**inputs, generation_config=generation,
                                                         do_sample=False, num_beams=1,
                                                         max_new_tokens=a.max_new_tokens)
                                raw = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                                             skip_special_tokens=True)[0]
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
