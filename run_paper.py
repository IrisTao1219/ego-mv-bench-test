"""Per-recipe-step, completion-first, regular-interval inference baseline."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

from run_inference import load_samples, save_json


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
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get(field)) is not bool:
        raise ValueError(f'{field} must be a JSON boolean')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('Nonempty reason required')
    if field == 'mistake' and value[field] and not str(value.get('feedback', '')).strip():
        raise ValueError('Corrective feedback required')
    return value


def prompt_for(stage, instruction, now, recent):
    context = f'Current recipe step: {instruction}\nVideo observation time: {now:.3f}s. '
    context += 'Images are timestamped in chronological order. Judge only this step from visible evidence. '
    if stage == 'completion':
        return context + (
            'Has the person successfully completed ALL requirements of this step? '
            'Explain why, using observed actions. Do not assume unseen actions occurred. '
            'Return only JSON, e.g. {"completed":false,"reason":"The action is still in progress."}. '
            'completed must be a boolean; use true only when completion is supported.'
        )
    return context + (
        'Has the person made, or are they about to make, a mistake that prevents completing this step? '
        'Check ingredient quantities (too much, too little when they stop), preparation '
        '(wrong size, shape or ingredient), technique (wrong tool or method), temperature '
        '(wrong heat setting), and timing (too short or too long). '
        'Intervene before an imminent irreversible mistake, or after a visible recoverable error. '
        'Do not report generic safety advice or invent hidden mistakes. '
        'Do not repeat the same mistake feedback already reported in the last 60 seconds: '
        + json.dumps(recent) + '. '
        'Return only JSON, e.g. {"mistake":false,"reason":"No error is visible.","feedback":""}. '
        'If mistake is true, supply a concise actionable English correction in feedback.'
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', default='/datasets/Hf_model/Qwen3-VL-2B-Instruct')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--video-ids', nargs='+')
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--max-seconds', type=float, default=0, help='Absolute video cutoff, 0 = all')
    p.add_argument('--interval', type=float, default=5)
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
    a.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(protocol='per_recipe_step', paper='arXiv:2606.09547v2',
                  prompt_version='completion_then_mistake_v1_adapted', trained_on_ego_comist=False)
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
            with (a.output_dir / f'{video_id}.jsonl').open('w') as log:
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
                        content, timestamps = [], []
                        for index in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                            ok, frame = cap.read()
                            if not ok:
                                raise RuntimeError(f'Cannot decode {video_id}, frame {index}')
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            image.thumbnail((a.max_side, a.max_side))
                            timestamps.append(index / fps)
                            content += [{'type': 'text', 'text': f'Frame {index/fps:.3f}s'},
                                        {'type': 'image', 'image': image}]
                        recent = [r for r in recent if requested-r['time'] <= 60]
                        event = None
                        for stage, field in [('completion', 'completed'), ('mistake', 'mistake')]:
                            prompt = prompt_for(stage, step['instruction'], requested, recent)
                            raw, error, parsed = '', None, None
                            began = time.monotonic()
                            if not a.dry_run:
                                inputs = processor.apply_chat_template(
                                    [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}],
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
                                    event = ('Success: ' + parsed['reason'] if stage == 'completion'
                                             else 'Feedback: ' + parsed['feedback'].strip())
                            log.write(json.dumps({'step_id': step_id, 'instruction': step['instruction'],
                                'timestamp': requested, 'frame_timestamps': timestamps, 'stage': stage,
                                'prompt': prompt, 'raw_response': raw, 'parsed': parsed,
                                'parse_error': error, 'accepted': event, 'dry_run': a.dry_run,
                                'latency_seconds': time.monotonic()-began})+'\n')
                            log.flush()
                            # On malformed completion, do not silently treat it as false.
                            if stage == 'completion' and not a.dry_run and (parsed is None or parsed[field]):
                                break
                        if event:
                            result['pred_texts'].append(event)
                            result['pred_timestamps'].append(requested)
                            if event.startswith('Feedback:'):
                                recent.append({'time': requested, 'feedback': event})
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
