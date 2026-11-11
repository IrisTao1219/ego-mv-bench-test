"""Causal, sampled-frame cooking assistant baseline. Python 3.10+."""
import argparse
import json
import math
import os
from pathlib import Path
import time


def build_prompt(now, history):
    return (
        f'You are a cautious live cooking assistant at video time {now:.3f}s. '
        'Images are ordered past-to-present. Report at most one NEW event. '
        'Choose exactly one type: Instruction, Feedback, Success, or None. '
        'Instruction: a clearly supported next action; do not invent a recipe. '
        'Feedback: a specific visible mistake and a concise correction. '
        'Success: visible completion of a specific previously instructed action. '
        'A tidy kitchen or objects being present is not success. Touching oven '
        'controls does not prove it reached its target temperature. '
        'Previous assistant messages may be wrong; they are not evidence. '
        'Do not repeat an event already reported. If uncertain or nothing new '
        'needs saying, output exactly {"type":"None","text":""}. '
        'Return only one JSON object with a single type string and brief English text. '
        'Valid format examples (illustrations only, not observations): '
        '{"type":"Instruction","text":"Place the bowl on a stable surface."} '
        '{"type":"Feedback","text":"The bowl is overflowing; stop pouring."} '
        '{"type":"Success","text":"The instructed lid is now closed."} '
        'Never join type names with a pipe character. '
        'Previous assistant events: ' + json.dumps(history[-12:])
    )


def is_recent_duplicate(message, now, history):
    normalized = ' '.join(message.casefold().split())
    return any(now - event['time'] <= 30 and
               ' '.join(event['message'].casefold().split()) == normalized
               for event in history)


def parse_answer(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Response must be an object')
    kind, message = value.get('type'), value.get('text')
    if kind not in ('Instruction', 'Feedback', 'Success', 'None'):
        raise ValueError('Unknown response type')
    if not isinstance(message, str) or (kind != 'None' and not message.strip()):
        raise ValueError('Missing response text')
    return kind, message.strip()


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


def frame_indices(current, fps, count, window):
    start = max(0, current - int(window * fps))
    if count == 1:
        return [current]
    return sorted(set(round(start + i * (current - start) / (count - 1)) for i in range(count)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-path', default='/datasets/Hf_model/Qwen3-VL-2B-Instruct')
    p.add_argument('--output-dir', type=Path, required=True, help='Use a new directory for each run')
    p.add_argument('--video-ids', nargs='+')
    p.add_argument('--limit', type=int, default=1, help='0 = all videos')
    p.add_argument('--interval', type=float, default=5, help='Seconds between calls')
    p.add_argument('--window', type=float, default=10, help='Past context in seconds')
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--max-side', type=int, default=448)
    p.add_argument('--max-seconds', type=float, default=0, help='0 = entire video')
    p.add_argument('--max-new-tokens', type=int, default=160)
    p.add_argument('--device-map', default='auto')
    p.add_argument('--dry-run', action='store_true', help='Decode frames without loading model')
    a = p.parse_args()
    if (not all(math.isfinite(v) for v in (a.interval, a.window, a.max_seconds))
            or a.interval <= 0 or a.window < 0 or a.frames < 1 or a.max_side < 32
            or a.max_seconds < 0 or a.limit < 0 or a.max_new_tokens < 1):
        p.error('Invalid sampling or generation parameters')
    samples = load_samples(a.data_root, a.video_ids, a.limit)
    import cv2
    from PIL import Image

    # Keep existing experiment results intact.
    a.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(a.output_dir / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    model = processor = None
    if not a.dry_run:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model_path, dtype='auto', device_map=a.device_map,
            attn_implementation='sdpa', local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)

    predictions = []
    for sample in samples:
        video_id = sample['video_id']
        cap = cv2.VideoCapture(str(a.data_root / 'videos' / f'{video_id}.MP4'))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or total <= 0:
                raise ValueError(f'Cannot decode video metadata: {video_id}')
            last_time = (total - 1) / fps
            if a.max_seconds:
                last_time = min(last_time, a.max_seconds)
            result = {'video_id': video_id, 'pred_texts': [], 'pred_timestamps': []}
            history = []
            with (a.output_dir / f'{video_id}.jsonl').open('w') as log:
                step = 0
                while step * a.interval <= last_time:
                    current = min(total - 1, int(step * a.interval * fps))
                    now = current / fps
                    content, times = [], []
                    for index in frame_indices(current, fps, a.frames, a.window):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f'Decode failed: {video_id}, frame {index}')
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image.thumbnail((a.max_side, a.max_side))
                        times.append(index / fps)
                        content.extend([{'type': 'text', 'text': f'Frame at {index / fps:.3f}s:'},
                                        {'type': 'image', 'image': image}])
                    prompt = build_prompt(now, history)
                    content.append({'type': 'text', 'text': prompt})
                    started = time.monotonic()
                    raw, error, accepted, suppressed = '', None, None, None
                    if not a.dry_run:
                        inputs = processor.apply_chat_template(
                            [{'role': 'user', 'content': content}], tokenize=True,
                            add_generation_prompt=True, return_dict=True, return_tensors='pt',
                        ).to(model.device)
                        with torch.inference_mode():
                            ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                        raw = processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:],
                                                     skip_special_tokens=True)[0]
                        try:
                            kind, text = parse_answer(raw)
                            if kind != 'None':
                                candidate = f'{kind}: {text}'
                                if is_recent_duplicate(candidate, now, history):
                                    suppressed = 'duplicate_within_30_seconds'
                                else:
                                    accepted = candidate
                                    result['pred_texts'].append(accepted)
                                    result['pred_timestamps'].append(now)
                                    history.append({'time': now, 'message': accepted})
                        except (ValueError, TypeError) as exc:
                            error = str(exc)
                    entry = {'video_id': video_id, 'timestamp': now, 'frame_timestamps': times,
                             'raw_response': raw, 'accepted': accepted, 'parse_error': error,
                             'suppressed_reason': suppressed,
                             'latency_seconds': time.monotonic() - started, 'dry_run': a.dry_run}
                    log.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    log.flush()
                    print(f'{video_id} {now:.2f}s: {accepted or error or "None"}', flush=True)
                    save_json(a.output_dir / f'{video_id}.prediction.json', result)
                    step += 1
            predictions.append(result)
            save_json(a.output_dir / 'predictions.json', predictions)
        finally:
            cap.release()


if __name__ == '__main__':
    main()
