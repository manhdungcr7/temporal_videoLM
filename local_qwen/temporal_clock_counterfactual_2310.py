"""Single-source temporal counterfactuals; diagnostics, not benchmark scores.

Extend the normalized 2310 clip to 14 s using its final frame and silence.
Translate intact audio by 7 s, or translate both media timelines by 7 s.
The latter is a circular video edit: it preserves sampled pixels but introduces
a seam. Same shapes, token layout and relative chronology within each event.
All labels for edits are diagnostic expectations, not SVHalluc annotations.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
sys.path.insert(0, str(ROOT / 'research_topics_learning/single_sample_prompt_lab'))
import temporal_audited_runner as audited
from runner import load, prepare_video, qwen_inputs, move_inputs

OUT = ROOT / 'local_qwen/temporal_clock_counterfactual_2310.json'
STATE_OUT = OUT.with_suffix('.states.pt')
FIXTURE_DIR = ROOT / 'local_qwen/temporal_clock_fixtures_2310'
QUESTION = 'When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No'
SWAPPED = 'When the speech is heard, is the narrated event happening at the same time in the video? (A) No (B) Yes'


def array_hash(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def make_fixture(source):
    FIXTURE_DIR.mkdir(exist_ok=True)
    target = FIXTURE_DIR / 'source_padded_14s.mp4'
    if not target.exists():
        args = [str(audited.FFMPEG), '-hide_banner', '-y', '-i', str(source),
                '-vf', 'tpad=stop_mode=clone:stop_duration=5', '-af', 'apad',
                '-t', '14', '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(target)]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    return target


def conditions(original, padded):
    a = np.asarray(padded['audio'][0])
    shift = 7 * 16000
    # Refuse to circularly move any non-negligible signal from the tail to front.
    tail_peak = float(np.abs(a[-shift:]).max())
    if tail_peak > 1e-4:
        raise ValueError(f'Audio tail is not silent: {tail_peak}')
    shifted = dict(padded, audio=[np.roll(a, shift)])
    frames = padded['videos'][0]
    frame_shift = int(round(7 * padded['fps']))
    if abs(frame_shift / padded['fps'] - 7) > .02:
        raise ValueError('Video/audio translations would disagree')
    both = dict(shifted, videos=[torch.roll(frames, frame_shift, dims=0)])
    yield 'original_10s', original, QUESTION, 'A', 0.0, 0.0
    yield 'padded_early', padded, QUESTION, 'A', 0.0, 0.0
    yield 'audio_late', shifted, QUESTION, 'B', 7.0, 0.0
    yield 'both_late', both, QUESTION, 'A', 7.0, 7.0
    yield 'padded_early_swapped', padded, SWAPPED, 'B', 0.0, 0.0
    yield 'audio_late_swapped', shifted, SWAPPED, 'A', 7.0, 0.0


def run(model, processor, media, prompt, expected, state_store,
        media_mode='av', max_new_tokens=24):
    if media is None:
        from lab_native_video import SYSTEM
        rendered = processor.apply_chat_template([
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
            add_generation_prompt=True, tokenize=False)
        inputs = processor(text=rendered, return_tensors='pt')
    else:
        inputs, rendered = qwen_inputs(processor, prompt, media, media_mode=media_mode)
    cpu_ids = inputs['input_ids'].clone()
    metadata = {
        'input_shape': list(cpu_ids.shape),
        'input_ids_sha256': array_hash(cpu_ids),
        'video_frames': int(media['videos'][0].shape[0]) if media and media_mode != 'audio' else 0,
        'fps': media['fps'] if media else None,
        'audio_duration_s': len(media['audio'][0]) / 16000 if media and media_mode != 'vision' else 0,
        'audio_sha256': array_hash(media['audio'][0]) if media and media_mode != 'vision' else None,
        'sampled_pixels_sha256': array_hash(media['videos'][0]) if media and media_mode != 'audio' else None,
        'video_grid_thw': inputs['video_grid_thw'].tolist() if 'video_grid_thw' in inputs else [],
        'audio_tokens': int((cpu_ids == processor.tokenizer.convert_tokens_to_ids(processor.audio_token)).sum()),
        'video_tokens': int((cpu_ids == processor.tokenizer.convert_tokens_to_ids(processor.video_token)).sum()),
    }
    inputs = move_inputs(inputs, model)
    handles = []
    raw_logits = []
    for i, layer in enumerate(model.model.layers):
        def capture(k):
            def hook(_m, _a, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if hidden.shape[1] > 1:
                    state_store[k] = hidden[0, -1].detach().float().cpu()
            return hook
        handles.append(layer.register_forward_hook(capture(i)))
    handles.append(model.lm_head.register_forward_pre_hook(
        lambda _m, args: (args[0][:, -1:, :], *args[1:])))
    def head_capture(_m, _a, output):
        if not raw_logits:
            raw_logits.append(output[0, -1].detach().float().cpu())
    handles.append(model.lm_head.register_forward_hook(head_capture))
    start = time.time()
    try:
        with torch.inference_mode():
            result = model.generate(**inputs, use_audio_in_video=media is not None and media_mode == 'av',
                do_sample=False, num_beams=1, use_cache=True, max_new_tokens=max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                return_dict_in_generate=True)
        ids = result.sequences[0, inputs['input_ids'].shape[1]:].tolist()
        response = processor.tokenizer.decode(ids, skip_special_tokens=True)
        choice = audited.parse_choice(response, 'TA')
        option_map = {k.upper(): v.title() for k, v in re.findall(
            r'\(([AB])\)\s*(Yes|No)\b', prompt, re.I)}
        verbal = re.match(r'^\s*(Yes|No)\b', response, re.I)
        if verbal and set(option_map) == {'A', 'B'} and len(set(option_map.values())) == 2:
            choice = next(k for k, v in option_map.items() if v == verbal.group(1).title())
        logits = raw_logits[0]
        candidate_ids = {s: processor.tokenizer.encode(s, add_special_tokens=False)
                         for s in ('A', 'B', 'Yes', 'No')}
        if any(len(v) != 1 for v in candidate_ids.values()):
            raise ValueError(f'Non-single-token candidates: {candidate_ids}')
        scores = {s: float(logits[v[0]]) for s, v in candidate_ids.items()}
        yes_letter = 'B' if prompt == SWAPPED else 'A'
        no_letter = 'A' if prompt == SWAPPED else 'B'
        record = {'question': prompt, 'rendered_prompt': rendered,
            'expected_choice': expected, 'expectation_provenance': 'diagnostic edit expectation; coarse source event timing',
            'metadata': metadata, 'raw_output': response, 'generated_ids': ids,
            'parsed_choice': choice,
            'first_token_logits': scores,
            'yes_minus_no_letter_logit': scores[yes_letter] - scores[no_letter],
            'literal_Yes_minus_No_logit': scores['Yes'] - scores['No'],
            'top_tokens': [{'id': k, 'text': processor.tokenizer.decode([k]), 'logit': float(logits[k])}
                           for k in torch.topk(logits, 6).indices.tolist()],
            'seconds': round(time.time() - start, 3),
            'stop_reason': 'eos' if ids[-1] == processor.tokenizer.eos_token_id else 'token_cap',
            'max_new_tokens': max_new_tokens, 'media_mode': media_mode if media else 'text'}
        return record
    finally:
        for handle in handles:
            handle.remove()
        if 'result' in locals():
            del result
        del inputs
        torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    row = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))[2310]
    if row['question'] != QUESTION:
        raise ValueError('Original question changed')
    source, normalization = audited.normalize_av_clip(audited.DATA / row['video_path'])
    fixture = make_fixture(source)
    original, padded = prepare_video(source), prepare_video(fixture)
    jobs = list(conditions(original, padded))
    result = {'idx': 2310, 'normalization': normalization,
        'checkpoint': 'Qwen/Qwen2.5-Omni-3B', 'revision': audited.REV,
        'quantization': 'NF4 FP16 compute', 'fixture': str(fixture),
        'warning': 'Single source. All transformed conditions are diagnostics; source event intervals remain coarse. Joint video roll introduces an edit seam. No benchmark score or general model claim.',
        'conditions': {}}
    model, processor = load()
    all_states = {}
    for name, media, prompt, expected, audio_delta, video_delta in jobs:
        print('Running', name, flush=True)
        state_store = {}
        rec = run(model, processor, media, prompt, expected, state_store)
        rec.update(audio_shift_s=audio_delta, video_shift_s=video_delta)
        result['conditions'][name] = rec
        all_states[name] = state_store
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save(all_states, STATE_OUT)
        print(json.dumps({'condition': name, 'expected': expected,
            'choice': rec['parsed_choice'], 'margin_yes_no': rec['yes_minus_no_letter_logit'],
            'output': rec['raw_output'], 'seconds': rec['seconds']}, ensure_ascii=True), flush=True)
    for name in ('audio_late', 'both_late'):
        ref = result['conditions']['padded_early']['metadata']
        cur = result['conditions'][name]['metadata']
        if cur['input_ids_sha256'] != ref['input_ids_sha256']:
            raise ValueError(f'Intervention changed token layout: {name}')
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
