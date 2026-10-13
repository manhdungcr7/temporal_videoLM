"""Diagnostic: test whether whole-interval captioning hides a brief action."""
import json
import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import audited, load, prepare_video, run

IDX = 1961
OUT = ROOT / 'local_qwen/temporal_frame_window_audit_1961.json'
WINDOWS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    rows = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))
    row = rows[IDX]
    path, normalization = audited.normalize_av_clip(audited.DATA / row['video_path'])
    media = prepare_video(path)
    frames, fps = media['videos'][0], media['fps']
    result = {'idx': IDX, 'normalization': normalization, 'question': row['question'],
        'sampled_fps': fps, 'whole_interval_caption': json.loads((ROOT / 'local_qwen/temporal_structured_pipeline_confirmation_ta.json').read_text(encoding='utf-8'))['samples'][0]['visual_observation'],
        'windows': [], 'warning': 'This is a targeted diagnostic after inspecting the held-out clip; do not treat it as a held-out performance result.'}
    model, processor = load()
    for start, end in WINDOWS:
        first = max(0, math.ceil(start * fps - 1e-8))
        last_inclusive = min(len(frames) - 1, int(end * fps + 1e-8))
        if last_inclusive < first:
            last_inclusive = first
        last = last_inclusive + 1
        selected = frames[first:last].clone()
        if len(selected) % 2:
            selected = torch.cat([selected, selected[-1:]], dim=0)
        local = dict(media, videos=[selected])
        prompt = f'Describe only the physical action and object visibly changing in this silent short video segment ({start:.1f}-{end:.1f} seconds). If no change is clear, say "no clear action". Do not infer from other parts of a recipe.'
        rec = run(model, processor, local, prompt, None, {}, media_mode='vision', max_new_tokens=48)
        item = {'interval_s': [start, end], 'indices_in_sampled_video': list(range(first, last)), 'caption': rec['raw_output']}
        result['windows'].append(item)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(item, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
