"""One fixed dense-frame probe at the brief narrated event interval."""
import json
from pathlib import Path
import sys
import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import audited, load, run

IDX = 1961
OUT = ROOT / 'local_qwen/temporal_dense_window_1961.json'
TIMES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    rows = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))
    path, normalization = audited.normalize_av_clip(audited.DATA / rows[IDX]['video_path'])
    container = av.open(str(path))
    stream = container.streams.video[0]
    selected = {}
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        nearest = min(TIMES, key=lambda x: abs(x - t))
        if abs(nearest - t) <= 0.13 and (nearest not in selected or abs(nearest-t) < abs(selected[nearest][0]-nearest)):
            rgb = frame.to_ndarray(format='rgb24')
            selected[nearest] = (t, torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
    container.close()
    if any(t not in selected for t in TIMES):
        raise RuntimeError(f'Missing requested frame timestamps: {sorted(set(TIMES)-set(selected))}')
    frames = torch.stack([selected[t][1] for t in TIMES])
    media = {'path': str(path), 'audio': [np.zeros(16000, dtype=np.float32)],
        'videos': [frames], 'fps': 4.0, 'max_pixels_per_frame': 160*28*28}
    model, processor = load()
    prompt = ('Describe the visible physical actions and the objects they affect in this short, silent video. '
        'Report only actions supported by visible change or hand-object interaction. If the frames show '
        'only an object already present, do not claim that it is being added or cooked. Keep the description to two sentences.')
    rec = run(model, processor, media, prompt, None, {}, media_mode='vision', max_new_tokens=72)
    result = {'idx': IDX, 'normalization': normalization,
        'sample_times_s': [{'requested': t, 'decoded': selected[t][0]} for t in TIMES],
        'sampled_fps': 4.0, 'raw_caption': rec['raw_output'],
        'warning': 'One targeted clip/window diagnostic, not an independent accuracy result.'}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
