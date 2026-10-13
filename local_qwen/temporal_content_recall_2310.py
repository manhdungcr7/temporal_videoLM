"""Content recognition on the entire original clip, not on an oracle event crop."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import audited, load, prepare_video, run, make_fixture, QUESTION

OUT = ROOT / 'local_qwen/temporal_content_recall_2310.json'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    row = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))[2310]
    source, normalization = audited.normalize_av_clip(audited.DATA / row['video_path'])
    media = prepare_video(source)
    padded = prepare_video(make_fixture(source))
    jobs = [
        ('whole_clip_visual_only', 'vision',
         'Describe the visible actions in this clip in chronological order. Use at most three short sentences.'),
        ('whole_clip_audio_only', 'audio',
         'Transcribe the spoken words in the audio.'),
    ]
    result = {'idx': 2310, 'checkpoint': 'Qwen/Qwen2.5-Omni-3B',
        'quantization': 'NF4 FP16 compute', 'normalization': normalization,
        'note': 'Content-recall diagnostics only; no human labels or event crops supplied to Qwen. A/B logits are not meaningful for these open tasks.',
        'conditions': {}}
    model, processor = load()
    all_states = {}
    jobs = [(name, mode, prompt, media, 96) for name, mode, prompt in jobs]
    for shift_s in (2, 4, 6):
        shift = shift_s * 16000
        audio = np.asarray(padded['audio'][0])
        if float(np.abs(audio[-shift:]).max()) > 1e-4:
            raise ValueError('Clock sweep would wrap non-silent audio')
        shifted_media = dict(padded, audio=[np.roll(audio, shift)])
        jobs.append((f'clock_sweep_{shift_s}s', 'av', QUESTION, shifted_media, 24))
    for name, mode, prompt, job_media, max_tokens in jobs:
        print('Running', name, flush=True)
        states = {}
        rec = run(model, processor, job_media, prompt, None, states,
                  media_mode=mode, max_new_tokens=max_tokens)
        if name.startswith('clock_sweep_'):
            rec['audio_shift_s'] = int(name.split('_')[-1][:-1])
            rec['expectation_provenance'] = 'No hard edit label assigned: intermediate offsets may straddle action boundaries.'
        result['conditions'][name] = rec
        all_states[name] = states
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save(all_states, OUT.with_suffix('.states.pt'))
        print(json.dumps({'condition': name, 'output': rec['raw_output'],
            'stop_reason': rec['stop_reason'], 'seconds': rec['seconds']}, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
