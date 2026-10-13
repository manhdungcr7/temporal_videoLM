"""Separate temporal-binding failure from interpretation of the question.

Uses the same single-source clock fixtures as temporal_clock_counterfactual_2310.
Diagnostic questions contain no externally supplied events, timestamps or labels.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import (
    audited, conditions, load, make_fixture, prepare_video, run, QUESTION)
import torch

OUT = ROOT / 'local_qwen/temporal_clock_wording_probes_2310.json'
INSTANCE = ('In this particular clip, while the audible sentence is being spoken, '
            'is the action it describes visibly taking place? (A) Yes (B) No')
ANYWHERE = ('Does the action described by the audible speech appear anywhere in the '
            'visual frames of this clip, regardless of when the speech is heard? (A) Yes (B) No')
DIRECT = ('When the speech is heard, is the narrated event happening at the same time '
          'in the video? Answer Yes or No.')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    row = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))[2310]
    source, normalization = audited.normalize_av_clip(audited.DATA / row['video_path'])
    original, padded = prepare_video(source), prepare_video(make_fixture(source))
    media_by_name = {name: media for name, media, *_ in conditions(original, padded)}
    media_by_name['no_media'] = None
    jobs = [('question_only_no_media', 'no_media', QUESTION, None),
            ('anywhere_early', 'padded_early', ANYWHERE, 'A'),
            ('anywhere_late', 'audio_late', ANYWHERE, 'A'),
            ('instance_early', 'padded_early', INSTANCE, 'A'),
            ('instance_late', 'audio_late', INSTANCE, 'B'),
            ('instance_joint', 'both_late', INSTANCE, 'A'),
            ('direct_early', 'padded_early', DIRECT, 'A'),
            ('direct_late', 'audio_late', DIRECT, 'B')]
    result = {'idx': 2310, 'checkpoint': 'Qwen/Qwen2.5-Omni-3B',
        'quantization': 'NF4 FP16 compute', 'normalization': normalization,
        'note': 'Diagnostic task reformulations. Single source; not benchmark scores. No external evidence supplied to Qwen.',
        'conditions': {}}
    all_states = {}
    model, processor = load()
    for name, media_name, prompt, expected in jobs:
        print('Running', name, flush=True)
        states = {}
        rec = run(model, processor, media_by_name[media_name], prompt, expected, states)
        rec['media_condition'] = media_name
        rec['diagnostic_margin'] = (rec['literal_Yes_minus_No_logit'] if name.startswith('direct_')
                                    else rec['yes_minus_no_letter_logit'])
        rec['diagnostic_margin_space'] = 'literal Yes/No' if name.startswith('direct_') else 'option letters mapped to Yes/No'
        all_states[name] = states
        result['conditions'][name] = rec
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save(all_states, OUT.with_suffix('.states.pt'))
        print(json.dumps({'condition': name, 'expected': expected, 'choice': rec['parsed_choice'],
            'margin_yes_no': rec['diagnostic_margin'], 'output': rec['raw_output'],
            'seconds': rec['seconds']}, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
