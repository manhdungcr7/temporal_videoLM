"""Choice-position controls and frozen repair on three other TA dev sources."""
import json
from pathlib import Path
import sys
import torch
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import audited, load, make_fixture, prepare_video, run, QUESTION
from temporal_activity_grounding_prototype_2310 import repair

OUT = ROOT / 'local_qwen/temporal_activity_grounding_validation_dev3.json'
CHOICE_OUT = ROOT / 'local_qwen/temporal_choice_position_controls_2310.json'
IDS = (1842, 1668, 1485)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    rows = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))
    manifest = json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))
    dev = {x['idx']: x for x in manifest['splits']['development']}
    if not set(IDS).issubset(dev) or any(dev[i]['task'] != 'TA' for i in IDS):
        raise ValueError('Only existing development TA sources may be used')
    model, processor = load()
    source, norm = audited.normalize_av_clip(audited.DATA / rows[2310]['video_path'])
    padded = prepare_video(make_fixture(source))
    controls = [
        ('reverse_option_order', '(B) No (A) Yes', 'A', {'A': 'Yes', 'B': 'No'}),
        ('reverse_order_swapped_meanings', '(B) Yes (A) No', 'B', {'A': 'No', 'B': 'Yes'}),
    ]
    choice_result = {'idx': 2310, 'normalization': norm,
        'note': 'Additional same-media diagnostic controls: distinguish option-letter preference from last-position preference.',
        'conditions': {}}
    stem = QUESTION.split(' (A)')[0]
    for name, suffix, expected, mapping in controls:
        print('Choice-position control', name, flush=True)
        rec = run(model, processor, padded, stem + ' ' + suffix, expected, {})
        rec['choice_meanings'] = mapping
        # parse_choice's verbal Yes/No fallback assumes A=Yes,B=No. Remap a
        # leading verbal answer when these controls use a different option map.
        verbal = re.match(r'^\s*(Yes|No)\b', rec['raw_output'], re.I)
        if verbal:
            meaning = verbal.group(1).title()
            rec['parsed_choice'] = next(k for k, v in mapping.items() if v == meaning)
            rec['parser_note'] = 'Leading verbal answer mapped using the options actually shown.'
        rec['parsed_meaning'] = mapping.get(rec['parsed_choice'])
        rec['yes_minus_no_letter_logit'] = (rec['first_token_logits']['A'] - rec['first_token_logits']['B']) * (1 if mapping['A'] == 'Yes' else -1)
        choice_result['conditions'][name] = rec
        CHOICE_OUT.write_text(json.dumps(choice_result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'condition': name, 'choice': rec['parsed_choice'],
            'meaning': rec['parsed_meaning'], 'output': rec['raw_output']}, ensure_ascii=True), flush=True)
    result = {'ids': list(IDS), 'protocol': 'Development only; three different sources. Repair prompts frozen from 2310. No confirmation samples used.',
        'checkpoint': 'Qwen/Qwen2.5-Omni-3B', 'quantization': 'NF4 FP16 compute',
        'warning': 'Small diagnostic pilot, not an estimate of general benchmark performance. RMS is an activity envelope, not a general VAD.',
        'samples': []}
    for i in IDS:
        print('Validation source', i, flush=True)
        row, item = rows[i], dev[i]
        path, norm = audited.normalize_av_clip(audited.DATA / item['video_path'])
        media = prepare_video(path)
        baseline = run(model, processor, media, row['question'], item['gold'], {})
        rec = repair(model, processor, media, row['question'], item['gold'])
        result['samples'].append({'idx': i, 'gold': item['gold'], 'source_id': item['source_id'],
            'normalization': norm, 'baseline': baseline, 'repair': rec})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'idx': i, 'gold': item['gold'], 'baseline': baseline['parsed_choice'],
            'repair': rec['final']['parsed_choice'], 'activity': rec['activity_span_s'],
            'audio': rec['audio_observation']['raw_output'],
            'visual': rec['visual_observation']['raw_output'],
            'final': rec['final']['raw_output']}, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
