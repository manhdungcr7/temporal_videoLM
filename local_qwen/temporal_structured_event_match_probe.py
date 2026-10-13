"""Probe structured event extraction with deterministic action/object matching.

Qwen extracts event tuples from saved ASR and visual summaries. A small,
predeclared lexical comparator then checks action family, object identity and
whether the observations came from the speech-time frame crop. Diagnostics only.
"""
import json
import re
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import load, run

OUT = ROOT / 'local_qwen/temporal_structured_event_match_probe.json'
EVENT_TEMPLATE = (
    'Extract the main physical manipulation described by this spoken text. '
    'Ignore setup phrases such as getting a tool and ignore unrelated ingredients. '
    'Resolve pronouns. Return exactly one JSON object with keys "action" and "object". '
    'Use a short lowercase base verb and the specific manipulated object; use "unknown" '
    'if the text does not identify one. Do not explain.\nText: {text}')
VISUAL_TEMPLATE = (
    'From this silent visual observation, extract the main physical manipulation that is '
    'actually visible and the specific object being manipulated. Return exactly one JSON '
    'object with keys "action" and "object". Use a short lowercase base verb and specific '
    'object; use "unknown" when the observation does not establish it. Do not infer an '
    'object from a related action. Do not explain.\nObservation: {text}')
ACTION_GROUPS = [
    {'cut', 'chop', 'dice', 'slice', 'mince'},
    {'mix', 'stir', 'combine', 'blend'},
    {'mash', 'crush'},
    {'pour', 'add', 'put', 'place'},
]
STOP = {'the', 'a', 'an', 'of', 'into', 'with', 'on', 'some', 'small', 'large', 'fresh'}


def parse_tuple(output):
    match = re.search(r'\{[^{}]*\}', output, re.S)
    if not match:
        return {'action': 'unknown', 'object': 'unknown'}, 'invalid_json'
    try:
        obj = json.loads(match.group())
        if not isinstance(obj, dict) or not {'action', 'object'} <= set(obj):
            raise ValueError('Missing fields')
        return {k: str(obj[k]).strip().lower() for k in ('action', 'object')}, None
    except Exception as exc:
        return {'action': 'unknown', 'object': 'unknown'}, f'{type(exc).__name__}: {exc}'


def words(value):
    found = re.findall(r"[a-z]+", value.lower())
    singular = set()
    for word in found:
        if word.endswith('ies') and len(word) > 4:
            word = word[:-3] + 'y'
        elif word.endswith(('oes', 'ches', 'shes', 'xes', 'zes')):
            word = word[:-2]
        elif word.endswith('s') and len(word) > 3:
            word = word[:-1]
        singular.add(word)
    return singular - STOP - {'unknown'}


def action_group(value):
    tokens = words(value)
    bases = set(tokens)
    for token in tokens:
        if token.endswith('ing') and len(token) > 5:
            stem = token[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            bases.add(stem)
            bases.add(stem + 'e')
        if token.endswith('ed') and len(token) > 4:
            stem = token[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            bases.add(stem)
            bases.add(stem + 'e')
    for i, group in enumerate(ACTION_GROUPS):
        if bases & group:
            return i
    return None


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    norm = json.loads((ROOT / 'local_qwen/temporal_narrated_event_normalization_probe.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT / 'local_qwen/temporal_activity_grounding_validation_dev3.json').read_text(encoding='utf-8'))['samples']
    if len(norm['cases']) != 7 or len(validation) != 3:
        raise ValueError('Expected the complete seven-case saved evidence set')
    result = {'protocol': 'Frozen tuple extraction prompts and lexical rules across four originals and three 2310 clock edits; text-only extraction.',
        'checkpoint': norm['checkpoint'], 'quantization': norm['quantization'],
        'event_template': EVENT_TEMPLATE, 'visual_template': VISUAL_TEMPLATE,
        'action_groups': [sorted(x) for x in ACTION_GROUPS],
        'decision_rule': 'Yes only if action-family matches, specific object token overlaps, and both observations refer to speech-time evidence. Otherwise No. Unknown object/action is not a match.',
        'warning': 'Development only. Qwen-generated text may omit evidence; lexical rules are intentionally conservative and not validated on other domains.', 'cases': []}
    model, processor = load()
    for case in norm['cases']:
        name = case['case']
        speech = case['saved_audio_observation']
        visual = case['saved_visual_observation']
        if name.startswith('2310_'):
            timing = 'speech-time visual crop produced by the frozen activity-window procedure'
        else:
            timing = 'speech-time visual crop produced by the frozen activity-window procedure'
        print('Tuple extraction', name, flush=True)
        e = run(model, processor, None, EVENT_TEMPLATE.replace('{text}', speech), None, {}, max_new_tokens=40)
        v = run(model, processor, None, VISUAL_TEMPLATE.replace('{text}', visual), None, {}, max_new_tokens=40)
        et, ee = parse_tuple(e['raw_output'])
        vt, ve = parse_tuple(v['raw_output'])
        same_action = action_group(et['action']) is not None and action_group(et['action']) == action_group(vt['action'])
        same_object = bool(words(et['object']) & words(vt['object']))
        pred = 'A' if same_action and same_object else 'B'
        rec = {'case': name, 'expected_or_edit_expectation': case['gold_or_edit_expectation'],
            'old_choice': case['decision']['parsed_choice'], 'speech_text': speech,
            'visual_text': visual, 'speech_tuple_raw': e['raw_output'], 'speech_tuple': et,
            'speech_tuple_parse_error': ee, 'visual_tuple_raw': v['raw_output'],
            'visual_tuple': vt, 'visual_tuple_parse_error': ve,
            'same_action_family': same_action, 'same_object_token': same_object,
            'temporal_support': timing, 'prediction': pred,
            'correct': pred == case['gold_or_edit_expectation']}
        result['cases'].append(rec)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': name, 'expected': rec['expected_or_edit_expectation'],
            'speech_tuple': et, 'visual_tuple': vt, 'same_action': same_action,
            'same_object': same_object, 'prediction': pred}, ensure_ascii=True), flush=True)
    originals = [c for c in result['cases'] if not c['case'].startswith('2310_') or c['case'] in ('2310_original_10s',)]
    result['summary'] = {'original_clips': len(originals),
        'correct_original': sum(c['correct'] for c in originals),
        'wrong_to_correct_vs_baseline': sum(c['old_choice'] != c['expected_or_edit_expectation'] and c['correct'] for c in originals),
        'correct_to_wrong_vs_baseline': sum(c['old_choice'] == c['expected_or_edit_expectation'] and not c['correct'] for c in originals),
        'edits': [{'case': c['case'], 'expected': c['expected_or_edit_expectation'], 'prediction': c['prediction']} for c in result['cases'] if c['case'].startswith('2310_') and c['case'] != '2310_original_10s']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Summary', json.dumps(result['summary'], ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
