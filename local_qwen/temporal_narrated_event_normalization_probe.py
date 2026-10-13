"""Frozen text-only probe: distinguish the narrated manipulation from setup.

Reuses saved Qwen observations. Not a fresh end-to-end benchmark run and not a
claim of novelty. All prompts are generic, contain no sample ID/gold/object cues.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import load, run

OUT = ROOT / 'local_qwen/temporal_narrated_event_normalization_probe.json'
EVENT_PROMPT = (
    'Identify the concrete physical action this utterance is primarily describing. '
    'Resolve pronouns to their objects. If it names an ingredient or says to get a tool '
    'before describing a manipulation, select the manipulation rather than the setup. '
    'Return only a short action-and-object phrase.\nUtterance: {utterance}')
MATCH_PROMPT = (
    'Narrated physical action: {event}\n'
    'Visual observations from frames during the same interval as the speech: {visual}\n'
    'Do the observed visual actions include the narrated physical action? '
    '(A) Yes (B) No\nSelect A or B, then give one short reason.')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    source = json.loads((ROOT / 'local_qwen/temporal_activity_grounding_prototype_2310.json').read_text(encoding='utf-8'))
    validation = json.loads((ROOT / 'local_qwen/temporal_activity_grounding_validation_dev3.json').read_text(encoding='utf-8'))
    if len(validation['samples']) != 3:
        raise ValueError('Wait for the validation evidence to finish')
    jobs = [(f'2310_{name}', rec['expected_choice'], rec)
            for name, rec in source['conditions'].items()]
    jobs += [(str(s['idx']), s['gold'], s['repair']) for s in validation['samples']]
    result = {'protocol': 'Text-only diagnostic rerun on saved Qwen evidence. Event extraction and matching prompts fixed before running all seven cases.',
        'checkpoint': 'Qwen/Qwen2.5-Omni-3B', 'quantization': 'NF4 FP16 compute',
        'event_template': EVENT_PROMPT, 'match_template': MATCH_PROMPT,
        'warning': 'This changes the interpretation of narrated event. Multiple narrated actions and action granularity can remain ambiguous. Not a new independent test split.',
        'cases': []}
    model, processor = load()
    for name, gold, rec in jobs:
        print('Event normalization', name, flush=True)
        audio = rec['audio_observation']['raw_output']
        visual = rec['visual_observation']['raw_output']
        event = run(model, processor, None, EVENT_PROMPT.replace('{utterance}', audio),
                    None, {}, max_new_tokens=48)
        decision = run(model, processor, None,
            MATCH_PROMPT.replace('{event}', event['raw_output']).replace('{visual}', visual),
            gold, {}, max_new_tokens=64)
        result['cases'].append({'case': name, 'gold_or_edit_expectation': gold,
            'saved_audio_observation': audio, 'saved_visual_observation': visual,
            'previous_prediction': rec['final']['parsed_choice'],
            'event_extraction': event, 'decision': decision})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': name, 'expected': gold,
            'event': event['raw_output'], 'prediction': decision['parsed_choice'],
            'output': decision['raw_output']}, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
