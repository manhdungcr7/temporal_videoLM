"""Test an explicit action+object+time comparator on saved dev evidence.

The single prompt is frozen across original samples and clock counterfactuals.
This is a text-only diagnostic replay over model-generated evidence, not a new
end-to-end score or independent test set.
"""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import load, run

OUT = ROOT / 'local_qwen/temporal_same_event_counterfactual_probe.json'
TEMPLATE = (
    'Judge the relation between the narrated physical event and the silent visual evidence.\n'
    'Narrated event: {event}\n'
    'Visual evidence from frames sampled at the same time as the speech: {visual}\n'
    'Compare these three things separately: (1) the physical action, (2) the object acted on, '
    'and (3) whether the action is visibly occurring in this time interval.\n'
    'Answer Yes only when the evidence supports the same action happening to the same object '
    'during this interval. A similar action on a different object is No. A related cooking step, '
    'a tool being obtained, or an action elsewhere in the clip is not enough. Do not infer that '
    'an unspecified red/white ingredient is the named object.\n'
    'Question: {question}\n'
    'Select A (Yes) or B (No), then give one short reason.')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    normalized = json.loads((ROOT / 'local_qwen/temporal_narrated_event_normalization_probe.json').read_text(encoding='utf-8'))
    clock = json.loads((ROOT / 'local_qwen/temporal_clock_counterfactual_2310.json').read_text(encoding='utf-8'))['conditions']
    validation = json.loads((ROOT / 'local_qwen/temporal_activity_grounding_validation_dev3.json').read_text(encoding='utf-8'))['samples']
    if len(normalized['cases']) != 7 or len(validation) != 3:
        raise ValueError('Expected the complete seven-case diagnostic evidence')
    prompt_by_case = {}
    for case in normalized['cases']:
        prompt_by_case[case['case']] = {
            'event': case['event_extraction']['raw_output'],
            'visual': case['saved_visual_observation'],
            'old_choice': case['decision']['parsed_choice'],
            'expected': case['gold_or_edit_expectation']}
    result = {'protocol': 'One frozen action+object+time comparator over saved Qwen evidence; text-only replay.',
        'checkpoint': normalized['checkpoint'], 'quantization': normalized['quantization'],
        'template': TEMPLATE, 'warning': 'Development diagnostics only. Evidence text is Qwen-generated; timing edits share one source and are not independent benchmark samples.',
        'cases': []}
    model, processor = load()
    for name, data in prompt_by_case.items():
        if name.startswith('2310_'):
            question = clock['padded_early']['question']
        else:
            idx = int(name)
            question = next(s for s in validation if s['idx'] == idx)['baseline']['question']
        prompt = TEMPLATE.replace('{event}', data['event']).replace('{visual}', data['visual']).replace('{question}', question)
        rec = run(model, processor, None, prompt, data['expected'], {}, max_new_tokens=64)
        result['cases'].append({'case': name, 'expected_or_edit_expectation': data['expected'],
            'old_comparator_choice': data['old_choice'], 'event': data['event'],
            'visual_observation': data['visual'], 'decision': rec})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': name, 'expected': data['expected'], 'old': data['old_choice'],
            'new': rec['parsed_choice'], 'output': rec['raw_output']}, ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
