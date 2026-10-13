"""Locked small confirmation pilot of action/object/time evidence matching."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local_qwen'))
from temporal_clock_counterfactual_2310 import audited, load, prepare_video, run
from temporal_activity_grounding_prototype_2310 import repair
from temporal_structured_event_match_probe import EVENT_TEMPLATE, VISUAL_TEMPLATE, parse_tuple, action_group, words

IDS = (1961, 2081)  # Fixed before inference: one TA Yes and one TA No from confirmation.
OUT = ROOT / 'local_qwen/temporal_structured_pipeline_confirmation_ta.json'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    rows = json.loads((audited.DATA / 'samples.json').read_text(encoding='utf-8'))
    split = json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['confirmation']
    by_id = {x['idx']: x for x in split}
    if any(i not in by_id or by_id[i]['task'] != 'TA' for i in IDS):
        raise ValueError('IDs must remain held-out confirmation TA examples')
    result = {'protocol': 'Locked Qwen2.5-Omni-3B pipeline: RMS activity window, separate audio/video evidence, structured event tuples, deterministic action/object comparison.',
        'checkpoint': 'Qwen/Qwen2.5-Omni-3B', 'quantization': 'NF4, FP16 compute',
        'confirmation_source_ids': [by_id[i]['source_id'] for i in IDS],
        'selected_ids_before_inference': list(IDS),
        'decision_rule': 'A only when action family and specific object overlap; evidence is captioned from frames selected within the audio activity interval. Otherwise B.',
        'caveat': 'Only two held-out TA clips; exploratory confirmation, not a benchmark result. RMS is not a general speech detector.', 'samples': []}
    model, processor = load()
    for i in IDS:
        row, meta = rows[i], by_id[i]
        path, normalization = audited.normalize_av_clip(audited.DATA / row['video_path'])
        media = prepare_video(path)
        print('Confirmation TA', i, flush=True)
        baseline = run(model, processor, media, row['question'], meta['gold'], {}, max_new_tokens=24)
        evidence = repair(model, processor, media, row['question'], meta['gold'])
        speech_text = evidence['audio_observation']['raw_output']
        visual_text = evidence['visual_observation']['raw_output']
        event_result = run(model, processor, None, EVENT_TEMPLATE.replace('{text}', speech_text), None, {}, max_new_tokens=40)
        visual_result = run(model, processor, None, VISUAL_TEMPLATE.replace('{text}', visual_text), None, {}, max_new_tokens=40)
        event, event_error = parse_tuple(event_result['raw_output'])
        visible, visual_error = parse_tuple(visual_result['raw_output'])
        same_action = action_group(event['action']) is not None and action_group(event['action']) == action_group(visible['action'])
        same_object = bool(words(event['object']) & words(visible['object']))
        prediction = 'A' if same_action and same_object else 'B'
        sample = {'idx': i, 'task': meta['task'], 'gold': meta['gold'], 'source_id': meta['source_id'],
            'normalization': normalization, 'question': row['question'],
            'baseline': baseline, 'activity_span_s': evidence['activity_span_s'],
            'audio_observation': speech_text, 'visual_observation': visual_text,
            'event_tuple_output': event_result['raw_output'], 'event_tuple': event, 'event_parse_error': event_error,
            'visual_tuple_output': visual_result['raw_output'], 'visual_tuple': visible, 'visual_parse_error': visual_error,
            'same_action_family': same_action, 'same_object_token': same_object,
            'structured_prediction': prediction,
            'baseline_correct': baseline['parsed_choice'] == meta['gold'],
            'structured_correct': prediction == meta['gold']}
        result['samples'].append(sample)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'idx': i, 'gold': meta['gold'], 'baseline': baseline['parsed_choice'],
            'speech_tuple': event, 'visual_tuple': visible, 'same_action': same_action,
            'same_object': same_object, 'prediction': prediction}, ensure_ascii=True), flush=True)
    b = sum(x['baseline_correct'] for x in result['samples'])
    p = sum(x['structured_correct'] for x in result['samples'])
    result['summary'] = {'n': len(IDS), 'baseline_correct': b, 'structured_correct': p,
        'wrong_to_correct': sum(not x['baseline_correct'] and x['structured_correct'] for x in result['samples']),
        'correct_to_wrong': sum(x['baseline_correct'] and not x['structured_correct'] for x in result['samples'])}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Summary', json.dumps(result['summary'], ensure_ascii=True), flush=True)
    print('Saved', OUT, flush=True)


if __name__ == '__main__':
    main()
