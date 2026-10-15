"""Check whether fitted features change decisions or only internal self-match."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def probability(scores,rotations):
    x=np.asarray(scores,dtype=float)/rotations;x-=x.max()
    p=np.exp(x);return p/p.sum()

def audit(phase,baseline,methods):
    folder=ROOT/'results'/phase
    rows=[json.loads(x) for s in (1,2) for x in (folder/f'results_s{s}.jsonl').read_text().splitlines()]
    assert len(rows)==40
    result={}
    for method in methods:
        changes=[r['idx'] for r in rows if r['predictions'][method]!=r['predictions'][baseline]]
        tv=[float(abs(probability(r['scores'][method],len(r['audit']))-
                      probability(r['scores'][baseline],len(r['audit']))).sum()/2) for r in rows]
        result[method]=dict(changed_predictions=len(changes),changed_ids=changes,
            mean_total_variation=float(np.mean(tv)),median_total_variation=float(np.median(tv)),
            max_total_variation=float(max(tv)))
    note='TV is over normalized geometric-mean option probabilities across rotations, not token-level sensor attention or an event-grounding metric.'
    out=dict(phase=phase,baseline=baseline,n=40,note=note,comparisons=result)
    (folder/'decision_sensitivity_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))

if __name__=='__main__':
    audit('latent_slot_opt40','L0_native',['L1_unoptimized_memory','L2_optimized_memory','L3_time_shuffled_target'])
    audit('latent_slot_read15','E4_optimized_read23_reference',['E1_unoptimized_read15','E2_optimized_read15','E3_shuffled_read15'])
    paths=[ROOT/'results/presentation_resample40'/f'results_s{s}.jsonl' for s in (1,2)]
    if all(p.exists() for p in paths) and sum(len(p.read_text().splitlines()) for p in paths)==40:
        audit('presentation_resample40','P0_original_native_reference',['P1_pyav_frame_index','P2_pyav_presentation_time'])
