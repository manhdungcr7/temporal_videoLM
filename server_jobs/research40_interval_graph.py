"""Question-independent interval relations, not a TA/TF answer program.

All source pairs retain separate contents and provenance. Relations describe
observation intervals only. No event-coreference, answer, or task is assumed.
"""
import json,time,gc
import numpy as np
import torch

R40_GRAPH_FILE=OUT/f'research40_interval_graph_s{R40_SERVER}.jsonl'

def r40_interval_relation(x,y,eps=0.04):
    a,b=x['start'],x['end'];c,d=y['start'],y['end']
    if b<c-eps:return 'before'
    if abs(b-c)<=eps:return 'meets'
    if a>d+eps:return 'after'
    if abs(a-d)<=eps:return 'met_by'
    same_start=abs(a-c)<=eps;same_end=abs(b-d)<=eps
    if same_start and same_end:return 'equal'
    if same_start:return 'starts' if b<d else 'started_by'
    if same_end:return 'finishes' if a>c else 'finished_by'
    if a<c and b>d:return 'contains'
    if a>c and b<d:return 'during'
    return 'overlaps' if a<c else 'overlapped_by'

def r40_interval_graph_text(nodes):
    audio=[n for n in nodes if n['kind'].startswith('audio')]
    video=[n for n in nodes if n['kind']=='visual']
    lines=[
        'Source observations and measured clock relationships. These relationships concern the time intervals '
        'of the observations, not an assumption that they describe the same event. '
        'Whether two descriptions refer to the same event must be established from their content. '
        'A visual note does not assert that its action fills the entire interval. '
        'Audio speech reports what was said, not necessarily what was then visible.']
    for i,n in enumerate(audio):lines.append(f"A{i}: speech heard [{n['start']:.2f},{n['end']:.2f}]s: {n['text']}")
    for j,n in enumerate(video):lines.append(f"V{j}: visible observation [{n['start']:.2f},{n['end']:.2f}]s: {n['text']}")
    lines.append('Clock links for all cross-source pairs (not filtered using the question):')
    for i,a in enumerate(audio):
        for j,v in enumerate(video):
            relation=r40_interval_relation(v,a)
            overlap=max(0,min(v['end'],a['end'])-max(v['start'],a['start']))
            dt=(v['start']+v['end']-a['start']-a['end'])/2
            if v['end']<=a['start']:
                prose=f"V{j} observation finishes {a['start']-v['end']:.2f}s before A{i} speech begins"
            elif v['start']>=a['end']:
                prose=f"V{j} observation begins {v['start']-a['end']:.2f}s after A{i} speech finishes"
            else:prose=f"V{j} observation and A{i} speech have overlapping clock intervals"
            lines.append(f"V{j}<->A{i}: {relation}; overlap={overlap:.2f}s; mid(V)-mid(A)={dt:+.2f}s; {prose}.")
    return '\n'.join(lines)+'\n\n'

faithful=r40_json_rows(OUT/f'research40_faithful_memory_s{R40_SERVER}.jsonl')
assert len(faithful)==len(R40_SAMPLES),'Complete faithful observations for all fixed IDs first.'
by_id={r['idx']:r for r in faithful}
baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
records=r40_json_rows(R40_GRAPH_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time();stem,texts=parse_question(r['question']);k=len(texts)
    supplement=r40_interval_graph_text(by_id[r['idx']]['observations'])
    payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
    raw=np.zeros(k);textonly=np.zeros(k);cot=np.zeros(k)
    state=r40_memory_analysis(payload,stem,supplement)
    for perm in rotations(k):
        choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
        lg,_,_,_=r40_memory_context(payload,stem,supplement,choices)
        v=lg[R40_LABELS[:k]].cpu().numpy();lp=v-np.logaddexp.reduce(v)
        v2=r40_text_logits(supplement+r40_prompt(stem,texts,perm))[:k];lp2=v2-np.logaddexp.reduce(v2)
        v3=r40_reason_answer(state,texts,perm);lp3=v3-np.logaddexp.reduce(v3)
        for slot,i in enumerate(perm):raw[i]+=lp[slot];textonly[i]+=lp2[slot];cot[i]+=lp3[slot]
    rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
        scores=dict(interval_graph_raw_av=raw.tolist(),interval_graph_text_reader=textonly.tolist(),
                    interval_graph_reasoning=cot.tolist()),analysis=state['text'],clock_graph=supplement,
        seconds=round(time.time()-started,2))
    with R40_GRAPH_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
    records.append(rec);done.add(r['idx'])
    merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
    r40_snapshot(merged,'interval_graph')
    print('INTERVAL_GRAPH_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','analysis')}),flush=True)
    del payload,state;gc.collect();torch.cuda.empty_cache()
RESULT=r40_snapshot(merged,'interval_graph_done')
print('INTERVAL_GRAPH_SUMMARY',json.dumps(RESULT),flush=True)
