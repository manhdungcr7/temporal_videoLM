"""Replace a salience peak with the event episode containing that peak.

Temporal support is the maximal contiguous positive visual-verification region
around the strongest chunk. Zero is the natural Yes-vs-No decision boundary.
No parameter uses task, class, speech overlap, or expected outcome.
"""


def episode_check(check,observations):
    c=dict(check)
    values=c['margins']
    best=int(np.argmax(values))
    if values[best]<=0:
        c['episode_interval']=None
        return c
    left,right=best,best
    while left>0 and values[left-1]>0:
        left-=1
    while right+1<len(values) and values[right+1]>0:
        right+=1
    interval=[observations[left]['start'],observations[right]['end']]
    audio=c['audio_interval']
    c['peak_interval']=c['visual_interval']
    c['episode_interval']=interval
    c['visual_interval']=interval
    c['episode_chunks']=list(range(left,right+1))
    c['relation']='precedes' if interval[1]<=audio[0] else 'follows' if interval[0]>=audio[1] else 'overlaps'
    return c


def episode_item(i,row):
    filename='cq_diagnostic_round8' if i in DEVELOPMENT_IDS else 'cq_provenance_controls'
    saved_records=[json.loads(line) for line in (OUT/f'{filename}_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in saved_records if x['idx']==i and 'error' not in x)
    observations=json.loads((OUT/f'cq_physical_memory_{i}.json').read_text())
    stem,options=parse_question(row['question'])
    checks=[episode_check(c,observations) for c in saved['checks']]
    memory=measured_graph(checks) if checks else saved['memory']
    rec=dict(set='development' if i in DEVELOPMENT_IDS else 'control',checks=checks,memory=memory,
             original_reason=saved['cq_verified_reason'])
    rec['cq_episode_graph']=text_only_scores(memory+stem,options)
    rec['cq_episode_reason'],rec['reasoning']=reason_scores(memory,stem,options)
    print('EPISODES',i,json.dumps(checks),flush=True)
    print('EPISODE_REASONING',i,rec['reasoning'],flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
DEVELOPMENT_IDS={x['idx'] for x in DEV_MANIFEST['selected']}
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_episode_graph','cq_episode_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round9_s{LAB_SERVER}.jsonl',episode_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND9_RESULT',json.dumps(RESULT),flush=True)
