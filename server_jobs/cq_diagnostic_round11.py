"""Explicit simultaneous-support facts distinguish onset from occupancy."""


def occupancy_memory(checks):
    memory=measured_graph(checks)
    lines=['Measured simultaneous-support facts for the same referenced events:']
    for j,c in enumerate(checks):
        if not c['supported']:
            lines.append(f'E{j}: no positive visual support available.')
            continue
        v,a=c['visual_interval'],c['audio_interval']
        left,right=max(v[0],a[0]),min(v[1],a[1])
        overlap=max(0.0,right-left)
        fraction=overlap/max(a[1]-a[0],1e-8)
        lines.append(f'E{j}: visual support interval {v}s; spoken-mention interval {a}s; '
                     f'simultaneous support {overlap:.2f}s ({fraction:.1%} of the spoken-mention interval).')
        if overlap>0:
            lines.append(f'Within [{left:.2f},{right:.2f}]s, the speech is heard AND the matching visible event is occurring.')
        else:
            lines.append(f'No shared supported time between the spoken mention and the matching visible occurrence of E{j}.')
    return memory+'\n'+'\n'.join(lines)+'\n'


def occupancy_item(i,row):
    records=[json.loads(line) for line in (OUT/f'cq_diagnostic_round9_s{LAB_SERVER}.jsonl').read_text().splitlines() if line.strip()]
    saved=next(x for x in records if x['idx']==i and 'error' not in x)
    memory=occupancy_memory(saved['checks']) if saved['checks'] else saved['memory']
    stem,options=parse_question(row['question'])
    rec=dict(set=saved['set'],checks=saved['checks'],memory=memory)
    rec['cq_occupancy_graph']=text_only_scores(memory+stem,options)
    rec['cq_occupancy_neutral'],rec['reasoning']=neutral_reason_scores(memory,stem,options)
    print('OCCUPANCY_REASONING',i,rec['reasoning'],flush=True)
    return rec


DEV_MANIFEST=json.loads((OUT/f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
control_path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not control_path.exists():control_path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(control_path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in DEV_MANIFEST['selected']+CONTROL_MANIFEST['selected']]
CONDITIONS=('cq_occupancy_graph','cq_occupancy_neutral')
R=run_loop(ITEMS,f'cq_diagnostic_round11_s{LAB_SERVER}.jsonl',occupancy_item,show=CONDITIONS,min_left=0)
RESULT={split:{c:{'accuracy':acc([r for r in R if r.get('set')==split],c)[0],
                 'n':acc([r for r in R if r.get('set')==split],c)[1]} for c in CONDITIONS}
        for split in ('development','control')}
print('ROUND11_RESULT',json.dumps(RESULT),flush=True)
