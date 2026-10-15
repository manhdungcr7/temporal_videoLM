"""Frozen provenance CQ on the six fixed class-collapse controls."""
path=OUT/f'cq_control_manifest_s{LAB_SERVER}.json'
if not path.exists():path=OUT/f'uploads/cq_control_manifest_s{LAB_SERVER}.json'
CONTROL_MANIFEST=json.loads(path.read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in CONTROL_MANIFEST['selected']]


def provenance_control_item(i,row):
    src,wave,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    rec=cq_provenance_answer(src,stem,options,OUT/f'cq_physical_memory_{i}.json',mode='verified_peak')
    print('CONTROL_GRAPH',i,json.dumps(rec['checks']),flush=True)
    print('CONTROL_REASONING',i,rec['reasoning'],flush=True)
    return rec


CONDITIONS=('cq_verified_graph','cq_verified_choices','cq_verified_reason')
R=run_loop(ITEMS,f'cq_provenance_controls_s{LAB_SERVER}.jsonl',provenance_control_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('PROVENANCE_CONTROL_RESULT',json.dumps(RESULT),flush=True)
