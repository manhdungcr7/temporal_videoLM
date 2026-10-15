"""Frozen beta=2 dual reader. Never select a reader from task/answer labels."""
CONTROL_MANIFEST=json.loads((OUT/f'cq_control_manifest_s{LAB_SERVER}.json').read_text())
ITEMS=[(d['idx'],DATA[d['idx']]) for d in CONTROL_MANIFEST['selected']]
BETA=2.0


def dual_control_item(i,row):
    rec=materialized_item(i,row)
    rec['cq_dual_readout']=(np.asarray(rec['cq_completed'])+BETA*np.asarray(rec['cq_memory'])).round(3).tolist()
    src,audio,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    rec['base']=score(prepare_video(src),stem,options)
    rec['beta']=BETA
    return rec


CONDITIONS=('base','cq_completed','cq_memory','cq_dual_readout')
R=run_loop(ITEMS,f'cq_dual_readout_controls_s{LAB_SERVER}.jsonl',dual_control_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('CONTROL_RESULT',json.dumps(RESULT),flush=True)
