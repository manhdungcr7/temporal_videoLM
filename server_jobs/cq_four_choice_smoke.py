"""Interface/paraphrase check, separate from benchmark accuracy."""
if LAB_SERVER!=1:
    raise ValueError('Run this check on server1')
index=2348
row=json.loads(fetch('samples.json').read_text(encoding='utf-8'))[index]
src,wave,dur,win=prep_clip(row,50)
question='Compared with the time at which the person spoke, when is the discussed event visible in the video?'
options=['Earlier only','During the speech','Later only','Not shown']
result=CQ_NS['cq_provenance_answer'](src,question,options,OUT/f'cq_physical_memory_{index}.json')
assert len(result['scores'])==4
(OUT/'cq_four_choice_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
RESULT={k:result[k] for k in ('answer','scores','mode','latency_sec')}
RESULT['four_choice_interface_passed']=True
print('FOUR_CHOICE_SMOKE',json.dumps(RESULT),flush=True)
