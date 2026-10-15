i=1454 if LAB_SERVER==1 else 2097
row=json.loads(fetch('samples.json').read_text())[i]
src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
result=NEURAL_NS['avbind_counterfactual_answer'](src,question,options,OUT/f'avbind_differential_slots_{i}.pt',OUT/f'avbind_raw_values_{i}.pt',gain=1)
RESULT=dict(idx=i,scores=result['scores'],predicted='ABCD'[result['prediction']],fixed_coefficient=1,gain=1)
print('DIFFERENTIAL_SLOT_SMOKE',json.dumps(RESULT),flush=True)
