from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
i=1454 if LAB_SERVER==1 else 2097
row=json.loads(fetch('samples.json').read_text())[i]
src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
before=ALL_ATTENTION_FUNCTIONS['sdpa']
ordinary=NEURAL_NS['avbind_latent_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',enabled=False)
zero=NEURAL_NS['avbind_kv_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',OUT/f'avbind_raw_values_{i}.pt',gain=0)
if max(abs(a-b) for a,b in zip(ordinary['scores'],zero['scores']))>.01:raise RuntimeError('KV zero-gain parity failed')
result=NEURAL_NS['avbind_kv_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',OUT/f'avbind_raw_values_{i}.pt',gain=1)
assert ALL_ATTENTION_FUNCTIONS['sdpa'] is before
RESULT=dict(idx=i,baseline=ordinary['scores'],binding=result['scores'],registry_restored=True)
print('INDEPENDENT_KV_SMOKE',json.dumps(RESULT),flush=True)
