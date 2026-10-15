"""One real clip first; verify zero-gain score parity and registry restoration."""
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
row=json.loads(fetch('samples.json').read_text())[1454 if LAB_SERVER==1 else 2097]
src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full');question,options=parse_question(row['question'])
i=1454 if LAB_SERVER==1 else 2097;original=ALL_ATTENTION_FUNCTIONS['sdpa']
plain=NEURAL_NS['avbind_latent_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',enabled=False)
zero=NEURAL_NS['avbind_attention_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',gain=0)
delta=max(abs(a-b) for a,b in zip(plain['scores'],zero['scores']))
if delta>.01:raise RuntimeError(f'Zero-gain baseline parity failed: {delta}')
boosted=NEURAL_NS['avbind_attention_answer'](src,question,options,OUT/f'avbind_raw_slots_{i}.pt',gain=2)
assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
RESULT=dict(idx=i,zero_gain_max_score_difference=delta,baseline=plain['scores'],boosted=boosted['scores'],registry_restored=True,
            max_bias=[r['max_key_bias'] for r in boosted['audits']])
(OUT/f'avbind_attention_smoke_s{LAB_SERVER}.json').write_text(json.dumps(RESULT),encoding='utf-8')
print('ATTENTION_SMOKE',json.dumps(RESULT),flush=True)
