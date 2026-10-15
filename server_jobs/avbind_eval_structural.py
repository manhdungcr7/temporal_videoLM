"""Unseen generated scenes, separate from any real AVQA performance."""
if not NEURAL_NS['BIND_METADATA'].get('trained'):
    raise RuntimeError('Training did not finish; this is not a valid trained evaluation')
held=json.loads((OUT/'avbind_structural_holdout.json').read_text());records=[]
path=OUT/'avbind_structural_eval.jsonl'
for i,row in enumerate(held):
    rec=dict(idx=i,family=row['metadata']['family'],gold=row['gold'])
    for mode in ('base','binding'):
        scores,audits=NEURAL_NS['binding_scores'](row['nodes'],row['question'],row['answer_options'],
                                               NEURAL_NS['BIND_READER'] if mode=='binding' else None)
        rec[mode]=scores;rec[mode+'_correct']=int(np.argmax(scores))==row['gold']
        if mode=='binding':rec['audits']=audits
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
    records.append(rec)
    print('BIND_STRUCTURAL',i,rec['family'],rec['base_correct'],rec['binding_correct'],flush=True)
RESULT=dict(n=len(records),base=sum(r['base_correct'] for r in records),
            binding=sum(r['binding_correct'] for r in records),textual_only=True)
print('BIND_STRUCTURAL_RESULT',json.dumps(RESULT),flush=True)
