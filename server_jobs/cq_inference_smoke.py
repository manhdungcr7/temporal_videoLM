"""Check the dataset-free API on a positive overlap case on each GPU."""
index=int(globals().get('SMOKE_ID',2348 if LAB_SERVER==1 else 2124))
row=json.loads(fetch('samples.json').read_text(encoding='utf-8'))[index]
src,wave,dur,win=prep_clip(row,50)
question,options=parse_question(row['question'])
result=CQ_NS['cq_provenance_answer'](src,question,options,OUT/f'cq_physical_memory_{index}.json')
result.update(idx=index,gold=row['answer'],correct='ABCD'[result['prediction']]==row['answer'])
(OUT/f'cq_inference_smoke_{index}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
RESULT={k:result[k] for k in ('idx','gold','answer','correct','scores','mode','latency_sec')}
RESULT['benchmark_variables_absent']=CQ_INFERENCE_METADATA['benchmark_variables_absent']
print('CQ_INFERENCE_SMOKE',json.dumps(RESULT),flush=True)
