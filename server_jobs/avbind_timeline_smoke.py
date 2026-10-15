from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
index=globals().get('TIMELINE_SMOKE_INDEX',1454 if LAB_SERVER==1 else 2097)
row=json.loads(fetch('samples.json').read_text())[index]
src=cut(fetch(row['video_path'].replace(BS,'/')),0,10**4,'full')
question,options=parse_question(row['question'])
original=ALL_ATTENTION_FUNCTIONS['sdpa']
path=OUT/f'avbind_timeline_smoke_r11_s{LAB_SERVER}.json'
saved=json.loads(path.read_text()) if path.exists() else {}
assert not saved or saved['idx']==index
result={name:saved[name] for name in ('plain','isolated') if name in saved}
for isolated,name in ((False,'plain'),(True,'isolated')):
    if name in result:continue
    result[name]=TIMELINE_NS['avbind_timeline_answer'](src,question,options,isolated=isolated)
    assert ALL_ATTENTION_FUNCTIONS['sdpa'] is original
    path.write_text(json.dumps(dict(idx=index,task=row['task_type_abbr'],gold=row['answer'],**result)),encoding='utf-8')
assert result['plain']['audits']==result['isolated']['audits']
record=dict(idx=index,task=row['task_type_abbr'],gold=row['answer'],**result)
path.write_text(json.dumps(record),encoding='utf-8')
RESULT=dict(idx=index,predictions={name:'ABCD'[value['prediction']] for name,value in result.items()},
            scores={name:value['scores'] for name,value in result.items()},file=path.name,matched_tokens=True)
print('TIMELINE_SMOKE',json.dumps(RESULT),flush=True)
