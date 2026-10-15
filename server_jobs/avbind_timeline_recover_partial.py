"""Preserve the completed plain condition from the persistent failed-job namespace."""
index=1454 if LAB_SERVER==1 else 2097
assert globals()['row']['task_type_abbr'] in ('TA','TF')
prior_result=globals().get('result',{})
path=OUT/f'avbind_timeline_smoke_r11_s{LAB_SERVER}.json'
if not path.exists() and isinstance(prior_result,dict) and 'plain' in prior_result:
    assert prior_result['plain']['isolated'] is False
    assert globals().get('index')==index
    record=dict(idx=index,task=row['task_type_abbr'],gold=row['answer'],plain=prior_result['plain'])
    path.write_text(json.dumps(record),encoding='utf-8')
RESULT=dict(file=path.name,recovered=path.exists(),scores=prior_result.get('plain',{}).get('scores'))
print('TIMELINE_PARTIAL_RECOVERED',json.dumps(RESULT),flush=True)
