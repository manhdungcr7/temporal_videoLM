LAB_SERVER=1
MANIFEST=json.loads((OUT/'uploads/cq_diagnostic_manifest_s1.json').read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in MANIFEST['selected']]
CONDITIONS=('cq_cross_source','cq_cross_source_grounded','cq_cross_source_reason','cq_cross_source_compact')
R=run_loop(ITEMS,'cq_diagnostic_round7_s1.jsonl',checked_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('CROSS_SOURCE_TA_RESULT',json.dumps(RESULT),flush=True)
