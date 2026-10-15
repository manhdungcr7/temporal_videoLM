"""Continue the exact frozen server1 set on server2 after tunnel loss."""
LAB_SERVER=1
MANIFEST=json.loads((OUT/'uploads/cq_diagnostic_manifest_s1.json').read_text())
ITEMS=[(x['idx'],DATA[x['idx']]) for x in MANIFEST['selected']]
old_records=[json.loads(line) for line in (OUT/'uploads/cq_diagnostic_round3_s1.jsonl').read_text().splitlines() if line.strip()]
for rec in old_records:
    if 'error' not in rec:
        (OUT/f'cq_physical_memory_{rec["idx"]}.json').write_text(json.dumps(rec['observations']),encoding='utf-8')

# Reproduce only missing physical extraction; use unchanged cached observations.
for i,row in ITEMS:
    cache=OUT/f'cq_physical_memory_{i}.json'
    if not cache.exists():
        src,wave,dur,win=prep_clip(row,50)
        observations=extraction_memory(src,dur)
        cache.write_text(json.dumps(observations),encoding='utf-8')
        print('RECOVERED_PHYSICAL',i,flush=True)
R=run_loop(ITEMS,'cq_diagnostic_round6_s1.jsonl',hierarchy_item,
           show=('cq_hierarchy','cq_hierarchy_graph','cq_hierarchy_reason'),min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]}
        for c in ('cq_hierarchy','cq_hierarchy_graph','cq_hierarchy_reason')}
print('RECOVERED_HIERARCHY_RESULT',json.dumps(RESULT),flush=True)
