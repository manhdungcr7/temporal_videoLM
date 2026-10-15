"""Three new development probes, excluding every reserved evaluation video."""
reserved=set()
for s in (1,2):
    manifest=json.loads((OUT/f'uploads/cq_large_manifest_s{s}.json').read_text())
    reserved.update(x['video_path'] for x in manifest['selected'])
reserved.update(json.loads((OUT/'uploads/cq_large_exclusions.json').read_text())['video_paths'])
rows=json.loads(fetch('samples.json').read_text());selected=[]
for gold in 'ABC':
    pool=[(i,row) for i,row in enumerate(rows) if row['data_type']=='temporal' and row['task_type_abbr']=='CMTB'
          and row['answer']==gold and row['video_path'].replace(BS,'/') not in reserved]
    random.Random(f'2026101402:CMTB:{gold}').shuffle(pool)
    for i,row in pool:
        src=fetch(row['video_path'].replace(BS,'/'))
        with av.open(str(src)) as c:duration=float(c.duration/av.time_base)
        if duration<=50:
            selected.append(dict(idx=i,task='CMTB',gold=gold,duration=duration,video_path=row['video_path'].replace(BS,'/')));break
    else:raise RuntimeError(f'No unused <=50s CMTB/{gold} clip remains')
(OUT/'avbind_cmtb_manifest.json').write_text(json.dumps(dict(selected=selected,seed=2026101402,
     selection='one per CMTB class, independent of predictions, excludes reserved400 videos'),indent=2),encoding='utf-8')
RESULT=dict(selected=selected)
print('CMTB_PROBES_SELECTED',json.dumps(RESULT),flush=True)
