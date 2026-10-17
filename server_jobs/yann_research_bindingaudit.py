"""R4 observed geometry and sensory-feature dependence, without changing fitted models."""
import importlib.util,json,torch
YR_DIR=OUT/'yann_temporal_research'
sp=importlib.util.spec_from_file_location('tam_geometry',OUT/'uploads/yann_tam_probe_model.py');tm=importlib.util.module_from_spec(sp);sp.loader.exec_module(tm)
rows=[]
for folder in ('binding48_v2','binding_evalextra'):
    for path in sorted((YR_DIR/folder).glob('*.pt')):
        if path.stem.endswith('_other'):continue
        c=torch.load(path,map_location='cpu',weights_only=True);e1,e2=c['endpoints'];truth=e1['interval'][0]<e2['interval'][0]
        start=[];center=[];earliest=[];supported=True
        for e in (e1,e2):
            d=tm.target_density(dict(spans=c['spans'],mods=c['mods'],source=e['source'],intervals=[e['interval']]),'cpu')
            if d is None:supported=False;break
            start.append(float(d@c['spans'][:,0]));center.append(float(d@c['spans'].mean(-1)));earliest.append(float(c['spans'][d>0,0].min()))
        rows.append(dict(video_id=c['video_id'],split=c['split'],variant=c['variant'],folder=folder,supported=supported,
            exact_event_center_order_correct=((sum(e1['interval'])<sum(e2['interval']))==truth),
            oracle_cell_start_mean_correct=(start[0]<start[1])==truth if supported else None,
            oracle_cell_center_mean_correct=(center[0]<center[1])==truth if supported else None,
            oracle_earliest_support_correct=(earliest[0]<earliest[1])==truth if supported else None))
summary={}
for split in ('train','validation','diagnostic'):
    rs=[r for r in rows if r['split']==split];summary[split]=dict(n=len(rs),supported=sum(r['supported'] for r in rs),
        counts={k:sum(bool(r[k]) for r in rs) for k in ('exact_event_center_order_correct','oracle_cell_start_mean_correct','oracle_cell_center_mean_correct','oracle_earliest_support_correct')})
RESULT=dict(server=YR_SERVER,summary=summary,rows=rows,note='Geometry-only diagnostics. Center-only errors do not prove that the full Fourier/memory representation lacks information. Earliest support is a coarse oracle, not deployment retrieval.')
(YR_DIR/'bindingaudit.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
