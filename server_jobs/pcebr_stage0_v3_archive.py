"""Audit raw v3 results, including seedwise frozen-context invariance."""
import collections
import hashlib
import json
import tarfile
import torch

_a3_folder=OUT/'pcebr_stage0_v3';_a3_dir=_a3_folder/'artifacts'
_a3_files=[];_a3_queries=[];_a3_probes=[];_a3_retrieval=collections.defaultdict(list)
for path in sorted(_a3_dir.glob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    _a3_files.append(dict(name=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if '_retrieval_' in path.name:
        metrics=_v3_orig_eval(obj['sim'],obj['examples'])
        assert metrics==obj['metrics'],('Reaggregate mismatch',path.name)
        sim=obj['sim'];ex=obj['examples'];norms=[x['content'].strip().lower() for x in ex]
        prefix=path.name.split('_retrieval_')[0]
        _a3_retrieval[prefix].append((path.name,sim,ex))
        for i,x in enumerate(ex):
            js=[j for j,y in enumerate(ex) if y['recording_id']==x['recording_id'] and y['source']==x['source']]
            gp=int(sim[i].argmax());lp=js[int(sim[i,js].argmax())]
            _a3_queries.append(dict(file=path.name,index=i,recording=x['recording_id'],source=x['source'],start=x['start'],
                global_pred=gp,local_pred=lp,global_correct=norms[i]==norms[gp],exact=lp==i,multipos=norms[i]==norms[lp],
                nonnull=not _v3_ns['is_null_description'](x['content'])))
    elif path.name.startswith('m2_probe_'):
        y=obj['y_heldout'];pred=obj['predictions'];rel=y<3
        _a3_probes.append(dict(file=path.name,n=len(y),correct=int((pred==y).sum()),n_rel=int(rel.sum()),correct_rel3=int((obj['predictions_3way']==y[rel]).sum()),
            per_class={str(c):dict(n=int((y==c).sum()),correct=int(((y==c)&(pred==c)).sum())) for c in range(4)}))
_a3_groupchecks=[]
for prefix,records in _a3_retrieval.items():
    records.sort(key=lambda x:x[0]);assert len(records)==26
    examples=records[0][2];groups=collections.defaultdict(list)
    for i,x in enumerate(examples):groups[(x['recording_id'],x['source'])].append(i)
    for seed in range(3):
        for ctxoffset,hieroffset,kind in [(0,3,'diag'),(4,6,'lowrank')]:
            ctx=records[2+seed*8+ctxoffset][1];hier=records[2+seed*8+hieroffset][1]
            cm=torch.stack([ctx[:,ix].max(1).values for ix in groups.values()],1)
            hm=torch.stack([hier[:,ix].max(1).values for ix in groups.values()],1)
            _a3_groupchecks.append(dict(prefix=prefix,seed=20261015+seed,kind=kind,
                max_error=float((cm-hm).abs().max()),winning_group_agreement=float((cm.argmax(1)==hm.argmax(1)).float().mean())))
_a3_audit=dict(server=PC_SERVER,files=_a3_files,raw_files=len(_a3_files),retrieval_queries=len(_a3_queries),
    retrieval_metrics_recomputed_exact=True,probes=_a3_probes,all_seed_group_checks=_a3_groupchecks)
(_a3_folder/'archive_audit.json').write_text(json.dumps(_a3_audit,indent=2),encoding='utf-8')
(_a3_folder/'retrieval_per_query.json').write_text(json.dumps(_a3_queries),encoding='utf-8')
archive_path=_a3_folder/'raw_artifacts.tar.gz'
with tarfile.open(archive_path,'w:gz') as archive:
    for path in sorted(_a3_dir.glob('*')):
        if path.is_file():archive.add(path,arcname='artifacts/'+path.name)
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v3/raw_artifacts.tar.gz',bytes=archive_path.stat().st_size,
    sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),raw_files=len(_a3_files),retrieval_queries=len(_a3_queries),
    probes=len(_a3_probes),all_group_max_errors_zero=all(x['max_error']==0 for x in _a3_groupchecks))
(_a3_folder/'archive_index.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('PCV3_ARCHIVE',json.dumps(RESULT),flush=True)
