"""Recompute stored metrics and archive all raw Stage-0 v2 artifacts."""
import ast
import hashlib
import json
import tarfile
import torch

_ar_folder=OUT/'pcebr_stage0_v2';_ar_dir=_ar_folder/'artifacts'
_ar_tree=ast.parse((OUT/'uploads/stage0_pcebr_mechanism_check_v2_original.py').read_text(encoding='utf-8'))
_ar_tree.body=[n for n in _ar_tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp))]
_ar_ns={'__name__':'pcv2_archive_eval','OUT':OUT}
exec(compile(_ar_tree,'pcv2_archive_eval','exec'),_ar_ns)
_ar_manifest=[];_ar_queries=[];_ar_probes=[]
for _ar_path in sorted(_ar_dir.glob('*.pt')):
    obj=torch.load(_ar_path,map_location='cpu',weights_only=True)
    _ar_manifest.append(dict(name=_ar_path.name,bytes=_ar_path.stat().st_size,sha256=hashlib.sha256(_ar_path.read_bytes()).hexdigest()))
    if _ar_path.name.endswith('_retrieval.pt'):
        metrics=_ar_ns['evaluate_sim_matrix'](obj['sim'],obj['examples'])
        assert metrics==obj['metrics'],('Retrieval reaggregate error',_ar_path.name)
        sim=obj['sim'];examples=obj['examples'];norms=[x['content'].strip().lower() for x in examples]
        for i,ex in enumerate(examples):
            g=[j for j,x in enumerate(examples) if x['recording_id']==ex['recording_id'] and x['source']==ex['source']]
            gp=int(sim[i].argmax());lp=g[int(sim[i,g].argmax())]
            _ar_queries.append(dict(file=_ar_path.name,index=i,recording=ex['recording_id'],source=ex['source'],
                start=ex['start'],global_pred=gp,local_pred=lp,global_correct=norms[i]==norms[gp],
                local_exact=abs(ex['start']-examples[lp]['start'])<1e-3,local_multipos=norms[i]==norms[lp],
                nonnull=not _ar_ns['is_null_description'](ex['content'])))
    elif _ar_path.name.startswith('m2_probe_'):
        y=obj['y_heldout'];p=obj['predictions']
        _ar_probes.append(dict(file=_ar_path.name,n=len(y),correct=int((p==y).sum()),
            per_class={str(k):dict(n=int((y==k).sum()),correct=int(((y==k)&(p==k)).sum())) for k in range(4)}))
_ar_audit=dict(server=PC_SERVER,raw_files=len(_ar_manifest),retrieval_queries=len(_ar_queries),
    retrieval_metrics_recomputed_exact=True,files=_ar_manifest,probes=_ar_probes)
(_ar_folder/'archive_audit.json').write_text(json.dumps(_ar_audit,indent=2),encoding='utf-8')
(_ar_folder/'retrieval_per_query.json').write_text(json.dumps(_ar_queries),encoding='utf-8')
_ar_tar=_ar_folder/'raw_artifacts.tar.gz'
with tarfile.open(_ar_tar,'w:gz') as archive:
    for path in sorted(_ar_dir.glob('*')):
        if path.is_file():archive.add(path,arcname='artifacts/'+path.name)
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v2/raw_artifacts.tar.gz',bytes=_ar_tar.stat().st_size,
    sha256=hashlib.sha256(_ar_tar.read_bytes()).hexdigest(),raw_files=len(_ar_manifest),retrieval_queries=len(_ar_queries),probes=_ar_probes)
(_ar_folder/'archive_index.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
print('PCV2_ARCHIVED',json.dumps(RESULT),flush=True)
