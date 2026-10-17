"""Save raw tensors and independently recompute reported counts."""
import hashlib
import json
import tarfile
import torch

v5_files=[]; v5_probes=[]; v5_nquery=0
for path in sorted((V5_DIR/'artifacts').glob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    v5_files.append(dict(name=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if '_retrieval_' in path.name:
        assert v5_orig_eval(obj['sim'],obj['examples'])==obj['metrics'],path.name
        v5_nquery+=len(obj['examples'])
    if path.name.startswith('m2_probe_'):
        y=obj['y_he'];p=obj['preds']; mask=y<3
        byclass={str(c):dict(correct=int(((y==c)&(p==c)).sum()),total=int((y==c).sum())) for c in range(4)}
        meta=obj['heldout_metadata'];fm={};rm={}
        for i,e in enumerate(meta):
            if e['pair_key'] is not None:
                key,side=e['pair_key'];(fm if side=='fwd' else rm)[key]=(e['label'],int(obj['logits_he'][i,:3].argmax()))
        paired=set(fm)&set(rm)
        hits=sum(fm[k][0]==fm[k][1] and rm[k][0]==rm[k][1] for k in paired)
        v5_probes.append(dict(file=path.name,correct4=int((p==y).sum()),total=len(y),correct3=int((obj['preds_3way']==y[mask]).sum()),total3=int(mask.sum()),by_class=byclass,paired_correct=hits,paired_total=len(paired),pred_counts=torch.bincount(p,minlength=4).tolist()))
v5_audit=dict(server=PC_SERVER,files=v5_files,raw_files=len(v5_files),retrieval_queries=v5_nquery,retrieval_metrics_recomputed_exact=True,probes=v5_probes)
v5_write('archive_audit.json',v5_audit)
archive=V5_DIR/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted((V5_DIR/'artifacts').glob('*')):
        if path.is_file():tar.add(path,arcname='artifacts/'+path.name)
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v5/raw_artifacts.tar.gz',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),raw_files=len(v5_files),retrieval_queries=v5_nquery)
v5_write('archive_index.json',RESULT)
print('PCV5_ARCHIVE',json.dumps(RESULT),flush=True)
