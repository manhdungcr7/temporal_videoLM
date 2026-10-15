"""Save raw tensors and independently recompute reported counts."""
import hashlib
import json
import tarfile
import torch

v4_files=[]; v4_probes=[]; v4_nquery=0
for path in sorted((V4_DIR/'artifacts').glob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    v4_files.append(dict(name=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if '_retrieval_' in path.name:
        assert v4_orig_eval(obj['sim'],obj['examples'])==obj['metrics'],path.name
        v4_nquery+=len(obj['examples'])
    if path.name.startswith('m2_probe_'):
        y=obj['y_he'];p=obj['preds']; mask=y<3
        byclass={str(c):dict(correct=int(((y==c)&(p==c)).sum()),total=int((y==c).sum())) for c in range(4)}
        meta=obj['heldout_metadata'];fm={};rm={}
        for i,e in enumerate(meta):
            if e['pair_key'] is not None:
                key,side=e['pair_key'];(fm if side=='fwd' else rm)[key]=(e['label'],int(obj['logits_he'][i,:3].argmax()))
        paired=set(fm)&set(rm)
        hits=sum(fm[k][0]==fm[k][1] and rm[k][0]==rm[k][1] for k in paired)
        v4_probes.append(dict(file=path.name,correct4=int((p==y).sum()),total=len(y),correct3=int((obj['preds_3way']==y[mask]).sum()),total3=int(mask.sum()),by_class=byclass,paired_correct=hits,paired_total=len(paired),pred_counts=torch.bincount(p,minlength=4).tolist()))
v4_audit=dict(server=PC_SERVER,files=v4_files,raw_files=len(v4_files),retrieval_queries=v4_nquery,retrieval_metrics_recomputed_exact=True,probes=v4_probes)
v4_write('archive_audit.json',v4_audit)
archive=V4_DIR/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted((V4_DIR/'artifacts').glob('*')):
        if path.is_file():tar.add(path,arcname='artifacts/'+path.name)
RESULT=dict(server=PC_SERVER,file='pcebr_stage0_v4/raw_artifacts.tar.gz',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),raw_files=len(v4_files),retrieval_queries=v4_nquery)
v4_write('archive_index.json',RESULT)
print('PCV4_ARCHIVE',json.dumps(RESULT),flush=True)
