"""Validate per-layer evidence and package the completed micro8 fix run."""
import hashlib
import json
import tarfile
import torch

d=OUT/'pcebr_v5_layerfix_micro8'
rows=json.loads((d/'results.json').read_text(encoding='utf-8'));assert len(rows)==4
checks=[]
for row in rows:
    raw=torch.load(d/'states'/f"{row['idx']}.pt",map_location='cpu',weights_only=True)
    assert torch.equal(raw['F1'][18]['fwd'],raw['original']['fwd'])
    assert row['native_replay_error']==0 and row['original_v5_probability_replay_error']<1e-7
    assert row['conditions']['zero_scale_replay']['logits']==row['conditions']['native']['logits']
    assert not set(raw['hook_rows'])&set(raw['option_rows'])
    for value in row['conditions'].values():
        logits=torch.tensor(value['logits']);prob=torch.softmax(logits,dim=-1)
        assert prob.tolist()==value['probabilities']
        assert 'ABCD'[int(prob.argmax())]==value['prediction']
        assert value['correct']==(value['prediction']==row['gold'])
    layerchecks={}
    for li,bank in raw['F2'].items():
        # Recompute every target layer's VO from the saved corresponding input bank.
        layer=MODEL_.model.layers[li]
        for name in ('V','A'):
            rebuilt=LF_NS['project_to_layer_vo_subspace'](layer,raw['raw_banks'][li][name])
            assert torch.equal(rebuilt,bank[name+'_vo'])
        b=lf_build(bank['V_vo'],bank['A_vo'],raw['original_banks']['lv'],raw['original_banks']['la'],raw['original_banks']['intervals'])
        assert torch.equal(b['fwd'],bank['fwd']) and torch.equal(b['rev'],bank['rev'])
        layerchecks[str(li)]=dict(own_vo_recomputed_exact=True,vector_rebuilt_exact=True,input_shape=raw['pre_input_audit'][li]['shape'],cos_original=float(bank['fwd']@raw['original']['fwd']))
    checks.append(dict(idx=row['idx'],layerchecks=layerchecks))
weights={}
for li in LF_ALL:
    layer=MODEL_.model.layers[li]
    weights[str(li)]={name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name,value in [('input_layernorm',layer.input_layernorm.weight),('v_proj',layer.self_attn.v_proj.weight),('o_proj',layer.self_attn.o_proj.weight)]}
hook=json.loads((d/'hook_audit.json').read_text(encoding='utf-8'))
assert hook['forward_hooks']==0 and hook['pre_hooks']==0 and hook['cuda_ok']
files=[]
for p in sorted(d.rglob('*')):
    if not p.is_file() or p.name in ('raw_artifacts.tar.gz','archive_audit.json','archive_index.json'):continue
    files.append(dict(name=str(p.relative_to(d)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
audit=dict(server=LF_SERVER,checks=checks,model=MODEL_.config._name_or_path,torch=torch.__version__,weights_sha256=weights,files=files,hooks=hook)
(d/'archive_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
archive=d/'raw_artifacts.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for item in files:tar.add(d/item['name'],arcname=item['name'])
RESULT=dict(server=LF_SERVER,file='pcebr_v5_layerfix_micro8/raw_artifacts.tar.gz',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),samples=len(rows))
(d/'archive_index.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8');print('LAYERFIX_ARCHIVE',json.dumps(RESULT),flush=True)
