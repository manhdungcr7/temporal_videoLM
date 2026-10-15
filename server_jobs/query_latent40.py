"""Question-conditioned continuous feedback, with question scaffold control.

No speech-window selection, answer-word clock, task router or modality handoff.
Weighted vocabulary embeddings are an existing primitive, not a novelty claim.
"""
import ast,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

QL_SERVER=globals().get('QL_SERVER',1)
SL_SERVER=QL_SERVER
_source=(OUT/'uploads/symmetric_latent40.py').read_text().split('RESULT=sl_run()',1)[0]
exec(compile(_source,'same_layer_definitions_only','exec'),globals())
_prototype_source=(OUT/'uploads/ccld_proposal_original.py').read_text()
_tree=ast.parse(_prototype_source)
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in
        ('hidden_to_manifold_thought_batch','extract_continuous_thoughts')]
assert len(_nodes)==2
exec(compile(ast.Module(body=_nodes,type_ignores=[]),'archived_prototype_math','exec'),globals())
QL_PROTOTYPE_SHA=hashlib.sha256(_prototype_source.encode()).hexdigest()
QL_DIR=OUT/'query_latent40';QL_DIR.mkdir(exist_ok=True)
QL_SHA=hashlib.sha256((OUT/'uploads/query_latent40.py').read_bytes()).hexdigest()
QL_METHODS=['Q0_native','Q1_global_prototype','Q2_question_prototype',
            'Q3_question_evidence_contrast','Q4_contrast_plus_joint','Q5_shuffled_thought']
QL_VALIDATED=False

def ql_forward(payload,bank,thought,selected,delta=None):
    hooks=[];last=-1 if bank is not None else payload['last_media']
    def inject(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;h=h.clone();q=h[0,last+1:].float()
        mask=selected.to(h.device);direction=thought.to(h.device)
        if bool(mask.any()):
            q[mask]=q[mask]+.20*q[mask].norm(dim=-1,keepdim=True).mean().clamp_min(1e-5)*direction
        h[0,last+1:]=q.to(h.dtype)
        return (h,)+out[1:] if isinstance(out,tuple) else h
    try:
        if thought is not None:
            for li in (0,1,2):hooks.append(MODEL_.model.layers[li].register_forward_hook(inject))
        return sl_forward(payload,prefix_bank=bank,deltas=delta)
    finally:
        for handle in hooks:handle.remove()

def ql_rotation(payload,k,stem,duration):
    global QL_VALIDATED
    full={}
    def capture(module,args,out):
        h=out[0] if isinstance(out,tuple) else out;full[22]=h[0].detach().float().cpu()
    handle=MODEL_.model.layers[22].register_forward_hook(capture)
    try:native,joint=sl_forward(payload,capture=True,build_bank=True)
    finally:handle.remove()
    bank=joint.pop('_prefix_bank');selected=sl_question_rows(payload,stem)
    cached,_=sl_forward(payload,prefix_bank=bank)
    prefix_error=float((native[SL_LABELS[:k]]-cached[SL_LABELS[:k]]).abs().max())
    use_bank=bank if prefix_error<=.03 else None
    replay=float((native[SL_LABELS[:k]]-payload['native_logits']).abs().max())
    assert replay<=.12,replay
    # Global prototype is independent of this dummy full-clip window; discard M9.
    _,global_thought=extract_continuous_thoughts(full[22],payload['sec'],payload['is_v'],
        payload['is_a'],payload['last_media'],(0.,duration))
    del full
    branches={}
    for name,absent in [('visual',('audio',)),('audio',('visual',)),('neither',('visual','audio'))]:
        mask=sl_mask(payload,absent);_,branches[name]=sl_forward(payload,mask=mask,capture=True)
        del mask;torch.cuda.empty_cache()
    pooled=torch.stack([joint[22][selected].mean(0),branches['neither'][22][selected].mean(0)])
    z=hidden_to_manifold_thought_batch(pooled,top_m=48,temp=.65)
    thought=F.normalize(z[0]-z[1],dim=-1)
    interaction=joint[22]-branches['visual'][22]-branches['audio'][22]+branches['neither'][22]
    projected=sl_project(interaction,branches['visual'][22]-branches['neither'][22],
                         branches['audio'][22]-branches['neither'][22])*selected[:,None]
    generator=torch.Generator().manual_seed(42)
    shuffled=thought[torch.randperm(len(thought),generator=generator)]
    conditions=[('Q1_global_prototype',global_thought,None),('Q2_question_prototype',z[0],None),
        ('Q3_question_evidence_contrast',thought,None),('Q4_contrast_plus_joint',thought,{22:projected}),
        ('Q5_shuffled_thought',shuffled,None)]
    logits={'Q0_native':native};validation={};fallback=[]
    for method,vector,delta in conditions:
        logits[method],_=ql_forward(payload,use_bank,vector,selected,delta)
        if not QL_VALIDATED and use_bank is not None:
            reference,_=ql_forward(payload,None,vector,selected,delta)
            error=float((reference[SL_LABELS[:k]]-logits[method][SL_LABELS[:k]]).abs().max())
            validation[method]=error
            if error>.03 or reference[SL_LABELS[:k]].argmax()!=logits[method][SL_LABELS[:k]].argmax():
                logits[method]=reference;fallback.append(method)
    if use_bank is not None:QL_VALIDATED=True
    return dict(scores={m:torch.log_softmax(lg[SL_LABELS[:k]],dim=-1).numpy().tolist() for m,lg in logits.items()},
        replay_error=replay,prefix_replay_error=prefix_error,prefix_used=use_bank is not None,
        intervention_cache_replay=validation,full_intervention_fallback=fallback,
        stem_rows=int(selected.sum()),tail_rows=len(selected),
        prototype_difference_norm=float((z[0]-z[1]).norm()),
        question_global_cosine=float(F.cosine_similarity(z[0],global_thought,dim=0)),
        prototype_source_sha256=QL_PROTOTYPE_SHA)

def ql_run():
    output=QL_DIR/f'results_s{QL_SERVER}.jsonl'
    done={json.loads(x)['idx'] for x in output.read_text().splitlines()} if output.exists() else set()
    for sample in SL_SAMPLES:
        idx=sample['idx']
        if idx in done:continue
        started=time.time();stem,texts=parse_question(sample['question']);k=len(texts)
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        totals={m:np.zeros(k) for m in QL_METHODS};audits=[]
        for rotation,perm in enumerate(rotations(k)):
            path=QL_DIR/f'{idx}_rot{rotation}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                print('QUERY_LATENT_ROT_START',QL_SERVER,idx,rotation,flush=True)
                payload=torch.load(SL_CACHE/f'{idx}_rot{rotation}.pt',map_location='cpu',weights_only=False)
                assert payload['manifest_sha256']==SL_SHA
                result=ql_rotation(payload,k,stem,sample['duration'])
                result.update(idx=idx,rotation=rotation,perm=list(perm),manifest_sha256=SL_SHA,runtime_sha256=QL_SHA)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
                del payload;gc.collect();torch.cuda.empty_cache()
            assert result['manifest_sha256']==SL_SHA and result['runtime_sha256']==QL_SHA
            for m,score in result['scores'].items():
                for slot,opt in enumerate(perm):totals[m][opt]+=score[slot]
            audits.append({k:v for k,v in result.items() if k!='scores'})
        rec=dict(idx=idx,task=sample['task'],gold=sample['gold'],status='ok',manifest_sha256=SL_SHA,
            runtime_sha256=QL_SHA,scores={m:s.tolist() for m,s in totals.items()},
            predictions={m:chr(65+int(s.argmax())) for m,s in totals.items()},
            audit=audits,seconds=round(time.time()-started,2))
        with output.open('a') as f:f.write(json.dumps(rec)+'\n')
        done.add(idx);print('QUERY_LATENT_SAMPLE',len(done),'/',len(SL_SAMPLES),json.dumps(rec),flush=True)
    return dict(server=QL_SERVER,completed=len(done),expected=len(SL_SAMPLES))

RESULT=ql_run()
print('QUERY_LATENT_COMPLETE',json.dumps(RESULT),flush=True)
