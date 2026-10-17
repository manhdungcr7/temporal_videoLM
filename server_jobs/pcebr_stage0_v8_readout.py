"""Replay saved contrasts, restrict intervention to final assistant token; fixed40."""
import ast,gc,hashlib,json,time,traceback
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling

BASE=OUT/'pcebr_stage0_v8_layerlocal'
D=OUT/'pcebr_stage0_v8_readout';D.mkdir(exist_ok=True)
if (D/'error.json').exists() and not (D/'initial_replay_error.json').exists():
    (D/'initial_replay_error.json').write_bytes((D/'error.json').read_bytes())
source=(BASE/'effective_source.py').read_text()
tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_stage0_m3_bivector_vo_tangent_routing')
route=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='run_caor_routed')
previous=json.loads((D/'samples.json').read_text()) if (D/'samples.json').exists() else []
rows=list(previous);completed={r['idx'] for r in rows}
prior=json.loads((BASE/'completion.json').read_text())
assert prior['n_samples']==20
versions={n:p._version for n,p in MODEL_.named_parameters()}
route_source=ast.unparse(route)+'\n';(D/'route_source.py').write_text(route_source)
try:
    for old in prior['samples']:
        idx=old['idx']
        if idx in completed:continue
        t0=time.time();model_llm=MODEL_
        payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
        raw=torch.load(BASE/'artifacts'/f'{idx}.pt',map_location='cpu',weights_only=False)
        device=MODEL_.get_input_embeddings().weight.device
        kw={k:v.to(device) for k,v in payload['decoder'].items()}
        last_m=int(payload['last_media']);prefix_len=last_m
        v_indices=torch.nonzero(payload['is_v'],as_tuple=False).squeeze(-1).long().cpu()
        a_indices=torch.nonzero(payload['is_a'],as_tuple=False).squeeze(-1).long().cpu()
        all_tail_rows_t=torch.arange(last_m+1,payload['tokens'],dtype=torch.long)
        last_only=torch.tensor([payload['tokens']-1],dtype=torch.long)
        mid_layers=(14,16,18,20,22,24)
        orig_impl=model_llm.model.layers[0].self_attn.config._attn_implementation
        orig_kernel=qwen_omni_modeling.ALL_ATTENTION_FUNCTIONS[orig_impl]
        labels=[PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD'][:len(payload['native_logits'])]
        with torch.inference_mode():
            out=model_llm.model(**kw,use_cache=True,return_dict=True)
            logits=model_llm.lm_head(out.last_hidden_state[:,-1:,:].to(model_llm.lm_head.weight.device))[0,-1].float().cpu()
            nat=F.softmax(logits[labels],dim=-1)
            legacy=out.past_key_values.to_legacy_cache() if hasattr(out.past_key_values,'to_legacy_cache') else out.past_key_values
            prefix_bank=tuple((k[:,:,:prefix_len].detach(),v[:,:,:prefix_len].detach()) for k,v in legacy)
            del out,legacy
        scope=dict(globals());exec(compile(ast.Module(body=[route],type_ignores=[]),'readout_same_route','exec'),scope)
        # Match the evaluated source exactly: profile gamma is rounded to six
        # decimals by aud_info; raw gamma is the pre-rounding extractor output.
        run=scope['run_caor_routed'];prof=raw['profiles']['supp'];scale=.042*prof['gamma_gate']
        def score(direction,rows_t):
            lg=run(mid_layers,prof,eta_scale=0.,clc_vec_1d=direction,clc_scale=scale,ct_scale=0.,target_rows=rows_t,use_prefix_cache=old['cache_used'])
            probs=F.softmax(lg[labels],dim=-1)
            return dict(pred='ABCD'[int(probs.argmax())],probabilities=probs.tolist())
        conditions=dict(native=dict(pred='ABCD'[int(nat.argmax())],probabilities=nat.tolist()))
        conditions['fulltail_shared22_mixed_replay']=score(raw['directions']['shared22_mixed_replay'],all_tail_rows_t)
        replay_errors={}
        for new_tag,old_tag in (('native','native'),('fulltail_shared22_mixed_replay','shared22_mixed_replay')):
            err=float((torch.tensor(conditions[new_tag]['probabilities'])-torch.tensor(old['conditions'][old_tag]['probabilities'])).abs().max())
            replay_errors[new_tag]=err;assert err<=1e-6,('replay',idx,new_tag,err)
        for tag,direction in raw['directions'].items():conditions['last_'+tag]=score(direction,last_only)
        record=dict(idx=idx,task=old['task'],gold=old['gold'],gamma=prof['gamma_gate'],conditions=conditions,source_artifact_sha256=old['artifact_sha256'],
            input_cache_sha256=old['input_cache_sha256'],prior_replay_errors=replay_errors,n_fulltail_positions=len(all_tail_rows_t),n_last_positions=1,
            cache_used=old['cache_used'],elapsed_sec=time.time()-t0)
        rows.append(record)
        tmp=D/'samples.tmp';tmp.write_text(json.dumps(rows,indent=2));tmp.replace(D/'samples.json')
        print('READOUT_SAMPLE',json.dumps(dict(idx=idx,gold=old['gold'],preds={k:v['pred'] for k,v in conditions.items()},elapsed_sec=record['elapsed_sec'])),flush=True)
        del raw,payload,scope,prefix_bank
        gc.collect();torch.cuda.empty_cache()
    assert len(rows)==20
    result=dict(server=PC_SERVER,n_samples=20,samples=rows,backbone_versions_unchanged=all(p._version==versions[n] for n,p in MODEL_.named_parameters()),
        parent_source_sha256=hashlib.sha256(source.encode()).hexdigest(),route_source_sha256=hashlib.sha256(route_source.encode()).hexdigest())
    assert result['backbone_versions_unchanged']
    (D/'completion.json').write_text(json.dumps(result,indent=2))
    RESULT=dict(n=20,scores={k:sum(r['conditions'][k]['pred']==r['gold'] for r in rows) for k in rows[0]['conditions']})
    print('READOUT_DONE',json.dumps(RESULT),flush=True)
except Exception:
    (D/'error.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));raise
