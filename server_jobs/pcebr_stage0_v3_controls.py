"""Read-only numerical/data-flow audits, separate from v3 gate metrics."""
import ast
import collections
import json
import torch
import torch.nn.functional as F

_c3_folder=OUT/'pcebr_stage0_v3';_c3_ns=_v3_ns
_c3_iv=torch.tensor([[2.*i,2.*i+2.] for i in range(5)])
_c3_zero=torch.zeros(5);_c3_tiny=_c3_zero.clone();_c3_tiny[0]=1e-4
_c3_tinylast=_c3_zero.clone();_c3_tinylast[-1]=1e-4
_c3_f=_c3_ns['compute_positive_debiased_relational_features']
_c3_flat=_c3_f(_c3_zero,_c3_zero,_c3_iv,_c3_iv)
_c3_peak=_c3_f(_c3_tiny,_c3_tiny,_c3_iv,_c3_iv)
_c3_order=_c3_f(_c3_tiny,_c3_tinylast,_c3_iv,_c3_iv)
_c3_output=dict(near_uniform_audit=dict(logit_peak=1e-4,uniform_phi=_c3_flat[0].tolist(),uniform_null=_c3_flat[1].tolist(),
    tiny_peak_same_slot_phi=_c3_peak[0].tolist(),tiny_peak_opposite_slots_phi=_c3_order[0].tolist(),tiny_peak_null_features=_c3_peak[1].tolist()))
_c3_bundle=_v3_bundles['avbind_pooled_pairs']
_c3_foil=[]
for split,xkey,tkey in [('train','train_ex','tr_txt'),('heldout','held_ex','he_txt')]:
    examples=_c3_bundle[xkey];zt=F.normalize(_c3_bundle[tkey].detach().cpu().float(),dim=-1)
    norms=[x['content'].strip().lower() for x in examples]
    for rec in sorted({x['recording_id'] for x in examples}):
        own=[i for i,x in enumerate(examples) if x['recording_id']==rec]
        other=[i for i,x in enumerate(examples) if x['recording_id']!=rec and not _c3_ns['is_null_description'](x['content'])]
        distinct=[i for i in other if norms[i] not in {norms[j] for j in own}]
        strict=[i for i in distinct if float((zt[i:i+1]@zt[own].T).max())<.55]
        _c3_foil.append(dict(split=split,recording=rec,distinct_captions=len(distinct),strict_cosine_foils=len(strict),fallback_used=not strict and bool(distinct)))
_c3_output['foil_selection']=_c3_foil
_c3_output['clock_ranges']=[]
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text(encoding='utf-8'))
for row in manifest[f'server{PC_SERVER}'][:4]:
    p=torch.load(OUT/'argon_latent40/decoder_cache'/f"{row['idx']}_rot0.pt",map_location='cpu',weights_only=True)
    clock=dict(idx=row['idx'],duration=row['duration'],decoded_audio_duration=row['decoded_audio_duration'])
    for source,mask in [('video',p['is_v']),('audio',p['is_a'])]:
        t=p['sec'][mask]
        b=torch.floor(t/2).long()
        clock[source]=dict(min_seconds=float(t.min()),max_seconds=float(t.max()),buckets=sorted(set(b.tolist())))
        actual_pos=p['decoder']['position_ids'][0,0].float()
        actual_base=actual_pos[p['is_a']].min()
        actual_seconds=(actual_pos-actual_base)/25.
        clock[source].update(captured_position_min_seconds=float(actual_seconds[mask].min()),
            captured_position_max_seconds=float(actual_seconds[mask].max()),
            helper_vs_captured_seconds_max_error=float((t-actual_seconds[mask]).abs().max()))
    _c3_output['clock_ranges'].append(clock)
tree=ast.parse(_v3_source.read_text(encoding='utf-8'))
builder=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_zero_shortcut_episodes')
_c3_output['sim_mat_load_lines']=[n.lineno for n in ast.walk(builder) if isinstance(n,ast.Name) and n.id=='sim_mat' and isinstance(n.ctx,ast.Load)]
_c3_output['hook_count_at_end']=sum(len(layer._forward_hooks) for layer in MODEL_.model.layers)
(_c3_folder/'controls.json').write_text(json.dumps(_c3_output,indent=2),encoding='utf-8')
RESULT=_c3_output
print('PCV3_CONTROLS',json.dumps(RESULT),flush=True)
