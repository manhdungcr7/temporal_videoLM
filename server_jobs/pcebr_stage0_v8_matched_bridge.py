"""Matched option permutation: identical retokenized trace and exact choice formatting."""
import ast,json,re,torch
from transformers import DynamicCache
D=OUT/'pcebr_stage0_v8/corrected'
src=(OUT/'uploads/stage0_pcebr_v8_corrected_standalone.py').read_text()
t=ast.parse(src);t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='RESULT' for x in n.targets))]
ns=dict(OUT=OUT);exec(compile(t,'bridge_layout_helpers','exec'),ns)
tok=PROC.tokenizer;emb=MODEL_.get_input_embeddings();dev=emb.weight.device
rows=[]
samples=json.loads((D/'samples.json').read_text())
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
lookup={s['idx']:s for s in manifest['selected']}
for row in samples:
    idx=row['idx'];sample=lookup[idx]
    payload=torch.load(OUT/'argon_latent40/decoder_cache'/f'{idx}_rot0.pt',map_location='cpu',weights_only=False)
    kw={k:v.to(dev) for k,v in payload['decoder'].items()}
    last=int(payload['last_media']);prefix=last
    stem=sample['question'].split('(A)')[0].split('A.')[0].strip()
    stem_rows,_,assistant_rows,_=ns['find_option_free_and_option_spans'](payload,stem,tok)
    keep=list(range(last+1))+stem_rows+assistant_rows
    e=kw['inputs_embeds'][:,keep,:].clone();pos=kw['position_ids'][:,:,keep].clone()
    end=last+len(stem_rows)
    gap=(kw['position_ids'][:,:,assistant_rows[0]:assistant_rows[0]+1]-pos[:,:,end:end+1])-1
    pos[:,:,end+1:]-=gap
    trace_ids=tok.encode(row['freeform_32tok_routed'],add_special_tokens=False)
    assert tok.decode(trace_ids,skip_special_tokens=True).strip()==row['freeform_32tok_routed']
    with torch.inference_mode():
        e=torch.cat([e,emb(torch.tensor([trace_ids],device=dev)).to(e.dtype)],dim=1)
        tracepos=pos[:,:,-1:]+torch.arange(1,len(trace_ids)+1,device=dev).view(1,1,-1)
        pos=torch.cat([pos,tracepos],dim=-1)
        nat=MODEL_.model(**kw,use_cache=True,return_dict=True)
        kv=nat.past_key_values.to_legacy_cache() if hasattr(nat.past_key_values,'to_legacy_cache') else nat.past_key_values
        bank=tuple((k[:,:,:prefix].detach(),v[:,:,:prefix].detach()) for k,v in kv)
        del nat,kv
    block=sample['question'][len(stem):].strip()
    markers=list(re.finditer(r'\(([ABCD])\)',block))
    pieces=[]
    for j,m in enumerate(markers):
        body=block[m.end():markers[j+1].start() if j+1<len(markers) else len(block)]
        x=re.fullmatch(r'(\s*)(.*?)(\s*)',body,flags=re.S)
        pieces.append((m.group(),x.group(1),x.group(2),x.group(3)))
    def render(perm):
        return block[:markers[0].start()]+''.join(label+lead+pieces[perm[j]][2]+trail for j,(label,lead,_,trail) in enumerate(pieces))
    assert render(list(range(len(pieces))))==block
    cyc=[(j+1)%len(pieces) for j in range(len(pieces))]
    labels=[tok.convert_tokens_to_ids(c) for c in 'ABCD'][:len(pieces)]
    def score(options):
        suffix='<|im_end|>\n<|im_start|>user\nUse the original recording and the analysis above to select the best answer. '+options+'\nAnswer with only the letter of the best answer.<|im_end|>\n<|im_start|>assistant\n'
        ids=tok.encode(suffix,add_special_tokens=False)
        with torch.inference_mode():
            ee=torch.cat([e,emb(torch.tensor([ids],device=dev)).to(e.dtype)],dim=1)
            pp=torch.cat([pos,pos[:,:,-1:]+torch.arange(1,len(ids)+1,device=dev).view(1,1,-1)],dim=-1)
            kk=tuple((k.to(l.self_attn.q_proj.weight.device),v.to(l.self_attn.q_proj.weight.device)) for l,(k,v) in zip(MODEL_.model.layers,bank))
            out=MODEL_.model(inputs_embeds=ee[:,prefix:,:],position_ids=pp[:,:,prefix:],attention_mask=torch.ones((1,ee.shape[1]),dtype=torch.long,device=dev),cache_position=torch.arange(prefix,ee.shape[1],device=dev),past_key_values=DynamicCache(ddp_cache_data=kk,config=MODEL_.model.config),use_cache=True,return_dict=True)
            lg=MODEL_.lm_head(out.last_hidden_state[:,-1:,:].to(MODEL_.lm_head.weight.device))[0,-1].float().cpu()
        return torch.softmax(lg[labels],dim=-1)
    p=score(block);pr=score(render(cyc));unperm=torch.zeros_like(pr)
    for j,k in enumerate(cyc):unperm[k]=pr[j]
    rows.append(dict(idx=idx,gold=row['gold_letter'],forward='ABCD'[int(p.argmax())],permuted_unmapped='ABCD'[int(unperm.argmax())],
        p_forward=p.tolist(),p_permuted_unmapped=unperm.tolist(),tv=float(.5*(p-unperm).abs().sum()),identity_render_exact=True,
        trace_retokenized=True,original_source_bridge_probability_error=float((p-torch.tensor(row['probabilities']['p_bridge'])).abs().max())))
    (D/'bridge_matched_format_control.json').write_text(json.dumps(rows,indent=2))
RESULT=dict(n=len(rows),forward_correct=sum(r['forward']==r['gold'] for r in rows),permuted_correct=sum(r['permuted_unmapped']==r['gold'] for r in rows))
print('V8_MATCHED_BRIDGE',json.dumps(RESULT),flush=True)
