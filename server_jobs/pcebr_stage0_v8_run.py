"""Observation wrapper for original CLRV v8; preserve main algorithm and archive controls."""
import ast, hashlib, json, time, traceback, gc, re
from pathlib import Path
import numpy as np
import torch

RUN_TAG=globals().get('V8_RUN_TAG','original')
D=OUT/'pcebr_stage0_v8'/RUN_TAG
D.mkdir(parents=True,exist_ok=True)
(D/'artifacts').mkdir(exist_ok=True)
source_name='stage0_pcebr_v8_logicfix.py' if RUN_TAG in ('logicfix','corrected') else 'stage0_pcebr_v8_original.py'
source=(OUT/'uploads'/source_name).read_text(encoding='utf-8')
expected_sha={'original':'fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895',
              'logicfix':'099e804046ac9c311bb260b1874a14b6e67d6948ac511e95befdaa8da5db0806',
              'corrected':'099e804046ac9c311bb260b1874a14b6e67d6948ac511e95befdaa8da5db0806'}[RUN_TAG]
assert hashlib.sha256(source.encode()).hexdigest()==expected_sha
word_rows=None
if RUN_TAG in ('logicfix','corrected'):
    helper_name='pcebr_stage0_v8_token_interface_v2.py' if RUN_TAG=='corrected' else 'pcebr_stage0_v8_token_interface.py'
    helper_text=(OUT/'uploads'/helper_name).read_text(encoding='utf-8')
    helper_ns={}
    exec(compile(helper_text,'v8_word_interface','exec'),helper_ns)
    source=helper_ns['patch_token_interface'](source)
    word_rows=helper_ns['word_rows']
    write_spec=dict(algorithm_source_sha256=expected_sha,token_helper_sha256=hashlib.sha256(helper_text.encode()).hexdigest(),effective_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        token_fix_reason='Original real tokenizer classifies BPE fragments of deictic words as concrete entities; corrected after first original micro outputs showed this interface failure',
        connector_fix='Prioritize relative/compared relation connectors; bare and inside an event is not an event boundary',
        performance_fix='Honor use_prefix_cache on zero-gate bypass; preserve full vs cached parity threshold')
    (D/'runtime_patch_spec.json').write_text(json.dumps(write_spec,indent=2))
    (D/'effective_source.py').write_text(source,encoding='utf-8')

def cpu(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [cpu(v) for v in x]
    return x

def write(name,x):
    (D/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

captures=[]
extraction_audits=[]
def capture_extract(v):
    def simple(k):
        z=v.get(k)
        if torch.is_tensor(z): return cpu(z).tolist() if z.numel()<200 else dict(shape=list(z.shape),min=float(z.min()),max=float(z.max()),norm_mean=float(z.norm(dim=-1).mean()))
        return z
    role=v['role_info']
    ids=v.get('input_ids_1d')
    tok=v.get('tok')
    old_pieces=[]
    if ids and tok:
        for r in v['query_rows']:
            piece=tok.decode([ids[r]],skip_special_tokens=True).strip().lower()
            if any(len(w)>=3 and w not in ns['STOPWORDS'] and w not in ns['DEICTIC_META_WORDS'] for w in re.findall(r'[a-z0-9]+',piece)):
                old_pieces.append(piece)
    top_tokens=[]
    if tok and torch.is_tensor(v.get('logit_gap_all')):
        for rr in v['logit_gap_all'][:2*v['K']].topk(5,dim=-1).indices.tolist():
            top_tokens.append([tok.decode([ii]) for ii in rr])
    extraction_audits.append(dict(role=role,specific_pieces=[tok.decode([ids[r]]) for r in role['specific_rows']] if ids and tok else [],
        original_piecewise_specific_pieces=old_pieces,
        top5_media_manifold_tokens=top_tokens,
        gate=v['excess_gate'],factorial=v['gate_factorial'],stats={k:simple(k) for k in ('mu_vocab','mat_spec','mat_all','scaf_1d','rel_dev_v','rel_dev_a','fg_mod_gate') if k in v}))
    write('extraction_audit.json',extraction_audits)
def capture(v):
    # Native trace and identical follow-up scorer: does CLRV beat ordinary two-turn inference?
    with torch.inference_mode():
        p_native_bridge=v['score_bridge_turn'](v['native_gen_emb'],v['native_gen_pos'],v['opt_block_text'],True)
        p_empty_bridge=v['score_bridge_turn'](v['freeform_embeds'],v['freeform_pos'],v['opt_block_text'],True)
    tok=PROC.tokenizer
    tail_ids=v['payload']['input_ids'][0,v['last_m']+1:].tolist()
    tail_text=tok.decode(tail_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    row=dict(v['rows_out'][-1])
    row.update(mc_pred_native_rollout_bridge='ABCD'[int(p_native_bridge.argmax())],
        mc_pred_empty_rollout_bridge='ABCD'[int(p_empty_bridge.argmax())],
        mc_native_rollout_bridge_correct=bool(int(p_native_bridge.argmax())==ord(row['gold_letter'])-65),
        mc_empty_rollout_bridge_correct=bool(int(p_empty_bridge.argmax())==ord(row['gold_letter'])-65),
        option_permutation_executed=len(v['opt_texts_by_idx'])==v['k_opts'],
        tail_roundtrip=tok(tail_text,add_special_tokens=False)['input_ids']==tail_ids,
        freeform_prompt=tok.decode(v['freeform_ids_1d'],skip_special_tokens=False),
        probabilities={k:cpu(x).tolist() for k,x in v.items() if k.startswith('p_') and torch.is_tensor(x) and x.ndim==1},
        p_native_bridge=p_native_bridge.tolist(),p_empty_bridge=p_empty_bridge.tolist())
    row.update(order_reversed_rollout_text_changed=v['ff_routed_text']!=v['ff_rev_text'],
        bridge_control_kind='order control changes latent routing; generated text need not actually reverse the semantic relation')
    raw={k:cpu(v[k]) for k in ('V18_raw','A18_raw','V18_vo','A18_vo','lg_v','lg_a','iv_t','vec_fwd','vec_rev','phi_fwd','phi_rev','slot_meta','aud_fwd','aud_rev','clc_dir_fwd','clc_dir_rev','h22_supp','h22_foil','h22_rev','dh_ev','dh_ord','prof_full_fwd','prof_foil_suppressed','prof_ord_sign_rev','clock_info','deltas_fwd','deltas_rev')}
    raw.update({k:cpu(x) for k,x in v.items() if k.startswith('p_') and torch.is_tensor(x)})
    raw.update(p_native_bridge=p_native_bridge,p_empty_bridge=p_empty_bridge)
    torch.save(raw,D/'artifacts'/f"m3_{row['idx']}.pt")
    captures.append(row)
    write('samples.json',captures)
    print('V8_SAMPLE_DONE',json.dumps({k:row[k] for k in ('idx','task','gold_letter','mc_pred_native','mc_pred_routed_direct','mc_pred_rollout_bridge_only','mc_pred_routed_fwd','mc_pred_native_rollout_bridge','confidence_gate_gamma_true_query','sample_elapsed_sec')}),flush=True)
    gc.collect();torch.cuda.empty_cache()

tree=ast.parse(source)
tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RESULT' for t in n.targets))]
class Observe(ast.NodeTransformer):
    def visit_Return(self,node):
        if isinstance(node.value,ast.Tuple) and isinstance(node.value.elts[0],ast.Name) and node.value.elts[0].id=='B18_ep2':
            return [ast.parse('V8_CAPTURE_EXTRACT(locals())').body[0],node]
        return node
    def visit_Assign(self,node):
        # Retain native generated embeddings instead of discarding them; no change to generation.
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='generate_freeform_with_profile' and isinstance(node.targets[0],ast.Tuple) and node.targets[0].elts[0].id=='ff_nat_text':
            node.targets[0]=ast.parse('ff_nat_text, native_gen_ids, native_gen_emb, native_gen_pos = None').body[0].targets[0]
        return self.generic_visit(node)
    def visit_For(self,node):
        self.generic_visit(node)
        body=[]
        for st in node.body:
            body.append(st)
            if isinstance(st,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='idx' for t in st.targets):
                body.append(ast.parse("print('V8_SAMPLE_START',idx,flush=True)").body[0])
            if isinstance(st,ast.Expr) and isinstance(st.value,ast.Call) and isinstance(st.value.func,ast.Attribute) and isinstance(st.value.func.value,ast.Name) and st.value.func.value.id=='rows_out' and st.value.func.attr=='append':
                body.append(ast.parse('V8_CAPTURE(locals())').body[0])
        node.body=body
        return node
tree=Observe().visit(tree);ast.fix_missing_locations(tree)
ns=dict(OUT=OUT,MODEL_=MODEL_,PROC=PROC,LAB_SERVER=PC_SERVER,V8_CAPTURE=capture,V8_CAPTURE_EXTRACT=capture_extract,V8_WORD_ROWS=word_rows)
exec(compile(tree,'original_v8_observed','exec'),ns)

# M0's synthetic GPU tensors were indexed into CPU slot-ID buffers. The real path
# already captures these tensors on CPU. Normalize this operational interface only.
original_extract=ns['extract_query_conditioned_salient_slots']
def extract_on_cpu(*args,**kwargs):
    args=list(args)
    for i in (0,1): args[i]=args[i].cpu()
    args[4]={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in args[4].items()}
    for k in ('h22_post_t','custom_q18','custom_q27'):
        if torch.is_tensor(kwargs.get(k)): kwargs[k]=kwargs[k].cpu()
    return original_extract(*args,**kwargs)
ns['extract_query_conditioned_salient_slots']=extract_on_cpu
iw=MODEL_.get_input_embeddings().weight
ow=MODEL_.lm_head.weight
stride=max(1,iw.shape[0]//1024)
ii=iw[::stride].detach().float().cpu()
oo=ow[::stride].detach().float().cpu()
write('embedding_geometry.json',dict(tie_word_embeddings=getattr(MODEL_.config,'tie_word_embeddings',None),same_storage=iw.data_ptr()==ow.data_ptr(),
    mean_input_norm=float(ii.norm(dim=-1).mean()),mean_output_norm=float(oo.norm(dim=-1).mean()),
    mean_same_token_input_output_cosine=float(torch.nn.functional.cosine_similarity(ii,oo,dim=-1).mean())))
del ii,oo
versions={n:p._version for n,p in MODEL_.named_parameters()}
try:
    m0=ns['run_algebraic_unit_checks']('cuda:0')
    write('m0_devicefix.json',m0)
    print('V8_M0_DEVICEFIX_PASS',flush=True)
    t0=time.time()
    result=ns['run_stage0_m3_bivector_vo_tangent_routing']()
    write('main_results.json',result)
    RESULT=dict(status='done',variant=RUN_TAG,n_samples=len(captures),seconds=time.time()-t0,
        backbone_parameters_unchanged=all(p._version==versions[n] for n,p in MODEL_.named_parameters()))
    write('completion.json',RESULT)
    print('V8_COMPLETE',json.dumps(RESULT),flush=True)
except Exception:
    write('error.json',dict(error=traceback.format_exc(),completed=len(captures)))
    raise
