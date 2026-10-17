"""Read-only real-tokenizer audit over the already-open research40 questions."""
import ast,hashlib,json
import torch
D=OUT/'pcebr_stage0_v8'
source=(OUT/'uploads/stage0_pcebr_v8_original.py').read_text(encoding='utf-8')
helpers={};exec(compile((OUT/'uploads/pcebr_stage0_v8_token_interface.py').read_text(),'word_interface','exec'),helpers)
def definitions(s,helper=None):
    t=ast.parse(s);t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='RESULT' for v in n.targets))]
    ns=dict(OUT=D,V8_WORD_ROWS=helper);exec(compile(t,'v8_parser_only','exec'),ns);return ns
old=definitions(source)
new=definitions(helpers['patch_token_interface'](source),helpers['word_rows'])
tok=PROC.tokenizer
manifest=json.loads((OUT/'uploads/research40_manifest.json').read_text())
rows=[]
for sample in manifest['selected']:
    stem=sample['question'].split('(A)')[0].split('A.')[0].strip()
    ids=tok.encode(stem,add_special_tokens=False);rr=list(range(len(ids)))
    rec=dict(idx=sample['idx'],question_stem=stem)
    for tag,ns in [('original',old),('wordfix',new)]:
        role=ns['split_query_into_endpoint_roles'](ids,rr,tok)
        rec[tag]=dict(role1_text=tok.decode([ids[r] for r in role['role1_rows']]),role2_text=tok.decode([ids[r] for r in role['role2_rows']]),
            specific_pieces=[tok.decode([ids[r]]) for r in role['specific_rows']],n_specific=len(role['specific_rows']),
            split_method=role.get('split_method'),is_deictic=role['is_purely_deictic_query'])
    rows.append(rec)
iw=MODEL_.get_input_embeddings().weight;ow=MODEL_.lm_head.weight
stride=max(1,iw.shape[0]//1024)
a=iw[::stride].detach().float().cpu();b=ow[::stride].detach().float().cpu()
geometry=dict(same_storage=iw.data_ptr()==ow.data_ptr(),tie_word_embeddings=getattr(MODEL_.config,'tie_word_embeddings',None),
    input_norm_mean=float(a.norm(dim=-1).mean()),output_norm_mean=float(b.norm(dim=-1).mean()),
    same_token_input_output_cosine_mean=float(torch.nn.functional.cosine_similarity(a,b,dim=-1).mean()))
out=dict(server=PC_SERVER,qa_evaluation=False,n_questions=len(rows),geometry=geometry,questions=rows)
(D/'real_tokenizer_parser_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
RESULT=dict(n_questions=len(rows),geometry=geometry)
print('V8_PARSER_AUDIT',json.dumps(RESULT),flush=True)
