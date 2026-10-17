"""Whole-word labels for real BPE token spans; no new stopwords or task switches."""
import re

def word_rows(input_ids_1d, stem_rows, tok):
    ids=[input_ids_1d[r] for r in stem_rows]
    # The synthetic unit tokenizer exposes word IDs, not a fast-tokenizer API.
    if not callable(tok):
        return [(r,tok.decode([i],skip_special_tokens=True).strip().lower(),
                 re.findall(r'[a-z0-9]+',tok.decode([i],skip_special_tokens=True).lower())) for r,i in zip(stem_rows,ids)]
    text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    if list(enc['input_ids']) != ids:
        raise ValueError('Cannot safely assign whole-word labels: tokenizer roundtrip mismatch')
    words=[(m.start(),m.end(),m.group().lower()) for m in re.finditer(r'[a-zA-Z0-9]+',text)]
    specials=set(tok.all_special_ids)
    rows=[]
    for r,i,(a,b) in zip(stem_rows,ids,enc['offset_mapping']):
        ws=[] if i in specials else [w for x,y,w in words if x<b and y>a]
        rows.append((r,text[a:b].strip().lower(),ws))
    return rows

def patch_token_interface(source):
    old="""    row_words = []
    for r in stem_rows:
        t_str = tok.decode([input_ids_1d[r]], skip_special_tokens=True).strip().lower() if tok is not None else ''
        words = re.findall(r'[a-z0-9]+', t_str)
        row_words.append((r, t_str, words))
"""
    new="""    row_words = V8_WORD_ROWS(input_ids_1d, stem_rows, tok)
    for r, t_str, words in row_words:
"""
    assert source.count(old)==1
    source=source.replace(old,new)
    # A temporal connector outranks a coordinator inside an event noun phrase.
    old='''    split_pos = None
    connector_found = False
    for idx_r, (r, t_str, words) in enumerate(row_words):'''
    new='''    split_pos = next((i for i,(_,_,ws) in enumerate(row_words) if 2<=i<=len(row_words)-2 and any(w in ('relative','compared') for w in ws)), None)
    connector_found = split_pos is not None
    for idx_r, (r, t_str, words) in ([] if split_pos is not None else enumerate(row_words)):'''
    assert source.count(old)==1
    source=source.replace(old,new).replace("('when', 'and', 'versus', 'vs')","('when', 'versus', 'vs')")
    # The original zero-gate bypass silently ignored use_prefix_cache and reran
    # the entire media prefix at every rollout token. Honor the existing flag.
    old="""                with torch.inference_mode():
                    o = model_llm.model(**use_kw, use_cache=False, return_dict=True)
                    return model_llm.lm_head(
"""
    new="""                with torch.inference_mode():
                    if use_prefix_cache:
                        L_bypass = use_kw['inputs_embeds'].shape[1]
                        bank_bypass = tuple((k.to(layer.self_attn.q_proj.weight.device),v.to(layer.self_attn.q_proj.weight.device)) for layer,(k,v) in zip(model_llm.model.layers,prefix_bank))
                        cache_bypass = DynamicCache(ddp_cache_data=bank_bypass,config=model_llm.model.config)
                        o = model_llm.model(inputs_embeds=use_kw['inputs_embeds'][:,prefix_len:,:],position_ids=use_kw['position_ids'][:,:,prefix_len:],attention_mask=torch.ones((1,L_bypass),dtype=torch.long,device=device),cache_position=torch.arange(prefix_len,L_bypass,device=device),past_key_values=cache_bypass,use_cache=True,return_dict=True)
                    else:
                        o = model_llm.model(**use_kw, use_cache=False, return_dict=True)
                    return model_llm.lm_head(
"""
    assert source.count(old)==1
    source=source.replace(old,new)
    return source
