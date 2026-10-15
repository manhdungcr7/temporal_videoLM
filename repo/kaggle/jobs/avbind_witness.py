"""R15 matched readout ablation, not a novelty or generalization claim.

Generate an option-blind, fallible evidence draft, then re-read all original
media to answer. Compare native AV with the existing guided record layout.
No task names, label semantics, modality-order planner or reference answers
enter either inference path. Keep the draft separate from measured evidence.
"""
import contextlib
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch


WITNESS_INSTRUCTION = (
    '\nBefore answering, briefly list the directly observed facts relevant to this '
    'question. Distinguish what is visible from what is heard. Include approximate '
    'recording times when relevant; mark uncertain or missing observations. '
    'Do not guess an answer. Use any evidence order appropriate to the question.'
)


def witness_identity(video, question, guided):
    return dict(media_sha256=hashlib.sha256(Path(video).read_bytes()).hexdigest(),
        question_sha256=hashlib.sha256(question.encode()).hexdigest(), guided=bool(guided),
        model_revision=GUIDE_METADATA['model_revision'],
        alignment_sha256=GUIDE_METADATA['source_sha256'] if guided else None,
        source_sha256=WITNESS_SOURCE_SHA, max_new_tokens=80, options_used=False)


def witness_tail_text(question, options=None, perm=None, draft=None):
    text=('A model-generated draft follows. It may be wrong; verify it against the '
          'original recording rather than treating it as an observation.\n'+draft+'\n' if draft else '')
    text+='[USER QUESTION]\n'+question
    if options is not None:
        text+=' '+' '.join(f'({letter}) {options[index]}' for letter,index in zip('ABCD',perm))
    return text


def witness_inputs(media, question, guided, nodes=None, raw=None, query=None,
                   offsets=None, options=None, perm=None, draft=None, drafting=False):
    effective=question+WITNESS_INSTRUCTION if drafting else question
    if guided:
        inp,segments,audit=context_clock_inputs(media,effective,options,perm,False,auxiliary_text=draft)
        provider,trace=guided_tail_provider(inp,segments,audit,question,query,offsets,nodes,raw)
        ctx=timeline_isolation(segments,audit['prefix_end'],
            decoder_modules=[layer.self_attn for layer in MODEL_.model.layers],tail_bias=provider)
        return inp,ctx,dict(layout=audit,guidance=trace)
    text=witness_tail_text(effective,options,perm,draft)
    inp,_=qwen_inputs(PROC,text,media,'av');inp=move_inputs(inp,MODEL_)
    positions,_=MODEL_.get_rope_index(inp['input_ids'],None,inp.get('video_grid_thw'),
        inp['attention_mask'],True,inp['feature_attention_mask'].sum(-1),inp['video_second_per_grid'])
    inp['position_ids']=positions
    return inp,contextlib.nullcontext(),dict(tokens=inp['input_ids'].shape[1],native_original_av=True)


def witness_decode(inp,ctx):
    length=inp['input_ids'].shape[1];mask=inp['attention_mask'];tokens=[]
    stops={TOK.eos_token_id,TOK.convert_tokens_to_ids('<|im_end|>')}
    head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
    try:
        with ctx,torch.no_grad():
            output=MODEL_(**inp,use_audio_in_video=True,use_cache=True)
            cache=output.past_key_values;position=inp['position_ids'][:,:,-1:]+1
            for step in range(80):
                token=int(output.logits[0,-1].argmax())
                if token in stops:break
                tokens.append(token)
                if step==79:break
                mask=torch.cat((mask,torch.ones_like(mask[:,-1:])),dim=1)
                output=MODEL_(input_ids=torch.tensor([[token]],device=inp['input_ids'].device),
                    attention_mask=mask,position_ids=position,past_key_values=cache,
                    cache_position=torch.tensor([length+step],device=inp['input_ids'].device),
                    use_audio_in_video=True,use_cache=True)
                cache=output.past_key_values;position=position+1
    finally:head.remove()
    return TOK.decode(tokens,skip_special_tokens=True).strip(),dict(tokens=len(tokens),truncated=len(tokens)==80)


def avbind_witness_answer(video,question,answer_options=None,guided=False,
                          slot_cache=None,witness_cache=None):
    started=time.time();identity=witness_identity(video,question,guided)
    media=prepare_video(video);kwargs={}
    if guided:
        nodes,raw=guided_raw_bank(video,slot_cache);query,offsets=question_content_tokens(question)
        kwargs=dict(nodes=nodes,raw=raw,query=query,offsets=offsets)
    saved=None;path=Path(witness_cache) if witness_cache else None
    if path and path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['identity']==identity,'Witness cache provenance mismatch'
    if saved is None:
        inp,ctx,audit=witness_inputs(media,question,guided,drafting=True,**kwargs)
        draft,decode_audit=witness_decode(inp,ctx)
        saved=dict(identity=identity,draft=draft,decode=decode_audit,draft_audit=audit)
        if path:
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(saved),encoding='utf-8');temporary.replace(path)
        del inp;torch.cuda.empty_cache()
    if answer_options is None:
        inp,ctx,audit=witness_inputs(media,question,guided,draft=saved['draft'],**kwargs)
        answer,decode_audit=witness_decode(inp,ctx);scores=prediction=None;audits=[audit]
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]];audits=[]
        for perm in rotations(len(answer_options)):
            inp,ctx,audit=witness_inputs(media,question,guided,options=answer_options,perm=perm,draft=saved['draft'],**kwargs)
            head=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
            try:
                with ctx,torch.no_grad():logits=MODEL_(**inp,use_audio_in_video=True,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
            audits.append(audit)
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del media;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,draft=saved,answer_audits=audits,
        question_type_routing=False,raw_media_retained=True,guided=guided,latency_sec=round(time.time()-started,2),
        warning='Fallible model draft, NOT observed ground truth; readout ablation only')
