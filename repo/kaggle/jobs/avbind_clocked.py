"""Shared physical occurrence clocks in native multimodal rotary coordinates.

Local raw temporal/spatial structure is preserved by translating each entire
record, not flattening its video grid or replacing raw data with captions.
The first media token at time t has temporal coordinate anchor+25*t for both
sources. Query/answer text follows the maximum record coordinate. No task
classifier, audio-first strategy, content interpretation or answer rule.
"""
import math
import time
import numpy as np
import torch


def canonical_record_positions(original, ids, segments, starts_seconds,
                               media_token_ids, prefix_end, ticks=25):
    if original.shape != (3,1,ids.numel()) or len(segments) != len(starts_seconds):
        raise ValueError('Clock-addressed records and native positions disagree')
    positions = original.clone()
    anchors = []
    preludes = []
    for (start,end), seconds in zip(segments, starts_seconds):
        if not (math.isfinite(seconds) and seconds >= 0 and prefix_end <= start < end <= ids.numel()):
            raise ValueError('Invalid recording-clock record')
        selected = torch.zeros(end-start,dtype=torch.bool,device=ids.device)
        for token_id in media_token_ids:
            selected |= ids[start:end] == token_id
        if not selected.any():
            raise ValueError('Record has no native media tokens')
        first = start + int(torch.where(selected)[0][0])
        anchors.append(first)
        preludes.append(int(original[0,0,first] - original[:,:,start:first+1].min()))
    origin = prefix_end + max(preludes) + 1
    for (start,end), seconds, first in zip(segments, starts_seconds, anchors):
        target = origin + round(seconds*ticks)
        translation = target - int(original[0,0,first])
        positions[:,:,start:end] += translation
    tail = segments[-1][1]
    query_origin = int(positions[:,:,:tail].max()) + 1
    positions[:,:,tail:] = (query_origin + torch.arange(ids.numel()-tail,device=positions.device))[None,None,:]
    return positions, dict(clock_origin=origin,question_origin=query_origin,
                           first_media_coordinates=[int(positions[0,0,i]) for i in anchors],ticks_per_second=ticks)


def clocked_inputs(records,question,options=None,perm=None):
    inp,segments,audit=timeline_inputs(records,question,options,perm)
    lengths=inp['feature_attention_mask'].sum(-1)
    positions,_=MODEL_.get_rope_index(inp['input_ids'],None,inp.get('video_grid_thw'),
        inp['attention_mask'],False,lengths,inp['video_second_per_grid'])
    starts=[r['start'] for r in records for _ in ('video','audio')]
    media_ids={TOK.convert_tokens_to_ids(PROC.video_token),TOK.convert_tokens_to_ids(PROC.audio_token)}
    positions,clock_audit=canonical_record_positions(positions,inp['input_ids'][0],segments,starts,media_ids,audit['prefix_end'])
    inp['position_ids']=positions
    audit.update(clock_audit)
    return inp,segments,audit


def avbind_clocked_answer(video_path,question,answer_options=None):
    started=time.time();records,duration=timeline_records(video_path);audits=[]
    decoder=[layer.self_attn for layer in MODEL_.model.layers]
    if answer_options is None:
        # Explicit cached decode maintains the new clock without relying on a
        # stale model-global rope_delta from an earlier native forward.
        inp,segments,audit=clocked_inputs(records,question)
        length=inp['input_ids'].shape[1];mask=inp['attention_mask'];tokens=[]
        stopping={TOK.eos_token_id,TOK.convert_tokens_to_ids('<|im_end|>')}
        with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder),torch.no_grad():
            output=MODEL_(**inp,use_audio_in_video=False,use_cache=True)
            cache=output.past_key_values;next_position=inp['position_ids'][:,:,-1:]+1
            for step in range(128):
                token=int(output.logits[0,-1].argmax())
                if token in stopping:break
                tokens.append(token)
                mask=torch.cat((mask,torch.ones_like(mask[:,-1:])),dim=1)
                output=MODEL_(input_ids=torch.tensor([[token]],device=inp['input_ids'].device),
                    attention_mask=mask,position_ids=next_position,past_key_values=cache,
                    cache_position=torch.tensor([length+step],device=inp['input_ids'].device),
                    use_audio_in_video=False,use_cache=True)
                cache=output.past_key_values;next_position=next_position+1
        answer=TOK.decode(tokens,skip_special_tokens=True).strip();scores=prediction=None;audits.append(audit)
    else:
        totals=np.zeros(len(answer_options));labels=[TOK.convert_tokens_to_ids(x) for x in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp,segments,audit=clocked_inputs(records,question,answer_options,perm)
            head=MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with timeline_isolation(segments,audit['prefix_end'],decoder_modules=decoder),torch.no_grad():
                    logits=MODEL_(**inp,use_audio_in_video=False,use_cache=False).logits[0,-1].float().cpu()
            finally:head.remove()
            lp=torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):totals[index]+=lp[position]
            audits.append(audit)
        scores=totals.round(3).tolist();prediction=int(np.argmax(scores));answer=answer_options[prediction]
    del records;torch.cuda.empty_cache()
    return dict(answer=answer,scores=scores,prediction=prediction,audits=audits,duration=duration,
        latency_sec=round(time.time()-started,2),question_type_routing=False,
        representation='native isolated raw records; shared physical rotary clock')
