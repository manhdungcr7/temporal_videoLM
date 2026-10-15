"""Native source/clock-addressed records with optional local-prefix isolation.

No generated captions, learned text-reader prior, external KV approximation,
question classifier, label interpretation, or modality-order reasoning plan.
Record serialization order is fixed infrastructure, not a reasoning order.
All query/answer rows may attend every earlier record. Local records can only
read the common system prefix and their own source-specific crop/address.
"""
import contextlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def timeline_records(video_path, interval=2):
    duration = len(decode_audio(video_path)) / 16000
    records = []
    for index in range(int(math.ceil(duration / interval))):
        start, end = index * interval, min((index+1) * interval, duration)
        if end-start < .2:
            continue
        crop = cut(video_path, start, end, f'avbind_latent_{index}')
        media = prepare_video(crop)
        records.append(dict(index=index, start=start, end=end, media=media))
    return records, duration


def timeline_grid_seconds(records, temporal_patch_size):
    return torch.tensor([temporal_patch_size / r['media']['fps'] for r in records],dtype=torch.float32)


def timeline_inputs(records, question, options=None, perm=None):
    content = []
    videos, audios, headers = [], [], []
    for record in records:
        start, end, index, media = record['start'], record['end'], record['index'], record['media']
        for source in ('video', 'audio'):
            header = f'\n[Observation {source[0].upper()}{index:03d}: {source}, {start:g} to {end:g} seconds]\n'
            headers.append(header)
            content.append(dict(type='text', text=header))
            if source == 'video':
                content.append(dict(type='video', video=media['path'], max_pixels=media['max_pixels_per_frame']))
                videos.append(media['videos'][0])
            else:
                content.append(dict(type='audio', audio=media['path']))
                audios.append(media['audio'][0])
            content.append(dict(type='text', text=f'\n[End of {source} observation {start:g} to {end:g} seconds]\n'))
    marker = '\n[USER QUESTION]\n'
    body = marker + question
    if options is not None:
        perm = tuple(range(len(options))) if perm is None else perm
        body += ' ' + ' '.join(f'({letter}) {options[i]}' for letter, i in zip('ABCD', perm))
    content.append(dict(type='text', text=body))
    system = (SYSTEM + ' Each source-labeled observation below has an interval in the original recording. '
              'An observation interval is not necessarily the full event duration. '
              'Heard descriptions and directly visible events are independent evidence.')
    messages = [dict(role='system', content=[dict(type='text', text=system)]),
                dict(role='user', content=content)]
    rendered = PROC.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    fps = records[0]['media']['fps']
    inp = PROC(text=rendered, audio=audios, videos=videos, images=None,
               return_tensors='pt', padding=True, videos_kwargs={'fps': fps, 'use_audio_in_video': False})
    # v4.57.6's processor divides by scalar fps even for multiple predecoded
    # videos. Preserve each crop's actual effective fps (short tails can differ),
    # without resampling its tensors or assigning an incorrect local clock.
    if len(inp['video_grid_thw']) != len(records):
        raise ValueError('Video grids and physical observations do not align')
    inp['video_second_per_grid'] = timeline_grid_seconds(records, PROC.video_processor.temporal_patch_size)
    if 'input_features' not in inp or 'pixel_values_videos' not in inp:
        raise ValueError('Timeline must contain both modalities')
    ids = inp['input_ids'][0].tolist()
    expanded = TOK.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    tokenized = TOK(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if tokenized['input_ids'] != ids:
        raise ValueError('Expanded timeline did not round-trip')
    offsets = tokenized['offset_mapping']
    boundaries = [expanded.index(header) for header in headers] + [expanded.rindex(marker)]
    starts = [next(i for i,(a,b) in enumerate(offsets) if b > position) for position in boundaries]
    segments = list(zip(starts[:-1], starts[1:]))
    if not segments or any(end <= start for start, end in segments):
        raise ValueError('Invalid local-record boundaries')
    # Verify the number of native padding tokens against the independent encoders.
    audio_id, video_id = TOK.convert_tokens_to_ids(PROC.audio_token), TOK.convert_tokens_to_ids(PROC.video_token)
    lengths = (inp['feature_attention_mask'].sum(-1)-1)//2+1
    expected_audio = int((((lengths-2)//2+1).clamp_min(0)).sum())
    expected_video = int((inp['video_grid_thw'].prod(-1).sum() // PROC.video_processor.merge_size**2))
    if ids.count(audio_id) != expected_audio or ids.count(video_id) != expected_video:
        raise ValueError('Timeline modality-token alignment mismatch')
    audit = dict(tokens=len(ids), video_tokens=expected_video, audio_tokens=expected_audio,
                 records=len(segments), prefix_end=starts[0], question_start=starts[-1])
    return move_inputs(inp, MODEL_), segments, audit


def local_record_allowed(rows, keys, prefix_end, start, end):
    """Causal keys in common prefix or this record, never another record/Q."""
    return (keys[None, :] <= rows[:, None]) & ((keys[None, :] < prefix_end) |
            ((keys[None, :] >= start) & (keys[None, :] < end)))


@contextlib.contextmanager
def timeline_isolation(segments, prefix_end, enabled=True, decoder_modules=None, tail_bias=None):
    if not enabled:
        yield
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    decoder_ids = None if decoder_modules is None else {id(module) for module in decoder_modules}
    def dispatch(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if decoder_ids is not None and id(module) not in decoder_ids:
            return original(module,query,key,value,attention_mask,dropout=dropout,
                            scaling=scaling,is_causal=is_causal,**kwargs)
        if dropout:
            raise ValueError('Timeline isolation is inference-only')
        length, klen = query.shape[-2], key.shape[-2]
        # Cached decode sees the immutable record states through ordinary attention.
        if length == 1:
            if tail_bias is not None:
                added=tail_bias(module,torch.tensor([klen-1],device=query.device),klen)
                if added is not None:
                    added=added.to(query.device,query.dtype)
                    attention_mask=(added if attention_mask is None else
                        added.masked_fill(~attention_mask,float('-inf')) if attention_mask.dtype==torch.bool else attention_mask+added)
            return original(module, query, key, value, attention_mask,
                            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        if length != klen:
            raise ValueError('Only full prefill and single-token decode are supported')
        if not (0 <= prefix_end <= length and segments and segments[-1][1] <= length
                and all(prefix_end <= a < b <= length for a,b in segments)):
            raise ValueError('Record boundaries do not belong to this decoder sequence')
        heads = query.shape[-3]
        if key.shape[-3] != heads:
            groups = heads // key.shape[-3]
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
        output = torch.empty_like(query)
        slices = [(0, prefix_end, None)] + [(a,b,(a,b)) for a,b in segments]
        slices.append((segments[-1][1], length, None))
        for start, end, local in slices:
            if end <= start:
                continue
            rows = torch.arange(start, end, device=query.device)
            keys = torch.arange(klen, device=query.device) if local is None else torch.cat((
                torch.arange(prefix_end, device=query.device), torch.arange(start,end,device=query.device)))
            allowed = keys[None, :] <= rows[:, None]
            mask = allowed
            if attention_mask is not None:
                supplied = attention_mask[..., :klen]
                if supplied.shape[-2] != 1:
                    supplied = supplied[..., rows, :]
                supplied = supplied[..., keys]
                mask = allowed & supplied if supplied.dtype == torch.bool else supplied.masked_fill(~allowed,float('-inf'))
            if local is None and start==segments[-1][1] and tail_bias is not None:
                added=tail_bias(module,rows,klen)
                if added is not None:
                    added=added.to(query.device,query.dtype)
                    mask=added.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else mask+added
            output[..., start:end, :] = F.scaled_dot_product_attention(
                query[..., start:end, :], key[..., keys, :], value[..., keys, :],
                attn_mask=mask, dropout_p=0, is_causal=False, scale=scaling)
        return output.transpose(1,2).contiguous(), None
    ALL_ATTENTION_FUNCTIONS.register('sdpa', dispatch)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)


def avbind_timeline_answer(video_path, question, answer_options=None, isolated=True):
    started = time.time()
    records, duration = timeline_records(video_path)
    audits = []
    if answer_options is None:
        inp, segments, audit = timeline_inputs(records, question)
        with timeline_isolation(segments, audit['prefix_end'], isolated,
                                [layer.self_attn for layer in MODEL_.model.layers]), torch.no_grad():
            output = MODEL_.generate(**inp, use_audio_in_video=False, use_cache=True,
                                      max_new_tokens=128, do_sample=False)
        answer = TOK.decode(output[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        scores = prediction = None
        audits.append(audit)
    else:
        totals = np.zeros(len(answer_options))
        labels = [TOK.convert_tokens_to_ids(letter) for letter in 'ABCD'[:len(answer_options)]]
        for perm in rotations(len(answer_options)):
            inp, segments, audit = timeline_inputs(records, question, answer_options, perm)
            handle = MODEL_.lm_head.register_forward_pre_hook(lambda _module,args:(args[0][:,-1:,:],))
            try:
                with timeline_isolation(segments, audit['prefix_end'], isolated,
                                        [layer.self_attn for layer in MODEL_.model.layers]), torch.no_grad():
                    logits = MODEL_(**inp,use_audio_in_video=False,use_cache=False).logits[0,-1].float().cpu()
            finally:
                handle.remove()
            lp = torch.log_softmax(logits[labels],-1).numpy()
            for position,index in enumerate(perm):
                totals[index] += lp[position]
            audits.append(audit)
        scores = totals.round(3).tolist()
        prediction = int(np.argmax(scores))
        answer = answer_options[prediction]
    del records
    torch.cuda.empty_cache()
    return dict(answer=answer,prediction=prediction,scores=scores,isolated=isolated,audits=audits,
                duration=duration,latency_sec=round(time.time()-started,2),question_type_routing=False,
                generated_memory_text=False,representation='native raw records with textual source/clock addresses')
