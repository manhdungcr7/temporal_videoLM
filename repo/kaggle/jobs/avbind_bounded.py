"""Null-aware bounded external readout: a calibration diagnostic.

Native and external attention are normalized separately, with external weight
at most 0.25, fixed before evaluation. Source-balanced priors preserve relative
occurrence selection. This does not establish a novel/general trained method.
"""
import math
import torch
import torch.nn.functional as F


def bounded_memory_read(query, key, value, memory_query, memory_key, memory_value,
                        bank_log_mass, raw_mask=None, scale=None, max_weight=.25):
    if not 0 <= max_weight <= 1:
        raise ValueError('Readout budget must lie in [0,1]')
    heads = query.shape[-3]
    if key.shape[-3] != heads:
        groups = heads // key.shape[-3]
        key = key.repeat_interleave(groups, dim=-3)
        value = value.repeat_interleave(groups, dim=-3)
    if memory_key.shape[-3] != heads:
        groups = heads // memory_key.shape[-3]
        memory_key = memory_key.repeat_interleave(groups, dim=-3)
        memory_value = memory_value.repeat_interleave(groups, dim=-3)
    native = F.scaled_dot_product_attention(query, key, value, attn_mask=raw_mask,
                                            dropout_p=0, is_causal=False, scale=scale)
    factor = query.shape[-1] ** -.5 if scale is None else scale
    logits = memory_query.float() @ memory_key.float().transpose(-2, -1) * factor + bank_log_mass
    present = torch.isfinite(bank_log_mass).any(-1, keepdim=True)
    logits = torch.where(present, logits, 0)
    weights = torch.softmax(logits, dim=-1) * present
    external = (weights @ memory_value.float()).to(query.dtype)
    # Scaling bank logits by N_native in R7 was responsible for budget growth;
    # the caller supplies a per-record log prior with that factor removed.
    prior = bank_log_mass.exp().sum(-1, keepdim=True).clamp(0, 1)
    budget = max_weight * prior
    result = (1-budget).to(query.dtype) * native + budget.to(query.dtype) * external
    return result, budget, weights


def avbind_bounded_answer(video_path, question, answer_options=None,
                          slot_cache=None, value_cache=None, gain=1.0):
    duration = len(decode_audio(video_path)) / 16000
    nodes, _ = differential_sensor_bank(video_path, duration, slot_cache)
    source_ids = torch.tensor([0 if node['source'] == 'video' else 1 for node in nodes])
    original = globals()['joint_memory_read']
    audits = []
    # Read the exact existing native token count once; do not use task labels.
    media = prepare_video(video_path)
    inp, _ = latent_question_inputs(media, question, answer_options)
    _, _, video, audio, _ = media_layout(MODEL_, PROC, inp)
    native_count = int((video | audio).sum())
    del media, inp

    def intervened(query, key, value, memory_query, memory_key, memory_value,
                   bank_log_mass, raw_mask=None, scale=None):
        corrected = equalize_source_log_mass(bank_log_mass, source_ids) - math.log(max(1, native_count))
        output, budget, weights = bounded_memory_read(query, key, value, memory_query,
            memory_key, memory_value, corrected, raw_mask, scale, max_weight=.25)
        audits.append({'native': float((1-budget).mean().detach().cpu()),
                       'external': float(budget.mean().detach().cpu()),
                       'external_by_source': {str(int(source)): float((budget * weights[..., source_ids.to(weights.device) == source]
                                                                      .sum(-1, keepdim=True)).mean().detach().cpu())
                                              for source in source_ids.unique()}})
        return output

    globals()['joint_memory_read'] = intervened
    try:
        result = avbind_counterfactual_answer(video_path, question, answer_options,
                                              slot_cache, value_cache, gain)
        result['readout_audits'] = audits
        result['max_external_weight'] = .25
        result['diagnostic_only'] = True
        return result
    finally:
        globals()['joint_memory_read'] = original
