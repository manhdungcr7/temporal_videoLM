"""Diagnostic source-prior intervention, not a proposed final paper method.

Equalize available source budgets without changing within-source occurrence
ratios or total non-null prior mass. Native content affinities remain free to
prefer either source. This is deliberately an ablation of transferred reader
calibration, with no question-type/label routing and no gold-label access.
"""
import math
import torch


def equalize_source_log_mass(log_mass, source_ids):
    if log_mass.shape[-1] != source_ids.numel():
        raise ValueError('Source identities must match external records')
    source_ids = source_ids.to(log_mass.device)
    masks = [source_ids == source for source in source_ids.unique()]
    totals = [torch.logsumexp(log_mass[..., mask], dim=-1, keepdim=True) for mask in masks]
    active = torch.stack([torch.isfinite(total) for total in totals]).sum(0)
    total = torch.logsumexp(log_mass, dim=-1, keepdim=True)
    result = torch.full_like(log_mass, float('-inf'))
    for mask, source_total in zip(masks, totals):
        valid = torch.isfinite(source_total) & torch.isfinite(total)
        adjustment = torch.where(valid, total - source_total - active.clamp_min(1).log(), 0)
        result[..., mask] = torch.where(valid, log_mass[..., mask] + adjustment, float('-inf'))
    return result


def final_row_memory_audit(query, key, memory_query, memory_key, log_mass,
                           source_ids, raw_mask=None, scale=None):
    """Actual joint-softmax shares, averaged over heads, last selected text row."""
    heads = query.shape[-3]
    if key.shape[-3] != heads:
        key = key.repeat_interleave(heads // key.shape[-3], dim=-3)
    if memory_key.shape[-3] != heads:
        memory_key = memory_key.repeat_interleave(heads // memory_key.shape[-3], dim=-3)
    scale = query.shape[-1] ** -.5 if scale is None else scale
    raw_logits = query[..., -1:, :].float() @ key.float().transpose(-2, -1) * scale
    if raw_mask is not None:
        mask = raw_mask[..., -1:, :]
        raw_logits = (raw_logits.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool
                      else raw_logits + mask.float())
    external_logits = memory_query[..., -1:, :].float() @ memory_key.float().transpose(-2, -1) * scale
    weights = torch.softmax(torch.cat((raw_logits, external_logits + log_mass), dim=-1), dim=-1)
    native_count = key.shape[-2]
    external = weights[..., native_count:]
    return {'native': float(weights[..., :native_count].sum(-1).mean().detach().cpu()),
            'external': float(external.sum(-1).mean().detach().cpu()),
            'external_by_source': {str(int(source)): float(external[..., source_ids.to(external.device) == source]
                                                        .sum(-1).mean().detach().cpu())
                                   for source in source_ids.unique()}}


def avbind_source_audit_answer(video_path, question, answer_options=None,
                               slot_cache=None, value_cache=None, gain=1.0):
    duration = len(decode_audio(video_path)) / 16000
    nodes, _ = differential_sensor_bank(video_path, duration, slot_cache)
    source_ids = torch.tensor([0 if node['source'] == 'video' else 1 for node in nodes])
    original = globals()['joint_memory_read']
    readout_audits = []

    def intervened(query, key, value, memory_query, memory_key, memory_value,
                   bank_log_mass, raw_mask=None, scale=None):
        corrected = equalize_source_log_mass(bank_log_mass, source_ids)
        output = original(query, key, value, memory_query, memory_key, memory_value,
                          corrected, raw_mask, scale)
        # Measure the actual read, not just the input reader prior.
        readout_audits.append(final_row_memory_audit(query, key, memory_query, memory_key,
                                                    corrected, source_ids, raw_mask, scale))
        return output

    globals()['joint_memory_read'] = intervened
    try:
        result = avbind_counterfactual_answer(video_path, question, answer_options,
                                              slot_cache, value_cache, gain)
        result['source_prior_intervention'] = 'equal source budget; original total null-aware mass and within-source ratios'
        result['readout_audits'] = readout_audits
        result['diagnostic_only'] = True
        return result
    finally:
        globals()['joint_memory_read'] = original
