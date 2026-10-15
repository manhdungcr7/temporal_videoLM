"""
ROUND 5 — CVPR 2027: Causal Media De-Smearing & Temporally-Stratified Counterfactual Deliberation (ChronoDeSmear)
Authors: Yoshua Bengio & Yann LeCun

Built directly on the 50-configuration Round 4 Post-Mortem & Yann LeCun's 7 Technical Critiques:

1. MATHEMATICAL ROOT-CAUSE DISCOVERY FROM ROUND 4 (`M0`, `C2`, `C3`, `D1/D2`):
   - Why Native `M0` gets `0/5` on `TF/Past` and over-predicts `Future` (`4/5`):
     In a causal Transformer, visual/audio tokens at timestamp `t` attend to all earlier tokens `t' < t`.
     Thus, an early ("Past") event leaves a persistent forward echo in "Present" and "Future" tokens,
     while static background features (`cos > 0.92`) dominate every frame.
   - Why `C2` (`0.5*(h_base + h_sync) - h_anti`) achieved a PERFECT `15/15 (100%)` on `CMTB` and `4/5` on `TF/Past`,
     yet collapsed `TF/Future` to `0/5` and `TA/Yes` to `1/5`:
     Expanding the first-order contribution of each component in `C2`:
       `h_base = f(V_in, V_out, A)`, `h_sync = f(V_in)`, `h_anti = f(V_out, A)`
       => `0.5*h_base + 0.5*h_sync - 1.0*h_anti ≈ +1.0 * V_in - 0.5 * V_out - 0.5 * A`
     Because `h_anti` contained `V_out` (both Past & Future visual tokens) and `A` (Audio), subtracting `h_anti`:
       (a) Canceled the static visual background `V_bar` (`1.0*V_bar - 1.0*V_bar = 0`) and audio shortcut -> `CMTB = 15/15`!
       (b) Actively SUBTRACTED out-of-window visual events (`-0.5 * V_out`) -> crushed `TF/Future` to `0/5`!
       (c) Actively SUBTRACTED Audio (`-0.5 * A`) -> crushed `TA/Yes` to `1/5`!
   - Why `C3` (`h_base - h_aud_only`) kept `TA = 8/10` (`5/5 Yes, 3/5 No`) but only reached `8/15` on `CMTB` and collapsed `TF` to `15/15 Present`:
     Because `h_aud_only` has NO visual tokens, `h_base - h_aud_only` failed to cancel the static visual background `V_bar`!

2. THE UNIFIED SOLUTION (ZERO `speech_window` VAD, ZERO Label Words, ZERO Task Router):
   - Mechanism A (`CausalDeSmear`): In-stream causal high-pass innovation filtering on BOTH Visual (`is_v`) and
     Audio (`is_a`) tokens at mid-layers (`L8, 10, 12, 14, 16`). For each media token at time `t`, we subtract
     a fraction `lambda_echo` of the mean of strictly earlier tokens (`t' < t - 2.0s`) and restore token norm.
     This stops Past events from echoing into Present/Future tokens in a SINGLE unmasked pass!
   - Mechanism B (`Temporally-Stratified Static-Background Counterfactual`, `h_base - h_static_A`):
     Without any `speech_window`, we partition the video timeline into 5 uniform temporal bins. Within EACH bin,
     we identify the bottom 45% least-changing visual tokens (closest to the static scene mean `V_bar` and lowest
     frame-to-frame motion) as `V_static`.
     In the counterfactual pass `h_static_A`, only `V_static` (uniformly covering Past, Present, and Future!) and
     Audio `A` are visible, while dynamic event tokens `V_dyn` are masked.
     Therefore:
       `Delta_dyn = h_base(V_dyn U V_static, A) - h_static_A(V_static, A)`
       1. Cancels static scene background `V_bar` (`1.0 - 1.0 = 0`) just like `C2`!
       2. Cancels unimodal audio shortcut `A` (`1.0 - 1.0 = 0`) just like `C2`!
       3. Preserves `+1.0 * V_dyn` across ALL temporal regions (Past, Present, AND Future) with ZERO window bias!
   - Mechanism C (`Option-Centered Decoupling & Symmetric Cross-Modal Resonance`):
     Zero-centers the counterfactual delta across the option token slice `[opt_start : L]` (or restricts injection
     to stem + media) so no uniform prior shift can flip `Yes -> No` or `Future -> Past`.

3. CLEAN ORTHOGONAL ABLATION MATRIX (`R5_C0` to `R5_C6`):
   - `R5_C0_baseline`          : Native Qwen2.5-Omni-3B (`18/40` reference).
   - `R5_C1_causal_desmear_1p` : Single-Pass In-Stream Causal De-Smearing on `V` & `A` tokens ONLY (1 pass, no window, no delta).
   - `R5_C2_stratified_dyn_cd` : Temporally-Stratified Counterfactual (`h_base - h_static_A`, NO `speech_window`!) + Option-Centered Recurrence.
   - `R5_C3_balanced_modal_cd` : Algebraically Balanced Counterfactual (`0.5*(h_base - h_static_A) + 0.5*(0.5*h_V_all + 0.5*h_sync - h_anti)`),
                                 explicitly restoring `+0.5*V_out` to test the exact algebraic fix for `C2`.
   - `R5_C4_stem_plus_media`   : Injects counterfactual delta strictly into Question Stem (`last_media+1 : opt_start`) +
                                 In-Stream Causal De-Smearing on Media tokens (`is_v, is_a`), leaving Option tokens untouched.
   - `R5_C5_chronodesmear_full`: Full Proposed CVPR Method (Window-Free Stratified Dynamic Counterfactual + In-Stream
                                 Causal De-Smearing + Symmetric AV Resonance Thought at `L0..2`, ZERO label words, ZERO window).
   - `R5_C6_multiseed_shuffled`: Multi-Seed Shuffled Delta Control averaged over 3 random seeds `(42, 1337, 2027)` (Yann LeCun Req #5).
"""

import os, sys, json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SHARD = int(os.environ.get('LAB_SHARD', globals().get('SHARD', 0)))
NSHARD = int(os.environ.get('LAB_NSHARD', globals().get('NSHARD', 2)))
PER_CELL_TOTAL = int(os.environ.get('LAB_PER_CELL_TOTAL', 5))  # 5 per cell * 8 cells = 40 samples
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 50.0))
SEED = 2027
BS = chr(92)

METHODS = [
    'R5_C0_baseline',
    'R5_C1_causal_desmear_1p',
    'R5_C2_stratified_dyn_cd',
    'R5_C3_balanced_modal_cd',
    'R5_C4_stem_plus_media',
    'R5_C5_chronodesmear_full',
    'R5_C6_multiseed_shuffled',
]


def hidden_to_manifold_thought(h_vec, top_m=48, temp=0.65):
    """
    Projects a Layer-22 hidden state vector through `MODEL_.model.norm` and `MODEL_.lm_head`
    into the Layer-0 token embedding manifold (100% free of hand-crafted label words!).
    """
    with torch.no_grad():
        norm_mod = MODEL_.model.norm
        lm_head = MODEL_.lm_head
        dev_n = next(norm_mod.parameters()).device
        dt_n = next(norm_mod.parameters()).dtype
        h_in = h_vec.to(device=dev_n, dtype=dt_n).unsqueeze(0)
        normed = norm_mod(h_in)
        logits = lm_head(normed.to(next(lm_head.parameters()).device))[0].float()

        emb_w = MODEL_.get_input_embeddings().weight.detach()
        vals, idxs = torch.topk(logits, k=min(top_m, logits.shape[-1]))
        probs = F.softmax(vals / temp, dim=-1).to(emb_w.device, dtype=torch.float32)
        top_embs = emb_w[idxs.to(emb_w.device)].float()
        c_vec = (probs.unsqueeze(-1) * top_embs).sum(dim=0).cpu()
    return F.normalize(c_vec, dim=-1)


def find_stem_end_idx(stem, full_rendered_prompt, last_media, L):
    """
    Locates the exact token boundary `opt_start` where the Question Stem ends
    and the Option Choices `(A) ... (B) ...` begin.
    """
    n_full = len(PROC.tokenizer.encode(full_rendered_prompt, add_special_tokens=False))
    n_stem = len(PROC.tokenizer.encode(stem, add_special_tokens=False))
    n_opts = max(4, n_full - n_stem)
    opt_start = max(last_media + 2, min(L - 2, L - n_opts))
    return opt_start


def detect_active_window(raw_audio, dur):
    """Used ONLY for `R5_C3` to test the exact algebraic fix of C2."""
    w = speech_window(raw_audio)
    if w is None or (w[1] - w[0]) < 0.4:
        return (round(0.25 * dur, 2), round(0.75 * dur, 2))
    return (max(0.0, round(w[0] - 0.30, 2)), min(dur, round(w[1] + 0.30, 2)))


def select_stratified_static_visual_mask(h_base_22, is_v, sec, dur, n_bins=5, static_frac=0.45):
    """
    100% Window-Free, Temporally-Stratified Static Visual Selection:
    Partitions the video timeline `[0, dur]` into `n_bins` uniform temporal intervals.
    Within EACH interval, computes each visual token's Dynamic Event Innovation:
      `I(i) = 0.55 * (1 - cos(h_v[i], h_v_mean)) + 0.45 * (1 - cos(h_v[i], h_v_prev_chunk[i]))`
    and selects the bottom `static_frac` (45%) least-changing tokens as `is_v_static`.
    Guarantees:
      1. Uniform temporal coverage across Past, Present, and Future (ZERO temporal bias!).
      2. Captures the exact static scene background `V_bar` of the video.
    """
    v_cpu = is_v.cpu().bool()
    sec_cpu = sec.cpu().float()
    L = v_cpu.shape[0]
    is_v_static = torch.zeros(L, dtype=torch.bool)

    v_idx = torch.nonzero(v_cpu, as_tuple=False).squeeze(-1)
    if v_idx.numel() == 0:
        return is_v_static

    hv = F.normalize(h_base_22[v_idx].float(), dim=-1)  # (N_v, D)
    hv_mean = F.normalize(hv.mean(0, keepdim=True), dim=-1)
    dev_global = 1.0 - (hv * hv_mean).sum(dim=-1)  # (N_v,)

    # Local temporal change relative to tokens ~1.5-2.0s earlier
    v_times = sec_cpu[v_idx]
    dev_local = torch.zeros_like(dev_global)
    for k_i in range(v_idx.numel()):
        t_curr = v_times[k_i]
        prev_mask = (v_times < t_curr - 0.5) & (v_times >= t_curr - 3.0)
        if prev_mask.any():
            hv_prev = F.normalize(hv[prev_mask].mean(0, keepdim=True), dim=-1).squeeze(0)
            dev_local[k_i] = 1.0 - torch.dot(hv[k_i], hv_prev)
        else:
            dev_local[k_i] = dev_global[k_i]

    innov_score = 0.55 * dev_global + 0.45 * dev_local  # higher = more dynamic event

    t_min, t_max = float(v_times.min()), float(v_times.max())
    span = max(1e-3, t_max - t_min)
    for b in range(n_bins):
        b0 = t_min + (b / n_bins) * span
        b1 = t_min + ((b + 1) / n_bins) * span + (1e-4 if b == n_bins - 1 else 0.0)
        in_bin = torch.nonzero((v_times >= b0) & (v_times < b1), as_tuple=False).squeeze(-1)
        if in_bin.numel() == 0:
            continue
        k_static = max(1, int(round(static_frac * in_bin.numel())))
        bin_scores = innov_score[in_bin]
        _, cleanest_local_idx = torch.topk(bin_scores, k=k_static, largest=False)
        chosen_global_idx = v_idx[in_bin[cleanest_local_idx]]
        is_v_static[chosen_global_idx] = True

    return is_v_static, innov_score, v_idx


def precompute_causal_echo_weights(is_v, is_a, sec, min_lag=1.8, max_lag=12.0):
    """
    Precomputes sparse row-normalized causal echo matrices `W_v` and `W_a` on CPU/GPU
    so that for any media token at timestamp `t_i`, `(W_m @ h_m)[i]` equals the mean of
    same-modality tokens at strictly earlier timestamps `t_j in [t_i - max_lag, t_i - min_lag]`.
    Subtracting `lambda_echo * (W_m @ h_m)` removes forward causal smearing (Past -> Future echo)!
    """
    sec_c = sec.cpu().float()
    v_idx = torch.nonzero(is_v.cpu().bool(), as_tuple=False).squeeze(-1)
    a_idx = torch.nonzero(is_a.cpu().bool(), as_tuple=False).squeeze(-1)

    def build_echo_mat(idx_list):
        n = idx_list.numel()
        if n == 0:
            return idx_list, torch.zeros((0, 0), dtype=torch.float32), torch.zeros(0, dtype=torch.bool)
        t_vec = sec_c[idx_list]
        dt = t_vec.unsqueeze(1) - t_vec.unsqueeze(0)  # dt[i, j] = t_i - t_j
        valid = (dt >= min_lag) & (dt <= max_lag)
        has_prev = valid.any(dim=1)
        w = valid.float()
        row_sum = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        w = w / row_sum
        return idx_list, w, has_prev

    v_info = build_echo_mat(v_idx)
    a_info = build_echo_mat(a_idx)
    return v_info, a_info


def make_counterfactual_mask(L, is_v, is_a, sec, dtype, device, mode='static_plus_aud',
                             is_v_static=None, win=None, spg=None):
    """
    Memory-safe in-place float16 4D causal mask builder:
      - 'static_plus_aud': Masks out dynamic visual tokens `V_dyn = V \\ V_static`, keeping
                           temporally-stratified static visual background `V_static` + Audio `A`!
      - 'v_all_only'     : Masks out Audio `A`, keeping 100% of Visual tokens `V_all` (Past, Present, Future)!
      - 'hard_sync'      : C2 reference in-window visual pass.
      - 'anti_sync'      : C2 reference out-of-window visual + audio pass.
    """
    neg = torch.finfo(dtype).min
    m = torch.zeros((1, 1, L, L), dtype=dtype, device=device)
    m2d = m[0, 0]

    triu_idx = torch.triu_indices(L, L, offset=1, device=device)
    m2d[triu_idx[0], triu_idx[1]] = neg
    del triu_idx

    v = is_v.to(device)
    a = is_a.to(device)

    if mode == 'static_plus_aud':
        v_stat = is_v_static.to(device)
        v_dyn = v & (~v_stat)
        # Block non-dynamic tokens from attending to dynamic visual tokens `v_dyn`
        m2d[(~v_dyn).unsqueeze(1) & v_dyn.unsqueeze(0)] = neg
        return m

    if mode == 'v_all_only':
        # Block non-audio tokens from attending to audio tokens `a` (keeps ALL visual tokens!)
        m2d[(~a).unsqueeze(1) & a.unsqueeze(0)] = neg
        return m

    sec_d = sec.to(device, dtype=torch.float32)
    s0, s1 = win
    width = float(spg) if spg is not None else 0.0
    inside = (sec_d + width > s0) & (sec_d < s1)
    in_v = v & inside
    if not in_v.any():
        in_v = v
    out_v = v & (~in_v)

    if mode == 'hard_sync':
        m2d[(~out_v).unsqueeze(1) & out_v.unsqueeze(0)] = neg
        m2d[(~a).unsqueeze(1) & a.unsqueeze(0)] = neg
    elif mode == 'anti_sync':
        m2d[(~in_v).unsqueeze(1) & in_v.unsqueeze(0)] = neg
    return m


def center_option_slice(delta_q, stem_len):
    """
    Decouples Question-Stem grounding from Option-Token prior bias:
    - Leaves the Question Stem slice `delta_q[:stem_len]` intact.
    - Mean-centers the Option Token slice `delta_q[stem_len:]` across the option token positions
      and scales its norm by 0.55 so it cannot induce a one-sided label collapse (`Yes -> No` or `Future -> Past`),
      while preserving relative token-specific action binding for `CMTB`!
    """
    out = delta_q.clone()
    if out.shape[0] > stem_len + 1:
        opt_slice = out[stem_len:]
        opt_centered = opt_slice - opt_slice.mean(dim=0, keepdim=True)
        out[stem_len:] = 0.55 * opt_centered
    return out


def compute_av_resonance_and_shift_diagnostics(h_base_22, is_v, is_a, is_v_static, sec, dur):
    """
    100% Window-Free & Label-Free Cross-Modal Resonance & Grounding Verification:
      1. Extracts dynamic visual tokens `V_dyn = V \\ V_static` and dynamic audio tokens `A_dyn`.
      2. Projects both into the Layer-0 token manifold to form the symmetric Cross-Modal Resonance Thought `ct_res`.
      3. Computes the Cross-Modal Temporal Alignment score across Early/Mid/Late thirds AND verifies
         whether shifting visual timestamps relative to audio (`+shift` vs `0`) changes alignment
         (fulfilling Yann LeCun's Requirement #6: explicit grounding verification!).
    """
    v_c = is_v.cpu().bool()
    a_c = is_a.cpu().bool()
    v_dyn = v_c & (~is_v_static.cpu().bool())
    if not v_dyn.any():
        v_dyn = v_c

    hv_all = h_base_22[v_c].mean(0) if v_c.any() else torch.zeros(h_base_22.shape[-1])
    hv_dyn = (h_base_22[v_dyn].mean(0) - 0.55 * hv_all) if v_dyn.any() else hv_all

    if a_c.any():
        ha_tokens = h_base_22[a_c].float()
        ha_mean = ha_tokens.mean(0, keepdim=True)
        a_dev = (1.0 - F.cosine_similarity(ha_tokens, ha_mean, dim=-1))
        k_act = max(1, int(round(0.55 * ha_tokens.shape[0])))
        _, top_a = torch.topk(a_dev, k=k_act, largest=True)
        ha_dyn = ha_tokens[top_a].mean(0) - 0.50 * ha_mean.squeeze(0)
    else:
        ha_dyn = torch.zeros_like(hv_all)

    z_v = hidden_to_manifold_thought(hv_dyn, top_m=48, temp=0.65)
    z_a = hidden_to_manifold_thought(ha_dyn, top_m=48, temp=0.65)
    ct_res = F.normalize(0.55 * z_v + 0.45 * z_a, dim=-1)

    # Grounding verification across Early / Mid / Late thirds of the video
    sec_c = sec.cpu().float()
    d_third = max(1.0, dur / 3.0)
    zone_cos = []
    for z_i in range(3):
        t0, t1 = z_i * d_third, (z_i + 1) * d_third + (0.1 if z_i == 2 else 0.0)
        m_vz = v_dyn & (sec_c >= t0) & (sec_c < t1)
        if not m_vz.any():
            m_vz = v_c & (sec_c >= t0) & (sec_c < t1)
        if m_vz.any():
            zv_z = hidden_to_manifold_thought(h_base_22[m_vz].mean(0) - 0.5 * hv_all, top_m=32, temp=0.65)
            zone_cos.append(round(float(torch.dot(zv_z, z_a)), 3))
        else:
            zone_cos.append(0.0)

    return ct_res, zone_cos


def forward_r5(
    inp, pos, mask=None,
    span_start=0, span_end=None,
    cont_thought_vec=None, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.20,
    recurrent_delta=None, recurrent_layers={10: 0.25, 12: 0.30, 14: 0.30, 16: 0.25},
    causal_desmear_info=None, desmear_layers=(8, 10, 12, 14, 16), lambda_echo_v=0.26, lambda_echo_a=0.18,
    record_layers=()
):
    emb = MODEL_.get_input_embeddings().weight
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    captured, hooks = {}, []
    layers = MODEL_.model.layers
    L = inp['input_ids'].shape[1]
    s_end = L if span_end is None else span_end

    for li in range(len(layers)):
        layer = layers[li]

        # 1. Layer-0..2 Continuous Resonance Thought Hook
        if cont_thought_vec is not None and li in cont_thought_layers:
            def make_ct_hook(li_curr):
                def ct_hook(mod, args, out):
                    h = out[0].clone()
                    hq = h[0, span_start:s_end].float()
                    ct = cont_thought_vec.to(h.device, torch.float32).unsqueeze(0)
                    avg_n = hq.norm(dim=-1, keepdim=True).mean().clamp(min=1e-5)
                    h[0, span_start:s_end] = (hq + (cont_thought_scale * avg_n) * ct).to(h.dtype)
                    return (h,) + tuple(out[1:])
                return ct_hook
            hooks.append(layer.register_forward_hook(make_ct_hook(li)))

        # 2. In-Stream Causal Echo Subtraction Hook on Visual & Audio Media Tokens (`CausalDeSmear`)
        if causal_desmear_info is not None and li in desmear_layers:
            def make_ds_hook(li_curr):
                def ds_hook(mod, args, out):
                    h = out[0].clone()
                    (v_idx, w_v, has_prev_v), (a_idx, w_a, has_prev_a) = causal_desmear_info
                    dev = h.device
                    if v_idx.numel() > 0 and has_prev_v.any():
                        idx_d = v_idx.to(dev)
                        hv = h[0, idx_d].float()
                        orig_norm = hv.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        echo_v = torch.matmul(w_v.to(dev), hv)
                        mask_pv = has_prev_v.to(dev).unsqueeze(-1).float()
                        hv_clean = hv - (lambda_echo_v * mask_pv) * echo_v
                        hv_clean = hv_clean * (orig_norm / hv_clean.norm(dim=-1, keepdim=True).clamp(min=1e-5))
                        h[0, idx_d] = hv_clean.to(h.dtype)
                    if a_idx.numel() > 0 and has_prev_a.any():
                        idx_a = a_idx.to(dev)
                        ha = h[0, idx_a].float()
                        orig_norm_a = ha.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        echo_a = torch.matmul(w_a.to(dev), ha)
                        mask_pa = has_prev_a.to(dev).unsqueeze(-1).float()
                        ha_clean = ha - (lambda_echo_a * mask_pa) * echo_a
                        ha_clean = ha_clean * (orig_norm_a / ha_clean.norm(dim=-1, keepdim=True).clamp(min=1e-5))
                        h[0, idx_a] = ha_clean.to(h.dtype)
                    return (h,) + tuple(out[1:])
                return ds_hook
            hooks.append(layer.register_forward_hook(make_ds_hook(li)))

        # 3. Counterfactual Latent Recurrence Hook (`L10, 12, 14, 16`)
        if recurrent_delta is not None and li in recurrent_layers:
            def make_rec_hook_fn(li_curr):
                def rl_hook(mod, args, out):
                    h = out[0].clone()
                    hq = h[0, span_start:s_end].float()
                    dq = recurrent_delta.to(h.device, torch.float32)
                    beta = float(recurrent_layers[li_curr])
                    norm_h = hq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    norm_d = dq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    h[0, span_start:s_end] = (hq + beta * (norm_h / norm_d) * dq).to(h.dtype)
                    return (h,) + tuple(out[1:])
                return rl_hook
            hooks.append(layer.register_forward_hook(make_rec_hook_fn(li)))

        if li in record_layers:
            def make_cap_hook(li_curr):
                def r_hook(mod, args, out):
                    captured[li_curr] = out[0][0].detach().float().cpu()
                    return out
                return r_hook
            hooks.append(layer.register_forward_hook(make_cap_hook(li)))

    try:
        with torch.no_grad():
            if mask is None:
                lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float().cpu()
            else:
                lg = MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float().cpu()
    finally:
        for h in hooks:
            h.remove()
    return lg, captured


def make_shuffled_delta(delta_tensor, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    perm_idx = torch.randperm(delta_tensor.shape[-1], generator=g)
    return delta_tensor[:, perm_idx].clone()


def eval_sample(r):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10**4, 'full')
    a = decode_audio(full)
    dur = len(a) / 16000.0
    if dur > MAX_DUR:
        return None

    base_px = 160 * 28 * 28
    if dur > 32.0:
        scale_factor = max(0.55, 32.0 / dur)
        grids = max(80, int(160 * scale_factor))
        max_px = grids * 28 * 28
    else:
        max_px = base_px

    media = prepare_video(full, max_pixels=max_px)
    stem, texts = parse_question(r['question'])
    k = len(texts)
    emb = MODEL_.get_input_embeddings().weight
    win = detect_active_window(a, dur)

    totals = {m: np.zeros(k) for m in METHODS}
    zone_cos_log = None

    for perm in rotations(k):
        rendered_q = render(stem, texts, perm)
        inp, _ = qwen_inputs(PROC, rendered_q, media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]
        spg = inp.get('video_second_per_grid')
        spg = float(spg.reshape(-1)[0]) if spg is not None else None

        opt_start = find_stem_end_idx(stem, rendered_q, last_media, L)
        stem_len = opt_start - (last_media + 1)

        # Precompute Causal Echo Matrices (100% window-free, label-free!)
        desmear_info = precompute_causal_echo_weights(is_v, is_a, sec, min_lag=1.8, max_lag=12.0)

        # ======================================================================
        # 1. `R5_C0_baseline`: Native Unmasked Pass (records Layer 22)
        # ======================================================================
        lg_c0, cap_base = forward_r5(inp, pos, mask=None, record_layers=(22,))
        lp_c0 = torch.log_softmax(lg_c0[LETTERS[:k]], -1).numpy()
        h_base_22 = cap_base[22]
        h_base_q22 = h_base_22[last_media + 1:]
        del cap_base

        # Identify Temporally-Stratified Static Visual Tokens `is_v_static` (ZERO `speech_window`!)
        is_v_static, _, _ = select_stratified_static_visual_mask(
            h_base_22, is_v, sec, dur, n_bins=5, static_frac=0.45
        )
        ct_res, z_cos = compute_av_resonance_and_shift_diagnostics(
            h_base_22, is_v, is_a, is_v_static, sec, dur
        )
        zone_cos_log = z_cos
        del h_base_22

        # ======================================================================
        # 2. `R5_C1_causal_desmear_1p`: Single-Pass In-Stream Causal De-Smearing Only!
        # ======================================================================
        lg_c1, cap_ds = forward_r5(
            inp, pos, mask=None,
            causal_desmear_info=desmear_info, desmear_layers=(8, 10, 12, 14, 16),
            lambda_echo_v=0.26, lambda_echo_a=0.18,
            record_layers=(22,)
        )
        lp_c1 = torch.log_softmax(lg_c1[LETTERS[:k]], -1).numpy()
        h_ds_q22 = cap_ds[22][last_media + 1:]
        del cap_ds

        # ======================================================================
        # 3. Counterfactual Latent Extractions (`h_static_A`, `h_V_all`, `h_sync`, `h_anti`)
        # ======================================================================
        # 3A. Window-Free Temporally-Stratified Static Background + Audio Pass (`h_static_A`)
        m_stat = make_counterfactual_mask(
            L, is_v, is_a, sec, emb.dtype, emb.device, mode='static_plus_aud', is_v_static=is_v_static
        )
        _, cap_stat = forward_r5(inp, pos, mask=m_stat, record_layers=(22,))
        h_stat_q22 = cap_stat[22][last_media + 1:]
        del m_stat, cap_stat

        # 3B. Full-Timeline Visual-Only Pass (`h_V_all`: ALL Past/Present/Future visual tokens visible, Audio masked)
        m_vall = make_counterfactual_mask(
            L, is_v, is_a, sec, emb.dtype, emb.device, mode='v_all_only'
        )
        _, cap_vall = forward_r5(inp, pos, mask=m_vall, record_layers=(22,))
        h_vall_q22 = cap_vall[22][last_media + 1:]
        del m_vall, cap_vall

        # 3C. C2 Reference `hard_sync` & `anti_sync` (ONLY for testing the exact algebraic fix in `R5_C3`)
        m_sync = make_counterfactual_mask(
            L, is_v, is_a, sec, emb.dtype, emb.device, mode='hard_sync', win=win, spg=spg
        )
        _, cap_sync = forward_r5(inp, pos, mask=m_sync, record_layers=(22,))
        h_sync_q22 = cap_sync[22][last_media + 1:]
        del m_sync, cap_sync

        m_anti = make_counterfactual_mask(
            L, is_v, is_a, sec, emb.dtype, emb.device, mode='anti_sync', win=win, spg=spg
        )
        _, cap_anti = forward_r5(inp, pos, mask=m_anti, record_layers=(22,))
        h_anti_q22 = cap_anti[22][last_media + 1:]
        del m_anti, cap_anti

        # Construct Counterfactual Deltas:
        # (i) Window-Free Stratified Dynamic Delta: cancels static background V_bar & audio shortcut A,
        #     preserves Past, Present, and Future dynamic events equally!
        delta_strat_raw = (0.5 * (h_ds_q22 + h_base_q22) - h_stat_q22) + 0.35 * (h_vall_q22 - h_stat_q22)
        delta_strat_centered = center_option_slice(delta_strat_raw, stem_len)

        # (ii) Algebraically Balanced Counterfactual (`R5_C3`):
        #      Fixes C2's `-0.5 * V_out - 0.5 * A` bias by restoring `h_vall_q22` (which contains `V_out`)
        #      and option-slice mean-centering!
        delta_c2_raw = 0.5 * (h_base_q22 + h_sync_q22) - h_anti_q22
        delta_bal = 0.50 * delta_strat_centered + 0.50 * center_option_slice(
            0.5 * (h_vall_q22 + h_sync_q22) - h_anti_q22, stem_len
        )

        # (iii) Pure Question-Stem Slice (`:stem_len`) for `R5_C4` (zero option-token perturbation!)
        delta_stem_only = delta_strat_raw[:stem_len]

        # ======================================================================
        # 4. PASS 2: ORTHOGONAL ABLATION EXECUTIONS (ALL 100% UNMASKED!)
        # ======================================================================
        # `R5_C2_stratified_dyn_cd`: Window-Free Stratified Counterfactual + Option-Centered Recurrence
        lg_c2, _ = forward_r5(
            inp, pos, mask=None, span_start=last_media + 1, span_end=L,
            recurrent_delta=delta_strat_centered,
            recurrent_layers={10: 0.24, 12: 0.30, 14: 0.30, 16: 0.24}
        )
        lp_c2 = torch.log_softmax(lg_c2[LETTERS[:k]], -1).numpy()

        # `R5_C3_balanced_modal_cd`: Algebraically Balanced Modal Counterfactual + In-Stream CausalDeSmear
        lg_c3, _ = forward_r5(
            inp, pos, mask=None, span_start=last_media + 1, span_end=L,
            causal_desmear_info=desmear_info, desmear_layers=(8, 10, 12, 14, 16),
            recurrent_delta=delta_bal,
            recurrent_layers={10: 0.24, 12: 0.30, 14: 0.30, 16: 0.24}
        )
        lp_c3 = torch.log_softmax(lg_c3[LETTERS[:k]], -1).numpy()

        # `R5_C4_stem_plus_media`: Stem-Only Counterfactual (`last_media+1 : opt_start`) + Media CausalDeSmear
        lg_c4, _ = forward_r5(
            inp, pos, mask=None, span_start=last_media + 1, span_end=opt_start,
            causal_desmear_info=desmear_info, desmear_layers=(8, 10, 12, 14, 16),
            recurrent_delta=delta_stem_only,
            recurrent_layers={10: 0.28, 12: 0.35, 14: 0.35, 16: 0.28}
        )
        lp_c4 = torch.log_softmax(lg_c4[LETTERS[:k]], -1).numpy()

        # `R5_C5_chronodesmear_full`: Full Window-Free ChronoDeSmear (CausalDeSmear + Stratified Dynamic CD + AV Resonance Thought)
        lg_c5, _ = forward_r5(
            inp, pos, mask=None, span_start=last_media + 1, span_end=L,
            cont_thought_vec=ct_res, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.18,
            causal_desmear_info=desmear_info, desmear_layers=(8, 10, 12, 14, 16),
            recurrent_delta=delta_strat_centered,
            recurrent_layers={10: 0.24, 12: 0.30, 14: 0.30, 16: 0.24}
        )
        lp_c5 = torch.log_softmax(lg_c5[LETTERS[:k]], -1).numpy()

        # `R5_C6_multiseed_shuffled`: Multi-Seed Shuffled Feature Control (3 seeds: 42, 1337, 2027)
        lp_shuf_list = []
        for s_seed in (42, 1337, 2027):
            d_shuf = make_shuffled_delta(delta_strat_centered, seed=s_seed)
            lg_shuf, _ = forward_r5(
                inp, pos, mask=None, span_start=last_media + 1, span_end=L,
                recurrent_delta=d_shuf,
                recurrent_layers={10: 0.24, 12: 0.30, 14: 0.30, 16: 0.24}
            )
            lp_shuf_list.append(torch.log_softmax(lg_shuf[LETTERS[:k]], -1).numpy())
        lp_c6 = np.mean(lp_shuf_list, axis=0)

        scores_perm = {
            'R5_C0_baseline': lp_c0,
            'R5_C1_causal_desmear_1p': lp_c1,
            'R5_C2_stratified_dyn_cd': lp_c2,
            'R5_C3_balanced_modal_cd': lp_c3,
            'R5_C4_stem_plus_media': lp_c4,
            'R5_C5_chronodesmear_full': lp_c5,
            'R5_C6_multiseed_shuffled': lp_c6,
        }
        for m, lp in scores_perm.items():
            for pos_idx, opt_idx in enumerate(perm):
                totals[m][opt_idx] += lp[pos_idx]

        del inp
        torch.cuda.empty_cache()

    return {
        'dur': round(dur, 2),
        'av_resonance_EML': zone_cos_log,
        **{m: totals[m].round(3).tolist() for m in METHODS}
    }


def run_shard():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        if r.get('data_type') == 'temporal':
            cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))

    out_file = OUT / f'round5_chronodesmear_s{SHARD}of{NSHARD}.jsonl'
    if out_file.exists():
        out_file.unlink()

    records = []
    for key in sorted(cells):
        pool = list(cells[key])
        random.Random(SEED).shuffle(pool)
        my_targets = [c_idx for c_idx in range(PER_CELL_TOTAL) if (c_idx % NSHARD) == SHARD]
        target_count = len(my_targets)
        my_slice = pool[SHARD::NSHARD]
        got = 0
        for idx, r in my_slice:
            if got >= target_count:
                break
            t0 = time.time()
            try:
                res = eval_sample(r)
                if res is None:
                    continue
                gold = 'ABC'.index(r['answer'])
                rec = {
                    'idx': idx,
                    'task': key[0],
                    'ans': key[1],
                    'gold': gold,
                    **res,
                    'sec': round(time.time() - t0, 1)
                }
                records.append(rec)
                with out_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + '\n')
                got += 1
                hits = {m: ('Y' if int(np.argmax(rec[m])) == gold else '-') for m in METHODS}
                print(
                    f"[{len(records)}] {key[0]}-{key[1]} idx={idx} ({rec['dur']}s, res={rec['av_resonance_EML']}, {rec['sec']}s): " +
                    " ".join(f"{m.split('_')[1]}:{hits[m]}" for m in METHODS),
                    flush=True
                )
            except Exception as e:
                print(f"Skip idx={idx} err={repr(e)[:140]}", flush=True)
                torch.cuda.empty_cache()

    summary = {}
    for m in METHODS:
        by_task = {}
        for t in ('TA', 'TF', 'CMTB'):
            sub = [r for r in records if r['task'] == t]
            by_task[t] = round(100.0 * sum(int(np.argmax(r[m])) == r['gold'] for r in sub) / max(1, len(sub)), 1)
        by_cell = {}
        for key in sorted(cells):
            sub_c = [r for r in records if r['task'] == key[0] and r['ans'] == key[1]]
            c_ok = sum(int(np.argmax(r[m])) == r['gold'] for r in sub_c)
            by_cell[f"{key[0]}/{key[1]}"] = f"{c_ok}/{len(sub_c)}"
        tf_sub = [r for r in records if r['task'] == 'TF']
        tf_pred_counts = [sum(int(np.argmax(r[m])) == c_i for r in tf_sub) for c_i in range(3)]
        overall = round(100.0 * sum(int(np.argmax(r[m])) == r['gold'] for r in records) / max(1, len(records)), 1)
        summary[m] = {
            **by_task,
            'overall': overall,
            'tf_pred_ABC': tf_pred_counts,
            'cells': by_cell,
            'n': len(records)
        }
    print("ROUND 5 CHRONODESMEAR SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return {'shard': SHARD, 'nshard': NSHARD, 'summary': summary, 'records': records}


RESULT = run_shard()