SHARD = 0
NSHARD = 2
TARGET_PER_CELL_SHARD = 2  # 2 per shard * 2 shards = 4 per cell (32 samples total, <=50s)
MAX_DUR = 50.0
SEED = 2027

import os, sys, json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BS = chr(92)
METHODS = [
    'R3_M0_base',                  # Native Qwen2.5-Omni-3B baseline
    'R3_M1_true_text_pmi',         # Base - 0.75 * True Text-Only Content-Free Prior (fixes Round 2 Bug #2)
    'R3_M2_true_gate_audio',       # True float16 Additive Hard Event-Window Gate + De-Smearing + PMI (fixes Bug #1)
    'R3_M3_recurrent_L22_to_L10',  # Recurrent-Depth Latent Loop (feeds Layer-22 Latent State back into Layers 10..16)
    'R3_M4_manifold_cont_thought', # Logit-Lens Vocabulary-Manifold Continuous Thought (Coconut) + Value-Clock + PMI
    'R3_M5_latent_handoff',        # Recurrent Latent Loop + Continuous Thought + Layer-14 Audio-to-Visual Hand-Off
    'R3_M6_latent_chrono_unified', # Full Unified LatentChrono-CoT (Self-Calibrated Latent Reasoning)
]

_PHRASE_EMB_CACHE = {}

def get_phrase_emb(text):
    if text not in _PHRASE_EMB_CACHE:
        emb_w = MODEL_.get_input_embeddings().weight.detach()
        ids = PROC.tokenizer.encode(text, add_special_tokens=False)
        t_ids = torch.tensor(ids, dtype=torch.long, device=emb_w.device)
        vec = F.normalize(emb_w[t_ids].float().mean(0).cpu(), dim=-1)
        _PHRASE_EMB_CACHE[text] = vec
    return _PHRASE_EMB_CACHE[text]


def text_only_logprobs(stem, texts, perm):
    k = len(texts)
    conv = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': render(stem, texts, perm)}]},
    ]
    text = PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp_t = PROC(text=text, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad():
        lg_t = MODEL_(**inp_t).logits[0, -1].float().cpu()
    del inp_t
    return torch.log_softmax(lg_t[LETTERS[:k]], -1).numpy()


def hidden_to_manifold_thought_batch(h_mat, top_m=48, temp=0.65):
    with torch.no_grad():
        norm_mod = MODEL_.model.norm
        lm_head = MODEL_.lm_head
        dev_n = next(norm_mod.parameters()).device
        dt_n = next(norm_mod.parameters()).dtype
        h_in = h_mat.to(device=dev_n, dtype=dt_n)
        normed = norm_mod(h_in)
        logits = lm_head(normed.to(next(lm_head.parameters()).device)).float()

        emb_w = MODEL_.get_input_embeddings().weight.detach()
        vals, idxs = torch.topk(logits, k=min(top_m, logits.shape[-1]), dim=-1)
        probs = F.softmax(vals / temp, dim=-1).to(emb_w.device, dtype=torch.float32)
        top_embs = emb_w[idxs.to(emb_w.device)].float()
        c_vecs = (probs.unsqueeze(-1) * top_embs).sum(dim=1).cpu()
    return F.normalize(c_vecs, dim=-1)


def detect_active_window(raw_audio, dur):
    w = speech_window(raw_audio)
    if w is None or (w[1] - w[0]) < 0.4:
        return (round(0.25 * dur, 2), round(0.75 * dur, 2))
    return (max(0.0, round(w[0] - 0.30, 2)), min(dur, round(w[1] + 0.30, 2)))


def build_latent_aligned_value_clock(h_layer22, sec, is_v, is_a, last_media, win, dur, chunk_s=2.0):
    sec_c = sec.cpu().float()
    v_c = is_v.cpu()
    a_c = is_a.cpu()
    ch = torch.floor(sec_c / chunk_s).long().clamp(min=0)
    n_chunks = int(ch[v_c].max().item()) + 1 if v_c.any() else 1

    s0, s1 = win
    in_win_a = a_c & (sec_c >= s0) & (sec_c <= s1)
    if not in_win_a.any():
        in_win_a = a_c

    ha_all = h_layer22[a_c].mean(0) if a_c.any() else torch.zeros(h_layer22.shape[-1])
    ha_win = h_layer22[in_win_a].mean(0) - 0.5 * ha_all

    q_slice = h_layer22[last_media + 1:-1] if (last_media + 1 < h_layer22.shape[0] - 1) else h_layer22[last_media:]
    hq_vec = q_slice.mean(0)

    hv_mean = h_layer22[v_c].mean(0) if v_c.any() else torch.zeros(h_layer22.shape[-1])
    in_win_v = v_c & (sec_c >= s0) & (sec_c <= s1)
    hv_sync = (h_layer22[in_win_v].mean(0) - 0.6 * hv_mean) if in_win_v.any() else hv_mean

    hv_chunks = []
    for c in range(n_chunks):
        mv = v_c & (ch == c)
        hv_c = (h_layer22[mv].mean(0) - 0.7 * hv_mean) if mv.any() else torch.zeros_like(hv_mean)
        hv_chunks.append(hv_c)

    batch_h = torch.stack(hv_chunks + [ha_win, hq_vec, hv_sync], dim=0)
    batch_z = hidden_to_manifold_thought_batch(batch_h, top_m=48, temp=0.65)

    Z_V = batch_z[:n_chunks]
    z_a_win = batch_z[n_chunks]
    z_q = batch_z[n_chunks + 1]
    z_v_sync = batch_z[n_chunks + 2]

    z_target = F.normalize(z_a_win + 0.5 * z_q, dim=-1)
    sim_c = F.cosine_similarity(Z_V, z_target.unsqueeze(0).expand_as(Z_V), dim=-1)
    sim_z = (sim_c - sim_c.mean()) / (sim_c.std() + 1e-5) if n_chunks > 1 else torch.zeros_like(sim_c)
    chunk_salience = torch.sigmoid(1.5 * sim_z)

    cont_thought_vec = F.normalize(0.50 * z_v_sync + 0.50 * z_a_win, dim=-1)

    e_past_p = get_phrase_emb("Past before earlier first")
    e_pres_p = get_phrase_emb("Present during simultaneous at the same time Yes")
    e_fut_p  = get_phrase_emb("Future after later subsequently next")
    e_no_p   = get_phrase_emb("No not at the same time different time")

    chunk_vecs = []
    for c in range(n_chunks):
        t_mid = (c + 0.5) * chunk_s
        e_sec = get_phrase_emb(f"Time {int(c * chunk_s)} seconds")
        if t_mid < s0 - 0.15:
            e_zone = F.normalize(e_past_p + 0.35 * e_no_p, dim=-1)
        elif t_mid > s1 + 0.15:
            e_zone = F.normalize(e_fut_p + 0.35 * e_no_p, dim=-1)
        else:
            e_zone = e_pres_p
        sal = float(0.5 + chunk_salience[c].item())
        v_clock = sal * F.normalize(0.30 * e_sec + 0.45 * e_zone + 0.25 * Z_V[c], dim=-1)
        chunk_vecs.append(v_clock)

    C_mat = torch.stack(chunk_vecs, dim=0)
    C_mat = F.normalize(C_mat - C_mat.mean(0, keepdim=True), dim=-1)

    ch_clamped = ch.clamp(min=0, max=n_chunks - 1)
    token_clock = C_mat[ch_clamped].clone()
    token_clock[~v_c] = 0.0
    return token_clock, cont_thought_vec, chunk_salience


def compute_dynamic_visual_foreground(h_layer, is_v, sec, dur, chunk_salience=None, chunk_s=2.0):
    v_c = is_v.cpu()
    L = h_layer.shape[0]
    fg_bias = torch.zeros(L, dtype=torch.float32)
    if not v_c.any():
        return fg_bias
    hv = h_layer[v_c]
    hv_mean = hv.mean(0, keepdim=True)
    nov = (hv - hv_mean).norm(dim=-1)
    nov_z = ((nov - nov.mean()) / (nov.std() + 1e-5)).clamp(-1.5, 2.0)

    sec_v = sec.cpu().float()[v_c]
    pos_comp = 0.80 * (sec_v / max(1.0, dur) - 0.5)
    bias_v = 0.75 * nov_z + pos_comp

    if chunk_salience is not None:
        ch_v = torch.floor(sec_v / chunk_s).long().clamp(min=0, max=len(chunk_salience) - 1)
        sal_v = (chunk_salience[ch_v] - 0.5) * 2.2
        bias_v = bias_v + sal_v

    fg_bias[v_c] = bias_v
    return fg_bias


def make_additive_mask(
    L, is_v, is_a, sec, last_media, win, dtype, device,
    spg=None, desmear_win_s=3.0, desmear_pen=3.5,
    gate_mode='none', out_v_pen=3.0, fg_bias=None
):
    neg = torch.finfo(dtype).min
    m = torch.zeros((1, 1, L, L), dtype=dtype, device=device)
    m2d = m[0, 0]

    triu_idx = torch.triu_indices(L, L, offset=1, device=device)
    m2d[triu_idx[0], triu_idx[1]] = neg
    del triu_idx

    v = is_v.to(device)
    a = is_a.to(device)
    media = v | a
    sec_d = sec.to(device, dtype=torch.float32)
    rows_q = (torch.arange(L, device=device) > last_media)

    med_idx = torch.nonzero(media, as_tuple=False).squeeze(-1)
    if med_idx.numel() > 0 and desmear_pen > 0:
        m0, m1 = int(med_idx.min().item()), int(med_idx.max().item()) + 1
        sec_m = sec_d[m0:m1]
        v_m = v[m0:m1]
        a_m = a[m0:m1]
        dt_m = (sec_m.unsqueeze(1) - sec_m.unsqueeze(0)).abs()

        sub = m2d[m0:m1, m0:m1]
        sync_va = v_m.unsqueeze(1) & a_m.unsqueeze(0) & (dt_m <= 1.8)
        sub[sync_va] = 0.0

        far_m = (dt_m > desmear_win_s) & (sub > neg * 0.5)
        sub[far_m] = sub[far_m] - float(desmear_pen)
        del sec_m, v_m, a_m, dt_m, sync_va, far_m

    if fg_bias is not None:
        fg_d = fg_bias.to(device=device, dtype=dtype)
        q_idx = torch.nonzero(rows_q, as_tuple=False).squeeze(-1)
        v_idx = torch.nonzero(v, as_tuple=False).squeeze(-1)
        if q_idx.numel() > 0 and v_idx.numel() > 0:
            m2d[q_idx[:, None], v_idx[None, :]] += fg_d[v_idx][None, :]

    if gate_mode != 'none' and win is not None:
        s0, s1 = win
        width = float(spg) if spg is not None else 0.0
        inside = (sec_d + width > s0) & (sec_d < s1)
        in_v = v & inside
        out_v = v & (~inside)

        if gate_mode == 'soft':
            m2d[rows_q.unsqueeze(1) & out_v.unsqueeze(0)] -= float(out_v_pen)
        elif gate_mode == 'hard':
            m2d[(~out_v).unsqueeze(1) & out_v.unsqueeze(0)] = neg
            m2d[(~a).unsqueeze(1) & a.unsqueeze(0)] = neg
        elif gate_mode == 'anti':
            m2d[(~in_v).unsqueeze(1) & in_v.unsqueeze(0)] = neg

    return m


def forward_latent_pass(
    inp, pos, mask=None,
    handoff_layer=None, is_a=None, last_media=0,
    clock_vec=None, clock_layers=(14, 16, 18, 20, 22), clock_scale=0.16,
    cont_thought_vec=None, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.22,
    recurrent_delta=None, recurrent_layers={10: 0.28, 12: 0.35, 14: 0.35, 16: 0.28},
    record_layers=()
):
    emb = MODEL_.get_input_embeddings().weight
    neg = torch.finfo(emb.dtype).min
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    captured, hooks = {}, []
    layers = MODEL_.model.layers
    L = inp['input_ids'].shape[1]

    mask_late = None
    if handoff_layer is not None and mask is not None and is_a is not None:
        mask_late = mask.clone()
        rows_q = (torch.arange(L, device=mask_late.device) > last_media)
        a_d = is_a.to(mask_late.device)
        mask_late[0, 0][rows_q.unsqueeze(1) & a_d.unsqueeze(0)] = neg

        if handoff_layer - 1 >= 0:
            def mutate_mask_hook(mod, args, out):
                mask.copy_(mask_late)
                return out
            hooks.append(layers[handoff_layer - 1].register_forward_hook(mutate_mask_hook))

    for li in range(len(layers)):
        layer = layers[li]

        if mask_late is not None and li >= handoff_layer:
            def make_handoff_pre(li_curr):
                def pre_h(mod, args, kwargs):
                    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                        am = kwargs['attention_mask']
                        kwargs['attention_mask'] = mask_late.to(device=am.device, dtype=am.dtype)
                    elif len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].ndim == 4:
                        am = args[1]
                        args = (args[0], mask_late.to(device=am.device, dtype=am.dtype)) + args[2:]
                    return args, kwargs
                return pre_h
            hooks.append(layer.register_forward_pre_hook(make_handoff_pre(li), with_kwargs=True))

        if cont_thought_vec is not None and li in cont_thought_layers:
            def make_ct_hook(li_curr):
                def ct_hook(mod, args, out):
                    h = out[0].clone()
                    hq = h[0, last_media + 1:].float()
                    ct = cont_thought_vec.to(h.device, torch.float32).unsqueeze(0)
                    avg_n = hq.norm(dim=-1, keepdim=True).mean().clamp(min=1e-5)
                    h[0, last_media + 1:] = (hq + (cont_thought_scale * avg_n) * ct).to(h.dtype)
                    return (h,) + tuple(out[1:])
                return ct_hook
            hooks.append(layer.register_forward_hook(make_ct_hook(li)))

        if clock_vec is not None and li in clock_layers:
            def make_clock_hook(li_curr):
                def c_hook(mod, args, out):
                    h = out[0].clone()
                    cv = clock_vec.to(h.device, torch.float32)
                    h_f = h[0].float()
                    active_m = (cv.norm(dim=-1) > 0)
                    if active_m.any():
                        avg_norm = h_f[active_m].norm(dim=-1, keepdim=True).mean().clamp(min=1e-5)
                        h_f = h_f + (clock_scale * avg_norm) * cv
                        h[0] = h_f.to(h.dtype)
                    return (h,) + tuple(out[1:])
                return c_hook
            hooks.append(layer.register_forward_hook(make_clock_hook(li)))

        if recurrent_delta is not None and li in recurrent_layers:
            def make_rec_loop_hook(li_curr):
                def rl_hook(mod, args, out):
                    h = out[0].clone()
                    hq = h[0, last_media + 1:].float()
                    dq = recurrent_delta.to(h.device, torch.float32)
                    beta = float(recurrent_layers[li_curr])
                    norm_h = hq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    norm_d = dq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    h[0, last_media + 1:] = (hq + beta * (norm_h / norm_d) * dq).to(h.dtype)
                    return (h,) + tuple(out[1:])
                return rl_hook
            hooks.append(layer.register_forward_hook(make_rec_loop_hook(li)))

        if li in record_layers:
            def make_rec_hook(li_curr):
                def r_hook(mod, args, out):
                    captured[li_curr] = out[0][0].detach().float().cpu()
                    return out
                return r_hook
            hooks.append(layer.register_forward_hook(make_rec_hook(li)))

    try:
        with torch.no_grad():
            if mask is None:
                lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float().cpu()
            else:
                lg = MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float().cpu()
    finally:
        for h in hooks:
            h.remove()
        if mask_late is not None:
            del mask_late
    return lg, captured


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

    for perm in rotations(k):
        lp_text_prior = text_only_logprobs(stem, texts, perm)

        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]
        spg = inp.get('video_second_per_grid')
        spg = float(spg.reshape(-1)[0]) if spg is not None else None

        # LOOP 1: LATENT PERCEPTION & VOCABULARY-MANIFOLD CONTINUOUS THOUGHT
        lg_base, cap_base = forward_latent_pass(inp, pos, mask=None, record_layers=(16, 22))
        lp_base = torch.log_softmax(lg_base[LETTERS[:k]], -1).numpy()

        clock_vec, cont_thought_vec, chunk_salience = build_latent_aligned_value_clock(
            cap_base[22], sec, is_v, is_a, last_media, win, dur
        )
        fg_bias = compute_dynamic_visual_foreground(cap_base[16], is_v, sec, dur, chunk_salience=chunk_salience)
        h_base_q22 = cap_base[22][last_media + 1:]
        del cap_base

        mask_hard = make_additive_mask(
            L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device,
            spg=spg, desmear_win_s=3.0, desmear_pen=3.5, gate_mode='hard', fg_bias=fg_bias
        )
        lg_hard, cap_sync = forward_latent_pass(
            inp, pos, mask=mask_hard, clock_vec=clock_vec, record_layers=(22,)
        )
        lp_hard = torch.log_softmax(lg_hard[LETTERS[:k]], -1).numpy()
        h_sync_q22 = cap_sync[22][last_media + 1:]
        del mask_hard, cap_sync

        mask_anti = make_additive_mask(
            L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device,
            spg=spg, desmear_win_s=0.0, desmear_pen=0.0, gate_mode='anti'
        )
        lg_anti, cap_anti = forward_latent_pass(inp, pos, mask=mask_anti, record_layers=(22,))
        h_anti_q22 = cap_anti[22][last_media + 1:]
        del mask_anti, cap_anti

        recurrent_delta_q22 = 0.5 * (h_base_q22 + h_sync_q22) - h_anti_q22

        # LOOP 2: LATENT-GUIDED RECURRENT GROUNDING & TEMPORAL VERIFICATION
        mask_desmear = make_additive_mask(
            L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device,
            spg=spg, desmear_win_s=3.0, desmear_pen=3.5, gate_mode='none', fg_bias=fg_bias
        )
        lg_rec, _ = forward_latent_pass(
            inp, pos, mask=mask_desmear, last_media=last_media,
            clock_vec=clock_vec,
            recurrent_delta=recurrent_delta_q22,
            recurrent_layers={10: 0.28, 12: 0.35, 14: 0.35, 16: 0.28}
        )
        lp_rec = torch.log_softmax(lg_rec[LETTERS[:k]], -1).numpy()

        lg_coconut, _ = forward_latent_pass(
            inp, pos, mask=mask_desmear, last_media=last_media,
            clock_vec=clock_vec,
            cont_thought_vec=cont_thought_vec, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.22
        )
        lp_coconut = torch.log_softmax(lg_coconut[LETTERS[:k]], -1).numpy()
        del mask_desmear

        mask_soft = make_additive_mask(
            L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device,
            spg=spg, desmear_win_s=3.0, desmear_pen=3.5, gate_mode='soft', out_v_pen=2.8, fg_bias=fg_bias
        )
        lg_handoff, _ = forward_latent_pass(
            inp, pos, mask=mask_soft,
            handoff_layer=14, is_a=is_a, last_media=last_media,
            clock_vec=clock_vec,
            cont_thought_vec=cont_thought_vec, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.20,
            recurrent_delta=recurrent_delta_q22,
            recurrent_layers={10: 0.28, 12: 0.35, 14: 0.35, 16: 0.28}
        )
        lp_handoff = torch.log_softmax(lg_handoff[LETTERS[:k]], -1).numpy()
        del mask_soft

        lp_m1 = lp_base - 0.75 * lp_text_prior
        lp_m2 = lp_hard - 0.45 * lp_text_prior
        lp_m3 = lp_rec - 0.75 * lp_text_prior
        lp_m4 = lp_coconut - 0.75 * lp_text_prior
        lp_m5 = lp_handoff - 0.70 * lp_text_prior

        prior_skew = float(np.max(lp_text_prior) - np.min(lp_text_prior))
        w_rel = float(np.clip((prior_skew - 0.30) / 0.55, 0.15, 0.85))
        lp_m6 = w_rel * (0.45 * lp_m5 + 0.30 * lp_m3 + 0.25 * lp_m4) + (1.0 - w_rel) * (0.60 * lp_m2 + 0.40 * lp_m5)

        scores_perm = {
            'R3_M0_base': lp_base,
            'R3_M1_true_text_pmi': lp_m1,
            'R3_M2_true_gate_audio': lp_m2,
            'R3_M3_recurrent_L22_to_L10': lp_m3,
            'R3_M4_manifold_cont_thought': lp_m4,
            'R3_M5_latent_handoff': lp_m5,
            'R3_M6_latent_chrono_unified': lp_m6,
        }
        for m, lp in scores_perm.items():
            for pos_idx, opt_idx in enumerate(perm):
                totals[m][opt_idx] += lp[pos_idx]

        del inp
        torch.cuda.empty_cache()

    return {
        'dur': round(dur, 2),
        'win': [win[0], win[1]],
        **{m: totals[m].round(3).tolist() for m in METHODS}
    }


def run_shard():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        if r.get('data_type') == 'temporal':
            cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))

    out_file = OUT / f'round3_latent_s{SHARD}of{NSHARD}.jsonl'
    if out_file.exists():
        out_file.unlink()

    records = []
    for key in sorted(cells):
        pool = list(cells[key])
        random.Random(SEED).shuffle(pool)
        my_slice = pool[SHARD::NSHARD]
        got = 0
        for idx, r in my_slice:
            if got >= TARGET_PER_CELL_SHARD:
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
                    f"[{len(records)}] {key[0]}-{key[1]} idx={idx} ({rec['dur']}s, win={rec['win']}, {rec['sec']}s): " +
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
        overall = round(100.0 * sum(int(np.argmax(r[m])) == r['gold'] for r in records) / max(1, len(records)), 1)
        summary[m] = {
            **by_task,
            'overall': overall,
            'delta_vs_base': round(overall - summary['R3_M0_base']['overall'], 1) if 'R3_M0_base' in summary else 0.0,
            'n': len(records)
        }
    print("ROUND 3 LATENT REASONING SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return {'shard': SHARD, 'nshard': NSHARD, 'summary': summary, 'records': records}


RESULT = run_shard()
