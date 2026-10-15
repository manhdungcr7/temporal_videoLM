"""
ROUND 4 — CVPR 2027: Counterfactual Cross-Modal Latent Deliberation (CCLD) & Orthogonal Ablation Suite
Authors: Yoshua Bengio & Yann LeCun

8 Orthogonal Configurations (All with `clock_vec = None` and `PMI = None`):
  - R4_C0_baseline          : 1-Pass Native Qwen2.5-Omni-3B.
  - R4_C1_m9_thought_ctrl   : Exact M9 reproduction (`ct_m9` at L0..2, `mask=None`).
  - R4_C2_m8_noclk_nopmi    : Exact M8 (`0.5*(h_base + h_sync) - h_anti` at L10..16, `mask=None`) with
                              `clock_vec` and `PMI` REMOVED (tests how much of M8's TF collapse was clock_vec vs h_anti).
  - R4_C3_causal_vis_rec    : 100% Window-Free, VAD-Free Causal Visual Recurrence:
                              `Delta_V = H_Q^(22)(V,A,Q) - H_Q^(22)(None_V,A,Q)` injected at L10..18 (`mask=None`).
  - R4_C4_shuffled_rec_ctrl : Yann LeCun's Shuffled-Vector Control for C3 (permuted dimensions, exact same L2 norm).
  - R4_C5_m9_plus_causal_rec: 100% Window-Free Unified (`ct_global` at L0..2 + `Delta_V` at L10..18, `mask=None`).
  - R4_C6_ccld_unmasked     : Full CCLD on Unmasked Pass 2 (`ct_global` at L0..2 +
                              `0.55*(h_base - h_aud_only) + 0.45*(h_sync - h_aud_only)` at L10..18, `mask=None`).
  - R4_C7_ccld_late_handoff : Same as C6 + Soft Late-Layer Audio-to-Question Damping (`-2.5` at Layers >= 16).
"""

SHARD = 0
NSHARD = 2
PER_CELL_TOTAL = 5  # 5 per cell * 8 cells = 40 samples total
MAX_DUR = 50.0
SEED = 2027

import os, sys, json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

BS = chr(92)
METHODS = [
    'R4_C0_baseline',
    'R4_C1_m9_thought_ctrl',
    'R4_C2_m8_noclk_nopmi',
    'R4_C3_causal_vis_rec',
    'R4_C4_shuffled_rec_ctrl',
    'R4_C5_m9_plus_causal_rec',
    'R4_C6_ccld_unmasked',
    'R4_C7_ccld_late_handoff',
]


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


def extract_continuous_thoughts(h_layer22, sec, is_v, is_a, last_media, win):
    sec_c = sec.cpu().float()
    v_c = is_v.cpu()
    a_c = is_a.cpu()
    s0, s1 = win

    in_win_a = a_c & (sec_c >= s0) & (sec_c <= s1)
    if not in_win_a.any():
        in_win_a = a_c
    ha_all = h_layer22[a_c].mean(0) if a_c.any() else torch.zeros(h_layer22.shape[-1])
    ha_win = h_layer22[in_win_a].mean(0) - 0.5 * ha_all

    hv_mean = h_layer22[v_c].mean(0) if v_c.any() else torch.zeros(h_layer22.shape[-1])
    in_win_v = v_c & (sec_c >= s0) & (sec_c <= s1)
    hv_sync = (h_layer22[in_win_v].mean(0) - 0.6 * hv_mean) if in_win_v.any() else hv_mean

    # 100% Window-Free Global Dynamic Visual & Audio Foreground
    hv_all = h_layer22[v_c]
    nov = (hv_all - hv_mean.unsqueeze(0)).norm(dim=-1)
    top_k = max(1, min(160, nov.numel() // 4))
    top_v_idx = torch.topk(nov, k=top_k).indices
    hv_dyn_global = hv_all[top_v_idx].mean(0) - 0.5 * hv_mean

    ha_tokens = h_layer22[a_c]
    nov_a = (ha_tokens - ha_all.unsqueeze(0)).norm(dim=-1)
    top_ka = max(1, min(48, nov_a.numel() // 3))
    top_a_idx = torch.topk(nov_a, k=top_ka).indices
    ha_dyn_global = ha_tokens[top_a_idx].mean(0) - 0.5 * ha_all

    batch_h = torch.stack([ha_win, hv_sync, hv_dyn_global, ha_dyn_global], dim=0)
    batch_z = hidden_to_manifold_thought_batch(batch_h, top_m=48, temp=0.65)
    z_a_win, z_v_sync, z_v_global, z_a_global = batch_z[0], batch_z[1], batch_z[2], batch_z[3]

    ct_m9 = F.normalize(0.50 * z_v_sync + 0.50 * z_a_win, dim=-1)
    ct_global = F.normalize(0.50 * z_v_global + 0.50 * z_a_global, dim=-1)
    return ct_m9, ct_global


def make_clean_mask(L, is_v, is_a, sec, last_media, win, dtype, device, spg=None,
                    gate_mode='none', aud_pen=2.5):
    neg = torch.finfo(dtype).min
    m = torch.zeros((1, 1, L, L), dtype=dtype, device=device)
    m2d = m[0, 0]

    triu_idx = torch.triu_indices(L, L, offset=1, device=device)
    m2d[triu_idx[0], triu_idx[1]] = neg
    del triu_idx

    v = is_v.to(device)
    a = is_a.to(device)
    sec_d = sec.to(device, dtype=torch.float32)

    if gate_mode == 'audio_only':
        # Visual-Blind Counterfactual: non-visual tokens cannot attend to visual tokens (0 window dependency!)
        m2d[(~v).unsqueeze(1) & v.unsqueeze(0)] = neg
    elif gate_mode in ('hard_sync', 'anti_sync') and win is not None:
        s0, s1 = win
        width = float(spg) if spg is not None else 0.0
        inside = (sec_d + width > s0) & (sec_d < s1)
        in_v = v & inside
        out_v = v & (~inside)
        if gate_mode == 'hard_sync':
            m2d[(~out_v).unsqueeze(1) & out_v.unsqueeze(0)] = neg
            m2d[(~a).unsqueeze(1) & a.unsqueeze(0)] = neg
        elif gate_mode == 'anti_sync':
            m2d[(~in_v).unsqueeze(1) & in_v.unsqueeze(0)] = neg

    return m


def forward_ccld(
    inp, pos, mask=None, last_media=0, is_a=None,
    cont_thought_vec=None, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.22,
    recurrent_delta=None, recurrent_layers={10: 0.25, 12: 0.30, 14: 0.30, 16: 0.25, 18: 0.20},
    late_audio_damp_layer=None, late_audio_pen=2.5,
    record_layers=()
):
    emb = MODEL_.get_input_embeddings().weight
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    captured, hooks = {}, []
    layers = MODEL_.model.layers
    L = inp['input_ids'].shape[1]

    mask_late = None
    created_mask = False
    if late_audio_damp_layer is not None and is_a is not None:
        if mask is None:
            mask = make_clean_mask(L, is_a, is_a, torch.zeros(L), last_media, None, emb.dtype, emb.device, gate_mode='none')
            created_mask = True
        mask_late = mask.clone()
        rows_q = (torch.arange(L, device=mask_late.device) > last_media)
        a_d = is_a.to(mask_late.device)
        mask_late[0, 0][rows_q.unsqueeze(1) & a_d.unsqueeze(0)] -= float(late_audio_pen)
        if late_audio_damp_layer - 1 >= 0:
            def mutate_mask_hook(mod, args, out):
                mask.copy_(mask_late)
                return out
            hooks.append(layers[late_audio_damp_layer - 1].register_forward_hook(mutate_mask_hook))

    for li in range(len(layers)):
        layer = layers[li]

        if mask_late is not None and li >= late_audio_damp_layer:
            def make_late_pre(li_curr):
                def pre_h(mod, args, kwargs):
                    if 'attention_mask' in kwargs:
                        am = kwargs['attention_mask']
                        dev = am.device if am is not None else args[0].device
                        kwargs['attention_mask'] = mask_late.to(device=dev, dtype=emb.dtype)
                    elif len(args) >= 2:
                        dev = args[0].device
                        args = (args[0], mask_late.to(device=dev, dtype=emb.dtype)) + args[2:]
                    return args, kwargs
                return pre_h
            hooks.append(layer.register_forward_pre_hook(make_late_pre(li), with_kwargs=True))

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

        if recurrent_delta is not None and li in recurrent_layers:
            def make_rec_hook_fn(li_curr):
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
        if mask_late is not None:
            del mask_late
        if created_mask and mask is not None:
            del mask
    return lg, captured


def make_shuffled_control(delta_tensor, seed=42):
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

    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        L = inp['input_ids'].shape[1]
        spg = inp.get('video_second_per_grid')
        spg = float(spg.reshape(-1)[0]) if spg is not None else None

        # ======================================================================
        # PASS 1: UN-CLOCKED LATENT STATE EXTRACTION (ALL `clock=None, pmi=None`)
        # ======================================================================
        lg_c0, cap_base = forward_ccld(inp, pos, mask=None, record_layers=(22,))
        lp_c0 = torch.log_softmax(lg_c0[LETTERS[:k]], -1).numpy()
        h_base_22 = cap_base[22]
        h_base_q22 = h_base_22[last_media + 1:]
        del cap_base

        ct_m9, ct_global = extract_continuous_thoughts(h_base_22, sec, is_v, is_a, last_media, win)
        del h_base_22

        # Visual-Blind (Audio-Only) Counterfactual Pass (`m[~v, v] = -inf`, 100% Window-Free!)
        mask_aud_only = make_clean_mask(L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device, spg=spg, gate_mode='audio_only')
        _, cap_aud_only = forward_ccld(inp, pos, mask=mask_aud_only, record_layers=(22,))
        h_aud_only_q22 = cap_aud_only[22][last_media + 1:]
        del mask_aud_only, cap_aud_only

        # Clean Synchronous Visual Pass (`hard_sync`, STRICTLY `clock_vec=None`!)
        mask_hard = make_clean_mask(L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device, spg=spg, gate_mode='hard_sync')
        _, cap_sync = forward_ccld(inp, pos, mask=mask_hard, record_layers=(22,))
        h_sync_q22 = cap_sync[22][last_media + 1:]
        del mask_hard, cap_sync

        # Anti-Sync Pass (`anti_sync`, STRICTLY `clock_vec=None` — used ONLY for C2 M8-no-clock control)
        mask_anti = make_clean_mask(L, is_v, is_a, sec, last_media, win, emb.dtype, emb.device, spg=spg, gate_mode='anti_sync')
        _, cap_anti = forward_ccld(inp, pos, mask=mask_anti, record_layers=(22,))
        h_anti_q22 = cap_anti[22][last_media + 1:]
        del mask_anti, cap_anti

        # Construct Orthogonal Recurrent Latent Deltas:
        delta_m8_noclk = 0.5 * (h_base_q22 + h_sync_q22) - h_anti_q22
        delta_causal_vis = h_base_q22 - h_aud_only_q22
        delta_shuffled = make_shuffled_control(delta_causal_vis)
        delta_ccld = 0.55 * (h_base_q22 - h_aud_only_q22) + 0.45 * (h_sync_q22 - h_aud_only_q22)

        # ======================================================================
        # PASS 2: ORTHOGONAL ABLATION & LATENT DELIBERATION (ALL `clock=None, pmi=None`)
        # ======================================================================
        # C1: Exact M9 Control (`ct_m9` at L0..2, mask=None)
        lg_c1, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            cont_thought_vec=ct_m9, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.22
        )
        lp_c1 = torch.log_softmax(lg_c1[LETTERS[:k]], -1).numpy()

        # C2: M8 Without Clock & Without PMI (`delta_m8_noclk` at L10..16, mask=None)
        lg_c2, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            recurrent_delta=delta_m8_noclk, recurrent_layers={10: 0.28, 12: 0.35, 14: 0.35, 16: 0.28}
        )
        lp_c2 = torch.log_softmax(lg_c2[LETTERS[:k]], -1).numpy()

        # C3: 100% Window-Free Causal Visual Recurrence (`delta_causal_vis` at L10..18, mask=None)
        lg_c3, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            recurrent_delta=delta_causal_vis, recurrent_layers={10: 0.25, 12: 0.30, 14: 0.30, 16: 0.25, 18: 0.20}
        )
        lp_c3 = torch.log_softmax(lg_c3[LETTERS[:k]], -1).numpy()

        # C4: Shuffled Vector Control (`delta_shuffled` at L10..18, mask=None)
        lg_c4, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            recurrent_delta=delta_shuffled, recurrent_layers={10: 0.25, 12: 0.30, 14: 0.30, 16: 0.25, 18: 0.20}
        )
        lp_c4 = torch.log_softmax(lg_c4[LETTERS[:k]], -1).numpy()

        # C5: 100% Window-Free Unified (`ct_global` at L0..2 + `delta_causal_vis` at L10..18, mask=None)
        lg_c5, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            cont_thought_vec=ct_global, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.20,
            recurrent_delta=delta_causal_vis, recurrent_layers={10: 0.25, 12: 0.30, 14: 0.30, 16: 0.25, 18: 0.20}
        )
        lp_c5 = torch.log_softmax(lg_c5[LETTERS[:k]], -1).numpy()

        # C6: Full CCLD Unmasked (`ct_global` at L0..2 + `delta_ccld` at L10..18, mask=None)
        lg_c6, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media,
            cont_thought_vec=ct_global, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.20,
            recurrent_delta=delta_ccld, recurrent_layers={10: 0.25, 12: 0.32, 14: 0.32, 16: 0.25, 18: 0.20}
        )
        lp_c6 = torch.log_softmax(lg_c6[LETTERS[:k]], -1).numpy()

        # C7: Full CCLD + Soft Late-Layer Audio Damping (`late_audio_damp_layer=16`, 100% Visual Timeline Unmasked!)
        lg_c7, _ = forward_ccld(
            inp, pos, mask=None, last_media=last_media, is_a=is_a,
            cont_thought_vec=ct_global, cont_thought_layers=(0, 1, 2), cont_thought_scale=0.20,
            recurrent_delta=delta_ccld, recurrent_layers={10: 0.25, 12: 0.32, 14: 0.32, 16: 0.25, 18: 0.20},
            late_audio_damp_layer=16, late_audio_pen=2.5
        )
        lp_c7 = torch.log_softmax(lg_c7[LETTERS[:k]], -1).numpy()

        scores_perm = {
            'R4_C0_baseline': lp_c0,
            'R4_C1_m9_thought_ctrl': lp_c1,
            'R4_C2_m8_noclk_nopmi': lp_c2,
            'R4_C3_causal_vis_rec': lp_c3,
            'R4_C4_shuffled_rec_ctrl': lp_c4,
            'R4_C5_m9_plus_causal_rec': lp_c5,
            'R4_C6_ccld_unmasked': lp_c6,
            'R4_C7_ccld_late_handoff': lp_c7,
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

    out_file = OUT / f'round4_ccld_s{SHARD}of{NSHARD}.jsonl'
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
                    f"[{len(records)}] {key[0]}-{key[1]} idx={idx} ({rec['dur']}s, {rec['sec']}s): " +
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
    print("ROUND 4 CCLD ABLATION SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return {'shard': SHARD, 'nshard': NSHARD, 'summary': summary, 'records': records}


RESULT = run_shard()

