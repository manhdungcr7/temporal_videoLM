# Round 2: LSTR-Relay on Server 2 (SHARD=1)
SHARD = 1
NSHARD = 2
MAX_DUR = 50.0
SEED = 2027

import os, sys, json, time, random, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io

BS = chr(92)
METHODS = [
    'C0_base',                  # Native Qwen2.5-Omni-3B baseline
    'C1_pmi_prior',             # Base - 0.85 * Content-Free Media-Masked Prior
    'C2_sync_deconfound',       # O(L*D) Active-Window Key Spotlight + Late-Layer Audio Damping
    'C3_vclock_pmi',            # Native Embedding/Value-Space Clock Injection + PMI Calibration
    'C4_coop_relay_L18_22',     # Multi-Layer Cooperative Question-Relay Steering (Layers 18, 20, 22)
    'C5_coop_relay_pmi',        # C4 Cooperative Relay Steering + Value-Clock + PMI Prior Calibration
    'C6_lstr_unified',          # Full Unified LSTR-Relay
]

# Torchvision fallback for missing read_video
def _read_video_cv2(filename, **kwargs):
    import cv2
    cap = cv2.VideoCapture(str(filename))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"cv2 could not read frames from {filename}")
    vframes = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    return vframes, torch.empty(0), {'video_fps': fps}
torchvision.io.read_video = _read_video_cv2

LETTERS = [PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD']
render = prompt_of

_PHRASE_EMB_CACHE = {}

def get_phrase_emb(text):
    if text not in _PHRASE_EMB_CACHE:
        emb_w = MODEL_.get_input_embeddings().weight.detach()
        ids = PROC.tokenizer.encode(text, add_special_tokens=False)
        t_ids = torch.tensor(ids, dtype=torch.long, device=emb_w.device)
        vec = F.normalize(emb_w[t_ids].float().mean(0).cpu(), dim=-1)
        _PHRASE_EMB_CACHE[text] = vec
    return _PHRASE_EMB_CACHE[text]


def analyze_temporal_streams(sec, is_v, is_a, raw_audio, chunk_s=2.0):
    sec_c = sec.cpu().float()
    v_c = is_v.cpu()
    a_c = is_a.cpu()
    media_c = v_c | a_c
    ch = torch.floor(sec_c / chunk_s).long().clamp(min=0)
    n_chunks = int(ch[media_c].max().item()) + 1 if media_c.any() else 1

    sr = 16000
    rms = np.zeros(n_chunks, dtype=np.float32)
    for c in range(n_chunks):
        s0, s1 = int(c * chunk_s * sr), int(min(len(raw_audio), (c + 1) * chunk_s * sr))
        if s1 > s0:
            seg = raw_audio[s0:s1]
            rms[c] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    db = 20.0 * np.log10(rms + 1e-9)
    if n_chunks > 1 and (np.max(db) - np.min(db)) > 1.5:
        lo, hi = np.percentile(db, 15), np.percentile(db, 90)
        thresh = lo + 0.45 * max(1e-3, hi - lo)
        w_aud = np.clip((db - thresh) / max(1.5, 0.25 * (hi - lo)), -1.0, 1.0)
    else:
        w_aud = np.zeros(n_chunks, dtype=np.float32)

    w_aud_t = torch.tensor(w_aud, dtype=torch.float32)
    p_aud = F.softmax(2.5 * w_aud_t, dim=0)
    t_idx = torch.arange(n_chunks, dtype=torch.float32)
    t_anchor = float((p_aud * t_idx).sum().item())
    return ch, n_chunks, w_aud_t, p_aud, t_anchor


def build_value_clock_and_displacement(h_mid, ch, n_chunks, is_v, is_a, last_media, w_aud_t, p_aud, t_anchor):
    v_c = is_v.cpu()
    a_c = is_a.cpu()
    media_c = v_c | a_c

    q_slice = h_mid[last_media + 1:-1] if (last_media + 1 < h_mid.shape[0] - 1) else h_mid[last_media:]
    hq = F.normalize(q_slice.mean(0, keepdim=True), dim=-1)

    v_vecs, a_vecs = [], []
    for c in range(n_chunks):
        mv = v_c & (ch == c)
        ma = a_c & (ch == c)
        v_vecs.append(h_mid[mv].mean(0) if mv.any() else h_mid[v_c].mean(0))
        a_vecs.append(h_mid[ma].mean(0) if ma.any() else h_mid[a_c].mean(0))
    V = torch.stack(v_vecs)
    A = torch.stack(a_vecs)

    V_centered = F.normalize(V - V.mean(0, keepdim=True), dim=-1)
    A_centered = F.normalize(A - A.mean(0, keepdim=True), dim=-1)

    a_anchor_vec = F.normalize((p_aud.unsqueeze(1) * A_centered).sum(0, keepdim=True) + 0.5 * hq, dim=-1)
    v_match = F.cosine_similarity(V_centered, a_anchor_vec.expand_as(V_centered), dim=-1)
    p_vis = F.softmax(4.0 * v_match, dim=0)
    t_idx = torch.arange(n_chunks, dtype=torch.float32)
    t_vis = float((p_vis * t_idx).sum().item())

    active_mask = (w_aud_t > 0.0).float()
    if active_mask.sum() == 0:
        active_mask[int(round(t_anchor))] = 1.0
    overlap_mass = float((p_vis * active_mask).sum().item())
    dt = t_vis - t_anchor

    e_yes    = get_phrase_emb("Yes same time simultaneous during present")
    e_no     = get_phrase_emb("No different time not simultaneous")
    e_past   = get_phrase_emb("Past before earlier previously")
    e_pres   = get_phrase_emb("Present during simultaneous at the same time")
    e_future = get_phrase_emb("Future after later subsequently")

    chunk_clocks = []
    for c in range(n_chunks):
        e_sec = get_phrase_emb(f"Time {int(c * 2)} seconds")
        offset = c - t_anchor
        if abs(offset) <= 0.85:
            e_rel = e_pres + 0.5 * e_yes
        elif offset < -0.85:
            e_rel = e_past + 0.3 * e_no
        else:
            e_rel = e_future + 0.3 * e_no
        chunk_clocks.append(F.normalize(0.4 * e_sec + 0.6 * e_rel, dim=-1))

    C_mat = torch.stack(chunk_clocks, dim=0)
    C_mat = F.normalize(C_mat - C_mat.mean(0, keepdim=True), dim=-1)

    ch_clamped = ch.clamp(min=0, max=n_chunks - 1)
    token_clock = C_mat[ch_clamped]
    token_clock[~media_c] = 0.0

    if overlap_mass >= 0.42 or abs(dt) <= 0.85:
        q_clock_hint = F.normalize(e_pres + e_yes, dim=-1)
    elif dt < -0.85:
        q_clock_hint = F.normalize(e_past + e_no, dim=-1)
    else:
        q_clock_hint = F.normalize(e_future + e_no, dim=-1)
    token_clock[last_media + 1:] = 0.65 * q_clock_hint.unsqueeze(0)

    return token_clock, overlap_mass, dt, v_match


def run_hooked_forward(
    inp, is_v, is_a, ch, last_media,
    w_aud=None, mode='base',
    clock_vec=None, clock_layers=(0, 1, 2), clock_scale=0.18,
    record_layers=(),
    steer_dict=None, steer_betas={18: 0.35, 20: 0.50, 22: 0.35}
):
    captured = {}
    hooks = []
    layers = MODEL_.model.layers
    v_mask = is_v.cpu()
    a_mask = is_a.cpu()
    media_mask = v_mask | a_mask
    L = inp['input_ids'].shape[1]

    token_k_scale = None
    if w_aud is not None and mode in ('pos_sync', 'neg_desync', 'vclock_all'):
        ch_c = ch.clamp(min=0, max=len(w_aud) - 1)
        w_tok = w_aud[ch_c]
        scale = torch.ones(L, dtype=torch.float32)
        if mode == 'pos_sync':
            scale[v_mask] = torch.exp(1.15 * w_tok[v_mask])
            scale[a_mask] = torch.exp(0.20 * w_tok[a_mask])
        elif mode == 'neg_desync':
            scale[v_mask] = torch.exp(-1.15 * w_tok[v_mask])
            scale[a_mask] = 1.35
        elif mode == 'vclock_all':
            scale[v_mask] = torch.exp(0.35 * w_tok[v_mask])
            scale[a_mask] = 1.20
        token_k_scale = scale

    for li in range(len(layers)):
        layer = layers[li]

        if mode == 'null_media' and li >= 8:
            def make_null_k(li_curr):
                def nk_hook(mod, args, out):
                    k_out = out.clone()
                    mm = media_mask.to(k_out.device)
                    if k_out.ndim == 3:
                        k_out[:, mm, :] = 0.0
                    else:
                        k_out[mm, :] = 0.0
                    return k_out
                return nk_hook
            hooks.append(layer.self_attn.k_proj.register_forward_hook(make_null_k(li)))
        elif token_k_scale is not None and 10 <= li <= 32:
            def make_k_hook(li_curr):
                def k_hook(mod, args, out):
                    sc = token_k_scale.to(out.device, out.dtype)
                    k_out = out * (sc.view(1, -1, 1) if out.ndim == 3 else sc.view(-1, 1))
                    am = a_mask.to(k_out.device)
                    if mode == 'pos_sync' and li_curr >= 18:
                        if k_out.ndim == 3:
                            k_out[:, am, :] *= 0.15
                        else:
                            k_out[am, :] *= 0.15
                    elif mode == 'neg_desync' and li_curr >= 18:
                        if k_out.ndim == 3:
                            k_out[:, am, :] *= 1.40
                        else:
                            k_out[am, :] *= 1.40
                    return k_out
                return k_hook
            hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))

        if (mode == 'null_media' and li >= 8) or (clock_vec is not None and li in clock_layers):
            def make_v_hook(li_curr):
                def v_hook(mod, args, out):
                    h = out[0].clone()
                    if mode == 'null_media' and li_curr >= 8:
                        h[:, media_mask.to(h.device), :] = 0.0
                    elif clock_vec is not None and li_curr in clock_layers:
                        cv = clock_vec.to(h.device, torch.float32)
                        h_f = h[0].float()
                        active_m = (cv.norm(dim=-1) > 0)
                        if active_m.any():
                            avg_norm = h_f[active_m].norm(dim=-1, keepdim=True).mean().clamp(min=1e-5)
                            h_f = h_f + (clock_scale * avg_norm) * cv
                            h[0] = h_f.to(h.dtype)
                    return (h,) + tuple(out[1:])
                return v_hook
            hooks.append(layer.register_forward_hook(make_v_hook(li)))

        if li in record_layers:
            def make_rec(li_curr):
                def r_hook(mod, args, out):
                    captured[li_curr] = out[0][0].detach().float().cpu()
                    return out
                return r_hook
            hooks.append(layer.register_forward_hook(make_rec(li)))

        if steer_dict is not None and li in steer_dict and li in steer_betas:
            def make_steer(li_curr):
                def s_hook(mod, args, out):
                    h = out[0].clone()
                    hq = h[0, last_media + 1:].float()
                    dq = steer_dict[li_curr].to(h.device, torch.float32)
                    beta = float(steer_betas[li_curr])
                    norm_h = hq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    norm_d = dq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    h[0, last_media + 1:] = (hq + beta * (norm_h / norm_d) * dq).to(h.dtype)
                    return (h,) + tuple(out[1:])
                return s_hook
            hooks.append(layer.register_forward_hook(make_steer(li)))

    try:
        with torch.no_grad():
            lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float().cpu()
    finally:
        for h in hooks:
            h.remove()
    return lg, captured


def eval_sample(r):
    orig = fetch(r['video_path'].replace(BS, '/'))
    full = cut(orig, 0, 10**4, 'full')
    a = decode_audio(full)
    dur = float(r.get('duration', len(a) / 16000.0))
    if dur > MAX_DUR:
        return None
        
    if dur > 35.0:
        max_patches = max(16, int(160 * (12.0 / dur)))
    else:
        max_patches = int(max(48, int(160 * min(1.0, 32.0 / dur))))
    max_pix = max_patches * 28 * 28
    
    gc.collect()
    torch.cuda.empty_cache()
    media = prepare_video(full, max_pixels=max_pix)
    
    stem, texts = parse_question(r['question'])
    k = len(texts)

    totals = {m: np.zeros(k) for m in METHODS}
    coop_layers = (18, 20, 22)

    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)

        ch, n_chunks, w_aud, p_aud, t_anchor = analyze_temporal_streams(sec, is_v, is_a, a)

        lg_base, cap_base = run_hooked_forward(inp, is_v, is_a, ch, last_media, mode='base', record_layers=coop_layers)
        lp_base = torch.log_softmax(lg_base[LETTERS[:k]], -1).numpy()

        lg_null, _ = run_hooked_forward(inp, is_v, is_a, ch, last_media, mode='null_media')
        lp_null = torch.log_softmax(lg_null[LETTERS[:k]], -1).numpy()

        clock_vec, overlap_mass, dt, _ = build_value_clock_and_displacement(
            cap_base[18], ch, n_chunks, is_v, is_a, last_media, w_aud, p_aud, t_anchor
        )

        lg_sync, cap_sync = run_hooked_forward(
            inp, is_v, is_a, ch, last_media, w_aud=w_aud, mode='pos_sync',
            clock_vec=clock_vec, record_layers=coop_layers
        )
        lp_sync = torch.log_softmax(lg_sync[LETTERS[:k]], -1).numpy()

        lg_neg, cap_neg = run_hooked_forward(
            inp, is_v, is_a, ch, last_media, w_aud=w_aud, mode='neg_desync',
            record_layers=coop_layers
        )
        lp_neg = torch.log_softmax(lg_neg[LETTERS[:k]], -1).numpy()

        lg_vclock, cap_vclock = run_hooked_forward(
            inp, is_v, is_a, ch, last_media, w_aud=w_aud, mode='vclock_all',
            clock_vec=clock_vec, record_layers=coop_layers
        )
        lp_vclock = torch.log_softmax(lg_vclock[LETTERS[:k]], -1).numpy()

        steer_dict = {
            li: (0.5 * (cap_sync[li][last_media + 1:] + cap_vclock[li][last_media + 1:]) - cap_neg[li][last_media + 1:])
            for li in coop_layers
        }
        lg_coop, _ = run_hooked_forward(
            inp, is_v, is_a, ch, last_media, w_aud=w_aud, mode='vclock_all',
            clock_vec=clock_vec, steer_dict=steer_dict, steer_betas={18: 0.35, 20: 0.50, 22: 0.35}
        )
        lp_coop = torch.log_softmax(lg_coop[LETTERS[:k]], -1).numpy()

        lp_c1 = lp_base - 0.85 * lp_null
        lp_c2 = lp_sync + 0.35 * (lp_sync - lp_neg) - 0.35 * lp_null
        lp_c3 = lp_vclock - 0.85 * lp_null
        lp_c4 = lp_coop
        lp_c5 = lp_coop - 0.85 * lp_null

        prior_spread = float(np.max(lp_null) - np.min(lp_null))
        w_rel = float(np.clip((prior_spread - 0.45) / 0.65, 0.15, 0.85))
        lp_c6 = (1.0 - w_rel) * lp_c2 + w_rel * (lp_c5 + 0.25 * (lp_coop - lp_neg))

        scores_perm = {
            'C0_base': lp_base,
            'C1_pmi_prior': lp_c1,
            'C2_sync_deconfound': lp_c2,
            'C3_vclock_pmi': lp_c3,
            'C4_coop_relay_L18_22': lp_c4,
            'C5_coop_relay_pmi': lp_c5,
            'C6_lstr_unified': lp_c6,
        }
        for m, lp in scores_perm.items():
            for pos_idx, opt_idx in enumerate(perm):
                totals[m][opt_idx] += lp[pos_idx]

        del inp
        gc.collect()
        torch.cuda.empty_cache()

    return {
        'dur': round(dur, 2),
        'overlap_mass': round(overlap_mass, 3),
        'dt_chunks': round(dt, 2),
        **{m: totals[m].round(3).tolist() for m in METHODS}
    }


def run_shard():
    samples = json.loads('[{"idx": 1517, "task": "CMTB", "gold": "A", "duration": 46.094, "video_path": "temporal/101/c9eELn4axpg/51d63f981bba3b9d.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) cut a lime squeeze on the avocado and stir (B) spread avocado mixture on one toast and on the other toast spread olive oil mayonnaise (C) cut the avocado and place in a bowl", "options": []}, {"idx": 1847, "task": "CMTB", "gold": "A", "duration": 48.078, "video_path": "temporal/212/HBUz55JRRm8/0801703b29731c47.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) melt fat in a pan (B) add carrots celery butter and salt to the pan (C) mash the boiled potatos with butter salt and pepper", "options": []}, {"idx": 1850, "task": "CMTB", "gold": "B", "duration": 49.068, "video_path": "temporal/212/HBUz55JRRm8/d16b9bea943290b9.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) mash the boiled potatos with butter salt and pepper (B) melt fat in a pan (C) cook the beef in the pan until brown", "options": []}, {"idx": 2196, "task": "CMTB", "gold": "B", "duration": 34.11, "video_path": "temporal/110/I1JgU6TK-yc/cbedc1c32c709cdc.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) add lemon juice crushed garlic grated parmesan salt and pepper to a bowl with butter milk and stir (B) pour some of it into a cup and put the rest into a bag with chicken (C) brush two slices of bread with oil top with salt and broil", "options": []}, {"idx": 1538, "task": "CMTB", "gold": "C", "duration": 29.162, "video_path": "temporal/313/CotdlwupDSI/b983fe2b4b244159.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) roll the salmon (B) place the salmon and tuna on a dish and garnish with green onions (C) pour some sauce", "options": []}, {"idx": 1962, "task": "CMTB", "gold": "C", "duration": 10.103, "video_path": "temporal/121/-GlSSp5ZOCQ/35ba1a64fda85c87.mp4", "question": "When the speech is heard, what visual action is occurring in the video? (A) remove all the clams from the shell (B) rinse and soak the clams (C) add bacon to the pot", "options": []}, {"idx": 2384, "task": "TA", "gold": "A", "duration": 21.2, "video_path": "temporal/306/GXnzgRC3sd4/e52fa34f6364f385.mp4", "question": "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No", "options": []}, {"idx": 2071, "task": "TA", "gold": "A", "duration": 23.116, "video_path": "temporal/112/UkqQAynrM2g/8b6103586e44849a.mp4", "question": "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No", "options": []}, {"idx": 1727, "task": "TA", "gold": "B", "duration": 33.159, "video_path": "temporal/117/oAE7nqQeMBQ/c99bca54850e3489.mp4", "question": "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No", "options": []}, {"idx": 1561, "task": "TA", "gold": "B", "duration": 38.014, "video_path": "temporal/207/nz_LHDf0uqE/b65cdc5ce322ca5d.mp4", "question": "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No", "options": []}, {"idx": 2373, "task": "TF", "gold": "A", "duration": 19.128, "video_path": "temporal/213/4Zl5NvXPi-0/7468d78d65a20909.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}, {"idx": 1433, "task": "TF", "gold": "A", "duration": 47.073, "video_path": "temporal/119/0ShsPjf9shQ/a93064d7110e0ac0.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}, {"idx": 1743, "task": "TF", "gold": "B", "duration": 16.073, "video_path": "temporal/221/cCWDR-jUv9U/f143a900a62df472.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}, {"idx": 2069, "task": "TF", "gold": "B", "duration": 18.103, "video_path": "temporal/112/UkqQAynrM2g/aaaca2229a42175a.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}, {"idx": 2306, "task": "TF", "gold": "C", "duration": 42.125, "video_path": "temporal/308/Ws7JgPJsVjs/5a6e2fb910d3c9c2.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}, {"idx": 1508, "task": "TF", "gold": "C", "duration": 27.081, "video_path": "temporal/108/7ebZWviUfUA/1a07af40c9fb36ff.mp4", "question": "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future", "options": []}]')
    print(f"=== Starting Round 2 on Server 2: {len(samples)} samples ===", flush=True)

    out_file = OUT / f'round2_lstr_s1of{NSHARD}.jsonl'
    done_indices = set()
    if out_file.exists():
        for line in out_file.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if 'idx' in rec: done_indices.add(rec['idx'])
                except Exception: pass
    print(f"Already completed {len(done_indices)} samples; remaining: {len(samples) - len(done_indices)}", flush=True)

    records = []
    for n, r in enumerate(samples):
        idx = r['idx']
        if idx in done_indices:
            continue
        t0 = time.time()
        try:
            res = eval_sample(r)
            if res is None:
                continue
            gold_char = r.get('gold', r.get('answer'))
            gold = ord(gold_char) - 65
            rec = {
                'idx': idx,
                'task': r['task'],
                'ans': gold_char,
                'gold': gold,
                **res,
                'sec': round(time.time() - t0, 1)
            }
            records.append(rec)
            with out_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            hits = {m: ('Y' if int(np.argmax(rec[m])) == gold else '-') for m in METHODS}
            print(
                f"[S2 {n+1}/{len(samples)}] {r['task']}-{gold_char} idx={idx} ({rec['dur']}s, dt={rec['dt_chunks']}, {rec['sec']}s): " +
                " ".join(f"{m.split('_')[0]}:{hits[m]}" for m in METHODS),
                flush=True
            )
        except Exception as e:
            print(f"[S2 {n+1}/{len(samples)}] Skip idx={idx} err={repr(e)[:120]}", flush=True)
            gc.collect()
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
            'delta_vs_base': round(overall - summary['C0_base']['overall'], 1) if 'C0_base' in summary else 0.0,
            'n': len(records)
        }
    print(f"=== Server 2 Summary ===", json.dumps(summary, indent=2), flush=True)
    return {'shard': SHARD, 'nshard': NSHARD, 'summary': summary, 'records': records}

RESULT = run_shard()
