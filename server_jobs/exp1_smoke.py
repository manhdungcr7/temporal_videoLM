"""Smoke test for Exp-1 methods (C0, C1, C2, C3) on 2 samples to verify execution and correctness."""
import json, time, numpy as np, torch
from pathlib import Path

# Load 2 samples from Server 1 manifest
manifest = json.loads((OUT / 'micro_dev_32_manifest.json').read_text())
samples = manifest['server1'][:2]
print(f"Loaded {len(samples)} smoke samples: {[s['idx'] for s in samples]}")

L4 = [TOK.convert_tokens_to_ids(c) for c in 'ABCD']

def run_smoke_item(r):
    src = fetch(r['video_path'].replace(BS, '/'))
    media = prepare_video(src)
    stem, texts = parse_question(r['question'])
    k = len(texts)
    print(f"\n--- Testing sample {r['idx']} ({r['task']}/{r['gold']}), k={k} options ---")
    
    # 1. C0: Native-Base
    t0 = time.time()
    tot_c0 = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        with torch.no_grad():
            lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot_c0[j] += lp[p_]
        del inp
    t_c0 = time.time() - t0
    pred_c0 = chr(65 + int(np.argmax(tot_c0)))
    print(f"C0 (Base): scores={tot_c0.round(3).tolist()}, pred={pred_c0}, time={t_c0:.2f}s")
    
    # 2. C2: RoPE-Desync Contrastive (alpha = 0.5 and 1.0)
    t0 = time.time()
    tot_c2_05 = np.zeros(k)
    tot_c2_10 = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        
        # Factual forward
        with torch.no_grad():
            lg_factual = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        
        # Desync forward: collapse temporal dimension of media tokens to minimum time
        pos_desync = pos.clone()
        media_m = (is_v | is_a).to(pos.device)
        pos_desync[0, 0][media_m] = pos[0, 0][media_m].min()
        
        inp_desync = {k_: v for k_, v in inp.items() if k_ != 'position_ids'}
        inp_desync['position_ids'] = pos_desync
        with torch.no_grad():
            lg_desync = MODEL_(**inp_desync, use_audio_in_video=True).logits[0, -1].float()
        
        # Contrastive logits: lg_factual + alpha * (lg_factual - lg_desync)
        lg_c2_05 = lg_factual + 0.5 * (lg_factual - lg_desync)
        lg_c2_10 = lg_factual + 1.0 * (lg_factual - lg_desync)
        
        lp_05 = torch.log_softmax(lg_c2_05[L4[:k]].cpu(), -1).numpy()
        lp_10 = torch.log_softmax(lg_c2_10[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm):
            tot_c2_05[j] += lp_05[p_]
            tot_c2_10[j] += lp_10[p_]
        del inp, inp_desync
    t_c2 = time.time() - t0
    pred_c2_05 = chr(65 + int(np.argmax(tot_c2_05)))
    pred_c2_10 = chr(65 + int(np.argmax(tot_c2_10)))
    print(f"C2 (a=0.5): scores={tot_c2_05.round(3).tolist()}, pred={pred_c2_05}, time={t_c2:.2f}s")
    print(f"C2 (a=1.0): scores={tot_c2_10.round(3).tolist()}, pred={pred_c2_10}")

    # 3. C3: Question-Relay Steering at Layer 20 (beta = 0.3)
    t0 = time.time()
    tot_c3 = np.zeros(k)
    STEER_LAYER = 20
    BETA = 0.3
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        media_m = (is_v | is_a).to(pos.device)
        
        # Branch 1: Desync forward, capture H_Q at STEER_LAYER
        pos_desync = pos.clone()
        pos_desync[0, 0][media_m] = pos[0, 0][media_m].min()
        inp_desync = {k_: v for k_, v in inp.items() if k_ != 'position_ids'}
        inp_desync['position_ids'] = pos_desync
        
        h_desync = None
        def hook_desync(module, args, out):
            nonlocal h_desync
            h = out[0] if isinstance(out, tuple) else out
            h_desync = h[:, last_media+1:].detach().clone()
        
        hook_handle = MODEL_.model.layers[STEER_LAYER].register_forward_hook(hook_desync)
        with torch.no_grad():
            _ = MODEL_(**inp_desync, use_audio_in_video=True)
        hook_handle.remove()
        
        # Branch 2: Factual forward, steer H_Q at STEER_LAYER
        def hook_steer(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            h_q = h[:, last_media+1:]
            delta = h_q - h_desync
            delta_norm = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6) * h_q.norm(dim=-1, keepdim=True)
            steered_q = h_q + BETA * delta_norm
            h[:, last_media+1:] = steered_q
            return (h,) + out[1:] if isinstance(out, tuple) else h
        
        hook_handle = MODEL_.model.layers[STEER_LAYER].register_forward_hook(hook_steer)
        with torch.no_grad():
            lg_c3 = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        hook_handle.remove()
        
        lp_c3 = torch.log_softmax(lg_c3[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot_c3[j] += lp_c3[p_]
        del inp, inp_desync
    t_c3 = time.time() - t0
    pred_c3 = chr(65 + int(np.argmax(tot_c3)))
    print(f"C3 (Steer L20 b=0.3): scores={tot_c3.round(3).tolist()}, pred={pred_c3}, time={t_c3:.2f}s")
    
    return {
        'idx': r['idx'],
        'task': r['task'],
        'gold': r['gold'],
        'c0': tot_c0.round(3).tolist(),
        'c2_05': tot_c2_05.round(3).tolist(),
        'c2_10': tot_c2_10.round(3).tolist(),
        'c3': tot_c3.round(3).tolist()
    }

results = []
for r in samples:
    res = run_smoke_item(r)
    results.append(res)

RESULT = results
print("SMOKE_TEST_COMPLETED_SUCCESSFULLY")
