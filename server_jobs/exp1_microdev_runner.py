"""Exp-1: 32 Micro-Dev Benchmark Runner across 2 Kaggle Servers.
Runs C0 (Native-Base), C1 (DeSmear+Clock), C2 (RoPE-Desync Contrastive a=0.5 & a=1.0), C3 (Relay Steering b=0.3).
Resumable, task-agnostic, saves JSONL incrementally.
"""
import json, time, math, numpy as np, torch
from pathlib import Path
import torchvision.io

import gc
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

# Load manifest
manifest = json.loads((OUT / 'micro_dev_32_manifest.json').read_text(encoding='utf-8'))
server_key = f'server{LAB_SERVER}'
samples = manifest[server_key]
print(f"=== Starting Exp-1 on Server {LAB_SERVER}: {len(samples)} samples ===", flush=True)

out_file = OUT / f'exp1_microdev_s{LAB_SERVER}.jsonl'
done_indices = set()
if out_file.exists():
    for line in out_file.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                if 'idx' in rec: done_indices.add(rec['idx'])
            except Exception: pass
print(f"Already completed {len(done_indices)} samples; remaining: {len(samples) - len(done_indices)}", flush=True)

L4 = [TOK.convert_tokens_to_ids(c) for c in 'ABCD']

def evaluate_sample(r):
    t_start = time.time()
    idx = r['idx']
    task = r['task']
    gold_char = r['gold']
    gold_idx = ord(gold_char) - 65
    
    gc.collect()
    torch.cuda.empty_cache()
    
    src = fetch(r['video_path'].replace(BS, '/'))
    dur = float(r['duration'])
    if dur > 35.0:
        max_patches = max(16, int(160 * (12.0 / dur)))
    else:
        max_patches = int(max(48, int(160 * min(1.0, 32.0 / dur))))
    max_pix = max_patches * 28 * 28
    torch.cuda.empty_cache()
    media = prepare_video(src, max_pixels=max_pix)
    stem, texts = parse_question(r['question'])
    k = len(texts)
    
    # ------------------ C1: DeSmear + Physical Clock ------------------
    tot_c1 = None
    pred_c1 = None
    t_c1 = 0
    try:
        t0 = time.time()
        # Call avbind_clocked_answer if available in globals
        if 'avbind_clocked_answer' in globals():
            c1_res = avbind_clocked_answer(src, stem, texts)
            tot_c1 = c1_res['scores']
            pred_c1 = chr(65 + int(np.argmax(tot_c1)))
            t_c1 = round(time.time() - t0, 2)
    except Exception as e:
        print(f"[{idx}] C1 error: {repr(e)[:150]}", flush=True)
    
    # ------------------ C0, C2, C3 via Optimized Rotations ------------------
    tot_c0 = np.zeros(k)
    tot_c2_05 = np.zeros(k)
    tot_c2_10 = np.zeros(k)
    tot_c3 = np.zeros(k)
    
    STEER_LAYER = 20
    BETA = 0.3
    
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, 'av')
        inp = move_inputs(inp, MODEL_)
        pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
        media_m = (is_v | is_a).to(pos.device)
        
        # --- Pass 1: Desync Forward (RoPE Time-Collapsed) ---
        # Captures H_Q at STEER_LAYER for C3 AND outputs lg_desync for C2
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
            lg_desync = MODEL_(**inp_desync, use_audio_in_video=True).logits[0, -1].float()
        hook_handle.remove()
        
        # --- Pass 2: Factual Plain Forward (C0 & C2) ---
        with torch.no_grad():
            lg_factual = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        
        # C0 logits
        lp_c0 = torch.log_softmax(lg_factual[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot_c0[j] += lp_c0[p_]
        
        # C2 Contrastive logits
        lg_c2_05 = lg_factual + 0.5 * (lg_factual - lg_desync)
        lg_c2_10 = lg_factual + 1.0 * (lg_factual - lg_desync)
        lp_05 = torch.log_softmax(lg_c2_05[L4[:k]].cpu(), -1).numpy()
        lp_10 = torch.log_softmax(lg_c2_10[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm):
            tot_c2_05[j] += lp_05[p_]
            tot_c2_10[j] += lp_10[p_]
            
        # --- Pass 3: Steered Factual Forward (C3) ---
        def hook_steer(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            h_q = h[:, last_media+1:]
            delta = h_q - h_desync
            delta_norm = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6) * h_q.norm(dim=-1, keepdim=True)
            h[:, last_media+1:] = h_q + BETA * delta_norm
            return (h,) + out[1:] if isinstance(out, tuple) else h
        
        hook_handle = MODEL_.model.layers[STEER_LAYER].register_forward_hook(hook_steer)
        with torch.no_grad():
            lg_c3 = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        hook_handle.remove()
        
        lp_c3 = torch.log_softmax(lg_c3[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot_c3[j] += lp_c3[p_]
        
        del inp, inp_desync
        torch.cuda.empty_cache()

    pred_c0 = chr(65 + int(np.argmax(tot_c0)))
    pred_c2_05 = chr(65 + int(np.argmax(tot_c2_05)))
    pred_c2_10 = chr(65 + int(np.argmax(tot_c2_10)))
    pred_c3 = chr(65 + int(np.argmax(tot_c3)))
    
    tot_sec = round(time.time() - t_start, 2)
    
    rec = {
        'idx': idx,
        'task': task,
        'gold': gold_char,
        'duration': r['duration'],
        'sec': tot_sec,
        'c0': tot_c0.round(3).tolist(),
        'c0_pred': pred_c0,
        'c0_correct': pred_c0 == gold_char,
        'c1': tot_c1,
        'c1_pred': pred_c1,
        'c1_correct': (pred_c1 == gold_char) if pred_c1 else None,
        'c2_05': tot_c2_05.round(3).tolist(),
        'c2_05_pred': pred_c2_05,
        'c2_05_correct': pred_c2_05 == gold_char,
        'c2_10': tot_c2_10.round(3).tolist(),
        'c2_10_pred': pred_c2_10,
        'c2_10_correct': pred_c2_10 == gold_char,
        'c3': tot_c3.round(3).tolist(),
        'c3_pred': pred_c3,
        'c3_correct': pred_c3 == gold_char,
    }
    return rec

# Run loop
for n, r in enumerate(samples):
    if r['idx'] in done_indices:
        continue
    try:
        rec = evaluate_sample(r)
        with out_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        
        # Status log
        mark_c0 = "OK" if rec['c0_correct'] else "FAIL"
        mark_c1 = ("OK" if rec['c1_correct'] else "FAIL") if rec['c1_correct'] is not None else "N/A"
        mark_c2 = "OK" if rec['c2_10_correct'] else "FAIL"
        mark_c3 = "OK" if rec['c3_correct'] else "FAIL"
        print(f"[S{LAB_SERVER} {n+1}/{len(samples)}] idx={r['idx']} {r['task']}/{r['gold']} | "
              f"C0:{rec['c0_pred']}({mark_c0}) C1:{rec['c1_pred']}({mark_c1}) "
              f"C2:{rec['c2_10_pred']}({mark_c2}) C3:{rec['c3_pred']}({mark_c3}) | {rec['sec']}s", flush=True)
    except Exception as e:
        print(f"[S{LAB_SERVER} {n+1}/{len(samples)}] ERROR on idx={r['idx']}: {repr(e)}", flush=True)
        torch.cuda.empty_cache()

print(f"=== Server {LAB_SERVER} Completed All Samples ===", flush=True)
RESULT = "DONE"
