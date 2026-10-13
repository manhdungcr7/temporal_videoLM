"""Shared helpers for the diagnostic notebooks (exec'd in the kaggle_rootcause.py namespace after the model is loaded).
Labels are only ever used for scoring/analysis; nothing here feeds a label to the model."""
import json, random, time
from pathlib import Path
import numpy as np, torch, soundfile as sf

L4 = [PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABCD']
SR = 16000


# ---------------------------------------------------------------- items
def sample_items(tasks, per_label, seed, max_dur=None):
    """Balanced per (task, gold letter); round-robin across cells so a deadline cut keeps the sample balanced."""
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        if r['data_type'] == 'temporal' and r['task_type_abbr'] in tasks: cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    pools = []
    for key in sorted(cells):
        pool = cells[key]; random.Random(seed).shuffle(pool); pools.append(pool[:per_label])
    items = []
    for k in range(max(len(p) for p in pools)):
        for p in pools:
            if k < len(p): items.append(p[k])
    return items


def all_temporal_items():
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    return [(i, r) for i, r in enumerate(rows) if r['data_type'] == 'temporal']


def prep_clip(r, max_dur=None):
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
    a = decode_audio(src); dur = len(a) / SR
    if max_dur is not None and dur > max_dur: raise RuntimeError('long')
    w = speech_window(a)
    if w is None: raise RuntimeError('no speech activity')
    return src, a, dur, w


def speech_first(w, dur): return (w[0] + w[1]) / 2 < dur / 2


# ---------------------------------------------------------------- scoring (up to 4 options, rotated orders summed per content)
def prompt_of(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABCD', perm))


def score(media, stem, texts, mode='av', fwd=None, on_logits=None):
    """fwd(inp) -> last-token logits (default: plain forward). on_logits(perm, logits) is called for every order."""
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, mode); inp = move_inputs(inp, MODEL_)
        if fwd is None:
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=(mode == 'av')).logits[0, -1].float()
        else: lg = fwd(inp)
        if on_logits is not None: on_logits(perm, lg)
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache()
    return tot.round(3).tolist()


def yes_minus_no(media, question, mode='vision'):
    s = score(media, question, ['Yes', 'No'], mode=mode); return round(s[0] - s[1], 3)


def layer_out(out):
    h = out[0] if isinstance(out, tuple) else out
    return h[0] if h.dim() == 3 else h


def forward_record(inp, layers, groups, use_audio=True):
    """plain forward; for each recorded layer (1-based output index) the mean hidden state of every boolean token group."""
    store, hooks = {}, []
    for li in layers:
        def hook(mod, args, out, li=li):
            h = layer_out(out)
            store[li] = {g: h[m.to(h.device)].float().mean(0).cpu() for g, m in groups.items() if bool(m.any())}
        hooks.append(MODEL_.model.layers[li - 1].register_forward_hook(hook))
    try:
        with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=use_audio).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()
    return lg, store


def token_groups(inp, dur, win):
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    sec, is_v, is_a = sec.cpu(), is_v.cpu(), is_a.cpu()
    L = inp['input_ids'].shape[1]; idx = torch.arange(L)
    inside = (sec >= win[0] - 1.0) & (sec <= win[1] + 1.0)
    g = {'last': idx == L - 1, 'q': idx > last_media,
         'v_first': is_v & (sec < dur / 2), 'v_second': is_v & (sec >= dur / 2),
         'a_speech': is_a & inside, 'a_rest': is_a & ~inside}
    return g, (pos, sec, is_v, is_a, last_media)


# ---------------------------------------------------------------- clip variants (ffmpeg, cached in CLIPS)
def with_audio(src, audio, tag):
    w = CLIPS / f'{Path(src).stem}_{tag}.wav'; sf.write(w, audio, SR)
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists(): ffmpeg('-i', str(src), '-i', str(w), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dst))
    return dst


def concat_clips(parts, tag, audio_from=None):
    dst = CLIPS / f'{Path(parts[0]).stem}_{tag}.mp4'
    if dst.exists(): return dst
    ins = []
    for p in parts: ins += ['-i', str(p)]
    n = len(parts)
    if audio_from is None:
        fc = ''.join(f'[{k}:v:0][{k}:a:0]' for k in range(n)) + f'concat=n={n}:v=1:a=1[v][a]'
        ffmpeg(*ins, '-filter_complex', fc, '-map', '[v]', '-map', '[a]', *ENC, str(dst))
    else:
        fc = ''.join(f'[{k}:v:0]' for k in range(n)) + f'concat=n={n}:v=1:a=0[v]'
        ffmpeg(*ins, '-i', str(audio_from), '-filter_complex', fc, '-map', '[v]', '-map', f'{n}:a:0', *ENC, '-shortest', str(dst))
    return dst


def halves(src, dur):
    return cut(src, 0, dur / 2, 'h1'), cut(src, dur / 2, dur, 'h2')


def pink(n, seed):
    X = np.fft.rfft(np.random.default_rng(seed).standard_normal(n)); f = np.arange(len(X), dtype='float64'); f[0] = 1
    x = np.fft.irfft(X / np.sqrt(f), n); return (x / (x.std() + 1e-12)).astype('float32')


def noisy_audio(a, w, snr_db=10.0, seed=0):
    """pink noise over the whole track at snr_db below the speech RMS: removes the benchmark's digital silence."""
    s = a[int(w[0] * SR): int(w[1] * SR)]; ps = float(np.sqrt(np.mean(s ** 2) + 1e-12))
    return np.clip(a + pink(len(a), seed) * ps * 10 ** (-snr_db / 20), -1, 1).astype('float32')


# ---------------------------------------------------------------- loop with resume + deadline
def run_loop(items, out_name, fn, show=(), min_left=1200):
    items = items[:int(globals().get('LIMIT', 10 ** 9))]          # LIMIT is set only by the smoke test
    res = OUT / out_name
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
    print('items', len(items), 'already done', len(done), flush=True)
    for n, (i, r) in enumerate(items):
        if i in done: continue
        if time_left() < min_left: print('deadline reached: stopping cleanly', flush=True); break
        t0 = time.time(); rec = dict(idx=i, task=r['task_type_abbr'], gold='ABCD'.index(r['answer']))
        try:
            rec.update(fn(i, r)); rec['sec'] = round(time.time() - t0)
        except torch.cuda.OutOfMemoryError:
            rec['error'] = 'OOM'
        except Exception as e:
            rec['error'] = repr(e)[:300]
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(n + 1, len(items), rec['task'], i, 'ERR', rec['error'][:80], flush=True); continue
        g = rec['gold']
        marks = ' '.join(f'{c}={"Y" if int(np.argmax(rec[c])) == g else "-"}' for c in show if isinstance(rec.get(c), list))
        print(n + 1, len(items), rec['task'], i, marks, rec['sec'], 's', flush=True)
    return [json.loads(l) for l in res.read_text().splitlines() if l.strip()] if res.exists() else []


def acc(R, cond, task=None, expect=None):
    """accuracy (%) and n of condition `cond`; expect(rec) gives the expected index (default: gold)."""
    v = [int(np.argmax(r[cond])) == (expect(r) if expect else r['gold']) for r in R
         if 'error' not in r and isinstance(r.get(cond), list) and (task is None or r['task'] == task) and (expect is None or expect(r) is not None)]
    return (round(100 * float(np.mean(v)), 1) if v else None, len(v))
