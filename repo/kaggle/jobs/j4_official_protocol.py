"""J4 — REPRODUCE THE PUBLISHED BASELINE ON ALL 983 TEMPORAL ITEMS (incl. clips > 42 s)
Why: every number so far uses our protocol (rotated options, letter log-probs) on clips <= 42 s. The paper reports
Qwen2.5-Omni TA 50.27 / TF 31.52 / CMTB 53.10 with generation. Before any SOTA claim we must (1) reproduce those numbers
with generation + parsing, (2) know our protocol's number on the same full set, (3) handle long clips instead of skipping.
Long clips: a fixed token budget. If 25*dur + 72*fps*dur > BUDGET, fps is lowered (>= 0.25), then pixels per frame.
TMRoPE keeps absolute time (25 ids / s) whatever the fps. On OOM the budget is halved once.
Per item: gen_official (question + "Answer with the option's letter from the given choices directly."), gen_raw (question
only), first-token letter log-probs in the original order (single), and rotated scores (our protocol)."""
import re, numpy as np, torch

BUDGET = int(globals().get('TOKEN_BUDGET', 7200))
SHARD, NSHARD = int(globals().get('SHARD', 0)), int(globals().get('NSHARD', 1))
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def prepare_budget(path, dur, budget):
    from qwen_omni_utils import process_mm_info
    import lab_native_video as LNV
    fps, mp = 2.0, LNV.VIDEO_MAX_PIXELS
    if 25 * dur + 72 * fps * dur > budget:
        fps = max(0.25, (budget / dur - 25) / 72)
        if 25 * dur + 72 * fps * dur > budget:
            mp = max(16 * 28 * 28, int(LNV.VIDEO_MAX_PIXELS * max(0.1, (budget / dur - 25) / (72 * fps))))
    ele = {'type': 'video', 'video': str(path), 'max_pixels': mp, 'fps': fps}
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': LNV.SYSTEM}]}, {'role': 'user', 'content': [ele, {'type': 'text', 'text': ''}]}]
    audios, images, videos, vk = process_mm_info(conv, use_audio_in_video=True, return_video_kwargs=True)
    f = vk['fps']; f = f[0] if isinstance(f, (list, tuple)) else f
    return {'path': str(path), 'audio': audios, 'videos': videos, 'fps': float(f), 'max_pixels_per_frame': mp}, dict(fps=round(fps, 3), max_pixels=mp)


def generate(media, prompt):
    inp, _ = qwen_inputs(PROC, prompt, media, 'av'); inp = move_inputs(inp, MODEL_)
    with torch.no_grad():
        o = MODEL_.generate(**inp, max_new_tokens=12, do_sample=False, use_audio_in_video=True,
                            return_dict_in_generate=True, output_logits=True)
    txt = PROC.batch_decode(o.sequences[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
    lg = o.logits[0][0].float()                     # raw first-step logits (before any logits processor)
    del inp; torch.cuda.empty_cache(); return txt, lg


def parse(txt, texts):
    """'(B)' > leading letter > unique option text > lone letter anywhere; -1 if none."""
    ok = lambda c: 'ABCD'.index(c) < len(texts)
    for pat in (r'\(([A-D])\)', r'^\s*([A-D])\s*(?:[.):,]|$)'):
        m = re.search(pat, txt)
        if m and ok(m.group(1)): return 'ABCD'.index(m.group(1))
    low = txt.lower()
    hits = [j for j, t in enumerate(texts) if t.lower() in low]
    if len(hits) == 1: return hits[0]
    m = re.search(r'\b([A-D])\b', txt)
    return 'ABCD'.index(m.group(1)) if m and ok(m.group(1)) else -1


def item(i, r):
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
    dur = len(decode_audio(src)) / SR
    stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2))
    for attempt, budget in enumerate((BUDGET, BUDGET // 2)):
        try:
            media, cfg = prepare_budget(src, dur, budget) if dur > 42 else (prepare_video(src), dict(fps=2.0, max_pixels=None))
            rec.update(cfg, budget=budget if dur > 42 else None)
            g_off, lg = generate(media, r['question'] + SUFFIX)
            rec['gen_official'] = g_off; rec['pred_official'] = parse(g_off, texts)
            rec['single'] = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy().round(3).tolist()
            g_raw, _ = generate(media, r['question'])
            rec['gen_raw'] = g_raw; rec['pred_raw'] = parse(g_raw, texts)
            rec['rotated'] = score(media, stem, texts)
            return rec
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == 1 or dur <= 42: raise
            rec['oom_retry'] = True


items = [x for n, x in enumerate(all_temporal_items()) if n % NSHARD == SHARD]
items = items[::2] + items[1::2]          # spread tasks/lengths so a deadline cut stays representative
R = run_loop(items, f'j4_official_shard{SHARD}.jsonl', item, show=('rotated', 'single'), min_left=900)

OK = [r for r in R if 'error' not in r]
summary = {}
for t in ('TA', 'TF', 'CMTB'):
    S = [r for r in OK if r['task'] == t]
    if not S: continue
    d = {'n': len(S), 'n_long(>42s)': sum(r['dur'] > 42 for r in S)}
    for c in ('official', 'raw'):
        d[f'gen_{c}'] = round(100 * np.mean([r[f'pred_{c}'] == r['gold'] for r in S]), 2)
        d[f'gen_{c}_unparsed%'] = round(100 * np.mean([r[f'pred_{c}'] == -1 for r in S]), 1)
    d['single'] = acc(S, 'single')[0]; d['rotated'] = acc(S, 'rotated')[0]
    d['rotated_short_only'] = acc([r for r in S if r['dur'] <= 42], 'rotated')[0]
    d['rotated_long_only'] = acc([r for r in S if r['dur'] > 42], 'rotated')[0]
    if t == 'TA':
        y = np.array([r['gold'] == 0 for r in S]); p = np.array([r['pred_official'] == 0 for r in S])
        prec = (y & p).sum() / max(1, p.sum()); rec_ = (y & p).sum() / max(1, y.sum())
        d['official_precision/recall/F1'] = [round(100 * prec, 1), round(100 * rec_, 1), round(200 * prec * rec_ / max(1e-9, prec + rec_), 1)]
    summary[t] = d
RESULT = {'summary': summary, 'errors': sum('error' in r for r in R), 'error_kinds': sorted({r['error'][:40] for r in R if 'error' in r})}
print(RESULT, flush=True)
