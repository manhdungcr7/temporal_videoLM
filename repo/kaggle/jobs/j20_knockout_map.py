"""J20 — KNOCKOUT MAP: depth x SOURCE x TIME.  Where inside the model does the answer read the audio, the video, the speech moment and
the rest of the video?  Needs the J8 helpers (Lay, forward).  Diagnostic only (uses the VAD speech window to define chunks; the method
built from it must take such signals from the model itself).
For every start layer L0 in J20_L0 and every knockout kind, the question/answer rows (all text after the media) are blocked from a set of
media tokens in every layer >= L0 (attention-mask knockout, as in Map the Flow / cross-modal information flow work):
    noA        all audio tokens            noV        all video tokens
    noSpeech   video+audio tokens of the 2-s chunks that overlap the speech window (the moment the question refers to)
    noRest     video+audio tokens of all other chunks
plus base, video-only input and audio-only input.  Rotated options, letter log-probs summed per option content.
Items: temporal TA/TF/CMTB (J20_PER_TEMPORAL per (task, gold)) + semantic GSA/FGSA/CSMB (J20_PER_SEMANTIC per (task, gold)); clips that
cannot be prepared are skipped BEFORE selection (counted).  Writes OUT/j20_knockout_map.jsonl."""
import json, random, numpy as np, torch
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J20_OUT = globals().get('J20_OUT', 'j20_knockout_map.jsonl')
L0S = tuple(globals().get('J20_L0', (8, 16, 24, 30)))
PER_T = int(globals().get('J20_PER_TEMPORAL', 2)); PER_S = int(globals().get('J20_PER_SEMANTIC', 2)); SEED20 = int(globals().get('J20_SEED', 2028))
NL20 = len(MODEL_.model.layers)
KINDS = ('noA', 'noV', 'noSpeech', 'noRest')


def prep_any(r):
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
    a = decode_audio(src); dur = len(a) / SR
    if dur > MAXD: raise RuntimeError('long')
    return src, a, dur, speech_window(a)


def prep20(r):
    return prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)


SKIPPED20 = []


def usable(r):
    try: prep20(r); return True
    except Exception as e: SKIPPED20.append((r['task_type_abbr'], r['answer'], repr(e)[:40])); return False


def balanced(tasks, per, seed):
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8')); cells = {}
    for i, r in enumerate(rows):
        if r['task_type_abbr'] in tasks: cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
    pools = []
    for key in sorted(cells):
        pool = cells[key][:]; random.Random(f'{seed}-{key}').shuffle(pool); got = []
        for i, r in pool:
            if usable(r): got.append((i, r))
            if len(got) == per: break
        pools.append(got)
    out = []
    for kk in range(max(len(p) for p in pools)):
        for p in pools:
            if kk < len(p): out.append(p[kk])
    return out


def knock_mask(lay, kind, w):
    cols = {'noA': lay.a, 'noV': lay.v}.get(kind)
    if cols is None:
        if w is None: return None
        t = torch.arange(lay.nc).float(); sp_chunks = ((2 * t + 2) > w[0]) & ((2 * t) < w[1])
        in_sp = sp_chunks[lay.chunk] & (lay.v | lay.a)
        cols = in_sp if kind == 'noSpeech' else ((lay.v | lay.a) & ~in_sp)
    if not bool(cols.any()): return None
    m = lay.m.clone(); rr = torch.where(lay.rows)[0]; cc = torch.where(cols)[0]
    m[rr.unsqueeze(1), cc.unsqueeze(0)] = float('-inf')
    return m


def item(i, r):
    src, a, dur, w = prep20(r)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2), data_type=r['data_type'], speech=None if w is None else [round(float(w[0]), 2), round(float(w[1]), 2)])
    rec['base'] = score(media, stem, texts)
    rec['base_mask'] = score(media, stem, texts, fwd=lambda inp: forward(inp, Lay(inp))[0])   # same 4-D-mask path as the knockouts, nothing blocked
    rec['vision_only'] = score(media, stem, texts, mode='vision'); rec['audio_only'] = score(media, stem, texts, mode='audio')
    cache = {}
    def fwd_for(kind, L0):
        def f(inp):
            lay = Lay(inp); m = knock_mask(lay, kind, w)
            if m is None: raise RuntimeError('empty knockout')
            special = {li: m for li in range(L0, NL20)}
            return forward(inp, lay, special=special)[0]
        return f
    for kind in KINDS:
        for L0 in L0S:
            try: rec[f'{kind}@{L0}'] = score(media, stem, texts, fwd=fwd_for(kind, L0))
            except RuntimeError as e:
                if 'empty knockout' not in str(e): raise
                rec[f'{kind}@{L0}'] = None
    return rec


items = balanced(('TA', 'TF', 'CMTB'), PER_T, SEED20) + balanced(('GSA', 'FGSA', 'CSMB'), PER_S, SEED20)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
import collections as _c
print('skipped before selection:', dict(_c.Counter((t, a_) for t, a_, _ in SKIPPED20)), flush=True)
R = run_loop(items, J20_OUT, item, show=('base',))
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)
