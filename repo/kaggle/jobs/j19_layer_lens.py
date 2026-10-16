"""J19 — LAYER LENS: what answer does each layer "hold", and does each layer lean on the video or on the audio?  Needs the J8 helpers (Lay).
For every answer rotation, the last token's residual stream after each of the 36 decoder layers is read out with the model's own final norm +
lm_head (logit lens), restricted to the answer letters, then summed per option content over rotations (same scoring as everywhere else).
Done for three inputs: full video+audio ('av'), video only ('vision'), audio only ('audio').  For 'av' we also record, per layer, the last
token's attention mass on video tokens, audio tokens and text (mean over heads, first rotation).
Items: temporal TA/TF/CMTB (J19_PER_TEMPORAL per (task, gold)) + semantic GSA/FGSA/CSMB (J19_PER_SEMANTIC per (task, gold)), seed J19_SEED.
Writes OUT/j19_layer_lens.jsonl.  Analysis: remote_tests/analyze_j19.py."""
import json, random, numpy as np, torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J19_OUT = globals().get('J19_OUT', 'j19_layer_lens.jsonl')
PER_T = int(globals().get('J19_PER_TEMPORAL', 2)); PER_S = int(globals().get('J19_PER_SEMANTIC', 3)); SEED19 = int(globals().get('J19_SEED', 2027))
NL = len(MODEL_.model.layers)
for _li, _layer in enumerate(MODEL_.model.layers): _layer.self_attn._ll_layer = _li
LL = {}


def mass_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = LL.get('cfg'); li = getattr(module, '_ll_layer', None)
    if not c or li is None: return out, None
    rep = query.shape[1] // key.shape[1]; k = key.repeat_interleave(rep, 1).float()
    sc = scaling if scaling is not None else query.shape[-1] ** -0.5
    s = (query[:, :, -1:, :].float() @ k.transpose(-1, -2)) * sc
    if attention_mask is not None: s = s + attention_mask[:, :, -1:, :].float().to(s.device)
    p = torch.softmax(s, -1)[0, :, 0, :].mean(0)                                  # [L] mean over heads
    c['mass'][li] = [float(p[c['v'].to(p.device)].sum()), float(p[c['a'].to(p.device)].sum()), float(p[c['t'].to(p.device)].sum())]
    return out, None


def lens_forward(inp, k, record_mass=False):
    """returns per-layer letter log-probs [NL, k], final log-probs [k], and (optionally) per-layer attention mass [NL, 3]."""
    got = {}
    def mk(li):
        def hk(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            got[li] = h[0, -1].detach()
        return hk
    hooks = [MODEL_.model.layers[li].register_forward_hook(mk(li)) for li in range(NL)]
    hooks.append(MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],)))
    old = None
    if record_mass:
        lay = Lay(inp); text = ~(lay.v | lay.a)
        LL['cfg'] = dict(v=lay.v, a=lay.a, t=text, mass={})
        old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = mass_attention
    try:
        with torch.no_grad():
            final = MODEL_(**inp, use_audio_in_video=('input_features' in inp and 'pixel_values_videos' in inp)).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()                                       # (incl. the lm_head row-selection hook) before the lens readout
        if old is not None: ALL_ATTENTION_FUNCTIONS['sdpa'] = old
    with torch.no_grad():
        norm = MODEL_.model.norm; head = MODEL_.lm_head
        lens = torch.stack([head(norm(got[li].to(next(norm.parameters()).device)[None]).to(head.weight.device))[0].float() for li in range(NL)])
    ids = L4[:k]
    lp = torch.log_softmax(lens[:, ids].cpu(), -1).numpy(); fp = torch.log_softmax(final[ids].cpu(), -1).numpy()
    mass = None
    if record_mass:
        m = LL.pop('cfg')['mass']; mass = [m[li] for li in range(NL)]
    return lp, fp, mass


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)
    media = prepare_video(src); stem, texts = parse_question(r['question']); k = len(texts)
    rec = dict(dur=round(dur, 2), data_type=r['data_type'])
    for mode in ('av', 'vision', 'audio'):
        tot = np.zeros((NL, k)); fin = np.zeros(k); mass = None; per_rot = []
        for perm in rotations(k):
            inp, _ = qwen_inputs(PROC, prompt_of(stem, texts, perm), media, mode); inp = move_inputs(inp, MODEL_)
            if mode == 'av' and 'n_video' not in rec:
                _l = Lay(inp); rec['n_video'] = int(_l.v.sum()); rec['n_audio'] = int(_l.a.sum()); rec['n_tokens'] = int(_l.L)
            lp, fp, ms = lens_forward(inp, k, record_mass=(mode == 'av' and mass is None))
            if ms is not None: mass = ms
            for slot, j in enumerate(perm): tot[:, j] += lp[:, slot]; fin[j] += fp[slot]
            per_rot.append(dict(perm=list(perm), letter_logprob=np.round(lp, 3).tolist()))   # by LETTER slot (for letter-position bias)
            del inp
        rec[f'rot_{mode}'] = per_rot
        torch.cuda.empty_cache()
        rec[f'lens_{mode}'] = np.round(tot, 3).tolist(); rec[f'final_{mode}'] = np.round(fin, 3).tolist()
        if mass is not None: rec['mass_av'] = np.round(np.array(mass), 4).tolist()
    return rec


def prep_any(r):
    src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
    a = decode_audio(src); dur = len(a) / SR
    if dur > MAXD: raise RuntimeError('long')
    return src, a, dur, None


def usable(r):
    """True if the clip can be prepared (some clips use an audio codec ffmpeg cannot decode, or are too long / have no speech).
    Unusable clips are skipped BEFORE selection and counted, so every cell is filled; the skip count is printed."""
    try:
        (prep_clip(r, MAXD) if r['data_type'] == 'temporal' else prep_any(r)); return True
    except Exception as e:
        SKIPPED.append((r['task_type_abbr'], r['answer'], repr(e)[:40])); return False


SKIPPED = []


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


items = balanced(('TA', 'TF', 'CMTB'), PER_T, SEED19) + balanced(('GSA', 'FGSA', 'CSMB'), PER_S, SEED19)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
import collections as _c
print('skipped before selection:', dict(_c.Counter((t, a) for t, a, _ in SKIPPED)), flush=True)
R = run_loop(items, J19_OUT, item, show=('final_av',))
RESULT = {'n_ok': sum('error' not in r for r in R), 'n': len(R)}
print(RESULT, flush=True)
