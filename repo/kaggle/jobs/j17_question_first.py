"""J17 — ROOT CAUSE "watch before knowing the question": does the model understand time better if it reads the question
(and its own plan) BEFORE the video+audio is encoded?  Needs the J8 helpers (Lay). Also records, for the qpre and qpre_plan inputs, the attention mass that video and audio tokens give to the prefix (per layer and 2-s chunk), and a check that the explicit 4-D mask path equals the native forward.
Under the causal mask the media tokens are encoded without ever seeing the question that comes after them. Here the question
stem (and optionally a short plan the model writes from the question alone) is placed in the user turn BEFORE the media.
Conditions (rotated options, letter log-probs summed per option content):
  base          [media] [stem + options]                                   (standard)
  qpre          [stem] [media] [stem + options]                            (question first)
  qpre_blind    same tokens as qpre, but media tokens may NOT attend to the prefix (the prefix is only repeated text)
  plan_after    [media] [plan + stem + options]                            (the plan text alone, read after the media)
  qpre_plan     [stem + plan] [media] [stem + options]                     (think, then watch)
The plan: the model, from the question stem only (no media), writes 1-2 sentences on what to see / hear and in what order.
qpre vs qpre_blind isolates "media encoded with the question in view"; qpre_plan vs plan_after isolates the same for the plan.
A contrastive readout (1+a)*qpre - a*base is computed offline. Report per (task, gold).  Writes OUT/j17_question_first.jsonl."""
import json, numpy as np, torch
if 'Lay' not in globals():                       # notebook run: load the J8 helpers (server jobs prepend them instead)
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J17_OUT = globals().get('J17_OUT', 'j17_question_first.jsonl')
PLAN_PROMPT = ('Question about a video with sound: "{q}"\nBefore watching, write one or two short sentences: what exactly must you '
               'see or hear to answer it, and in what temporal order? Do not answer the question.')
TOK = PROC.tokenizer


def inputs_pre(prefix, prompt, media):
    """user turn = [prefix text] [video with its audio] [prompt text] (prefix None -> standard layout)."""
    from lab_native_video import conversation_for
    conv = conversation_for(media['path'], prompt, 'av', media['max_pixels_per_frame'])
    if prefix: conv[1]['content'].insert(0, {'type': 'text', 'text': prefix})
    rendered = PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = PROC(text=rendered, audio=media['audio'], images=None, videos=media['videos'], return_tensors='pt', padding=True,
               videos_kwargs={'fps': media['fps'], 'use_audio_in_video': True})
    if 'pixel_values_videos' not in inp or 'input_features' not in inp: raise RuntimeError('media missing')
    return move_inputs(inp, MODEL_)


def prefix_span(inp, lay):
    """(prefix, media_rows): the user text before <|vision_bos|><|audio_bos|> (system prompt excluded, it stays visible), and
    the media tokens plus their start/end markers."""
    ids = inp['input_ids'][0].cpu(); media = lay.v | lay.a; first = int(torch.where(media)[0].min())
    starts = torch.where(ids == TOK.convert_tokens_to_ids('<|im_start|>'))[0]
    user_start = int(starts[starts < first].max()) + 3                         # skip '<|im_start|>', 'user', '\n'
    pre = torch.zeros(lay.L, dtype=torch.bool); pre[user_start:first - 2] = True
    rows = torch.zeros(lay.L, dtype=torch.bool); rows[first - 2:int(torch.where(media)[0].max()) + 3] = True
    return pre, rows


def mask_forward(inp, block_prefix):
    lay = Lay(inp); m = lay.m.clone()
    if block_prefix:
        pre, rows = prefix_span(inp, lay)
        m[rows.nonzero().squeeze(1).unsqueeze(1), pre.nonzero().squeeze(1).unsqueeze(0)] = float('-inf')
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    with torch.no_grad():
        return MODEL_(**kw, attention_mask=lay.t(m), position_ids=lay.pos.to(lay.dev), use_audio_in_video=True).logits[0, -1].float()


prefix_blind_forward = lambda inp: mask_forward(inp, True)


# ---- mechanism: how much attention do video / audio tokens give to the text placed before them? (per layer, per 2-s chunk)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
for _li, _layer in enumerate(MODEL_.model.layers): _layer.self_attn._pm_layer = _li
PM = {}


def pm_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
    c = PM.get('cfg'); li = getattr(module, '_pm_layer', None)
    if not c or li is None: return out, None
    rep = query.shape[1] // key.shape[1]; k = key.repeat_interleave(rep, 1).float()
    sc = scaling if scaling is not None else query.shape[-1] ** -0.5
    pre = c['pre'].to(query.device); mass = torch.zeros(query.shape[2], device=query.device)
    for r in c['row_chunks']:
        r = r.to(query.device)
        s = (query[:, :, r, :].float() @ k.transpose(-1, -2)) * sc + attention_mask[:, :, r, :].float().to(query.device)
        mass[r] = torch.softmax(s, -1)[..., pre].sum(-1)[0].mean(0)            # mean over heads
    c['rec'][li] = mass.cpu()
    return out, None


def prefix_mass(inp):
    """[layers, chunks] mean attention mass on the prefix, separately for video and audio tokens."""
    lay = Lay(inp); pre, _ = prefix_span(inp, lay); media_idx = torch.where(lay.v | lay.a)[0]
    PM['cfg'] = dict(pre=pre, row_chunks=list(media_idx.split(512)), rec={})
    old = ALL_ATTENTION_FUNCTIONS['sdpa']; ALL_ATTENTION_FUNCTIONS['sdpa'] = pm_attention
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],))
    try:
        with torch.no_grad(): MODEL_(**kw, attention_mask=lay.t(lay.m), position_ids=lay.pos.to(lay.dev), use_audio_in_video=True)
    finally:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = old; hk.remove()
    rec = PM.pop('cfg')['rec']; M = torch.stack([rec[l] for l in range(len(MODEL_.model.layers))])        # [layers, L]
    out = {}
    for name, sel in (('video', lay.v), ('audio', lay.a)):
        out[name] = [[round(float(M[l, sel & (lay.chunk == c)].mean()), 5) if (sel & (lay.chunk == c)).any() else None
                      for c in range(lay.nc)] for l in range(M.shape[0])]
    out['prefix_tokens'] = int(pre.sum())
    return out


def score_pre(media, prefix, stem, texts, fwd=None, plan_in_prompt=None):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        prompt = prompt_of(stem, texts, perm)
        if plan_in_prompt: prompt = plan_in_prompt + '\n' + prompt
        inp = inputs_pre(prefix, prompt, media)
        if fwd is None:
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
        else: lg = fwd(inp)
        lp = torch.log_softmax(lg[L4[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache()
    return tot.round(3).tolist()


def make_plan(stem):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': PLAN_PROMPT.format(q=stem)}]}]
    rendered = PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = move_inputs(PROC(text=rendered, return_tensors='pt', padding=True), MODEL_)
    with torch.no_grad(): out = MODEL_.generate(**inp, max_new_tokens=48, do_sample=False)
    txt = TOK.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return txt.split('\n')[0][:300]


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    plan = make_plan(stem)
    rec = dict(dur=round(dur, 2), plan=plan)
    rec['base'] = score(media, stem, texts)
    rec['qpre'] = score_pre(media, stem, stem, texts)
    rec['qpre_blind'] = score_pre(media, stem, stem, texts, fwd=prefix_blind_forward)
    rec['plan_after'] = score_pre(media, None, stem, texts, plan_in_prompt=plan)
    rec['qpre_plan'] = score_pre(media, stem + '\n' + plan, stem, texts)
    # sanity: the explicit 4-D-mask path without blocking must reproduce the native forward
    inp = inputs_pre(stem, prompt_of(stem, texts, tuple(range(len(texts)))), media)
    with torch.no_grad(): native = MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
    rec['check_4d_maxdiff'] = round(float((mask_forward(inp, False) - native)[L4].abs().max()), 4)
    rec['mass_qpre'] = prefix_mass(inp); del inp
    inp = inputs_pre(stem + '\n' + plan, prompt_of(stem, texts, tuple(range(len(texts)))), media)
    rec['mass_qpre_plan'] = prefix_mass(inp); del inp
    rec['speech'] = [round(float(w[0]), 2), round(float(w[1]), 2)]
    return rec


items = sample_items(tuple(globals().get('J17_TASKS', ('TA', 'TF', 'CMTB'))), PER_LABEL, seed=SEED)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
C = ('base', 'qpre', 'qpre_blind', 'plan_after', 'qpre_plan')
R = run_loop(items, J17_OUT, item, show=C)
OK = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v))
RESULT = {}
for t in ('TA', 'TF', 'CMTB'):
    for g in sorted({r['gold'] for r in OK if r['task'] == t}):
        S = [r for r in OK if r['task'] == t and r['gold'] == g]
        RESULT[f'{t}_{"ABCD"[g]}'] = {c: f'{sum(am(r[c]) == g for r in S)}/{len(S)}' for c in C}
print(RESULT, flush=True)
