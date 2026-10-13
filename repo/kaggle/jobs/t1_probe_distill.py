"""T1 — STAGE-1 PILOT of learned probe tokens (team proposal). The model is frozen; ONLY m probe embeddings are trained.
A probe block of m tokens is placed after the media of a short clip (4 s). Probe rows may read the media; the question and
answer rows may read ONLY the probes (not the media). Objective = self-distillation: the answer distribution when the model
reads the probes must match the answer distribution when it reads the clip directly (teacher = same frozen model, media
visible, no probes). Data: random 4-s clips of Daily-Omni videos (not SVHalluc), generic questions, teacher answers
generated greedily. Reported: held-out token agreement / KL for (a) trained probes, (b) untrained probes, (c) no media at
all (text prior), so we know whether embeddings alone can carry the clip's content. Writes OUT/t1_probe_distill.json and
OUT/t1_probes.pt."""
import json, random, tarfile, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download

M = int(globals().get('M_PROBES', 8)); N_TRAIN = int(globals().get('N_TRAIN_CLIPS', 160)); N_EVAL = int(globals().get('N_EVAL_CLIPS', 40))
STEPS = int(globals().get('STEPS', 800)); LR = float(globals().get('LR', 5e-3)); CLIP_S = 4.0
QUESTIONS = ['What is happening in this video? Answer in one sentence.', 'What action is the person doing?',
             'What objects can be seen?', 'What sound or speech is heard?']
TOK = PROC.tokenizer
PROBE_WORD = ' probe'
PROBE_TXT = PROBE_WORD * M
for p_ in MODEL_.parameters(): p_.requires_grad_(False)          # frozen model: only the probe embeddings learn
emb_layer = MODEL_.get_input_embeddings(); D = emb_layer.weight.shape[1]

# ---------------- data: 4-s clips of Daily-Omni videos
DO = Path('/kaggle/working/daily_omni'); DO.mkdir(parents=True, exist_ok=True)
vid_dir = DO / 'videos'
if not vid_dir.exists():
    with tarfile.open(hf_hub_download('liarliar/Daily-Omni', 'Videos.tar', repo_type='dataset')) as t: t.extractall(vid_dir)
vids = sorted(p for p in vid_dir.rglob('*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov'))
rng = random.Random(0); rng.shuffle(vids)


def make_clip(vp, k):
    full = cut(vp, 0, 10 ** 4, 'full'); dur = len(decode_audio(full)) / SR
    t0 = rng.uniform(0, max(0.0, dur - CLIP_S)); c = cut(full, t0, t0 + CLIP_S, f'c{k}')
    return prepare_video(c)


def prompt_ids(media, text):
    inp, _ = qwen_inputs(PROC, text, media, 'av'); return move_inputs(inp, MODEL_)


def teacher(media, q):
    """greedy teacher answer (<=16 tokens) with the media visible, and its per-token log-probs."""
    inp = prompt_ids(media, q)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=16, do_sample=False, use_audio_in_video=True)
    ans = o[0, inp['input_ids'].shape[1]:].tolist()
    ans = [t for t in ans if t not in (TOK.eos_token_id, TOK.pad_token_id)][:16] or [o[0, -1].item()]
    full = dict(inp); full['input_ids'] = torch.cat([inp['input_ids'], torch.tensor([ans], device=inp['input_ids'].device)], 1)
    full['attention_mask'] = torch.ones_like(full['input_ids'])
    L0 = inp['input_ids'].shape[1]; sel = torch.arange(L0 - 1, L0 - 1 + len(ans))
    hk = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],))
    try:
        with torch.no_grad(): lg = MODEL_(**full, use_audio_in_video=True).logits[0].float()
    finally: hk.remove()
    return ans, torch.log_softmax(lg, -1).cpu()


probe = torch.nn.Parameter(emb_layer.weight[TOK.encode(' the', add_special_tokens=False)[0]].detach().float().clone().repeat(M, 1)
                           + 0.01 * torch.randn(M, D, device=emb_layer.weight.device))
init_probe = probe.detach().clone()


def student_logits(media, q, ans, mode='probe', P=None):
    """mode: 'probe' (answer reads probes only) | 'none' (answer reads neither media nor probes)."""
    inp = prompt_ids(media, PROBE_TXT + '\n' + q)
    ids = torch.cat([inp['input_ids'], torch.tensor([ans], device=inp['input_ids'].device)], 1)
    full = dict(inp); full['input_ids'] = ids; full['attention_mask'] = torch.ones_like(ids)
    lay = Lay(full); L = ids.shape[1]; seq = ids[0].tolist()
    pid = TOK.encode(PROBE_WORD, add_special_tokens=False)[0]
    ppos = [j for j in range(lay.last_media + 1, L) if seq[j] == pid][:M]
    if len(ppos) != M: raise RuntimeError('probe tokens not found')
    pcols = torch.zeros(L, dtype=torch.bool); pcols[ppos] = True
    media = lay.v | lay.a; m = lay.m.clone()
    after = torch.arange(L) > max(ppos)
    m[after.nonzero().squeeze(1).unsqueeze(1), media.nonzero().squeeze(1).unsqueeze(0)] = float('-inf')   # Q/A rows: no media
    if mode == 'none': m[after.nonzero().squeeze(1).unsqueeze(1), pcols.nonzero().squeeze(1).unsqueeze(0)] = float('-inf')
    P_ = probe if P is None else P
    def hk_emb(mod, args, out):
        out = out.clone(); out[0, ppos] = P_.to(out.dtype).to(out.device); return out
    L0 = inp['input_ids'].shape[1]; sel = torch.arange(L0 - 1, L0 - 1 + len(ans))
    h1 = emb_layer.register_forward_hook(hk_emb)
    h2 = MODEL_.lm_head.register_forward_pre_hook(lambda mod, args: (args[0][:, sel.to(args[0].device), :],))
    kw = {k: v for k, v in full.items() if k != 'attention_mask'}
    try:
        lg = MODEL_(**kw, attention_mask=lay.t(m), position_ids=lay.pos.to(lay.dev), use_audio_in_video=True).logits[0].float()
    finally:
        h1.remove(); h2.remove()
    return torch.log_softmax(lg, -1)


def build(n, tag):
    data = []; t0 = time.time()
    for k in range(n):
        try:
            media = make_clip(vids[k % len(vids)] if tag == 'train' else vids[-1 - k], f'{tag}{k}')
            for q in QUESTIONS:
                ans, lp = teacher(media, q); data.append((media, q, ans, lp))
        except Exception as e:
            print('skip', tag, k, repr(e)[:80], flush=True)
        if k % 20 == 0: print(tag, k, len(data), round(time.time() - t0), 's', flush=True)
    return data


train = build(N_TRAIN, 'train'); evald = build(N_EVAL, 'eval')
print('train examples', len(train), 'eval examples', len(evald), flush=True)


def evaluate(mode, P=None):
    agree, kls = [], []
    with torch.no_grad():
        for media, q, ans, tlp in evald:
            slp = student_logits(media, q, ans, mode, P).cpu()
            agree += (slp.argmax(-1) == tlp.argmax(-1)).float().tolist()
            kls.append(float((tlp.exp() * (tlp - slp)).sum(-1).mean()))
    return round(100 * float(np.mean(agree)), 1), round(float(np.mean(kls)), 3)


log = {'before': {'probe_untrained': evaluate('probe', init_probe), 'no_media_no_probe': evaluate('none')}}
print('BEFORE', log['before'], flush=True)
opt = torch.optim.Adam([probe], lr=LR); t0 = time.time(); hist = []
for step in range(1, STEPS + 1):
    media, q, ans, tlp = train[random.randrange(len(train))]
    with torch.enable_grad():
        slp = student_logits(media, q, ans, 'probe')
        loss = (tlp.to(slp.device).exp() * (tlp.to(slp.device) - slp)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([probe], 1.0); opt.step()
    hist.append(float(loss))
    if step % 50 == 0:
        print('step', step, 'kl', round(float(np.mean(hist[-50:])), 3), round(time.time() - t0), 's',
              'mem', [round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 1) for d in range(torch.cuda.device_count())], flush=True)
    if step % 200 == 0 or step == STEPS:
        log[f'step{step}'] = evaluate('probe'); print('EVAL', step, log[f'step{step}'], flush=True)
        torch.save(probe.detach().cpu(), OUT / 't1_probes.pt')
(OUT / 't1_probe_distill.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
RESULT = log
print(RESULT, flush=True)
