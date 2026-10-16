"""J18 — ROOT CAUSE "does TMRoPE carry time for this omni model?"  Needs the J8 helpers (Lay).
Architecture (Qwen2.5-Omni-3B, mrope_section [16,24,24], head_dim 128, theta 1e6, 25 ids/s): for VIDEO tokens the time id only
enters the 16 highest-frequency rotary pairs (periods 0.25-6.4 s); the other 48 pairs carry row/column + a constant. AUDIO tokens
carry the time id in all 64 pairs. Following "Causality Matters" (2508.11576), we test whether the model actually USES these ids,
separately for video and audio. Token ORDER is never changed here, only the position ids:
  base         original ids
  v_tconst     every video token gets the temporal id of the first video frame (no positional time for vision)
  v_tshuffle   the 2-s chunks of VIDEO temporal ids are permuted (positional time contradicts the token order)
  a_tconst     every audio token gets the id of the first audio token (all three sections; no positional time for audio)
  av_tconst    both
Text ids after the media are left unchanged. Report per (task, gold).  Writes OUT/j18_tmrope_time.jsonl."""
import json, numpy as np, torch
if 'Lay' not in globals():
    exec((ROOT / 'j8_acar.py').read_text(encoding='utf-8').split(chr(10) + 'items = sample_items')[0], globals())

J18_OUT = globals().get('J18_OUT', 'j18_tmrope_time.jsonl')
GEN = torch.Generator().manual_seed(0)


def pos_forward(inp, kind):
    lay = Lay(inp); pos = lay.pos.clone().cpu()                                 # [3, 1, L]; masks (lay.v, lay.chunk) are CPU too
    v, a = lay.v, lay.a
    if kind in ('v_tconst', 'av_tconst'):
        pos[0, 0, v] = pos[0, 0, v].min()
    if kind == 'v_tshuffle':
        t0 = pos[0, 0, v].clone(); ch = lay.chunk[v]
        have = torch.unique(ch)                                                 # chunks that contain video tokens (audio can be longer)
        lut = torch.arange(lay.nc); lut[have] = have[torch.randperm(len(have), generator=GEN)]   # permute only among them
        base_t = int(t0.min()) + 50 * torch.arange(lay.nc)                      # chunk c starts 2 s = 50 ids after chunk 0
        pos[0, 0, v] = t0 - base_t[ch] + base_t[lut[ch]]                        # keeps the offset inside each chunk
    if kind in ('a_tconst', 'av_tconst'):
        pos[:, 0, a] = pos[:, 0, a].min()
    kw = {k_: v_ for k_, v_ in inp.items() if k_ != 'attention_mask'}
    with torch.no_grad():
        return MODEL_(**kw, attention_mask=lay.t(lay.m), position_ids=pos.to(lay.dev), use_audio_in_video=True).logits[0, -1].float()


def item(i, r):
    src, a, dur, w = prep_clip(r, MAXD)
    media = prepare_video(src); stem, texts = parse_question(r['question'])
    rec = dict(dur=round(dur, 2))
    for kind in ('base', 'v_tconst', 'v_tshuffle', 'a_tconst', 'av_tconst'):
        rec[kind] = score(media, stem, texts, fwd=lambda inp, kind=kind: pos_forward(inp, kind))
    return rec


items = sample_items(tuple(globals().get('J18_TASKS', ('TA', 'TF', 'CMTB'))), PER_LABEL, seed=SEED)
SHARD = globals().get('J12_SHARD')
if SHARD: items = items[SHARD[0]::SHARD[1]]
C = ('base', 'v_tconst', 'v_tshuffle', 'a_tconst', 'av_tconst')
R = run_loop(items, J18_OUT, item, show=C)
OK = [r for r in R if 'error' not in r]
am = lambda v: int(np.argmax(v))
RESULT = {}
for t in ('TA', 'TF', 'CMTB'):
    for g in sorted({r['gold'] for r in OK if r['task'] == t}):
        S = [r for r in OK if r['task'] == t and r['gold'] == g]
        RESULT[f'{t}_{"ABCD"[g]}'] = {c: f'{sum(am(r[c]) == g for r in S)}/{len(S)}' for c in C}
RESULT['answer_changed_vs_base'] = {c: round(float(np.mean([am(r[c]) != am(r['base']) for r in OK])), 3) for c in C[1:]}
print(RESULT, flush=True)
