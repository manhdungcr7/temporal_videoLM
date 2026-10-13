"""Step-by-step debug + activation patching on one CMTB clip (local, NF4, original option order).
Runs:
  plain     : native model                                 (wrong answer expected)
  clean     : gate_audio (video outside the speech window and all audio hidden from non-audio rows)  (right answer)
  patch_fwd : plain run, but at layer L the hidden states of the question/answer rows are REPLACED by the clean ones
              -> from which layer on does carrying the clean question representation fix the answer?
  patch_rev : clean run, but at layer L the question rows are replaced by the plain ones
              -> from which layer on does the plain question representation already carry the error?
Also: logit lens (letters) per layer for plain and clean, and the per-layer cosine distance between the two runs'
question-row states. Everything is written to patching_<idx>.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import media_layout, gated_mask

rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
idx = int(sys.argv[1]); r = rows[idx]
p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']; win = (c['first'] * dur, c['last'] * dur)
model, proc = runner.load()
layers = model.model.layers; NL = len(layers)
media = prepare_video(cut(BASE / p, 0, dur, 'full'))
inp, _ = qwen_inputs(proc, r['question'], media, 'av'); inp = move_inputs(inp, model)
pos, sec, is_v, is_a, last_media = media_layout(model, proc, inp)
L = inp['input_ids'].shape[1]; emb = model.get_input_embeddings().weight
spg = float(inp['video_second_per_grid'].reshape(-1)[0])
k = 3 if '(C)' in r['question'] else 2
letters = [proc.tokenizer.convert_tokens_to_ids(x) for x in 'ABC'[:k]]
rows_q = torch.arange(L, device=emb.device) > last_media
mask_plain = gated_mask(L, is_v, sec, last_media, win, 'none', 0.0, emb.dtype, emb.device, spg, 'all')
mask_clean = gated_mask(L, is_v, sec, last_media, win, 'hard', 0.0, emb.dtype, emb.device, spg, 'all', is_a, True)
kw = {kk: v for kk, v in inp.items() if kk != 'attention_mask'}
posd = pos.to(emb.device)


def run(mask, record=False, patch=None):
    """patch = (layer, tensor[rows_q, d]) replaces the question rows of that layer's output."""
    store = {}; hooks = []
    for li in range(NL):
        def hook(mod, args, out, li=li):
            h = out[0]
            if patch is not None and patch[0] == li:
                h = h.clone(); h[0, rows_q.to(h.device)] = patch[1].to(h.device, h.dtype)
                out = (h,) + tuple(out[1:])
            if record: store[li] = h[0, rows_q.to(h.device)].detach().float().cpu()
            return out
        hooks.append(layers[li].register_forward_hook(hook))
    try:
        with torch.no_grad():
            o = model(**kw, attention_mask=mask, position_ids=posd, use_audio_in_video=True, output_hidden_states=record)
    finally:
        for h in hooks: h.remove()
    lg = o.logits[0, -1].float()
    lens = None
    if record:
        lens = []
        for h in o.hidden_states[1:]:
            x = h[0, -1].to(model.lm_head.weight.device, model.lm_head.weight.dtype)
            x = model.model.norm(x.to(model.model.norm.weight.device)).to(model.lm_head.weight.device)
            lens.append([round(float(v), 3) for v in model.lm_head(x).float()[letters]])
        lens[-1] = [round(float(lg[t]), 3) for t in letters]
    return [round(float(lg[t]), 3) for t in letters], store, lens


t0 = time.time()
plain, H_plain, lens_plain = run(mask_plain, record=True)
clean, H_clean, lens_clean = run(mask_clean, record=True)
gold = 'ABC'.index(r['answer'])
out = dict(idx=idx, question=r['question'], gold=r['answer'], dur=dur, window=win, L=L, n_question_rows=int(rows_q.sum()),
           plain=plain, clean=clean, lens_plain=lens_plain, lens_clean=lens_clean)
out['cos_dist_question_rows'] = [round(float(1 - torch.nn.functional.cosine_similarity(H_plain[i], H_clean[i], dim=-1).mean()), 4) for i in range(NL)]
probe_layers = list(range(0, NL, 3)) + [NL - 1]
out['patch_fwd'] = {}; out['patch_rev'] = {}
for li in probe_layers:
    out['patch_fwd'][li] = run(mask_plain, patch=(li, H_clean[li]))[0]
    out['patch_rev'][li] = run(mask_clean, patch=(li, H_plain[li]))[0]
    print('layer', li, 'fwd', out['patch_fwd'][li], 'rev', out['patch_rev'][li], flush=True)
out['sec'] = round(time.time() - t0)
Path(__file__).with_name(f'patching_{idx}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
pick = lambda v: 'ABC'[int(np.argmax(v))]
print('gold', r['answer'], '| plain', pick(plain), plain, '| clean', pick(clean), clean)
print('fwd patch (plain + clean question rows at layer L):', {li: pick(v) for li, v in out['patch_fwd'].items()})
print('rev patch (clean + plain question rows at layer L):', {li: pick(v) for li, v in out['patch_rev'].items()})
print('cos distance plain vs clean question rows by layer:', out['cos_dist_question_rows'])
