"""Minimal local check of attention gating on the shortest CMTB clips (one option order, NF4)."""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import gated_last_logits
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
model, proc = runner.load()
ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
res = []
for i in map(int, sys.argv[1].split(',')):
    r = rows[i]; p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']
    win = (c['first'] * dur, c['last'] * dur)
    media = prepare_video(cut(BASE / p, 0, dur, 'full'))
    inp, _ = qwen_inputs(proc, r['question'], media, 'av'); inp = move_inputs(inp, model)
    out = dict(idx=i, gold=r['answer'], win=[round(x, 1) for x in win])
    with torch.no_grad(): plain = model(**inp, use_audio_in_video=True).logits[0, -1].float()
    out['plain'] = [round(x, 2) for x in plain[ids].tolist()]
    for name, mode, scope in (('none', 'none', 'all'), ('hard_text', 'hard', 'text'), ('hard_all', 'hard', 'all')):
        lg, info = gated_last_logits(model, proc, inp, win, mode, 0, scope=scope)
        out[name] = [round(x, 2) for x in lg[ids].tolist()]
    out['pred'] = {k: 'ABC'[max(range(3), key=lambda j: out[k][j])] for k in ('plain', 'none', 'hard_text', 'hard_all')}
    res.append(out); print(json.dumps(out), flush=True)
    torch.cuda.empty_cache()
Path(__file__).with_name('gate_min_check.json').write_text(json.dumps(res, indent=1))
