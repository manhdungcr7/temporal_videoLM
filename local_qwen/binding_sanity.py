"""Local sanity check of the synchronous-binding masks on one short clip (NF4)."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import bound_last_logits, media_layout
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
i = int(sys.argv[1]); r = rows[i]; p = r['video_path'].replace(chr(92), '/'); dur = AUD['clips'][p]['dur']
model, proc = runner.load()
media = prepare_video(cut(BASE / p, 0, dur, 'full'))
inp, _ = qwen_inputs(proc, r['question'], media, 'av'); inp = move_inputs(inp, model)
ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
pos, sec, is_v, is_a, last_media = media_layout(model, proc, inp)
ch_v = torch.floor(sec[is_v] / 2.0); ch_a = torch.floor(sec[is_a] / 2.0)
print('chunks video', int(ch_v.min()), int(ch_v.max()), '| audio', int(ch_a.min()), int(ch_a.max()))
with torch.no_grad(): plain = model(**inp, use_audio_in_video=True).logits[0, -1].float()
out = {'gold': r['answer'], 'plain': plain[ids].tolist()}
out['explicit_no_change (restrict=False,bidir=False)'] = bound_last_logits(model, proc, inp, restrict=False, bidir=False)[ids].tolist()
out['bind_local d0'] = bound_last_logits(model, proc, inp, delta=0)[ids].tolist()
out['bind_local d1'] = bound_last_logits(model, proc, inp, delta=1)[ids].tolist()
out['bidir_only d0'] = bound_last_logits(model, proc, inp, delta=0, restrict=False, bidir=True)[ids].tolist()
out['bind_local_bidir d0'] = bound_last_logits(model, proc, inp, delta=0, bidir=True)[ids].tolist()
for k, v in out.items(): print(k, [round(x, 3) for x in v] if isinstance(v, list) else v)
