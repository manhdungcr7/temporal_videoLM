"""Sanity + first signal for speech-anchored attention gating on a few short CMTB clips (local NF4)."""
import json, sys, random
from pathlib import Path
import torch, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import gated_last_logits

rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
OUT = Path(__file__).with_name('gate_sanity.jsonl')
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

def parse(q):
    import re
    parts = re.split(r'\s*\(([ABC])\)\s*', q); return parts[0].strip(), [parts[2].strip(), parts[4].strip(), parts[6].strip()]

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cm = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == 'CMTB' and AUD['clips'][r['video_path'].replace(chr(92), '/')]['dur'] <= 20]
    random.Random(1).shuffle(cm); sel = cm[:n]
    model, proc = runner.load()
    ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    for i, r in sel:
        p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']; win = (c['first'] * dur, c['last'] * dur)
        media = prepare_video(cut(BASE / p, 0, dur, 'full'))
        stem, texts = parse(r['question']); gold = 'ABC'.index(r['answer'])
        rec = dict(idx=i, gold=gold, dur=round(dur, 1), win=[round(x, 1) for x in win])
        scores = {k: np.zeros(3) for k in ('plain', 'none', 'hard', 'soft4')}
        for k_perm, perm in enumerate(PERMS):
            prompt = stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))
            inp, _ = qwen_inputs(proc, prompt, media, 'av'); inp = move_inputs(inp, model)
            with torch.no_grad():
                plain = model(**inp, use_audio_in_video=True).logits[0, -1].float()
            res = {'plain': plain}
            for mode, beta, key in (('none', 0, 'none'), ('hard', 0, 'hard'), ('soft', 4.0, 'soft4')):
                lg, info = gated_last_logits(model, proc, inp, win, mode, beta)
                res[key] = lg
            if k_perm == 0:
                rec['max_abs_diff_none_vs_plain'] = float((res['none'] - res['plain']).abs().max())
                rec['layout'] = info
            for key, lg in res.items():
                lp = torch.log_softmax(lg[ids], -1).cpu().numpy()
                for pos, j in enumerate(perm): scores[key][j] += lp[pos]
            torch.cuda.empty_cache()
        rec.update({k: dict(pred=int(np.argmax(v)), content=v.round(2).tolist()) for k, v in scores.items()})
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(i, 'gold', gold, {k: rec[k]['pred'] for k in scores}, 'diff', round(rec['max_abs_diff_none_vs_plain'], 4), rec['layout'], flush=True)

if __name__ == '__main__':
    main()
