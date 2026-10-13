"""Local test on short CMTB clips: is the residual error (gate 62% vs crop 96%) caused by the audio content itself?
conditions (3 rotated option orders each, summed per option content):
  plain       : whole clip, audio+video
  gate        : out-of-window video hidden from every position
  audio_block : audio hidden from all non-audio positions, NO window gate
  gate_audio  : both"""
import json, sys, random, re, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD
from speech_gate import gated_last_logits
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
OUT = Path(__file__).with_name('gate_audio_ablation.jsonl')
def parse(q):
    p = re.split(r'\s*\(([ABC])\)\s*', q); return p[0].strip(), [p[2].strip(), p[4].strip(), p[6].strip()]
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pool = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == 'CMTB' and AUD['clips'][r['video_path'].replace(chr(92), '/')]['dur'] <= 15]
    random.Random(2).shuffle(pool); pick = pool[:n]
    done = {json.loads(l)['idx'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    model, proc = runner.load()
    ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for i, r in pick:
        if i in done: continue
        t0 = time.time()
        p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']; win = (c['first'] * dur, c['last'] * dur)
        media = prepare_video(cut(BASE / p, 0, dur, 'full')); stem, texts = parse(r['question']); gold = 'ABC'.index(r['answer'])
        sc = {k: np.zeros(3) for k in ('plain', 'gate', 'audio_block', 'gate_audio')}
        for perm in perms:
            prompt = stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))
            inp, _ = qwen_inputs(proc, prompt, media, 'av'); inp = move_inputs(inp, model)
            with torch.no_grad(): lg = {'plain': model(**inp, use_audio_in_video=True).logits[0, -1].float()}
            lg['gate'] = gated_last_logits(model, proc, inp, win, 'hard')[0]
            lg['audio_block'] = gated_last_logits(model, proc, inp, win, 'none', block_audio=True)[0]
            lg['gate_audio'] = gated_last_logits(model, proc, inp, win, 'hard', block_audio=True)[0]
            for k, v in lg.items():
                lp = torch.log_softmax(v[ids].cpu(), -1).numpy()
                for pos, j in enumerate(perm): sc[k][j] += lp[pos]
            del inp; torch.cuda.empty_cache()
        rec = dict(idx=i, gold=gold, dur=round(dur, 1), **{k: dict(pred=int(np.argmax(v)), content=v.round(2).tolist()) for k, v in sc.items()}, sec=round(time.time() - t0))
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(i, 'gold', gold, {k: rec[k]['pred'] for k in sc}, rec['sec'], flush=True)
if __name__ == '__main__': main()
