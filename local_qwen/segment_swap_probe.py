"""Causal test of temporal position bias: same content, swapped segment order.
For a short CMTB clip with speech window S and muted part Q, build
  orig    : the original order (re-encoded CFR)
  swapped : the two segments in the opposite order (each segment keeps its own audio)
The gold answer (action visible while speech is heard) is unchanged; only its position in the clip flips.
Scored with 3 rotated option orders, under audio+video ('av') and video only ('vision')."""
import json, sys, random, re, subprocess, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD, FFMPEG, CLIPS
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
OUT = Path(__file__).with_name('segment_swap_probe.jsonl')
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

def parse(q):
    p = re.split(r'\s*\(([ABC])\)\s*', q); return p[0].strip(), [p[2].strip(), p[4].strip(), p[6].strip()]

def concat(a, b, dst):
    if not dst.exists():
        subprocess.run([str(FFMPEG), '-hide_banner', '-y', '-i', str(a), '-i', str(b), '-filter_complex',
                        '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]', '-map', '[v]', '-map', '[a]',
                        '-r', '30', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(dst)],
                       check=True, capture_output=True)
    return dst

def score(model, proc, media, stem, texts, mode, ids):
    tot = np.zeros(3)
    for perm in PERMS:
        prompt = stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))
        inp, _ = qwen_inputs(proc, prompt, media, mode); inp = move_inputs(inp, model)
        with torch.no_grad(): lg = model(**inp, use_audio_in_video=(mode == 'av')).logits[0, -1].float()
        lp = torch.log_softmax(lg[ids].cpu(), -1).numpy()
        for pos, j in enumerate(perm): tot[j] += lp[pos]
        del inp; torch.cuda.empty_cache()
    return tot

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pool = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == 'CMTB' and AUD['clips'][r['video_path'].replace(chr(92), '/')]['dur'] <= 20]
    random.Random(3).shuffle(pool)
    early = [x for x in pool if AUD['clips'][x[1]['video_path'].replace(chr(92), '/')]['centroid'] < 0.5][:n // 2]
    late = [x for x in pool if AUD['clips'][x[1]['video_path'].replace(chr(92), '/')]['centroid'] >= 0.5][:n // 2]
    done = {json.loads(l)['idx'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    model, proc = runner.load(); ids = [proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    for i, r in early + late:
        if i in done: continue
        t0 = time.time()
        p = r['video_path'].replace(chr(92), '/'); c = AUD['clips'][p]; dur = c['dur']; s0, s1 = c['first'] * dur, c['last'] * dur
        speech_first = c['centroid'] < 0.5
        seg_s = cut(BASE / p, s0, s1, 'segS')
        seg_q = cut(BASE / p, s1, dur, 'segQ') if speech_first else cut(BASE / p, 0, s0, 'segQ')
        orig = concat(seg_s, seg_q, CLIPS / f'{Path(p).stem}_orig2.mp4') if speech_first else concat(seg_q, seg_s, CLIPS / f'{Path(p).stem}_orig2.mp4')
        swap = concat(seg_q, seg_s, CLIPS / f'{Path(p).stem}_swap.mp4') if speech_first else concat(seg_s, seg_q, CLIPS / f'{Path(p).stem}_swap.mp4')
        stem, texts = parse(r['question']); gold = 'ABC'.index(r['answer'])
        rec = dict(idx=i, gold=gold, dur=round(dur, 1), speech_first_in_orig=speech_first)
        for name, clip in (('orig', orig), ('swap', swap)):
            media = prepare_video(clip)
            for mode in ('av', 'vision'):
                s = score(model, proc, media, stem, texts, mode, ids)
                rec[f'{name}_{mode}'] = dict(pred=int(np.argmax(s)), content=s.round(2).tolist())
        rec['sec'] = round(time.time() - t0)
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(i, 'gold', gold, 'speech first in orig', speech_first, {k: rec[k]['pred'] for k in rec if k.startswith(('orig_', 'swap_'))}, rec['sec'], flush=True)

if __name__ == '__main__': main()
