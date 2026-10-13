"""Step 1: does Qwen2.5-Omni-3B know WHEN the speech is heard?
For short temporal clips (CFR-normalised whole clip, audio+video):
  q_ts   : SVHalluc paper's own prompt -> start/end seconds; IoU with the RMS audio-active window.
  q_half : 'first half or second half' forced choice, scored by first-token logits of the two words.
  q_ts_audio : the same timestamp prompt with audio only (no video).
RMS window = reference (analysis only)."""
import json, sys, re, time, random
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import runner
from lab_native_video import prepare_video, qwen_inputs, move_inputs
from temporal_anchor_probe import cut, BASE, AUD

OUT = Path(__file__).with_name('speech_localization_probe.jsonl')
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))
Q_TS = 'When is the speech heard in the audio? Provide the start and end timestamps (in seconds).'
Q_HALF = 'Is the speech heard in the first half or in the second half of the video? Answer with one word: first or second.'

def gen(model, proc, prompt, media, mode, n=40):
    inp, _ = qwen_inputs(proc, prompt, media, mode); inp = move_inputs(inp, model)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=n, do_sample=False, use_audio_in_video=(mode == 'av'))
    return proc.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]

def half_scores(model, proc, media):
    inp, _ = qwen_inputs(proc, Q_HALF, media, 'av'); inp = move_inputs(inp, model)
    with torch.no_grad():
        lg = model(**inp, use_audio_in_video=True).logits[0, -1].float()
    s = {}
    for w in ('first', 'First', 'second', 'Second'):
        ids = proc.tokenizer.encode(w, add_special_tokens=False)
        s[w] = float(lg[ids[0]])
    return max(s['first'], s['First']), max(s['second'], s['Second'])

def nums(t):
    v = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', t)]
    return (v[0], v[1]) if len(v) >= 2 else None

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); uni = max(a[1], b[1]) - min(a[0], b[0])
    return inter / uni if uni > 0 else 0

def main():
    paths = sorted({r['video_path'].replace(chr(92), '/') for r in rows if r['data_type'] == 'temporal'})
    short = [p for p in paths if AUD['clips'][p]['dur'] <= 25]
    random.Random(0).shuffle(short)
    first = [p for p in short if AUD['clips'][p]['centroid'] < 0.5][:12]
    second = [p for p in short if AUD['clips'][p]['centroid'] >= 0.5][:12]
    done = {json.loads(l)['path'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    model, proc = runner.load()
    for p in first + second:
        if p in done: continue
        c = AUD['clips'][p]; dur = c['dur']; ref = (c['first'] * dur, c['last'] * dur)
        t0 = time.time()
        full = cut(BASE / p, 0, dur, 'full'); media = prepare_video(full)
        rec = dict(path=p, dur=round(dur, 1), ref=[round(x, 1) for x in ref], ref_half='first' if c['centroid'] < 0.5 else 'second')
        try:
            t = gen(model, proc, Q_TS, media, 'av'); ab = nums(t)
            rec['q_ts'] = dict(text=t, span=ab, iou=round(iou(ab, ref), 3) if ab else None)
            sf, ss = half_scores(model, proc, media)
            rec['q_half'] = dict(first=sf, second=ss, pred='first' if sf > ss else 'second')
            t = gen(model, proc, Q_TS, media, 'audio'); ab = nums(t)
            rec['q_ts_audio'] = dict(text=t, span=ab, iou=round(iou(ab, ref), 3) if ab else None)
        except Exception as e:
            rec['error'] = repr(e)[:200]
        rec['sec'] = round(time.time() - t0); torch.cuda.empty_cache()
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(p[-30:], rec.get('ref'), rec.get('q_ts', {}).get('span'), rec.get('q_half', {}).get('pred'), rec['ref_half'], rec['sec'], flush=True)

if __name__ == '__main__':
    main()
