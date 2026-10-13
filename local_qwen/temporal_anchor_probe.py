"""Anchor-then-look probe on CMTB.
Conditions per item (same checkpoint, greedy, fresh context):
  base         : CFR-normalised full clip, audio+video, original question.
  speech_win   : only the audio-active window, video only, "which action is occurring in this video".
  silent_win   : only the muted window, video only, same question (control).
Speech window = span of audio-active 50 ms frames (RMS audit). Labels are used only for scoring."""
import json, sys, subprocess, time, re
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent))
import runner  # sets PATH/ffmpeg, PyAV librosa patch, loader
from lab_native_video import prepare_video, qwen_inputs, move_inputs

BASE = Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark')
HERE = Path(__file__).parent
FFMPEG = Path(r'F:\A_star_paper_NLP_ACL\research_topics_learning\single_sample_prompt_lab\validation_ffmpeg\ffmpeg.exe')
CLIPS = HERE / 'anchor_clips'; CLIPS.mkdir(exist_ok=True)
OUT = HERE / 'temporal_anchor_probe.jsonl'
AUD = json.loads((HERE / 'temporal_structure_audit.json').read_text(encoding='utf-8'))
rows = json.loads((BASE / 'samples.json').read_text(encoding='utf-8'))

def cut(src, start, end, tag):
    dst = CLIPS / f'{Path(src).stem}_{tag}.mp4'
    if not dst.exists():
        cmd = [str(FFMPEG), '-hide_banner', '-y', '-ss', f'{start:.2f}', '-to', f'{end:.2f}', '-i', str(src),
               '-map', '0:v:0', '-map', '0:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'ultrafast',
               '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'aresample=async=1:first_pts=0', str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
    return dst

def options(q):
    return re.findall(r'\(([ABC])\)\s*(.*?)(?=\s*\([ABC]\)|$)', q)

def score(model, proc, prompt, media, mode):
    inp, _ = qwen_inputs(proc, prompt, media, mode)
    inp = move_inputs(inp, model)
    with torch.no_grad():
        out = model(**inp, use_audio_in_video=(mode == 'av'))
    logits = out.logits[0, -1].float()
    ids = {L: proc.tokenizer.convert_tokens_to_ids(L) for L in 'ABC'}
    s = {L: float(logits[i]) for L, i in ids.items()}
    return max(s, key=s.get), s

def main():
    items = [(i, r) for i, r in enumerate(rows) if r['task_type_abbr'] == 'CMTB']
    sel = []
    for g in 'ABC':
        pool = [(i, r) for i, r in items if r['answer'] == g and AUD['clips'][r['video_path'].replace('\\', '/')]['dur'] <= 30]
        sel += pool[:8]
    done = {json.loads(l)['idx'] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    model, proc = runner.load()
    for i, r in sel:
        if i in done: continue
        p = r['video_path'].replace('\\', '/'); c = AUD['clips'][p]; dur = c['dur']
        s0, s1 = c['first'] * dur, c['last'] * dur
        silent = (0.0, max(s0 - 0.05, 0.5)) if c['centroid'] >= 0.5 else (min(s1 + 0.05, dur - 0.5), dur)
        src = BASE / p
        full = cut(src, 0, dur, 'full'); sw = cut(src, s0, s1, 'speech'); qw = cut(src, *silent, 'silent')
        opts = '\n'.join(f'({L}) {t}' for L, t in options(r['question']))
        q_vis = f'Which visual action is occurring in this video?\n{opts}\nAnswer with the option letter.'
        rec = dict(idx=i, gold=r['answer'], dur=round(dur, 1), speech=[round(s0, 1), round(s1, 1)], silent=[round(x, 1) for x in silent])
        t = time.time()
        for name, path, prompt, mode in (('base', full, r['question'], 'av'), ('speech_win', sw, q_vis, 'vision'),
                                         ('silent_win', qw, q_vis, 'vision')):
            try:
                pred, s = score(model, proc, prompt, prepare_video(path), mode)
                rec[name] = dict(pred=pred, logits=s)
            except Exception as e:
                rec[name] = dict(pred=None, error=repr(e)[:200])
            torch.cuda.empty_cache()
        rec['sec'] = round(time.time() - t)
        with OUT.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(i, r['answer'], {k: rec[k]['pred'] for k in ('base', 'speech_win', 'silent_win')}, rec['sec'], flush=True)

if __name__ == '__main__':
    main()
