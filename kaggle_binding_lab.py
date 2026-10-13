"""Human-inspired binding lab: does changing HOW audio and video of the same moment are bound help?
Usage: python kaggle_binding_lab.py ROOT OUT PER_LABEL [SHARD_ID NUM_SHARDS]
Items: SVHalluc temporal (TA/TF/CMTB) AND semantic (GSA/FGSA/CMSB) so that a gain on one family cannot hide a loss on
the other. Every condition is scored with rotated option orders, summed per option content. No training, one forward.
  base        original model
  bind0       cross-modal attention only between tokens of the SAME 2-s chunk (causal kept)
  bind0_bi    bind0 + video tokens may also read the audio of their own chunk (it comes later in the sequence)
  bind1_bi    same with +-1 chunk tolerance
  bi_only     only add video->same-chunk-audio edges, no restriction
  balance     question/answer rows: audio keys get +log(n_video/n_audio) so both streams get equal prior attention
Labels are used only for scoring."""
import os, sys, json, re, subprocess, time, random
from pathlib import Path
import numpy as np
import torch

os.environ['FORCE_QWENVL_VIDEO_READER'] = 'decord'
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PER_LABEL = int(sys.argv[3])
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MAX_DUR = float(os.environ.get('LAB_MAX_DUR', 42.0 if torch.cuda.device_count() >= 2 else 28.0))
sys.path.insert(0, str(ROOT))
import av, librosa
from huggingface_hub import hf_hub_download

DATASET, REVISION = 'zcs15/SVHalluc', '211b4f14ab66ae9557b69e302d1f090e0e610839'
MODEL, MREV = 'Qwen/Qwen2.5-Omni-3B', 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
BS = chr(92); FF = os.environ.get('FFMPEG_BIN', 'ffmpeg')


def decode_audio(path):
    chunks = []
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        for fr in c.decode(audio=0):
            for p in rs.resample(fr): chunks.append(p.to_ndarray().reshape(-1))
        for p in rs.resample(None): chunks.append(p.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype('float32', copy=False)


_orig = librosa.load
def _load(path, sr=None, offset=0.0, duration=None, **k):
    if str(path).lower().endswith('.mp4'):
        a = decode_audio(path)
        return a[int(sr * offset): None if duration is None else int(sr * (offset + duration))], sr
    return _orig(path, sr=sr, offset=offset, duration=duration, **k)
librosa.load = _load

from lab_native_video import prepare_video, qwen_inputs, move_inputs
from speech_gate import media_layout, binding_mask, balance_mask
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

CLIPS = OUT / 'clips'; CLIPS.mkdir(parents=True, exist_ok=True)


def fetch(name):
    local = os.environ.get('LAB_LOCAL_DATA')
    if local and (Path(local) / name).exists(): return Path(local) / name
    return Path(hf_hub_download(DATASET, name, repo_type='dataset', revision=REVISION))


def normalised(src):
    dst = CLIPS / f'{Path(src).stem}_full.mp4'
    if not dst.exists():
        subprocess.run([FF, '-hide_banner', '-y', '-i', str(src), '-map', '0:v:0', '-map', '0:a:0', '-vf', 'fps=30', '-fps_mode', 'cfr',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                        '-af', 'aresample=async=1:first_pts=0', str(dst)], check=True, capture_output=True)
    return dst


def parse_question(q):
    parts = re.split(r'\s*\(([ABC])\)\s*', q)
    return parts[0].strip(), [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]


def rotations(k): return [tuple((s + j) % k for j in range(k)) for s in range(k)]
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))


MODEL_ = PROC = LETTERS = None
CONDS = {'bind0': dict(delta=0, restrict=True, bidir=False), 'bind0_bi': dict(delta=0, restrict=True, bidir=True),
         'bind1_bi': dict(delta=1, restrict=True, bidir=True), 'bi_only': dict(delta=0, restrict=False, bidir=True)}


def load_model():
    cache = os.environ.get('LAB_CACHE')
    kw = dict(revision=MREV, torch_dtype=torch.float16, attn_implementation='sdpa', cache_dir=cache)
    if os.environ.get('LAB_NF4'):
        from transformers import BitsAndBytesConfig
        kw.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                  bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    elif torch.cuda.device_count() >= 2: kw.update(device_map='balanced', max_memory={0: '10GiB', 1: '10GiB'})
    else: kw.update(device_map={'': 0})
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL, **kw).eval()
    return m, Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=MREV, cache_dir=cache)


def logits_for(inp, cond):
    if cond == 'base':
        with torch.no_grad(): return MODEL_(**inp, use_audio_in_video=True).logits[0, -1].float()
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]
    if cond == 'balance': mask = balance_mask(L, is_v, is_a, last_media, emb.dtype, emb.device)
    else: mask = binding_mask(L, is_v, is_a, sec, emb.dtype, emb.device, **CONDS[cond])
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    with torch.no_grad():
        return MODEL_(**kw, attention_mask=mask, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()


def run_item(r):
    src = normalised(fetch(r['video_path'].replace(BS, '/')))
    dur = len(decode_audio(src)) / 16000
    if dur > MAX_DUR: raise RuntimeError('skipped: longer than MAX_DUR')
    media = prepare_video(src)
    stem, texts = parse_question(r['question']); k = len(texts)
    conds = ['base', 'bind0', 'bind0_bi', 'bind1_bi', 'bi_only', 'balance']
    tot = {c: np.zeros(k) for c in conds}
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, 'av'); inp = move_inputs(inp, MODEL_)
        for c in conds:
            lp = torch.log_softmax(logits_for(inp, c)[LETTERS[:k]].cpu(), -1).numpy()
            for pos, j in enumerate(perm): tot[c][j] += lp[pos]
        del inp; torch.cuda.empty_cache()
    return dict(dur=round(dur, 2), **{c: v.round(3).tolist() for c, v in tot.items()})


def main():
    global MODEL_, PROC, LETTERS
    rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
    cells = {}
    for i, r in enumerate(rows):
        t = r['task_type_abbr'].replace('CSMB', 'CMSB')
        if (r['data_type'] == 'semantic' and '(A) Yes (B) No' not in r['question']): continue
        cells.setdefault((t, r['answer']), []).append((i, r))
    items = []
    for key in sorted(cells):
        pool = cells[key]; random.Random(0).shuffle(pool); items += pool[:PER_LABEL]
    items.sort(key=lambda x: x[0])
    mine = [x for k, x in enumerate(items) if k % NSHARD == SHARD]
    if os.environ.get('LAB_ONLY_IDS'):
        want = {int(v) for v in os.environ['LAB_ONLY_IDS'].split(',')}; mine = [x for x in enumerate(rows) if x[0] in want]
    res = OUT / f'binding_shard{SHARD}of{NSHARD}.jsonl'
    done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
    MODEL_, PROC = load_model(); LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
    print('model ready; items', len(mine), 'done', len(done), flush=True)
    for n, (i, r) in enumerate(mine):
        if i in done: continue
        t0 = time.time(); task = r['task_type_abbr'].replace('CSMB', 'CMSB')
        rec = dict(idx=i, task=task, family=r['data_type'], gold='ABC'.index(r['answer']))
        try:
            rec.update(run_item(r)); rec['sec'] = round(time.time() - t0)
        except Exception as e:
            rec['error'] = repr(e)[:300]
        torch.cuda.empty_cache()
        with res.open('a') as f: f.write(json.dumps(rec) + '\n')
        if 'error' in rec: print(n + 1, '/', len(mine), task, i, rec['error'][:60], flush=True); continue
        g = rec['gold']; ok = lambda c: 'Y' if int(np.argmax(rec[c])) == g else '-'
        print(n + 1, '/', len(mine), task, i, ' '.join(f'{c}:{ok(c)}' for c in ('base', 'bind0', 'bind0_bi', 'bind1_bi', 'bi_only', 'balance')), rec['sec'], 's', flush=True)


if __name__ == '__main__':
    main()
