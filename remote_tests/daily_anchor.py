"""NATURAL-DATA TEST of the CMTB root cause and of the routing fix (Daily-Omni, real videos, no synthetic silence).
Question subset: a quoted utterance + a visual target, e.g. "What action is the man performing when he says 'Are you ready
to rock?'" (Daily-Omni AV Event Alignment / Context understanding). This is the natural counterpart of SVHalluc CMTB.
Fully automatic anchor: Whisper word timestamps on the clip's audio + fuzzy match of the quoted words -> window [t0, t1]
(fixed component, no labels). Window widened by 1 s each side. Items whose quote is not found (match < 0.6) are kept and
flagged; every condition degrades to base there.
Conditions (4 options, rotated, summed per option content):
  base          native model
  anchor_text   "Note: the quoted words are spoken from t0 to t1 s." + question       (SVHalluc: no effect)
  crop          only the window's video, no audio  (= TFO-style pipeline crop, arXiv 2609.04242)
  crop_av       the window's video with its audio
  route_t       attention routing in layers 13-27: text rows read video only inside the window (time)
  route_ta      route_t + audio blocked for the text rows (time + source)      (SVHalluc CMTB: 51 -> 100)
Decision: route_ta or route_t > base by a CI-backed margin on natural data -> the temporal-selection cause and the fix are
not SVHalluc artefacts. Writes OUT/daily_anchor.jsonl."""
import json, re, tarfile, time, difflib
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download

LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABCD']
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABCD', perm))
MID = list(range(13, 28))

DO = Path('/kaggle/working/daily_omni'); DO.mkdir(parents=True, exist_ok=True)
qa = json.loads(Path(hf_hub_download('liarliar/Daily-Omni', 'qa.json', repo_type='dataset')).read_text(encoding='utf-8'))
vid_dir = DO / 'videos'
if not vid_dir.exists():
    with tarfile.open(hf_hub_download('liarliar/Daily-Omni', 'Videos.tar', repo_type='dataset')) as t: t.extractall(vid_dir)
paths = {p.stem: p for p in vid_dir.rglob('*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')}
print('videos', len(paths), flush=True)

QUOTE = re.compile(r"""(?:(?<=\s)|^)['"‘“](.{4,}?)['"’”](?![A-Za-z])""")
VIS = re.compile(r"\b(doing|visual|visible|shown|show|seen|see|action|on screen|appear\w*|happening|performing|holding|scene|displayed|wearing|gesture|looking)\b", re.I)
AUDIO_TARGET = re.compile(r"\b(statement|say|said|sound|music|hear|heard)\b.*\?$", re.I)
items = []
for i, q in enumerate(qa):
    m = QUOTE.search(q['Question'])
    if m and VIS.search(q['Question']) and q['Type'] in ('AV Event Alignment', 'Context understanding'):
        items.append((i, q, m.group(1)))
items.sort(key=lambda x: (x[1]['video_duration'] != '30s', x[0]))          # 30-s clips first
print('items', len(items), flush=True)

ASR = None
def asr_words(path):
    global ASR
    if ASR is None:
        from transformers import pipeline
        ASR = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=0, torch_dtype=torch.float16, chunk_length_s=30)
    a = decode_audio(path)
    out = ASR({'raw': a, 'sampling_rate': 16000}, return_timestamps='word', generate_kwargs={'task': 'transcribe'})
    return [(c['text'], c['timestamp'][0], c['timestamp'][1]) for c in out['chunks'] if c['timestamp'][0] is not None], len(a) / 16000


norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower()).split()


def find_quote(words, quote):
    q = norm(quote); W = [(norm(w), s, e) for w, s, e in words]; flat = []
    for toks, s, e in W:
        for t in toks: flat.append((t, s, e))
    best = (0.0, None)
    for L in range(max(1, len(q) - 2), len(q) + 3):
        for st in range(0, max(1, len(flat) - L + 1)):
            seg = flat[st: st + L]
            if not seg: continue
            r = difflib.SequenceMatcher(None, q, [t for t, _, _ in seg]).ratio()
            if r > best[0]: best = (r, (seg[0][1], seg[-1][2] if seg[-1][2] is not None else seg[-1][1] + 0.5))
    return best


def prep(path, dur):
    if dur <= 42: return prepare_video(path)
    from qwen_omni_utils import process_mm_info
    import lab_native_video as LNV
    fps = max(0.5, (7200 / dur - 25) / 72)
    ele = {'type': 'video', 'video': str(path), 'max_pixels': LNV.VIDEO_MAX_PIXELS, 'fps': fps}
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': LNV.SYSTEM}]}, {'role': 'user', 'content': [ele, {'type': 'text', 'text': ''}]}]
    au, im, vi, vk = process_mm_info(conv, use_audio_in_video=True, return_video_kwargs=True)
    f = vk['fps']; f = f[0] if isinstance(f, (list, tuple)) else f
    return {'path': str(path), 'audio': au, 'videos': vi, 'fps': float(f), 'max_pixels_per_frame': LNV.VIDEO_MAX_PIXELS}


def route_logits(inp, window, block_audio):
    emb = MODEL_.get_input_embeddings().weight
    pos, sec, is_v, is_a, last_media = media_layout(MODEL_, PROC, inp)
    L = inp['input_ids'].shape[1]; spg = float(inp['video_second_per_grid'].reshape(-1)[0])
    base = gated_mask(L, is_v, sec, last_media, window, 'none', 0.0, emb.dtype, emb.device, spg, 'all')
    routed = gated_mask(L, is_v, sec, last_media, window, 'hard', 0.0, emb.dtype, emb.device, spg, 'all', is_a, block_audio)
    def pre(mod, args, kwargs):
        am_ = kwargs.get('attention_mask'); kwargs['attention_mask'] = routed.to(am_.device if am_ is not None else routed.device); return args, kwargs
    hooks = [MODEL_.model.layers[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True) for li in MID]
    kw = {k: v for k, v in inp.items() if k != 'attention_mask'}
    try:
        with torch.no_grad(): return MODEL_(**kw, attention_mask=base, position_ids=pos.to(emb.device), use_audio_in_video=True).logits[0, -1].float()
    finally:
        for h in hooks: h.remove()


def sc(media, stem, texts, mode='av', window=None, block_audio=False):
    k = len(texts); tot = np.zeros(k)
    for perm in rotations(k):
        inp, _ = qwen_inputs(PROC, render(stem, texts, perm), media, mode); inp = move_inputs(inp, MODEL_)
        if window is not None: lg = route_logits(inp, window, block_audio)
        else:
            with torch.no_grad(): lg = MODEL_(**inp, use_audio_in_video=(mode == 'av')).logits[0, -1].float()
        lp = torch.log_softmax(lg[LETTERS[:k]].cpu(), -1).numpy()
        for p_, j in enumerate(perm): tot[j] += lp[p_]
        del inp
    torch.cuda.empty_cache(); return tot.round(3).tolist()


res = OUT / 'daily_anchor.jsonl'
done = {json.loads(l)['i'] for l in res.read_text().splitlines() if l.strip()} if res.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, q, quote) in enumerate(items):
    if i in done: continue
    t0 = time.time(); rec = dict(i=i, type=q['Type'], dur_tag=q['video_duration'], gold='ABCD'.index(q['Answer']), quote=quote)
    try:
        vp = next((p for k, p in paths.items() if q['video_id'] in k), None)
        if vp is None: raise RuntimeError('video not found')
        src = cut(vp, 0, 10 ** 4, 'full')
        words, dur = asr_words(src); score_, win = find_quote(words, quote)
        rec.update(dur=round(dur, 2), match=round(score_, 3), win=None if win is None else [round(win[0], 2), round(win[1], 2)])
        texts = [re.sub(r'^[A-D]\.\s*', '', c).strip() for c in q['Choice']]; stem = q['Question']
        media = prep(src, dur)
        rec['base'] = sc(media, stem, texts)
        if win is not None and score_ >= 0.6:
            w = (max(0.0, win[0] - 1.0), min(dur, win[1] + 1.0)); rec['window'] = [round(w[0], 2), round(w[1], 2)]
            rec['anchor_text'] = sc(media, f'Note: the quoted words are spoken from {w[0]:.1f} to {w[1]:.1f} seconds of the video. ' + stem, texts)
            rec['route_t'] = sc(media, stem, texts, window=w)
            rec['route_ta'] = sc(media, stem, texts, window=w, block_audio=True)
            crop_src = cut(src, w[0], w[1], f'anchor{i}')
            mc = prepare_video(crop_src)
            rec['crop'] = sc(mc, stem, texts, mode='vision')
            rec['crop_av'] = sc(mc, stem, texts)
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:300]; torch.cuda.empty_cache()
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    if 'error' in rec: print(n + 1, len(items), i, 'ERR', rec['error'][:80], flush=True); continue
    print(n + 1, len(items), i, rec['dur_tag'], 'match', rec['match'], {c: am(rec[c]) == rec['gold'] for c in ('base', 'anchor_text', 'route_t', 'route_ta', 'crop', 'crop_av') if c in rec}, rec['sec'], 's', flush=True)

R = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
A = [r for r in R if 'route_ta' in r]
RESULT = {'items': len(R), 'anchored': len(A), 'errors': sum('error' in r for r in R),
          'acc_on_anchored': {c: round(100 * float(np.mean([am(r[c]) == r['gold'] for r in A])), 1) for c in ('base', 'anchor_text', 'route_t', 'route_ta', 'crop', 'crop_av')} if A else None,
          'acc_all_items_base': round(100 * float(np.mean([am(r['base']) == r['gold'] for r in R if 'base' in r])), 1) if R else None}
LETTERS = [PROC.tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
def render(stem, texts, perm): return stem + ' ' + ' '.join(f'({L}) {texts[j]}' for L, j in zip('ABC', perm))
print(RESULT, flush=True)
