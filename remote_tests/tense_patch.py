"""WHERE does the speech tense enter the TF decision?  (activation patching, same video, only the tense of the speech differs)
Uses the stimuli of tense_intervention.jsonl (sentences already written there). For each TF item with valid rewrites:
  clean   = clip whose speech is the FUTURE rewrite   (model leans to answer C = Future)
  corrupt = clip whose speech is the PAST rewrite     (model leans to answer A = Past)
Original option order (A Past, B Present, C Future). Metric m = logit(C) - logit(A) at the last token.
  fwd(l): run corrupt, replace the question-row hidden states after layer l by those of the clean run  -> recovery (m_patch - m_corrupt)/(m_clean - m_corrupt)
  rev(l): run clean,   replace by the corrupt run                                                       -> loss     (m_clean - m_patch)/(m_clean - m_corrupt)
Only items with m_clean - m_corrupt > 1 are used (the tense moves the logits). Writes OUT/tense_patch.jsonl."""
import json, time
import numpy as np, soundfile as sf, torch, subprocess

if 'TTS_MODEL' not in globals():
    from transformers import VitsModel, AutoTokenizer
    TTS_TOK = AutoTokenizer.from_pretrained('facebook/mms-tts-eng')
    TTS_MODEL = VitsModel.from_pretrained('facebook/mms-tts-eng').to('cuda:0').eval()


def tts(text):
    ids = TTS_TOK(text, return_tensors='pt').to('cuda:0')
    with torch.no_grad(): wav = TTS_MODEL(**ids).waveform[0].float().cpu().numpy()
    sr = TTS_MODEL.config.sampling_rate
    if sr != 16000:
        import librosa as _l; wav = _l.resample(wav, orig_sr=sr, target_sr=16000)
    return wav / (np.abs(wav).max() + 1e-6) * 0.5


def with_speech(src, a, s0, s1, wav, tag):
    out = a.copy(); i0, i1 = int(s0 * 16000), int(s1 * 16000); out[i0:i1] = 0.0
    seg = wav[: len(out) - i0]; out[i0:i0 + len(seg)] = seg
    wpath = CLIPS / f'{src.stem}_{tag}.wav'; sf.write(wpath, out, 16000); dst = CLIPS / f'{src.stem}_{tag}.mp4'
    ffmpeg('-i', str(src), '-i', str(wpath), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dst))
    return dst


src_rows = [json.loads(l) for l in (OUT / 'tense_intervention.jsonl').read_text().splitlines() if l.strip()]
src_rows = [r for r in src_rows if r['task'] == 'TF' and 'error' not in r]
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
res = OUT / 'tense_patch.jsonl'
done = {json.loads(l)['idx'] for l in res.read_text().splitlines()} if res.exists() else set()
NL = len(MODEL_.model.layers); LAYS = list(range(0, NL, 2)) + [NL - 1]
print('TF items with rewrites', len(src_rows), flush=True)
for n, sr_ in enumerate(src_rows):
    i = sr_['idx']
    if i in done: continue
    t0 = time.time(); rec = dict(idx=i, gold=sr_['gold'])
    try:
        r = rows[i]; src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
        a = decode_audio(src); s0, s1 = sr_['win']; stem, texts = parse_question(r['question'])
        inps, Ss, lg = {}, {}, {}
        for k in ('future', 'past'):
            media = prepare_video(with_speech(src, a, s0, s1, tts(sr_['sentences'][k]), 'tts_' + k))
            inp, _ = qwen_inputs(PROC, render(stem, texts, (0, 1, 2)), media, 'av'); inps[k] = move_inputs(inp, MODEL_)
            lg[k], Ss[k] = masked_forward(inps[k], 'plain', None, record_layers=list(range(1, NL + 1)))
        m = lambda l: float(l[LETTERS[2]] - l[LETTERS[0]])
        mc, mp = m(lg['future']), m(lg['past']); rec.update(m_clean=round(mc, 3), m_corrupt=round(mp, 3))
        if mc - mp > 1.0:
            rec['fwd'], rec['rev'] = {}, {}
            for li in LAYS:
                rec['fwd'][li] = round((m(masked_forward(inps['past'], 'plain', None, patch=(li, Ss['future'][li + 1][2]))[0]) - mp) / (mc - mp), 3)
                rec['rev'][li] = round((mc - m(masked_forward(inps['future'], 'plain', None, patch=(li, Ss['past'][li + 1][2]))[0])) / (mc - mp), 3)
        del inps; torch.cuda.empty_cache(); rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:300]
    with res.open('a') as f: f.write(json.dumps(rec) + '\n')
    print(n + 1, len(src_rows), 'ERR ' + rec['error'][:60] if 'error' in rec else {k: rec.get(k) for k in ('m_clean', 'm_corrupt', 'sec')}, flush=True)
R = [json.loads(l) for l in res.read_text().splitlines()]; P = [r for r in R if 'fwd' in r]
RESULT = {'items': len(R), 'with_tense_effect': len(P),
          'fwd_recovery_by_layer': {l: round(float(np.mean([r['fwd'][l] for r in P])), 2) for l in P[0]['fwd']} if P else None,
          'rev_loss_by_layer': {l: round(float(np.mean([r['rev'][l] for r in P])), 2) for l in P[0]['rev']} if P else None}
print(RESULT, flush=True)
