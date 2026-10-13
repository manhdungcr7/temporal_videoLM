TENSE_N_PER = 8
"""Causal tense intervention on real SVHalluc clips (runs on the remote GPU server namespace).
For each TA/TF item: video frames untouched; the speech window's audio is replaced by synthetic speech (MMS-TTS English)
of the SAME action said in different tenses, placed at the same onset; everything else in the audio stays as in the clip.
  tts_orig    : TTS of the Whisper transcript (controls for the synthetic voice)
  tts_future  : "<future-tense rewrite>"   tts_present : "<present progressive rewrite>"   tts_past : "<past/completed rewrite>"
Rewrites are produced by the same Qwen model, text only, and printed for inspection. Whisper and the rewrites are analysis
tools for building the stimuli; the evaluated model only ever sees video + audio + the original question.
Scored for TF with the original Past/Present/Future question and with the before/during/after wording, for TA with the
original question. Option orders rotated."""
import json, random, time, re, subprocess
import numpy as np, soundfile as sf, torch

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


def gen_text(prompt, n=48):
    conv = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    text = PROC.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = PROC(text=text, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad(): o = MODEL_.generate(**inp, max_new_tokens=n, do_sample=False)
    return PROC.batch_decode(o[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip().split('\n')[0].strip().strip('"')


REWRITE = {
    'future': 'Rewrite this cooking narration as ONE short sentence in which the speaker says they WILL do the main action LATER (future tense, e.g. "Next, I will ..."). Keep the same action and objects. Output only the sentence.\nNarration: ',
    'present': 'Rewrite this cooking narration as ONE short sentence in which the speaker says they ARE doing the main action RIGHT NOW (present progressive, e.g. "Now I am ..."). Keep the same action and objects. Output only the sentence.\nNarration: ',
    'past': 'Rewrite this cooking narration as ONE short sentence in which the speaker says they have ALREADY FINISHED the main action (past tense, e.g. "I already ..."). Keep the same action and objects. Output only the sentence.\nNarration: '}
BDA_Q = 'Is the action that the speaker talks about shown in the video before, during, or after the speech?'


def with_speech(src, a, s0, s1, wav, tag):
    """original audio, speech window replaced by wav starting at s0 (the rest of the window silenced)."""
    out = a.copy(); i0, i1 = int(s0 * 16000), int(s1 * 16000)
    out[i0:i1] = 0.0
    seg = wav[: len(out) - i0]; out[i0:i0 + len(seg)] = seg
    wpath = CLIPS / f'{src.stem}_{tag}.wav'; sf.write(wpath, out, 16000)
    dst = CLIPS / f'{src.stem}_{tag}.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-y', '-i', str(src), '-i', str(wpath), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(dst)],
                   check=True, capture_output=True)
    return dst, round(len(seg) / 16000, 2)


rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
N_PER = int(globals().get('TENSE_N_PER', 8))
cells = {}
for i, r in enumerate(rows):
    if r['task_type_abbr'] in ('TA', 'TF'): cells.setdefault((r['task_type_abbr'], r['answer']), []).append((i, r))
items = []
for key in sorted(cells):
    pool = cells[key]; random.Random(11).shuffle(pool); items += pool[:N_PER]
res_path = OUT / 'tense_intervention.jsonl'
done = {json.loads(l)['idx'] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
am = lambda v: int(np.argmax(v))
for n, (i, r) in enumerate(items):
    if i in done: continue
    t0 = time.time(); task = r['task_type_abbr']
    rec = dict(idx=i, task=task, gold='ABC'.index(r['answer']))
    try:
        src = cut(fetch(r['video_path'].replace(BS, '/')), 0, 10 ** 4, 'full')
        a = decode_audio(src); dur = len(a) / 16000
        if dur > MAX_DUR: raise RuntimeError('long')
        s0, s1 = speech_window(a)
        tr = whisper_text(a, s0, s1)
        sents = {'orig': tr}
        for k, p in REWRITE.items(): sents[k] = gen_text(p + tr)
        rec.update(dur=round(dur, 2), win=[round(s0, 2), round(s1, 2)], transcript=tr, sentences=sents)
        stem, texts = parse_question(r['question'])
        for k, s in sents.items():
            clip, sec_len = with_speech(src, a, s0, s1, tts(s), 'tts_' + k)
            media = prepare_video(clip)
            rec[f'{k}_len'] = sec_len
            rec[f'{k}_q'] = scores(media, stem, texts)
            if task == 'TF': rec[f'{k}_bda'] = scores(media, BDA_Q, ['Before', 'During', 'After'])
        rec['sec'] = round(time.time() - t0)
    except Exception as e:
        rec['error'] = repr(e)[:300]
    with res_path.open('a') as f: f.write(json.dumps(rec) + '\n')
    if 'error' in rec: print(n + 1, len(items), task, i, rec['error'][:60], flush=True); continue
    print(n + 1, len(items), task, i, 'gold', rec['gold'], {k: am(rec[f'{k}_q']) for k in sents}, rec['sec'], 's', flush=True)
    print('    ', {k: v[:70] for k, v in sents.items()}, flush=True)
RESULT = {'written': str(res_path)}
