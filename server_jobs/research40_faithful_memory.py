"""Repair observation fidelity, then compare raw-AV and text-memory readouts.

ASR/VAD are generic perception tools, never question-dependent event finders.
No transcript annotations, benchmark category, answer constraints, or gold enters inference.
"""
import json, time, gc, sys, subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import pipeline

R40_FAITH_FILE=OUT/f'research40_faithful_memory_s{R40_SERVER}.jsonl'
R40_ASR_DIR=OUT/'research40_asr';R40_ASR_DIR.mkdir(exist_ok=True)
try:
    from silero_vad import load_silero_vad,get_speech_timestamps
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','silero-vad'],check=True,timeout=120)
    from silero_vad import load_silero_vad,get_speech_timestamps
R40_VAD=load_silero_vad().cpu()
R40_ASR=pipeline('automatic-speech-recognition',model='openai/whisper-small',device=0,torch_dtype=torch.float16)
print('FAITHFUL_SOURCE_MODELS_READY',flush=True)

def r40_asr_nodes(path,cache_path):
    if cache_path.exists():return json.loads(cache_path.read_text())
    wave=decode_audio(path)
    regions=get_speech_timestamps(torch.from_numpy(wave),R40_VAD,sampling_rate=16000,
        min_speech_duration_ms=150,min_silence_duration_ms=200,speech_pad_ms=80)
    nodes=[]
    for region in regions:
        lo,hi=region['start'],region['end'];a,b=lo/16000,hi/16000
        res=R40_ASR({'raw':wave[lo:hi],'sampling_rate':16000},return_timestamps='word',
                    chunk_length_s=30,generate_kwargs={'task':'transcribe'})
        words=[]
        for chunk in res.get('chunks',[]):
            start,end=chunk.get('timestamp',(None,None))
            if start is None:start=0
            if end is None:end=b-a
            words.append(dict(start=max(a,min(b,a+float(start))),end=max(a,min(b,a+float(end))),text=chunk['text']))
        nodes.append(dict(kind='audio speech',start=a,end=b,text=res['text'].strip(),words=words,
                          tool='Silero VAD + multilingual Whisper-small'))
    cache_path.write_text(json.dumps(nodes,indent=2),encoding='utf-8')
    return nodes

def r40_clean_visual(nodes):
    result=[]
    for node in nodes:
        if node['kind']!='visual':continue
        # Keep only the first assistant observation. Never ingest a generated next-user turn.
        text=node['text'].split('\nHuman:')[0].split('\nUser:')[0].split('<|im_end|>')[0].strip()
        text=text.split('\n')[0].strip()
        result.append(dict(node,text=text))
    return result

def r40_faithful_visual_observations(row):
    cache=R40_MEMORY_DIR/f"{row['idx']}_4s.json"
    if cache.exists():return json.loads(cache.read_text())
    pixels=base_by_id[row['idx']]['pixels']
    media=prepare_video(fetch(row['video_path']),max_pixels=pixels)
    nodes=[]
    for start in np.arange(0,row['decoded_audio_duration'],4.0):
        start=float(start);end=min(row['decoded_audio_duration'],start+4.0)
        if end-start<.05:continue
        nodes.append(dict(kind='visual',start=start,end=end,text=r40_observe(media,start,end,'visual')))
        cache.write_text(json.dumps(nodes,indent=2),encoding='utf-8')
    del media;gc.collect();torch.cuda.empty_cache()
    return nodes

baseline=r40_json_rows(R40_BASE_FILE);base_by_id={r['idx']:r for r in baseline}
assert len(baseline)==len(R40_SAMPLES)
records=r40_json_rows(R40_FAITH_FILE);done={r['idx'] for r in records}
for r in R40_SAMPLES:
    if r['idx'] in done:continue
    started=time.time()
    try:
        src=fetch(r['video_path'])
        observations=r40_faithful_visual_observations(r)
        video=r40_clean_visual(observations)
        audio=r40_asr_nodes(src,R40_ASR_DIR/f"{r['idx']}.json")
        nodes=sorted(video+audio,key=lambda n:(n['start'],n['kind']))
        supplement=r40_memory_text(nodes)
        stem,texts=parse_question(r['question']);k=len(texts)
        payload=torch.load(R40_CACHE/f"{r['idx']}_rot0.pt",map_location='cpu',weights_only=False);payload['sample_idx']=r['idx']
        raw=np.zeros(k);textonly=np.zeros(k);cot=np.zeros(k)
        state=r40_memory_analysis(payload,stem,supplement)
        for perm in rotations(k):
            choices=' '.join(f'({chr(65+s)}) {texts[i]}' for s,i in enumerate(perm))
            lg,_,_,_=r40_memory_context(payload,stem,supplement,choices)
            v=lg[R40_LABELS[:k]].cpu().numpy();lp=v-np.logaddexp.reduce(v)
            v2=r40_text_logits(supplement+r40_prompt(stem,texts,perm))[:k];lp2=v2-np.logaddexp.reduce(v2)
            v3=r40_reason_answer(state,texts,perm);lp3=v3-np.logaddexp.reduce(v3)
            for slot,i in enumerate(perm):raw[i]+=lp[slot];textonly[i]+=lp2[slot];cot[i]+=lp3[slot]
        rec=dict(idx=r['idx'],task=r['task'],gold=r['gold'],status='ok',manifest_sha256=R40_MANIFEST['manifest_sha256'],
                 scores=dict(faithful_memory_raw_av=raw.tolist(),faithful_memory_text_reader=textonly.tolist(),
                             faithful_memory_reasoning=cot.tolist(),faithful_dual_reader=(.5*(raw+textonly)).tolist()),
                 observations=nodes,analysis=state['text'],seconds=round(time.time()-started,2))
        with R40_FAITH_FILE.open('a') as f:f.write(json.dumps(rec)+'\n')
        records.append(rec);done.add(r['idx'])
        merged=[dict(rr,scores=dict(base_by_id[rr['idx']]['scores'],**rr['scores'])) for rr in records]
        r40_snapshot(merged,'faithful_source_memory')
        print('FAITHFUL_MEMORY_SAMPLE',json.dumps({k:rec[k] for k in ('idx','task','gold','scores','seconds','analysis')}),flush=True)
        del payload,state;gc.collect();torch.cuda.empty_cache()
    except Exception:
        import traceback
        print('FAITHFUL_MEMORY_ERROR',r['idx'],traceback.format_exc(),flush=True)
        torch.cuda.empty_cache();raise
RESULT=r40_snapshot(merged,'faithful_memory_done')
print('FAITHFUL_MEMORY_SUMMARY',json.dumps(RESULT),flush=True)
