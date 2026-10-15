"""Physical CQ extraction and source-aware event memory; generic QA readout.

Each two-second modality summary is generated in a fresh native context from
that modality of that crop only. This diagnoses latent/raw-state contamination
and supplies a cacheable deployable counterpart to independent CQ branches.
"""
import math, re

MANIFEST = json.loads((OUT / f'cq_diagnostic_manifest_s{LAB_SERVER}.json').read_text())
DATA = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
ITEMS = [(d['idx'], DATA[d['idx']]) for d in MANIFEST['selected']]
EXTRACT_VIDEO = ('Describe the visible actions and objects in this short video in one brief factual phrase. '
                 'Distinguish preparing or holding something from actually performing an action. '
                 'Report only what is directly visible.')
EXTRACT_AUDIO = ('Transcribe the spoken words heard in this short audio. '
                 'If there is no intelligible speech, briefly name the sound instead. '
                 'Do not invent missing words.')
GENERIC_REASON = ('Use the source-tagged observations to answer the question. '
                  'First identify the referenced events and their observed intervals. '
                  'Treat spoken descriptions as audio evidence, never as proof of a visible action. '
                  'Compare observed intervals and keep preparing, performing and completed states distinct. ')


def native_generate(media, prompt, mode, limit=24):
    inp, _ = qwen_inputs(PROC, prompt, media, mode)
    inp = move_inputs(inp, MODEL_)
    with torch.no_grad():
        seq = MODEL_.generate(**inp, use_audio_in_video=(mode=='av'), max_new_tokens=limit,
                              do_sample=False, use_cache=True)
    text = TOK.decode(seq[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    del inp, seq
    return text


def text_generate(prompt, limit=200):
    messages = [{'role':'system','content':[{'type':'text','text':SYSTEM}]},
                {'role':'user','content':[{'type':'text','text':prompt}]}]
    chat = PROC.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inp = PROC(text=chat, return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    with torch.no_grad():
        seq = MODEL_.generate(**inp, max_new_tokens=limit, do_sample=False, use_cache=True)
    text = TOK.decode(seq[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    return text


def extraction_memory(src, dur):
    observations = []
    for t in range(int(math.ceil(dur/2))):
        start, end = 2*t, min(2*t+2, dur)
        if end-start < 0.2:
            continue
        chunk = cut(src, start, end, f'cq_phys_{t}')
        media = prepare_video(chunk)
        visual = native_generate(media, EXTRACT_VIDEO, 'vision', 24)
        wave = media['audio'][0]
        rms = float(np.sqrt(np.mean(np.asarray(wave)**2)))
        heard = 'silence' if rms < 1e-4 else native_generate(media, EXTRACT_AUDIO, 'audio', 32)
        observations.append(dict(start=round(start,2),end=round(end,2),video=visual,audio=heard,rms=round(rms,6)))
        print('PHYSICAL_CQ', round(start,1), visual, '|', heard, flush=True)
        del media
        torch.cuda.empty_cache()
    return observations


def source_memory(observations):
    # Separate tables prevent adjacency from implying that narration is visible.
    seen = ['VIDEO observations (directly visible, sound not provided):']
    heard = ['AUDIO observations (only heard, frames not provided):']
    for x in observations:
        interval = f'[{x["start"]},{x["end"]}]s'
        seen.append(interval+' '+x['video'])
        heard.append(interval+' '+x['audio'])
    return '\n'.join(seen)+'\n\n'+'\n'.join(heard)+'\n'


def reason_scores(memory, stem, options):
    # Evidence deliberation is produced once without answer options, so no
    # option's phrasing can influence the event description or timestamps.
    reasoning = text_generate(memory + '\nQuestion: '+stem+'\n'+GENERIC_REASON+
                              'Write a brief evidence analysis with timestamps. Do not give an answer option.', 180)
    return text_only_scores(memory+'\nEvidence analysis: '+reasoning+'\nQuestion: '+stem, options), reasoning


def physical_item(i,row):
    src, audio, dur, win = prep_clip(row,50)
    stem, options = parse_question(row['question'])
    cache = OUT/f'cq_physical_memory_{i}.json'
    if cache.exists():
        observations = json.loads(cache.read_text())
    else:
        observations = extraction_memory(src,dur)
        cache.write_text(json.dumps(observations,indent=2),encoding='utf-8')
    memory = source_memory(observations)
    rec = dict(dur=round(dur,3),observations=observations,memory=memory,options=options)
    rec['cq_physical'] = text_only_scores(memory+stem,options)
    rec['cq_physical_grounded'] = text_only_scores(memory+GENERIC_REASON+stem,options)
    rec['cq_physical_reason'], rec['reasoning'] = reason_scores(memory,stem,options)
    print('EVIDENCE_ANALYSIS',i,rec['reasoning'],flush=True)
    return rec


CONDITIONS = ('cq_physical','cq_physical_grounded','cq_physical_reason')
R = run_loop(ITEMS,f'cq_diagnostic_round3_s{LAB_SERVER}.jsonl',physical_item,show=CONDITIONS,min_left=0)
RESULT = {c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND3_RESULT',json.dumps(RESULT),flush=True)
