"""Hierarchical provenance CQ: two-second visual memory + utterance audio nodes.

Each utterance is generated from audio only without question/options. Generic
interval edges retain both time and source; the model answers the unchanged
question. Speech VAD supplies measurements, never labels or answer mappings.
Requires round3 helpers as a prelude. Cached raw captions remain for auditing.
"""
import importlib.metadata, hashlib, sys

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','silero-vad'],check=True,timeout=180)
    from silero_vad import load_silero_vad, get_speech_timestamps
VAD_MODEL=load_silero_vad()
VAD_VERSION=importlib.metadata.version('silero-vad')
print('VAD_VERSION',VAD_VERSION,flush=True)
DIALOGUE_STOP=re.compile(r'\n\s*(Human|User|Assistant|Question)\s*:',re.I)


def clean_observation(text):
    text=DIALOGUE_STOP.split(text)[0]
    text=re.split(r'If you have any other|feel free to|What do you think|Well, that.s all',text,flags=re.I)[0]
    return text.strip()


def utterance_nodes(src,wave,dur):
    stamps=get_speech_timestamps(torch.from_numpy(wave.copy()).float(),VAD_MODEL,
                                  sampling_rate=16000,return_seconds=True)
    merged=[]
    for stamp in stamps:
        start,end=float(stamp['start']),float(stamp['end'])
        if merged and start-merged[-1][1] <= 0.4:
            merged[-1][1]=end
        else:
            merged.append([start,end])
    nodes=[]
    for n,(start,end) in enumerate(merged):
        t0,t1=max(0,start-0.12),min(dur,end+0.12)
        path=cut(src,t0,t1,f'cq_utterance_{n}')
        media=prepare_video(path)
        transcript=clean_observation(native_generate(media,
            'Transcribe the spoken words exactly. Output only the transcript, no commentary.', 'audio',64))
        event=clean_observation(native_generate(media,
            'What action or event is described by the spoken words? Give only one brief factual phrase. '
            'Use only the audio, do not infer what is visible.','audio',28))
        nodes.append(dict(start=round(start,2),end=round(end,2),words=transcript,reference=event))
        print('UTTERANCE_CQ',json.dumps(nodes[-1]),flush=True)
        del media
    return nodes


def hierarchical_memory(observations,nodes,edges=False):
    lines=['VIDEO nodes: direct visual observations, with original-video timestamps.']
    for k,x in enumerate(observations):
        lines.append(f'V{k} [{x["start"]},{x["end"]}]s: '+clean_observation(x['video']))
    lines.append('\nSPEECH nodes: only heard words; described events are not visual observations.')
    for j,node in enumerate(nodes):
        lines.append(f'A{j} [{node["start"]},{node["end"]}]s: words="{node["words"]}"; '
                     f'described event="{node["reference"]}"')
    # Preserve non-speech acoustic evidence so the representation is not a
    # speech-only gate. These are explicitly separated from spoken events.
    lines.append('\nOther AUDIO observations, not established speech:')
    for x in observations:
        if x['audio']=='silence' or any(x['start']<n['end'] and n['start']<x['end'] for n in nodes):
            continue
        lines.append(f'[{x["start"]},{x["end"]}]s: '+clean_observation(x['audio']))
    if edges:
        lines.append('\nInterval relations computed from measured timestamps (not event identity):')
        for j,node in enumerate(nodes):
            for k,x in enumerate(observations):
                rel='precedes' if x['end']<=node['start'] else 'follows' if x['start']>=node['end'] else 'overlaps'
                lines.append(f'V{k} {rel} A{j}')
    return '\n'.join(lines)+'\n'


def hierarchy_item(i,row):
    src,wave,dur,win=prep_clip(row,50)
    stem,options=parse_question(row['question'])
    observations=json.loads((OUT/f'cq_physical_memory_{i}.json').read_text())
    nodes=utterance_nodes(src,wave,dur)
    memory=hierarchical_memory(observations,nodes)
    graph=hierarchical_memory(observations,nodes,True)
    rec=dict(dur=round(dur,3),utterances=nodes,memory=memory,graph=graph,vad_version=VAD_VERSION)
    rec['cq_hierarchy']=text_only_scores(memory+stem,options)
    rec['cq_hierarchy_graph']=text_only_scores(graph+GENERIC_REASON+stem,options)
    rec['cq_hierarchy_reason'],rec['reasoning']=reason_scores(graph,stem,options)
    print('HIERARCHY_REASONING',i,rec['reasoning'],flush=True)
    return rec


CONDITIONS=('cq_hierarchy','cq_hierarchy_graph','cq_hierarchy_reason')
R=run_loop(ITEMS,f'cq_diagnostic_round6_s{LAB_SERVER}.jsonl',hierarchy_item,show=CONDITIONS,min_left=0)
RESULT={c:{'accuracy':acc(R,c)[0],'n':acc(R,c)[1]} for c in CONDITIONS}
print('ROUND6_RESULT',json.dumps(RESULT),flush=True)
