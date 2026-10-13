"""Synchronized AV-window preprocessing pilot on two confirmation TA samples."""
import json,sys,time,subprocess,hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_qwen'))
sys.path.insert(0,str(ROOT/'research_topics_learning/single_sample_prompt_lab'))
from temporal_clock_counterfactual_2310 import audited,load,run
from runner import prepare_video

IDS=(1961,2081) # Confirmation labels: A (Yes), B (No).
WINDOWS=((0.0,2.5),(2.0,4.5),(4.0,6.5))
OUT=ROOT/'local_qwen/temporal_preprocess_sync_window_probe.json'
DIR=ROOT/'local_qwen/temporal_sync_windows'
QUESTION='When the speech is heard in this synchronized segment, is the narrated event happening at the same time in the video? (A) Yes (B) No'

def make_window(source,start,end,target):
    if not target.exists():
        cmd=[str(audited.FFMPEG),'-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(source),'-t',str(end-start),
             '-map','0:v:0','-map','0:a:0','-vf','fps=30','-fps_mode','cfr','-c:v','libx264','-preset','ultrafast',
             '-crf','24','-pix_fmt','yuv420p','-af','aresample=async=1:first_pts=0','-c:a','aac','-movflags','+faststart',str(target)]
        subprocess.run(cmd,check=True,capture_output=True)
    return target

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace');torch.set_num_threads(4);DIR.mkdir(exist_ok=True)
    rows=json.loads((audited.DATA/'samples.json').read_text(encoding='utf-8'))
    conf=json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['confirmation'];by_id={x['idx']:x for x in conf}
    model,processor=load();all_results=[]
    for idx in IDS:
        item=rows[idx];meta=by_id[idx];source,norm=audited.normalize_av_clip(audited.DATA/item['video_path'])
        pieces=[]
        for j,(start,end) in enumerate(WINDOWS):
            clip=make_window(source,start,end,DIR/f'{idx}_{j}_{start:.1f}_{end:.1f}.mp4')
            media=prepare_video(clip)
            prompt=QUESTION+'\nJudge only evidence within this segment; do not infer an event from speech alone.'
            started=time.time()
            rec=run(model,processor,media,prompt,meta['gold'],{},max_new_tokens=20)
            item_result={'window_s':[start,end],'frame_count':int(media['videos'][0].shape[0]),
                'fps':media['fps'],'audio_duration_s':len(media['audio'][0])/16000,
                'video_tokens':rec['metadata']['video_tokens'],'audio_tokens':rec['metadata']['audio_tokens'],
                'raw_output':rec['raw_output'],'parsed_choice':rec['parsed_choice'],'elapsed_s':round(time.time()-started,2),
                'video_sha256':hashlib.sha256(media['videos'][0].numpy().tobytes()).hexdigest(),
                'audio_sha256':hashlib.sha256(media['audio'][0].tobytes()).hexdigest()}
            pieces.append(item_result)
            print(json.dumps({'idx':idx,**item_result},ensure_ascii=True),flush=True)
            del media,rec;torch.cuda.empty_cache()
        valid=[p['parsed_choice'] for p in pieces if p['parsed_choice'] in ('A','B')]
        aggregate='A' if 'A' in valid else ('B' if valid and all(x=='B' for x in valid) else None)
        all_results.append({'idx':idx,'task':meta['task'],'gold':meta['gold'],'source_id':meta['source_id'],
            'question':item['question'],'aggregation':'A if any synchronized window predicts A; B if every window predicts B',
            'aggregate_prediction':aggregate,'aggregate_correct':aggregate==meta['gold'],'windows':pieces,
            'normalization':{k:v for k,v in norm.items() if k!='command'},
            'warning':'Exploratory three-window TA pilot; the aggregation rule and prompt are not validated.'})
        OUT.write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding='utf-8')
    del model,processor;torch.cuda.empty_cache()
    print('Saved',OUT,flush=True)

if __name__=='__main__':main()
