"""Controlled raw-vs-CFR preprocessing pair on a dev clip with Decord PTS reversals."""
import json,sys,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_qwen'))
sys.path.insert(0,str(ROOT/'research_topics_learning/single_sample_prompt_lab'))
from temporal_clock_counterfactual_2310 import audited,load,run
from runner import prepare_video

IDX=2300
OUT=ROOT/'local_qwen/temporal_preprocess_normalization_pair_probe.json'

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace');torch.set_num_threads(4)
    rows=json.loads((audited.DATA/'samples.json').read_text(encoding='utf-8'))
    meta=next(x for x in json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['development'] if x['idx']==IDX)
    item=rows[IDX];source=audited.DATA/item['video_path'];normalized,norm=audited.normalize_av_clip(source)
    model,processor=load();results=[]
    for label,path in (('raw_source',source),('cfr30_normalized',normalized)):
        media=prepare_video(path);started=time.time()
        rec=run(model,processor,media,item['question'],meta['gold'],{},max_new_tokens=24)
        results.append({'idx':IDX,'task':meta['task'],'gold':meta['gold'],'condition':label,
            'path':str(path),'frame_count':int(media['videos'][0].shape[0]),'fps':media['fps'],
            'audio_duration_s':len(media['audio'][0])/16000,'video_hash':rec['metadata']['sampled_pixels_sha256'],
            'audio_hash':rec['metadata']['audio_sha256'],'video_tokens':rec['metadata']['video_tokens'],
            'raw_output':rec['raw_output'],'parsed_choice':rec['parsed_choice'],
            'elapsed_s':round(time.time()-started,2)})
        OUT.write_text(json.dumps({'gold':meta['gold'],'question':item['question'],
            'normalization':{k:v for k,v in norm.items() if k!='command'},'conditions':results,
            'warning':'Paired diagnostic on one development example; raw-vs-normalized also changes audio re-encoding.'},
            ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(results[-1],ensure_ascii=True),flush=True)
        del media,rec;torch.cuda.empty_cache()
    del model,processor;torch.cuda.empty_cache();print('Saved',OUT,flush=True)

if __name__=='__main__':main()
