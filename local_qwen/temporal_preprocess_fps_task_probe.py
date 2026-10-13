"""Controlled 2-vs-4 FPS test on one previously wrong CMTB and TF sample."""
import json, sys, time
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_qwen'))
sys.path.insert(0,str(ROOT/'research_topics_learning/single_sample_prompt_lab'))
from temporal_clock_counterfactual_2310 import audited, load, run
from runner import prepare_video

IDS=(1917,2300) # Dev examples: CMTB (gold C) and TF (gold B).
FPS_VALUES=(2.0,4.0)
OUT=ROOT/'local_qwen/temporal_preprocess_fps_task_probe.json'
PIXELS=160*28*28

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4)
    rows=json.loads((audited.DATA/'samples.json').read_text(encoding='utf-8'))
    dev=json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['development']
    by_id={x['idx']:x for x in dev}
    model,processor=load();results=[]
    for idx in IDS:
        item=rows[idx];meta=by_id[idx]
        path,norm=audited.normalize_av_clip(audited.DATA/item['video_path'])
        base=prepare_video(path)
        for fps in FPS_VALUES:
            conv=[{'role':'system','content':[{'type':'text','text':'You are Qwen.'}]},
                  {'role':'user','content':[{'type':'video','video':str(path),'max_pixels':PIXELS,'fps':fps},
                                            {'type':'text','text':''}]}]
            _,_,videos,vkw=process_mm_info(conv,use_audio_in_video=False,return_video_kwargs=True)
            actual=vkw['fps'][0] if isinstance(vkw['fps'],(list,tuple)) else vkw['fps']
            media=dict(base,videos=videos,fps=float(actual))
            started=time.time()
            rec=run(model,processor,media,item['question'],meta['gold'],{},max_new_tokens=24)
            result={'idx':idx,'task':meta['task'],'gold':meta['gold'],'fps':fps,'reported_fps':float(actual),
                    'frame_count':int(videos[0].shape[0]),'video_grid_thw':rec['metadata']['video_grid_thw'],
                    'video_tokens':rec['metadata']['video_tokens'],'audio_tokens':rec['metadata']['audio_tokens'],
                    'input_shape':rec['metadata']['input_shape'],'raw_output':rec['raw_output'],
                    'parsed_choice':rec['parsed_choice'],'elapsed_s':round(time.time()-started,2),
                    'normalization':{k:v for k,v in norm.items() if k!='command'},
                    'warning':'One development example per task; diagnostic only, not a score.'}
            results.append(result);OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(result,ensure_ascii=True),flush=True)
            del media,videos,rec;torch.cuda.empty_cache()
    del model,processor;torch.cuda.empty_cache()
    print('Saved',OUT,flush=True)

if __name__=='__main__':main()
