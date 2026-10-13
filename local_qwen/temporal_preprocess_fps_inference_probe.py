"""One-time inference probe: change only native Qwen video sampling FPS."""
import json, sys, time
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_qwen'))
sys.path.insert(0,str(ROOT/'research_topics_learning/single_sample_prompt_lab'))
from temporal_clock_counterfactual_2310 import audited, load, run
from runner import prepare_video

IDS=(1961,2081) # Confirmation: one gold-Yes and one gold-No TA example.
FPS_VALUES=(1.0,2.0,4.0)
OUT=ROOT/'local_qwen/temporal_preprocess_fps_inference_probe.json'
PIXELS=160*28*28

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4)
    rows=json.loads((audited.DATA/'samples.json').read_text(encoding='utf-8'))
    confirm=json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['confirmation']
    by_id={x['idx']:x for x in confirm}
    model,processor=load()
    results=[]
    for idx in IDS:
        item=rows[idx]; meta=by_id[idx]
        norm_path,norm=audited.normalize_av_clip(audited.DATA/item['video_path'])
        base=prepare_video(norm_path) # Hold the exact full-clip audio waveform fixed.
        for fps in FPS_VALUES:
            conversation=[
                {'role':'system','content':[{'type':'text','text':'You are Qwen.'}]},
                {'role':'user','content':[{'type':'video','video':str(norm_path),'max_pixels':PIXELS,'fps':fps},
                                          {'type':'text','text':''}]},
            ]
            _,_,videos,vkw=process_mm_info(conversation,use_audio_in_video=False,return_video_kwargs=True)
            reported=vkw['fps'][0] if isinstance(vkw['fps'],(list,tuple)) else vkw['fps']
            media=dict(base,videos=videos,fps=float(reported))
            t=time.time()
            rec=run(model,processor,media,item['question'],meta['gold'],{},max_new_tokens=24)
            out={
                'idx':idx,'task':meta['task'],'gold':meta['gold'],'source_id':meta['source_id'],
                'target_fps':fps,'reported_fps':float(reported),'frames':int(videos[0].shape[0]),
                'video_grid_thw':rec.get('metadata',{}).get('video_grid_thw'),
                'audio_duration_s':len(base['audio'][0])/16000,
                'raw_output':rec.get('raw_output'),'parsed_choice':rec.get('parsed_choice'),
                'input_shape':rec.get('metadata',{}).get('input_shape'),
                'video_tokens':rec.get('metadata',{}).get('video_tokens'),
                'audio_tokens':rec.get('metadata',{}).get('audio_tokens'),
                'elapsed_s':round(time.time()-t,2),'normalization':{k:v for k,v in norm.items() if k!='command'},
                'warning':'Tiny diagnostic confirmation probe; not benchmark performance.'
            }
            results.append(out); OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(out,ensure_ascii=True),flush=True)
            del media,videos,rec
            torch.cuda.empty_cache()
    del model,processor
    torch.cuda.empty_cache()
    print('Saved',OUT,flush=True)

if __name__=='__main__': main()
