"""Same-visual-token-budget spatial-vs-temporal resolution check on TA 1961."""
import json,sys,time
from pathlib import Path
import torch
from qwen_omni_utils import process_mm_info

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_qwen'))
sys.path.insert(0,str(ROOT/'research_topics_learning/single_sample_prompt_lab'))
from temporal_clock_counterfactual_2310 import audited,load,run
from runner import prepare_video

IDX=1961
OUT=ROOT/'local_qwen/temporal_preprocess_resolution_tradeoff_probe.json'
FPS=1.0; PIXELS=320*28*28

def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace');torch.set_num_threads(4)
    rows=json.loads((audited.DATA/'samples.json').read_text(encoding='utf-8'))
    meta=next(x for x in json.loads(audited.SPLIT_FILE.read_text(encoding='utf-8'))['splits']['confirmation'] if x['idx']==IDX)
    item=rows[IDX];path,norm=audited.normalize_av_clip(audited.DATA/item['video_path'])
    base=prepare_video(path)
    conv=[{'role':'system','content':[{'type':'text','text':'You are Qwen.'}]},
          {'role':'user','content':[{'type':'video','video':str(path),'max_pixels':PIXELS,'fps':FPS},
                                    {'type':'text','text':''}]}]
    _,_,videos,vkw=process_mm_info(conv,use_audio_in_video=False,return_video_kwargs=True)
    actual=vkw['fps'][0] if isinstance(vkw['fps'],(list,tuple)) else vkw['fps']
    media=dict(base,videos=videos,fps=float(actual),max_pixels_per_frame=PIXELS)
    model,processor=load();start=time.time()
    rec=run(model,processor,media,item['question'],meta['gold'],{},max_new_tokens=24)
    out={'idx':IDX,'task':meta['task'],'gold':meta['gold'],'fps':FPS,'reported_fps':float(actual),
         'frames':int(videos[0].shape[0]),'resolution_each_frame':list(videos[0].shape[-2:]),
         'max_pixels_per_frame':PIXELS,'video_grid_thw':rec['metadata']['video_grid_thw'],
         'video_tokens':rec['metadata']['video_tokens'],'audio_tokens':rec['metadata']['audio_tokens'],
         'input_shape':rec['metadata']['input_shape'],'raw_output':rec['raw_output'],
         'parsed_choice':rec['parsed_choice'],'elapsed_s':round(time.time()-start,2),
         'normalization':{k:v for k,v in norm.items() if k!='command'},
         'comparison':'1 FPS at doubled pixels/frame approximately matches 2 FPS default visual-token budget; tests spatial detail vs temporal density.',
         'warning':'Single diagnostic confirmation sample, not a performance result.'}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
