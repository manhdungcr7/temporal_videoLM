import os,sys,json,time
from pathlib import Path
ROOT=Path(r'F:\A_star_paper_NLP_ACL');LAB=ROOT/'research_topics_learning/single_sample_prompt_lab'
os.environ['PATH']=str(LAB/'validation_ffmpeg')+os.pathsep+os.environ['PATH']
os.environ['FORCE_QWENVL_VIDEO_READER']='decord'
sys.path.insert(0,str(LAB))
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor,BitsAndBytesConfig
import librosa
from lab_media import decode_audio
_orig=librosa.load
def _load(path,sr=None,offset=0.0,duration=None,**k):
    if str(path).lower().endswith('.mp4'):
        a=decode_audio(path);assert sr==16000
        return a[int(sr*offset):None if duration is None else int(sr*(offset+duration))],sr
    return _orig(path,sr=sr,offset=offset,duration=duration,**k)
librosa.load=_load
from lab_native_video import conversation_for,prepare_video,qwen_inputs,move_inputs
REV='f75b40e3da2003cdd6e1829b1f420ca70797c34e';CACHE=str(ROOT/'local_qwen/hf_cache')
idx=int(sys.argv[1]);mode=sys.argv[2]
rows=json.loads((ROOT/'data/svhalluc_benchmark/samples.json').read_text(encoding='utf-8'))
row=rows[idx];path=ROOT/'data/svhalluc_benchmark'/row['video_path']
t=time.time()
q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
kw=dict(revision=REV,cache_dir=CACHE,torch_dtype=torch.float16,attn_implementation='sdpa')
if mode=='nf4':kw.update(quantization_config=q,device_map={'':0})
else:kw.update(device_map='auto',max_memory={0:'3GiB','cpu':'11GiB'})
model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-Omni-3B',**kw).eval()
proc=Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-3B',revision=REV,cache_dir=CACHE)
print('load s',round(time.time()-t),'gpu GiB',torch.cuda.memory_allocated()/2**30,flush=True)
media=prepare_video(path)
inputs,_=qwen_inputs(proc,row['question'],media)
print('tokens',inputs['input_ids'].shape,flush=True)
inputs=move_inputs(inputs,model)
t=time.time()
with torch.no_grad():
    out=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_audio_in_video=True)
txt=proc.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
print('gen s',round(time.time()-t),'peak GiB',torch.cuda.max_memory_allocated()/2**30)
print(row['question'],'| gold',row['answer'],'| out',txt)
