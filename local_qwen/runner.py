"""Load NF4 Thinker once; run baseline or prompt over a list of sample indices; append JSONL."""
import os,sys,json,time,re,random
from pathlib import Path
ROOT=Path(r'F:\A_star_paper_NLP_ACL');LAB=ROOT/'research_topics_learning/single_sample_prompt_lab'
os.environ['PATH']=str(LAB/'validation_ffmpeg')+os.pathsep+os.environ['PATH']
os.environ['FORCE_QWENVL_VIDEO_READER']='decord'
sys.path.insert(0,str(LAB))
import torch,librosa
from lab_media import decode_audio
_o=librosa.load
def _l(path,sr=None,offset=0.0,duration=None,**k):
    if str(path).lower().endswith('.mp4'):
        a=decode_audio(path);return a[int(sr*offset):None if duration is None else int(sr*(offset+duration))],sr
    return _o(path,sr=sr,offset=offset,duration=duration,**k)
librosa.load=_l
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor,BitsAndBytesConfig
from lab_native_video import prepare_video,qwen_inputs,move_inputs
REV='f75b40e3da2003cdd6e1829b1f420ca70797c34e';CACHE=str(ROOT/'local_qwen/hf_cache')
def load():
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    m=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-Omni-3B',revision=REV,cache_dir=CACHE,
        dtype=torch.float16,attn_implementation='sdpa',quantization_config=q,device_map={'':0}).eval()
    p=Qwen2_5OmniProcessor.from_pretrained('Qwen/Qwen2.5-Omni-3B',revision=REV,cache_dir=CACHE)
    return m,p
def parse(t):
    t=t.strip()
    m=re.search(r'Final answer:\s*\(?([AB])\b',t,re.I) or re.match(r'\(?([AB])\b',t)
    return m.group(1).upper() if m else None
if __name__=='__main__':
    ids=json.load(open(sys.argv[1]));tmpl=Path(sys.argv[2]).read_text(encoding='utf-8') if sys.argv[2]!='none' else None
    out=Path(sys.argv[3]);maxnew=int(sys.argv[4])
    rows=json.loads((ROOT/'data/svhalluc_benchmark/samples.json').read_text(encoding='utf-8'))
    done={json.loads(l)['idx'] for l in out.read_text().splitlines()} if out.exists() else set()
    model,proc=load()
    for i in ids:
        if i in done:continue
        r=rows[i];t=time.time()
        try:
            media=prepare_video(ROOT/'data/svhalluc_benchmark'/r['video_path'])
            prompt=tmpl.replace('{question}',r['question']) if tmpl else r['question']
            inp,_=qwen_inputs(proc,prompt,media);inp=move_inputs(inp,model)
            with torch.no_grad():o=model.generate(**inp,max_new_tokens=maxnew,do_sample=False,use_audio_in_video=True)
            txt=proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            rec=dict(idx=i,gold=r['answer'],pred=parse(txt),text=txt,tokens=int(inp['input_ids'].shape[1]),sec=round(time.time()-t))
        except Exception as e:
            rec=dict(idx=i,gold=r['answer'],pred=None,text='ERR '+repr(e)[:200],tokens=0,sec=0)
        torch.cuda.empty_cache()
        with out.open('a') as f:f.write(json.dumps(rec)+'\n')
