"""Persistent, frozen Qwen Omni worker: one model load, fresh context for each prompt."""
from __future__ import annotations
import argparse
import gc
import json
import re
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

WIRE = sys.stdout
PREFIX = "__SVHLAB_JSON__"
MODEL = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"


def emit(value):
    WIRE.write(PREFIX + json.dumps(value, ensure_ascii=False) + "\n"); WIRE.flush()


def parse_final(text, complete):
    """Read the explicit final choice, with support for older prompt formats."""
    if not complete: return None
    final_labels = re.findall(r"(?i)\bFinal answer\s*:", text)
    if len(final_labels)>1:return None
    final_match = re.search(r"(?i)\bFinal answer\s*:\s*([AB](?:\s*\((?:Yes|No)\))?)\s*[.!?]?\s*$",text)
    if final_labels and not final_match:return None
    final_fields=[final_match.group(1)] if final_match else []
    fields = re.findall(r"(?im)^\s*4\.\s*Original question\s*:\s*([^\n]+)", text)
    if len(fields)>1:return None
    terminal = re.search(r"(?i)\b(?:the\s+)?answer\s+is\s+([AB](?:\s*\((?:Yes|No)\))?)[.!]?\s*$",text)
    value=(final_fields[0] if final_fields else fields[0] if fields else terminal.group(1) if terminal else text).strip()
    match=re.match(r"^(?:Answer\s*:\s*)?([AB])(?=$|[\s().:—-])",value,re.I)
    if not match or re.search(r"\bor\s+[AB](?:\b|\s*\()",value,re.I):return None
    letter=match.group(1).upper()
    meaning=re.match(r"\s*\((Yes|No)\)",value[match.end():],re.I)
    if meaning and (letter,meaning.group(1).lower()) not in (("A","yes"),("B","no")):return None
    return letter


def parse_baseline(text,complete):
    if not complete:return None
    value=text.strip()
    match=re.match(r"^(?:(?:The\s+)?(?:answer|option)\s*(?:is|:)\s*)?([AB])\b",value,re.I)
    if match:
        if re.search(r"\bor\s+[AB]\b",value,re.I):return None
        letter=match.group(1).upper()
        meaning=re.match(r"\s*\((Yes|No)\)",value[match.end():],re.I)
        if meaning and (letter,meaning.group(1).lower()) not in (('A','yes'),('B','no')):return None
        return letter
    match=re.match(r"^(Yes|No)\b",value,re.I)
    return ('A' if match.group(1).lower()=='yes' else 'B') if match else None


def final_format_issues(text, complete):
    issues = []
    if not complete: issues.append("Output hit the token limit: final answer is not counted.")
    if complete and parse_final(text, complete) is None:
        issues.append("Could not identify one unambiguous Final answer: A (Yes) or B (No), or a supported older answer format.")
    return issues


class Engine:
    def __init__(self, output):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(2); torch.manual_seed(20261009)
        if torch.cuda.device_count() < 2: raise RuntimeError("Enable Kaggle T4 x2")
        config = Qwen2_5OmniThinkerConfig.from_pretrained(MODEL, revision=REVISION)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, config=config, torch_dtype=torch.float16,
            attn_implementation="sdpa", device_map="balanced",
            max_memory={0:"10GiB",1:"10GiB"}, low_cpu_mem_usage=True).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        if any(str(d) in ("cpu","disk") for d in self.model.hf_device_map.values()):
            raise RuntimeError("Unexpected CPU offload; check both GPUs are available")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=REVISION)
        self.meta = {"model":MODEL,"revision":REVISION,"device_map":self.model.hf_device_map,
                     "runtime_parameters":sum(p.numel() for p in self.model.parameters()),
                     "diagnostic_support":1}
        self.media = self.row = None
        (self.output / "model.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def sample(self, request):
        from lab_native_video import prepare_video
        allowed = {"cmd","request_id","item_id","question","path"}
        if set(request)-allowed: raise ValueError("Sample request has unexpected fields, possibly a label")
        self.media = None; self.row = None
        media = prepare_video(request['path'])
        self.media = media
        self.row = {k:request[k] for k in ("item_id","question","path")}
        result = {**self.row,'frame_count':int(media['videos'][0].shape[0]),
                  'video_shape':list(media['videos'][0].shape),'sampled_fps':media['fps'],
                  'audio_duration_s':len(media['audio'][0])/16000,
                  'preprocessing':'Official qwen-omni-utils 0.0.8, direct MP4 path, default temporal sampling, bounded max_pixels per frame for T4',
                  'max_pixels_per_frame':media['max_pixels_per_frame'],
                  'same_media_for_both_conditions':True}
        (self.output/(request["item_id"]+"_media.json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result

    def generate(self, prompt, limit, baseline=False, media_mode='av', diagnostic=False):
        import torch
        from lab_native_video import qwen_inputs, move_inputs
        if self.media is None: raise RuntimeError("Select/load one sample before inference")
        if not isinstance(limit,int) or not 16 <= limit <= 1024: raise ValueError("Token limit must be 16..1024")
        inputs = None
        started = time.perf_counter()
        try:
            inputs, rendered = qwen_inputs(self.processor, prompt, self.media,media_mode)
            capacity=self.model.config.text_config.max_position_embeddings
            if inputs['input_ids'].shape[-1]+limit > capacity:
                raise RuntimeError(f'Native video + prompt exceed the model context ({capacity}); no media was silently truncated')
            inputs = move_inputs(inputs,self.model)
            for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
            eos = {self.processor.tokenizer.eos_token_id}
            original_eos = self.model.generation_config.eos_token_id
            if isinstance(original_eos,int): eos.add(original_eos)
            elif original_eos: eos.update(original_eos)
            eos.discard(None)
            # No previous chat or past_key_values is supplied; only the model/media are reused.
            with torch.inference_mode():
                output = self.model.generate(**inputs, use_audio_in_video=media_mode=='av', do_sample=False,
                    num_beams=1, use_cache=True, max_new_tokens=limit, eos_token_id=sorted(eos),
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_logits=False)
            ids = output.sequences[0,inputs['input_ids'].shape[-1]:]
            complete = bool(len(ids) and int(ids[-1]) in eos)
            raw = self.processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
            result = {"item_id":self.row['item_id'],"question":self.row['question'],"prompt":prompt,
                "rendered_prompt":rendered,"raw":raw,"complete":complete,"generated_tokens":len(ids),
                "max_new_tokens":limit,"input_tokens":int(inputs['input_ids'].shape[-1]),
                "prediction":None if diagnostic else parse_baseline(raw,complete) if baseline else parse_final(raw,complete),
                "format_issues":final_format_issues(raw,complete) if not (baseline or diagnostic) else [],
                "latency_s":time.perf_counter()-started,
                "peak_allocated_GiB":[round(torch.cuda.max_memory_allocated(i)/2**30,3) for i in range(torch.cuda.device_count())],
                "method":"diagnostic" if diagnostic else "baseline" if baseline else "prompt","fresh_context":True,
                "media_mode":media_mode,"audio_provided":media_mode in ('av','audio'),
                "video_provided":media_mode in ('av','vision'),
                "max_pixels_per_frame":self.media['max_pixels_per_frame'],
                "video_shape":list(self.media['videos'][0].shape) if media_mode!='audio' else None,
                "run_id":uuid4().hex[:12]}
            result['decision_interface']='Read the freely generated answer; no logit adjustment or forced A/B selection'
            result['media_preprocessing']='official_direct_video_defaults'
            path=self.output/(self.row['item_id']+'_'+result['method']+'_'+result['run_id']+'.json')
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            result['result_path']=str(path)
            return result
        finally:
            inputs=None; gc.collect(); torch.cuda.empty_cache()

    def dispatch(self, request):
        command=request['cmd']
        if command=='sample':return self.sample(request)
        if command=='baseline':
            return self.generate(self.row['question'],request.get('max_new_tokens',384),True)
        if command=='prompt':return self.generate(request['prompt'],request.get('max_new_tokens',384))
        if command=='diagnostic':
            if set(request)-{'cmd','request_id','prompt','media_mode','max_new_tokens','max_pixels'}:
                raise ValueError('Unexpected diagnostic fields; no labels are accepted')
            mode=request.get('media_mode','av')
            if mode not in ('av','vision','audio'):raise ValueError('Unknown media mode')
            if self.media is None:raise RuntimeError('Select one sample first')
            current=self.media
            pixels=request.get('max_pixels',current['max_pixels_per_frame'])
            if pixels not in (current['max_pixels_per_frame'],2*current['max_pixels_per_frame']):
                raise ValueError('Diagnostic resolution must be unchanged or doubled')
            try:
                if pixels!=current['max_pixels_per_frame']:
                    from lab_native_video import prepare_video
                    self.media=prepare_video(self.row['path'],pixels)
                    if self.media['fps']!=current['fps'] or self.media['videos'][0].shape[0]!=current['videos'][0].shape[0]:
                        raise RuntimeError('Resolution probe changed temporal sampling')
                return self.generate(request['prompt'],request.get('max_new_tokens',192),media_mode=mode,diagnostic=True)
            finally:self.media=current
        if command=='ping':return self.meta
        raise ValueError('Unknown command: '+command)


def main():
    sys.stdout = sys.stderr  # Library prints/progress cannot corrupt the JSON protocol.
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mock',action='store_true')
    args=parser.parse_args()
    try:
        engine=Engine(args.output) if not args.mock else None
        emit({'event':'ready','model':MODEL,'revision':REVISION,'mock':args.mock})
    except Exception:
        emit({'event':'fatal','error':traceback.format_exc()});return
    for line in sys.stdin:
        request_id=None
        try:
            request=json.loads(line);request_id=request.get('request_id')
            if request['cmd']=='close':emit({'request_id':request_id,'ok':True,'result':{}});break
            result=engine.dispatch(request) if engine else {'mock':True,'echo':request}
            emit({'request_id':request_id,'ok':True,'result':result})
        except Exception:
            emit({'request_id':request_id,'ok':False,'error':traceback.format_exc()})


if __name__=='__main__':main()
