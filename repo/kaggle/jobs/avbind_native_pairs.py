"""Question-blind native AV / observation-text pairs for adapter pretraining.

Pseudo annotations are NOT ground truth event labels. Source/window provenance
is known from physical crops; text can be noisy. No benchmark question, answer,
or category enters feature extraction. Separation is by parent recording ID,
not just clip filename, to avoid another clip of an evaluation recording.
"""
from pathlib import PurePosixPath
import re
import torch
import torch.nn.functional as F


def recording_key(path):
    parent=PurePosixPath(str(path).replace('\\','/')).parent
    return parent.name if len(parent.name)==11 else str(parent)


def native_pair_feature(media,mode):
    prompt='Observe this recording carefully and describe its directly available content.'
    inp,_=qwen_inputs(PROC,prompt,media,mode);inp=move_inputs(inp,MODEL_);captured={}
    class Ready(Exception):pass
    def capture(module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['value']=F.layer_norm(hidden[0,-1].float(),(hidden.shape[-1],)).detach().cpu()
        raise Ready()
    handle=MODEL_.model.layers[27].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_audio_in_video=False,use_cache=False)
    except Ready:pass
    finally:handle.remove()
    return captured['value']


def native_pair_description(media,mode):
    from transformers import StoppingCriteria,StoppingCriteriaList
    prompt=('Transcribe intelligible speech verbatim and briefly identify other audible sounds. '
            'If there is no speech, describe only the sounds. Give only directly heard content, no conversation.'
            if mode=='audio' else 'Describe directly visible entities, actions and changes. '
            'Do not infer speech, sounds or events outside the supplied frames. One concise factual observation.')
    inp,_=qwen_inputs(PROC,prompt,media,mode);inp=move_inputs(inp,MODEL_);length=inp['input_ids'].shape[1]
    class StopDialogue(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=TOK.decode(input_ids[0,length:],skip_special_tokens=True)
            return bool(re.search(r'(?:\n|\b)(?:Human|Assistant|User):',text))
    handle=MODEL_.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],))
    try:
        with torch.no_grad():sequence=MODEL_.generate(**inp,use_audio_in_video=False,use_cache=True,
            do_sample=False,max_new_tokens=48,stopping_criteria=StoppingCriteriaList([StopDialogue()]))
    finally:handle.remove()
    text=TOK.decode(sequence[0,length:],skip_special_tokens=True).strip()
    text=re.split(r'(?:\n|\b)(?:Human|Assistant|User):',text)[0].strip()
    return text


def native_pair_text_feature(content):
    messages=[dict(role='system',content=[dict(type='text',text=SYSTEM)]),
              dict(role='user',content=[dict(type='text',text='An independently observed fact: '+content)])]
    chat=PROC.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    inp=PROC(text=chat,return_tensors='pt').to(MODEL_.get_input_embeddings().weight.device)
    encoded=TOK(chat,return_offsets_mapping=True)
    assert encoded['input_ids']==inp['input_ids'][0].tolist()
    begin=chat.index(content);end=begin+len(content)
    indices=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>begin and a<end]
    if not indices:raise ValueError('Empty annotation span')
    captured={}
    class Ready(Exception):pass
    def capture(module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured['value']=F.layer_norm(hidden[0,indices].float().mean(0),(hidden.shape[-1],)).detach().cpu()
        raise Ready()
    handle=MODEL_.model.layers[27].register_forward_hook(capture)
    try:
        with torch.no_grad():MODEL_(**inp,use_cache=False)
    except Ready:pass
    finally:handle.remove()
    return captured['value']
