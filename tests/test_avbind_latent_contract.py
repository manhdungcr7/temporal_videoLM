"""Input isolation/interface checks; not tests of learned AV QA accuracy."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

SOURCE=Path('repo/kaggle/jobs/avbind_latent.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE)

class LatentContract(unittest.TestCase):
    def test_sensor_bank_never_accepts_question_or_choices(self):
        functions={n.name:n for n in TREE.body if isinstance(n,ast.FunctionDef)}
        for name in ('latent_sensor_feature','latent_sensor_bank'):
            names={n.id for n in ast.walk(functions[name]) if isinstance(n,ast.Name)}
            self.assertFalse(names&{'question','answer_options','gold','task','options'})
            methods={n.func.attr for n in ast.walk(functions[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertNotIn('generate',methods)
        args=functions['avbind_latent_answer'].args
        self.assertEqual(ast.literal_eval(args.defaults[0]),None)

    def test_native_prompt_matches_plain_question_choices(self):
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='latent_question_inputs')
        strings=[n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertFalse(any('Answer with' in value or 'observations' in value or 'caption' in value for value in strings))

    def test_raw_capture_does_not_generate_and_removes_hook(self):
        calls=[]
        class IdentityLayer(torch.nn.Module):
            def forward(self,x):return x
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=type('Backbone',(),{})()
                self.model.layers=torch.nn.ModuleList([IdentityLayer() for _ in range(28)])
            def forward(self,**kwargs):
                calls.append(kwargs)
                value=torch.tensor([[[1.,2.,3.,4.],[4.,3.,2.,1.]]])
                for layer in self.model.layers:value=layer(value)
                return value
        model=Model()
        def inputs(proc,prompt,media,mode):
            calls.append(dict(prompt=prompt,mode=mode))
            return dict(),None
        namespace=dict(qwen_inputs=inputs,move_inputs=lambda inp,_model:inp,PROC=None,MODEL_=model,
                       torch=torch,F=F,BIND_LAYER=28)
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='latent_sensor_feature')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'latent_contract','exec'),namespace)
        for mode in ('vision','audio'):
            vector=namespace['latent_sensor_feature'](dict(),mode)
            torch.testing.assert_close(vector,F.layer_norm(torch.tensor([4.,3.,2.,1.]),(4,)))
            self.assertEqual(calls[-2]['mode'],mode)
            self.assertNotIn('question',calls[-2]['prompt'].lower())
            self.assertFalse(calls[-1]['use_audio_in_video'])
            self.assertEqual(len(model.model.layers[27]._forward_hooks),0)

    def test_unicode_question_span_excludes_options(self):
        class ByteTokenizer:
            def convert_tokens_to_ids(self,name):return {'video':999,'audio':998}[name]
            def decode(self,ids,**kwargs):return bytes(ids).decode('utf-8')
            def __call__(self,text,**kwargs):
                offsets=[]
                for i,character in enumerate(text):offsets.extend([(i,i+1)]*len(character.encode('utf-8')))
                return dict(input_ids=list(text.encode('utf-8')),offset_mapping=offsets)
        question='Trong cảnh người đàn ông đang quét nhà thì ổng nói gì?'
        tail='\n'+question+' (A) tùy chọn (B) lựa chọn\n'
        namespace=dict(TOK=ByteTokenizer(),PROC=SimpleNamespace(video_token='video',audio_token='audio'))
        node=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='native_query_span')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'unicode_span','exec'),namespace)
        ids=[999,998]+list(tail.encode('utf-8'))
        positions=namespace['native_query_span'](dict(input_ids=torch.tensor([ids])),question)
        self.assertEqual(bytes([ids[i] for i in positions]).decode('utf-8'),question)

if __name__=='__main__':unittest.main()
