"""CPU checks for causal/source contracts of the new latent intervention."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn


def definitions():
    source=Path(__file__).resolve().parents[1]/'server_jobs/latent_reopen40.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)
              and node.name in ('lr_local_mask','lr_forward')]
    namespace={'torch':torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


class Block(nn.Module):
    def forward(self,hidden_states,attention_mask=None,**kwargs):
        return (hidden_states+.01,)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Block() for _ in range(24)])
        self.fail=False;self.last=None
    def forward(self,inputs_embeds,attention_mask=None,**kwargs):
        h=inputs_embeds
        for layer in self.layers:h=layer(hidden_states=h,attention_mask=attention_mask)[0]
        if self.fail:raise RuntimeError('deliberate decoder failure')
        self.last=h.detach().clone()
        return SimpleNamespace(last_hidden_state=h)


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(10,8);self.model=Decoder();self.lm_head=nn.Linear(8,5)
    def get_input_embeddings(self):return self.embedding


class LatentReopenContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);self.ns=definitions();self.model=Toy();self.ns['MODEL_']=self.model
        self.payload=dict(tokens=8,last_media=5,
            is_v=torch.tensor([False,False,True,False,True,False,False,False]),
            is_a=torch.tensor([False,False,False,True,False,True,False,False]),
            sec=torch.tensor([0.,0.,0.,.1,8.,8.1,0.,0.]),
            decoder={'inputs_embeds':torch.randn(1,8,8),
                     'position_ids':torch.arange(8).reshape(1,1,8),
                     'attention_mask':torch.ones(1,8)})

    def test_only_distant_causal_cross_source_edges_change(self):
        mask=self.ns['lr_local_mask'](self.payload)[0,0]
        neg=torch.finfo(mask.dtype).min
        for row in range(8):
            for col in range(8):
                if col>row:self.assertEqual(float(mask[row,col]),neg)
                elif (row,col) in ((4,3),(5,2)):self.assertEqual(float(mask[row,col]),-3.5)
                else:self.assertEqual(float(mask[row,col]),0.)

    def test_global_question_keeps_every_media_key(self):
        mask=self.ns['lr_local_mask'](self.payload)[0,0]
        self.assertTrue(torch.equal(mask[6:,:6],torch.zeros(2,6)))

    def test_zero_reflection_is_exact_identity_and_media_untouched(self):
        forward=self.ns['lr_forward']
        base,_=forward(self.payload);native=self.model.model.last.clone()
        zero={18:torch.zeros(2,8),22:torch.zeros(2,8)}
        result,_=forward(self.payload,reflection=zero)
        torch.testing.assert_close(base,result,rtol=0,atol=0)
        delta={18:torch.ones(2,8),22:torch.ones(2,8)}
        forward(self.payload,reflection=delta)
        torch.testing.assert_close(native[:,:6],self.model.model.last[:,:6],rtol=0,atol=0)
        self.assertFalse(torch.equal(native[:,6:],self.model.model.last[:,6:]))

    def test_cleanup_after_failure(self):
        self.model.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            self.ns['lr_forward'](self.payload,mask=self.ns['lr_local_mask'](self.payload),capture=True)
        for layer in self.model.model.layers:
            self.assertFalse(layer._forward_hooks);self.assertFalse(layer._forward_pre_hooks)


if __name__=='__main__':unittest.main()
