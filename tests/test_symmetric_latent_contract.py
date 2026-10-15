"""Geometry and source-access checks for the independent latent correction."""
import ast
from pathlib import Path
import unittest
import torch
from test_latent_reopen_contract import Toy

class SymmetricContract(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'server_jobs/symmetric_latent40.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sl_project','sl_mask','sl_forward','sl_question_rows')]
        self.ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),self.ns)
        torch.manual_seed(9);self.model=Toy();self.ns['MODEL_']=self.model
        self.payload=dict(tokens=8,last_media=5,
            is_v=torch.tensor([0,0,1,0,1,0,0,0],dtype=torch.bool),
            is_a=torch.tensor([0,0,0,1,0,1,0,0],dtype=torch.bool),
            decoder={'inputs_embeds':torch.randn(1,8,8),'position_ids':torch.arange(8).reshape(1,1,8),
                     'attention_mask':torch.ones(1,8)})

    def test_projection_removes_span_and_is_source_swap_invariant(self):
        v=torch.randn(5,32);a=torch.randn(5,32);i=torch.randn(5,32)
        result=self.ns['sl_project'](i,v,a)
        torch.testing.assert_close((result*v).sum(-1),torch.zeros(5),atol=2e-6,rtol=0)
        torch.testing.assert_close((result*a).sum(-1),torch.zeros(5),atol=2e-6,rtol=0)
        torch.testing.assert_close(result,self.ns['sl_project'](i,a,v),atol=1e-6,rtol=1e-6)

    def test_rank_deficient_and_zero_bases_are_finite(self):
        v=torch.randn(5,32);i=torch.randn(5,32)
        result=self.ns['sl_project'](i,v,2*v)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close((result*v).sum(-1),torch.zeros(5),atol=2e-6,rtol=0)
        torch.testing.assert_close(self.ns['sl_project'](i,torch.zeros_like(v),torch.zeros_like(v)),i)

    def test_all_masks_are_causal_and_block_indirect_paths(self):
        for absent in [('visual',),('audio',),('visual','audio')]:
            mask=self.ns['sl_mask'](self.payload,absent)[0,0]
            upper=torch.ones(8,8,dtype=torch.bool).triu_(1)
            self.assertTrue(torch.all(mask[upper]==torch.finfo(mask.dtype).min))
            allowed=mask==0;reach=allowed.clone()
            for _ in range(8):reach|=(reach.float()@allowed.float())>0
            for source in absent:
                keys=self.payload['is_v' if source=='visual' else 'is_a']
                self.assertFalse(reach[6:,keys].any())

    def test_small_deltas_never_inflated_and_media_unchanged(self):
        fn=self.ns['sl_forward'];fn(self.payload);base=self.model.model.last.clone()
        d=torch.ones(2,8)*1e-4
        fn(self.payload,deltas={22:d});result=self.model.model.last
        torch.testing.assert_close(base[:,:6],result[:,:6],rtol=0,atol=0)
        torch.testing.assert_close(result[:,6:]-base[:,6:],d[None],rtol=.005,atol=3e-7)

    def test_large_delta_is_bounded(self):
        fn=self.ns['sl_forward'];_,states=fn(self.payload,capture=True)
        native=self.model.model.last.clone();fn(self.payload,deltas={22:torch.ones(2,8)*1e4},gain=.15)
        change=(self.model.model.last[0,6:]-native[0,6:]).norm(dim=-1)
        self.assertTrue(torch.all(change<=.15*states[22].norm(dim=-1)+1e-6))

    def test_zero_feedback_identity_and_hook_cleanup(self):
        fn=self.ns['sl_forward'];base,_=fn(self.payload)
        result,_=fn(self.payload,deltas={22:torch.zeros(2,8)})
        torch.testing.assert_close(base,result,rtol=0,atol=0)
        self.model.model.fail=True
        with self.assertRaises(RuntimeError):fn(self.payload,mask=self.ns['sl_mask'](self.payload,('audio',)),capture=True)
        for layer in self.model.model.layers:
            self.assertFalse(layer._forward_hooks);self.assertFalse(layer._forward_pre_hooks)

    def test_question_stem_excludes_choices_by_exact_embedding_match(self):
        from types import SimpleNamespace
        self.ns['PROC']=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kwargs:[2]))
        self.payload['decoder']['inputs_embeds'][0,6]=self.model.embedding.weight[2].detach()
        selected=self.ns['sl_question_rows'](self.payload,'arbitrary question')
        self.assertEqual(selected.tolist(),[True,False])

if __name__=='__main__':unittest.main()
