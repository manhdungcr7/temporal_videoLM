"""Continuous thoughts must target the stem, without direct choice steering."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F
from test_latent_reopen_contract import Toy

ROOT=Path(__file__).resolve().parents[1]

def functions(path,names,ns):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body
           if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

class QueryLatentContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2);self.model=Toy();self.ns={'torch':torch,'F':F,'MODEL_':self.model}
        functions(ROOT/'server_jobs/symmetric_latent40.py',{'sl_forward'},self.ns)
        functions(ROOT/'server_jobs/query_latent40.py',{'ql_forward'},self.ns)
        self.payload=dict(tokens=8,last_media=5,
            decoder={'inputs_embeds':torch.randn(1,8,8),'position_ids':torch.arange(8).reshape(1,1,8),
                     'attention_mask':torch.ones(1,8)})

    def test_only_stem_receives_direct_feedback(self):
        self.ns['sl_forward'](self.payload);base=self.model.model.last.clone()
        self.ns['ql_forward'](self.payload,None,torch.ones(8),torch.tensor([True,False]))
        result=self.model.model.last
        torch.testing.assert_close(base[:,:6],result[:,:6],rtol=0,atol=0)
        torch.testing.assert_close(base[:,7:],result[:,7:],rtol=0,atol=0)
        self.assertFalse(torch.equal(base[:,6],result[:,6]))

    def test_cleanup_and_zero_thought_identity(self):
        base,_=self.ns['sl_forward'](self.payload)
        result,_=self.ns['ql_forward'](self.payload,None,torch.zeros(8),torch.tensor([True,False]))
        torch.testing.assert_close(base,result,rtol=0,atol=0)
        self.model.model.fail=True
        with self.assertRaises(RuntimeError):
            self.ns['ql_forward'](self.payload,None,torch.ones(8),torch.tensor([True,False]))
        for layer in self.model.model.layers:
            self.assertFalse(layer._forward_hooks);self.assertFalse(layer._forward_pre_hooks)

    def test_global_prototype_is_window_independent(self):
        functions(ROOT/'results/argon_ccld40/proposal_original.py',{'extract_continuous_thoughts'},self.ns)
        self.ns['hidden_to_manifold_thought_batch']=lambda h,**kw:F.normalize(h,dim=-1)
        h=torch.randn(8,16);v=torch.tensor([0,0,1,0,1,0,0,0],dtype=torch.bool)
        a=torch.tensor([0,0,0,1,0,1,0,0],dtype=torch.bool);sec=torch.tensor([0.,0.,0.,.1,8.,8.1,0.,0.])
        _,first=self.ns['extract_continuous_thoughts'](h,sec,v,a,5,(0.,2.))
        _,second=self.ns['extract_continuous_thoughts'](h,sec,v,a,5,(8.,9.))
        torch.testing.assert_close(first,second,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
