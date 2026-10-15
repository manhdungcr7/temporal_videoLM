"""Causal masking, counterfactual leakage and residual norms for CCLD."""
import ast
from pathlib import Path
import unittest
import torch
from test_latent_reopen_contract import Toy

ROOT=Path(__file__).resolve().parents[1]

def extract(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    ns={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


class CCLDContract(unittest.TestCase):
    def setUp(self):
        self.ns=extract(ROOT/'server_jobs/argon_ccld40.py',{'cc_forward'})
        self.ns.update(extract(ROOT/'results/argon_ccld40/proposal_original.py',
                               {'make_clean_mask','make_shuffled_control'}))
        torch.manual_seed(7);self.model=Toy();self.ns['MODEL_']=self.model
        self.v=torch.tensor([0,0,1,0,1,0,0,0],dtype=torch.bool)
        self.a=torch.tensor([0,0,0,1,0,1,0,0],dtype=torch.bool)
        self.sec=torch.tensor([0.,0.,0.,.1,8.,8.1,0.,0.])
        self.payload=dict(tokens=8,last_media=5,is_v=self.v,is_a=self.a,
            decoder={'inputs_embeds':torch.randn(1,8,8),'position_ids':torch.arange(8).reshape(1,1,8),
                     'attention_mask':torch.ones(1,8)})

    def mask(self,mode):
        return self.ns['make_clean_mask'](8,self.v,self.a,self.sec,5,(0.,2.),
            torch.float32,torch.device('cpu'),gate_mode=mode)[0,0]

    def test_all_controls_preserve_causality(self):
        upper=torch.ones(8,8,dtype=torch.bool).triu(1)
        for mode in ('none','audio_only','hard_sync','anti_sync'):
            mask=self.mask(mode)
            self.assertTrue(torch.all(mask[upper]==torch.finfo(mask.dtype).min))

    def test_visual_blind_blocks_indirect_visual_paths(self):
        allowed=self.mask('audio_only')==0
        # Reachability across multiple layers, including visual->audio->query.
        reach=allowed.clone()
        for _ in range(8):reach|=(reach.float()@allowed.float())>0
        self.assertFalse(reach[6:,self.v].any())
        self.assertTrue(reach[6:,self.a].any())

    def test_shuffled_control_retains_each_row_norm(self):
        x=torch.randn(7,64);shuffled=self.ns['make_shuffled_control'](x)
        torch.testing.assert_close(x.norm(dim=-1),shuffled.norm(dim=-1))
        self.assertFalse(torch.equal(x,shuffled))

    def test_native_replay_and_query_only_feedback(self):
        fn=self.ns['cc_forward'];base,_=fn(self.payload);native=self.model.model.last.clone()
        zero,_=fn(self.payload,delta=torch.zeros(2,8),recurrent_layers={10:.25,18:.20})
        torch.testing.assert_close(base,zero,rtol=0,atol=0)
        fn(self.payload,thought=torch.ones(8),delta=torch.ones(2,8),recurrent_layers={10:.25,18:.20})
        torch.testing.assert_close(native[:,:6],self.model.model.last[:,:6],rtol=0,atol=0)
        self.assertFalse(torch.equal(native[:,6:],self.model.model.last[:,6:]))

    def test_late_damping_starts_at_exact_block16(self):
        observed={};handles=[]
        for li,layer in enumerate(self.model.model.layers):
            def capture(m,args,kwargs,li=li):
                observed[li]=kwargs.get('attention_mask').clone()
            handles.append(layer.register_forward_pre_hook(capture,with_kwargs=True))
        # Capture after the intervention pre-hook by reordering hook registration.
        for handle in handles:handle.remove()
        original=type(self.model.model.layers[0]).forward
        for li,layer in enumerate(self.model.model.layers):
            def forward(hidden_states,attention_mask=None,li=li,**kwargs):
                observed[li]=attention_mask.clone()
                return (hidden_states+.01,)
            layer.forward=forward
        self.ns['cc_forward'](self.payload,late_audio=True)
        self.assertEqual(observed[15].ndim,2)
        self.assertTrue(torch.all(observed[16][0,0,6:,self.a]==-2.5))

    def test_hook_cleanup_on_error(self):
        self.model.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            self.ns['cc_forward'](self.payload,late_audio=True,thought=torch.ones(8),capture=True)
        for layer in self.model.model.layers:
            self.assertFalse(layer._forward_hooks);self.assertFalse(layer._forward_pre_hooks)


if __name__=='__main__':unittest.main()
