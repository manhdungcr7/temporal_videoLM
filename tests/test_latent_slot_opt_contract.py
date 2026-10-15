"""Memory visibility must preserve causal edges and source/time support."""
import ast
from pathlib import Path
import unittest
import torch

class SlotOptimizationContract(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'server_jobs/latent_slot_opt40.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
               and n.name in ('lo_mask','lo_slots')]
        self.ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),self.ns)

    def test_memory_never_opens_future_query_edges(self):
        mask=self.ns['lo_mask'](3,4,2,torch.float32,torch.device('cpu'),memory_bias=2.)[0,0]
        self.assertEqual(tuple(mask.shape),(3,9))
        self.assertTrue(torch.all(mask[0,7:]==torch.finfo(torch.float32).min))
        self.assertEqual(float(mask[1,8]),torch.finfo(torch.float32).min)
        self.assertTrue(torch.all(mask[:,4:6]==2.))

    def test_disabled_memory_is_identical_to_native_after_removing_virtual_keys(self):
        fn=self.ns['lo_mask'];native=fn(3,4,0,torch.float32,torch.device('cpu'))
        augmented=fn(3,4,2,torch.float32,torch.device('cpu'),memory_enabled=False)
        keep=torch.tensor([0,1,2,3,6,7,8])
        torch.testing.assert_close(native,augmented[...,keep],rtol=0,atol=0)

    def test_source_blind_slot_cannot_read_omitted_source_keys(self):
        forbidden=torch.tensor([False,True,False,True,False,False])
        mask=self.ns['lo_mask'](2,4,0,torch.float32,torch.device('cpu'),forbidden=forbidden)
        self.assertTrue(torch.all(mask[...,1]==torch.finfo(torch.float32).min))
        self.assertTrue(torch.all(mask[...,3]==torch.finfo(torch.float32).min))

    def test_slot_pooling_never_mixes_sources_or_time_groups(self):
        payload=dict(is_v=torch.tensor([1,0,1,0,1,0],dtype=torch.bool),
            is_a=torch.tensor([0,1,0,1,0,1],dtype=torch.bool),sec=torch.tensor([0.,.1,.9,1.,3.,3.1]),
            decoder={'inputs_embeds':torch.arange(24,dtype=torch.float32).reshape(1,6,4),
                     'position_ids':torch.arange(6).reshape(1,1,6).repeat(3,1,1)})
        h=torch.arange(24,dtype=torch.float32).reshape(6,4)*2
        visual=self.ns['lo_slots'](payload,'video',h)
        audio=self.ns['lo_slots'](payload,'audio',h)
        self.assertEqual(len(visual),2);self.assertEqual(len(audio),2)
        torch.testing.assert_close(visual[0]['target'],h[[0,2]].mean(0))
        torch.testing.assert_close(audio[0]['target'],h[[1,3]].mean(0))
        self.assertLess(visual[0]['time'],visual[1]['time'])

if __name__=='__main__':unittest.main()
