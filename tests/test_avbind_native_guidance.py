import unittest
import torch
import torch.nn.functional as F
from repo.kaggle.jobs.avbind_native_guidance import indexed_occurrence_bias,native_late_guidance
from repo.kaggle.jobs.avbind_alignment import occurrence_token_bias
import repo.kaggle.jobs.avbind_native_guidance as native


class NativeGuidanceTests(unittest.TestCase):
    def test_noncontiguous_native_tokens_keep_identity(self):
        native.occurrence_token_bias=occurrence_token_bias
        query=torch.eye(2);projected=torch.eye(2);indices=[torch.tensor([1,4]),torch.tensor([2,5,6])]
        actual,_=indexed_occurrence_bias(query,projected,indices,8)
        self.assertTrue((actual[:,[0,3,7]]==0).all())
        packed,_=occurrence_token_bias(query,projected,[(0,2),(2,5)],5)
        torch.testing.assert_close(actual[:,[1,4,2,5,6]],packed)

    def test_native_causality_selected_rows_and_measured_attention(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa'];module=object();sensor=object();torch.manual_seed(12)
        q=torch.randn(1,4,9,8);k=torch.randn(1,2,9,8);v=torch.randn_like(k)
        keys=k.repeat_interleave(2,-3);values=v.repeat_interleave(2,-3)
        causal=torch.ones(9,9,dtype=torch.bool).tril()
        def ordinary(m,q,k,v,mask,scaling=None,**kwargs):
            if k.shape[-3]!=q.shape[-3]:k=k.repeat_interleave(2,-3);v=v.repeat_interleave(2,-3)
            mask=causal if mask is None else causal & mask if mask.dtype==torch.bool else mask.masked_fill(~causal,float('-inf'))
            return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scaling).transpose(1,2),None
        ALL_ATTENTION_FUNCTIONS.register('sdpa',ordinary)
        bias=torch.zeros(2,9);bias[:,[1,3]]=.6;answer=torch.zeros(1,9);answer[:,[2,4]]=.7
        try:
            for supplied in (None,torch.ones(1,1,9,9,dtype=torch.bool),torch.zeros(1,1,9,9)):
                audit=[]
                with native_late_guidance([module],[6,7],bias,answer,[torch.tensor([1,3]),torch.tensor([2,4])],audit):
                    actual,_=ALL_ATTENTION_FUNCTIONS['sdpa'](module,q,k,v,supplied,scaling=.3)
                    untouched,_=ALL_ATTENTION_FUNCTIONS['sdpa'](sensor,q,k,v,supplied,scaling=.3)
                full=torch.zeros(9,9);full[[6,7]]=bias;full[8]=answer[0]
                expected=F.scaled_dot_product_attention(q,keys,values,attn_mask=full.masked_fill(~causal,float('-inf')),scale=.3).transpose(1,2)
                torch.testing.assert_close(actual,expected)
                torch.testing.assert_close(actual[:,:6],untouched[:,:6])
                self.assertEqual(len(audit),1);self.assertTrue(audit[0]['actual_native_attention'])
                self.assertGreater(audit[0]['after'][1],audit[0]['before'][1])
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

    def test_zero_gain_identity_and_registry_restore(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa'];module=object();sentinel=torch.randn(1,3,1,2)
        def ordinary(*args,**kwargs):return sentinel,None
        ALL_ATTENTION_FUNCTIONS.register('sdpa',ordinary)
        try:
            with native_late_guidance([module],[1],torch.zeros(1,3),torch.zeros(1,3),[],[],gain=0):
                actual,_=ALL_ATTENTION_FUNCTIONS['sdpa'](module,sentinel,sentinel,sentinel,None)
                self.assertIs(actual,sentinel)
            self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],ordinary)
            with self.assertRaises(RuntimeError):
                with native_late_guidance([module],[1],torch.zeros(1,3),torch.zeros(1,3),[],[]):
                    raise RuntimeError('simulate failure')
            self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],ordinary)
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)


if __name__=='__main__':unittest.main()
