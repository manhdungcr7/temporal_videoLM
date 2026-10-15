"""Causality, null behavior and source/token accounting; not QA accuracy."""
import unittest
import torch
import torch.nn.functional as F
from repo.kaggle.jobs.avbind_attention import addressed_key_bias,segmented_sdpa

class AddressedAttentionTests(unittest.TestCase):
    def tensors(self):
        torch.manual_seed(3)
        return torch.randn(1,4,17,8),torch.randn(1,2,17,8),torch.randn(1,2,17,8)

    def test_zero_bias_matches_grouped_query_causal_attention(self):
        q,k,v=self.tensors();selected=torch.zeros(17,dtype=torch.bool);selected[10:13]=True;selected[-1]=True
        actual=segmented_sdpa(q,k,v,selected,torch.zeros(17))
        expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,is_causal=True)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_positive_bias_matches_full_causal_reference(self):
        q,k,v=self.tensors();selected=torch.zeros(17,dtype=torch.bool);selected[10:13]=True;selected[-1]=True
        bias=torch.linspace(0,2,17)
        mask=torch.zeros(17,17).masked_fill(~torch.ones(17,17,dtype=torch.bool).tril(),float('-inf'))
        mask=mask+selected[:,None]*bias[None]
        actual=segmented_sdpa(q,k,v,selected,bias)
        expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,attn_mask=mask)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_existing_masks_are_preserved(self):
        for boolean in (False,True):
            q,k,v=self.tensors();selected=torch.arange(17)>12;bias=torch.linspace(0,1,17)
            allowed=torch.ones(17,17,dtype=torch.bool).tril();allowed[:,3]=False
            original=allowed if boolean else torch.zeros(17,17).masked_fill(~allowed,float('-inf'))
            reference=torch.zeros(17,17).masked_fill(~allowed,float('-inf'))+selected[:,None]*bias[None]
            actual=segmented_sdpa(q,k,v,selected,bias,original,causal=False)
            expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,attn_mask=reference)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_cached_decode_leaves_new_text_keys_unbiased(self):
        q,k,v=self.tensors();q=q[:,:,-1:];bias=torch.ones(14)
        actual=segmented_sdpa(q,k,v,torch.ones(1,dtype=torch.bool),bias,causal=False)
        expected=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,attn_mask=F.pad(bias,(0,3)).view(1,17))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_future_tokens_cannot_change_selected_earlier_row(self):
        q,k,v=self.tensors();selected=torch.ones(17,dtype=torch.bool);bias=torch.ones(17)
        original=segmented_sdpa(q,k,v,selected,bias)
        k[:,:,8:]+=100;v[:,:,8:]-=50
        changed=segmented_sdpa(q,k,v,selected,bias)
        torch.testing.assert_close(original[:,:,:8],changed[:,:,:8],rtol=1e-5,atol=1e-6)

    def test_token_count_accounting_and_null_gate(self):
        nodes=[dict(source='video',interval=[0,2]),dict(source='audio',interval=[0,2])]
        seconds=torch.zeros(7);video=torch.tensor([1,1,1,1,0,0,0],dtype=torch.bool)
        audio=torch.tensor([0,0,0,0,1,1,0],dtype=torch.bool)
        support=torch.ones(2,2)
        bias=addressed_key_bias(nodes,support,torch.ones(2),seconds,video,audio,gain=2)
        # Equal slot support gives equal EXTRA mass despite unequal token counts.
        extra=bias.expm1()
        torch.testing.assert_close(extra[video].sum(),extra[audio].sum())
        self.assertEqual(float(bias[-1]),0)
        zero=addressed_key_bias(nodes,support,torch.zeros(2),seconds,video,audio,gain=2)
        torch.testing.assert_close(zero,torch.zeros(7))

    def test_each_slot_stays_attached_to_source_and_interval(self):
        nodes=[dict(source='video',interval=[0,2]),dict(source='audio',interval=[2,4])]
        sec=torch.tensor([0.,1.,2.,3.,0.,1.,2.,3.,9.]);video=torch.arange(9)<4;audio=(torch.arange(9)>=4)&(torch.arange(9)<8)
        support=torch.tensor([[1.,0.],[0.,0.]])
        bias=addressed_key_bias(nodes,support,torch.ones(2),sec,video,audio)
        self.assertTrue((bias[:2]>0).all());self.assertTrue((bias[2:]==0).all())
        shifted=addressed_key_bias([dict(n,interval=[t+100 for t in n['interval']]) for n in nodes],support,torch.ones(2),sec+100,video,audio)
        torch.testing.assert_close(bias,shifted)

if __name__=='__main__':unittest.main()
