"""Explicit repeated keys preserve GQA attention, including masks/gradients."""
import unittest
import torch
import torch.nn.functional as F

class GroupedQueryEquivalence(unittest.TestCase):
    def test_causal_and_masked_attention_and_gradients(self):
        for causal,mask_kind in ((True,None),(False,None),(False,'bool'),(False,'float')):
            with self.subTest(causal=causal,mask=mask_kind):
                torch.manual_seed(17)
                q=torch.randn(1,4,7,8,requires_grad=True)
                k=torch.randn(1,2,7,8,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
                mask=None
                if mask_kind=='bool':mask=torch.ones(7,7,dtype=torch.bool).tril()
                if mask_kind=='float':mask=torch.zeros(7,7).masked_fill(~torch.ones(7,7,dtype=torch.bool).tril(),float('-inf'))
                a=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,is_causal=causal,attn_mask=mask)
                b=F.scaled_dot_product_attention(q,k.repeat_interleave(2,dim=1),v.repeat_interleave(2,dim=1),is_causal=causal,attn_mask=mask)
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True)
                gb=torch.autograd.grad(b.square().sum(),(q,k,v))
                for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
