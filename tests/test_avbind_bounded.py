import unittest
import torch
import torch.nn.functional as F
from repo.kaggle.jobs.avbind_bounded import bounded_memory_read


class BoundedReadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.q = torch.randn(1,4,3,8)
        self.k = torch.randn(1,2,9,8)
        self.v = torch.randn_like(self.k)
        self.mk = torch.randn(1,2,5,8)
        self.mv = torch.randn_like(self.mk)
        self.mask = torch.arange(9)[None,:] <= torch.tensor([6,7,8])[:,None]

    def test_null_is_exact_native_and_finite(self):
        output,budget,_ = bounded_memory_read(self.q,self.k,self.v,self.q,self.mk,self.mv,
            torch.full((1,1,1,5),float('-inf')),self.mask)
        native = F.scaled_dot_product_attention(self.q,self.k.repeat_interleave(2,-3),
            self.v.repeat_interleave(2,-3),attn_mask=self.mask)
        torch.testing.assert_close(output,native,rtol=0,atol=0)
        self.assertEqual(float(budget.sum()),0)
        self.assertTrue(torch.isfinite(output).all())

    def test_budget_cap_and_convex_read(self):
        output,budget,weights = bounded_memory_read(self.q,self.k,self.v,self.q,self.mk,self.mv,
            torch.zeros(1,1,1,5),self.mask)
        self.assertLessEqual(float(budget.max()),.25)
        native = F.scaled_dot_product_attention(self.q,self.k.repeat_interleave(2,-3),
            self.v.repeat_interleave(2,-3),attn_mask=self.mask)
        external = weights @ self.mv.repeat_interleave(2,-3)
        torch.testing.assert_close(output,.75*native+.25*external)

    def test_small_non_null_prior_scales_budget_without_token_count(self):
        prior = torch.full((1,1,1,5),.01).log()
        _,budget,_ = bounded_memory_read(self.q,self.k,self.v,self.q,self.mk,self.mv,prior,self.mask)
        self.assertAlmostEqual(float(budget.item()),.0125,places=6)

    def test_raw_causal_mask_preserved(self):
        prior = torch.zeros(1,1,1,5)
        first,_,_ = bounded_memory_read(self.q,self.k,self.v,self.q,self.mk,self.mv,prior,self.mask)
        changed = self.v.clone();changed[:,:,-1] += 100
        second,_,_ = bounded_memory_read(self.q,self.k,changed,self.q,self.mk,self.mv,prior,self.mask)
        torch.testing.assert_close(first[:,:,:2],second[:,:,:2])
        self.assertGreater(float((first[:,:,2]-second[:,:,2]).abs().max()),1)
