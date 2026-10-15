import unittest
import torch
from repo.kaggle.jobs.avbind_alignment import NativeSemanticAlignment,semantic_pair_loss,occurrence_token_bias


class NativeAlignmentTests(unittest.TestCase):
    def test_initial_projection_preserves_layer_normalized_content(self):
        torch.manual_seed(4);model=NativeSemanticAlignment(8,3);x=torch.randn(2,4,8);sources=torch.tensor([[0,1,0,1],[1,1,0,0]])
        torch.testing.assert_close(model(x,sources),torch.nn.functional.layer_norm(x,(8,)))
        self.assertTrue(torch.isfinite(model(x,sources)).all())

    def test_multiple_identical_descriptions_are_not_false_negatives(self):
        projected=torch.eye(4,requires_grad=True);positive=torch.ones(4,4,dtype=torch.bool)
        loss=semantic_pair_loss(projected,projected.detach(),positive);self.assertAlmostEqual(float(loss.detach()),0,places=5)
        loss.backward();self.assertIsNotNone(projected.grad)

    def test_token_count_normalization_equalizes_added_mass_not_logits(self):
        query=torch.tensor([[1.,0.]])
        records=torch.tensor([[1.,0.],[0.,1.]])
        bias,audit=occurrence_token_bias(query,records,[(0,4),(4,6)],7)
        extra=bias.expm1();ratio=extra[:,:4].sum()/extra[:,4:6].sum()
        expected=audit['record_weights'][0,0]/audit['record_weights'][0,1]
        torch.testing.assert_close(ratio,expected,rtol=1e-5,atol=1e-5)
        self.assertEqual(float(bias[0,6]),0)

    def test_ambiguous_query_affinities_disable_boost(self):
        query=torch.ones(2,8);records=torch.ones(4,8)
        bias,audit=occurrence_token_bias(query,records,[(0,2),(2,4),(4,6),(6,8)],10)
        torch.testing.assert_close(bias,torch.zeros_like(bias),rtol=0,atol=1e-6)

    def test_occurrence_order_permutation_does_not_change_scores(self):
        torch.manual_seed(2);query=torch.randn(3,8);records=torch.randn(3,8)
        bias,audit=occurrence_token_bias(query,records,[(0,2),(2,5),(5,7)],9)
        order=torch.tensor([2,0,1]);permuted,_=occurrence_token_bias(query,records[order],[(5,7),(0,2),(2,5)],9)
        torch.testing.assert_close(bias,permuted)
