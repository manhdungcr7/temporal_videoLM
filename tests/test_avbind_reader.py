"""CPU structural checks only: no answer accuracy or learned performance claims.

Run from the workspace root: python -B -m unittest discover -s tests -p test_avbind_reader.py
Only torch and the Python standard library are required.
"""

import math
import unittest

import torch

from repo.kaggle.jobs.avbind_reader import AVBindReader


class AVBindReaderStructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.reader = AVBindReader(dim=12, rank=4, num_sources=3).cpu().eval()
        self.queries = torch.randn(2, 3, 12)
        self.content = torch.randn(2, 5, 12)
        self.sources = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 2, 0]])
        self.intervals = torch.tensor([
            [[0., 1.], [2., 4.], [5., 6.], [8., 10.], [11., 13.]],
            [[1., 3.], [4., 5.], [7., 9.], [10., 12.], [14., 15.]],
        ], dtype=torch.float64)
        self.validity = torch.tensor([
            [True, True, False, True, True],
            [True, False, True, True, True],
        ])

    def run_reader(self, **changes):
        inputs = dict(queries=self.queries, memory_content=self.content,
                      source_ids=self.sources, intervals=self.intervals,
                      validity=self.validity)
        inputs.update(changes)
        return self.reader(**inputs)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def assertMetadata(self, hop, content, sources, intervals, validity):
        self.assertTrue(torch.equal(hop.content, content))
        self.assertTrue(torch.equal(hop.source_ids, sources))
        self.assertTrue(torch.equal(hop.intervals, intervals))
        self.assertTrue(torch.equal(hop.validity, validity))
        self.assertEqual(hop.intervals.dtype, intervals.dtype)

    def test_permutation_equivariance_and_record_binding(self):
        original = self.run_reader()
        # Different permutation per batch also exercises both pairwise axes.
        permutation = torch.tensor([[3, 0, 4, 2, 1], [2, 4, 1, 0, 3]])
        batch = torch.arange(2)[:, None]
        content = self.content[batch, permutation]
        sources = self.sources[batch, permutation]
        intervals = self.intervals[batch, permutation]
        validity = self.validity[batch, permutation]
        permuted = self.run_reader(memory_content=content, source_ids=sources,
                                   intervals=intervals, validity=validity)
        self.assertClose(permuted.output, original.output)
        self.assertClose(permuted.residual, original.residual)
        for before, after in zip(original.hops, permuted.hops):
            self.assertMetadata(after, content, sources, intervals, validity)
            for field in ("content_logits", "address_logits", "temporal_logits",
                          "support_logits", "support", "pool_weights", "read_weights"):
                expected = getattr(before, field).gather(
                    2, permutation[:, None, :].expand(2, 3, 5)
                )
                self.assertClose(getattr(after, field), expected)
            for field in ("null_logits", "predicted_null_support", "null_support",
                          "null_gate", "residual_gate", "delta"):
                self.assertClose(getattr(after, field), getattr(before, field))

    def test_time_translation_retains_exact_original_metadata(self):
        original = self.run_reader()
        shifted_intervals = self.intervals + 1_000_000.125
        shifted = self.run_reader(intervals=shifted_intervals)
        for before, after in zip(original.hops, shifted.hops):
            self.assertMetadata(before, self.content, self.sources,
                                self.intervals, self.validity)
            self.assertMetadata(after, self.content, self.sources,
                                shifted_intervals, self.validity)
        # This module's latent reads use a relative origin. End-to-end model
        # invariance is a separate test, since query features may encode time.
        self.assertClose(original.hops[0].content_logits, shifted.hops[0].content_logits)
        self.assertClose(original.residual, shifted.residual)
        for before,after in zip(original.hops,shifted.hops):
            self.assertClose(before.support_logits,after.support_logits)
        self.assertTrue(torch.isfinite(shifted.residual).all())

    def test_multi_occurrence_independent_support_without_timestamp_pooling(self):
        # Zero logits deterministically yield several independent 0.5 supports.
        # This checks structure, not whether an untrained reader finds evidence.
        with torch.no_grad():
            for parameter in self.reader.parameters():
                parameter.zero_()
        content = self.content[:, :1].expand(-1, 4, -1).clone()
        sources = torch.tensor([[0, 0, 1, 2], [2, 1, 1, 0]])
        intervals = torch.tensor([
            [[0., 1.], [20., 21.], [40., 44.], [60., 61.]],
            [[2., 3.], [22., 23.], [42., 45.], [62., 63.]],
        ], dtype=torch.float64)
        validity = torch.ones(2, 4, dtype=torch.bool)
        result = self.run_reader(memory_content=content, source_ids=sources,
                                 intervals=intervals, validity=validity)
        self.assertEqual(result.residual.shape, (2, 3, 12))
        self.assertEqual(len(result.hops), 2)
        for hop in result.hops:
            self.assertEqual(hop.support.shape, (2, 3, 4))
            self.assertClose(hop.support, torch.full((2, 3, 4), 0.5))
            self.assertClose(hop.support.sum(-1), torch.full((2, 3), 2.0))
            self.assertEqual(hop.null_support.shape, (2, 3))
            self.assertMetadata(hop, content, sources, intervals, validity)
            self.assertEqual(hop.intervals.shape, (2, 4, 2))

    def test_occurrence_edit_remains_attached_and_semantic_lookup_separate(self):
        original = self.run_reader()
        edited_content = self.content.clone()
        edited_sources = self.sources.clone()
        edited_intervals = self.intervals.clone()
        edited_content[0, 1] += 1
        edited_sources[0, 1] = 2
        edited_intervals[0, 1] = torch.tensor([100., 103.], dtype=torch.float64)
        result = self.run_reader(memory_content=edited_content,
                                 source_ids=edited_sources, intervals=edited_intervals)
        for hop in result.hops:
            self.assertMetadata(hop, edited_content, edited_sources,
                                edited_intervals, self.validity)
        metadata_only = self.run_reader(source_ids=edited_sources, intervals=edited_intervals)
        self.assertClose(original.hops[0].content_logits, metadata_only.hops[0].content_logits)
        unchanged = torch.tensor([0, 2, 3, 4])
        self.assertClose(original.hops[0].content_logits[0, :, unchanged],
                         result.hops[0].content_logits[0, :, unchanged])

    def test_all_invalid_padding_is_finite_and_forces_null(self):
        result = self.run_reader(
            memory_content=torch.full_like(self.content, float("nan")),
            source_ids=torch.full_like(self.sources, -999),
            intervals=torch.full_like(self.intervals, float("nan")),
            validity=torch.zeros_like(self.validity),
            null_support_override=torch.zeros(2, 3, 2),
        )
        self.assertTrue(torch.equal(result.output, self.queries))
        self.assertTrue(torch.equal(result.residual, torch.zeros_like(self.queries)))
        for hop in result.hops:
            for field in ("content_logits", "address_logits", "temporal_logits",
                          "support_logits", "support", "pool_weights", "read_weights",
                          "null_logits", "predicted_null_support", "null_support",
                          "null_gate", "residual_gate", "delta"):
                self.assertTrue(torch.isfinite(getattr(hop, field)).all(), field)
            self.assertTrue(torch.equal(hop.null_support, torch.ones(2, 3)))
            self.assertTrue(torch.equal(hop.predicted_null_support, torch.ones(2, 3)))
            self.assertTrue(torch.equal(hop.support, torch.zeros(2, 3, 5)))
            self.assertTrue(torch.equal(hop.residual_gate, torch.zeros(2, 3)))

    def test_second_hop_compares_occurrences_before_relation_pooling(self):
        with torch.no_grad():
            for parameter in self.reader.parameters():
                parameter.zero_()
            self.reader.hops[1].address_query.weight[0, 0] = 1
            self.reader.hops[1].relation_query.weight[0, 0] = 1
        queries = torch.zeros(1, 1, 12)
        queries[..., 0] = 1
        intervals = torch.tensor([[[0., 1.], [10., 11.]]], dtype=torch.float64)
        result = self.run_reader(
            queries=queries, memory_content=torch.zeros(1, 2, 12),
            source_ids=torch.zeros(1, 2, dtype=torch.long), intervals=intervals,
            validity=torch.ones(1, 2, dtype=torch.bool),
            null_support_override=torch.zeros(1, 1, 2),
        )
        self.assertClose(result.hops[0].read_weights, torch.full((1, 1, 2), 0.5))
        expected = 0.5 * math.log1p(10) / math.sqrt(3)
        self.assertClose(result.hops[1].temporal_logits,
                         torch.tensor([[[-expected, expected]]]))
        # Comparing against a mean timestamp (5) would yield log1p(5),
        # which is distinct from the encoded per-occurrence relation above.
        self.assertNotAlmostEqual(expected, math.log1p(5) / math.sqrt(3))
        self.assertTrue(torch.equal(result.hops[1].intervals, intervals))

    def test_empty_memory_and_mixed_validity(self):
        empty = self.run_reader(memory_content=self.content[:, :0],
                                source_ids=self.sources[:, :0],
                                intervals=self.intervals[:, :0],
                                validity=self.validity[:, :0])
        self.assertTrue(torch.equal(empty.output, self.queries))
        for hop in empty.hops:
            self.assertEqual(hop.support.shape, (2, 3, 0))
            self.assertTrue(torch.equal(hop.null_support, torch.ones(2, 3)))
        validity = self.validity.clone()
        validity[0] = False
        mixed = self.run_reader(validity=validity)
        self.assertTrue(torch.equal(mixed.output[0], self.queries[0]))
        for hop in mixed.hops:
            self.assertTrue(torch.equal(hop.support[~validity[:, None, :].expand(2, 3, 5)],
                                        torch.zeros((~validity).sum().item() * 3)))
            self.assertTrue(torch.isfinite(hop.delta).all())

    def test_causal_null_gate_disables_residual_with_valid_memory(self):
        enabled = self.run_reader(null_support_override=torch.zeros(2, 3, 2))
        disabled = self.run_reader(null_support_override=torch.ones(2, 3, 2))
        self.assertGreater(enabled.residual.abs().sum().item(), 0)
        self.assertTrue(torch.equal(disabled.output, self.queries))
        self.assertTrue(torch.equal(disabled.residual, torch.zeros_like(self.queries)))
        self.assertClose(enabled.hops[0].support, disabled.hops[0].support)
        self.assertGreater(disabled.hops[0].support.sum().item(), 0)
        for hop in disabled.hops:
            self.assertTrue(torch.equal(hop.null_gate, torch.zeros(2, 3)))
            self.assertTrue(torch.equal(hop.delta, torch.zeros_like(self.queries)))
            self.assertTrue(torch.equal(hop.read_weights, torch.zeros(2, 3, 5)))
        self.assertTrue(torch.equal(disabled.hops[1].temporal_logits, torch.zeros(2, 3, 5)))
        # Disable just hop two: the total residual is exactly hop one's delta.
        override = torch.zeros(2, 3, 2)
        override[..., 1] = 1
        second_disabled = self.run_reader(null_support_override=override)
        self.assertClose(second_disabled.residual, second_disabled.hops[0].delta)
        self.assertTrue(torch.equal(second_disabled.hops[1].delta, torch.zeros_like(self.queries)))
        override[..., 0] = 1
        override[..., 1] = 0
        first_disabled = self.run_reader(null_support_override=override)
        self.assertClose(first_disabled.residual, first_disabled.hops[1].delta)
        self.assertTrue(torch.equal(first_disabled.hops[1].temporal_logits, torch.zeros(2, 3, 5)))
        # Also exercise the learned null-head path without an override.
        with torch.no_grad():
            for hop in self.reader.hops:
                hop.null_head.weight.zero_()
                hop.null_head.bias.fill_(100)
        predicted_null = self.run_reader()
        self.assertTrue(torch.equal(predicted_null.output, self.queries))
        self.assertTrue(torch.equal(predicted_null.residual, torch.zeros_like(self.queries)))
        for hop in predicted_null.hops:
            self.assertTrue(torch.equal(hop.null_support, torch.ones(2, 3)))

    def test_adapter_gradients_with_frozen_backbone_features(self):
        # Inputs model precomputed frozen features; only adapter parameters
        # receive gradients. This is an autograd check, not an optimization run.
        result = self.run_reader()
        result.residual.square().sum().backward()
        self.assertIsNone(self.queries.grad)
        self.assertIsNone(self.content.grad)
        for name, parameter in self.reader.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.reader.hops[1].relation_query.weight.grad.abs().sum().item(), 0)

    def test_rank_is_adjustable_and_default_parameter_count(self):
        adjustable = AVBindReader(dim=12, rank=7, num_sources=3)
        result = adjustable(self.queries, self.content, self.sources,
                            self.intervals, self.validity)
        self.assertEqual(result.output.shape, self.queries.shape)
        default = AVBindReader(dim=2048, rank=64, num_sources=2)
        self.assertEqual(sum(p.numel() for p in default.parameters()), 1_311_494)

    def test_invalid_input_rejected(self):
        bad_intervals = self.intervals.clone()
        bad_intervals[0, 0] = torch.tensor([5., 1.])
        with self.assertRaises(ValueError):
            self.run_reader(intervals=bad_intervals)
        with self.assertRaises(ValueError):
            self.run_reader(source_ids=torch.full_like(self.sources, 3))
        with self.assertRaises(ValueError):
            self.run_reader(validity=self.validity.float())
        with self.assertRaises(ValueError):
            self.run_reader(null_support_override=torch.full((2, 3, 2), 1.5))


if __name__ == "__main__":
    unittest.main()
