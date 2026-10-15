"""Untrained joint AV binding architecture prototype; no novelty claim.

This adapter accepts frozen-backbone features without owning a backbone or
detaching inputs. Source IDs are arbitrary categorical labels, with no modality
ordering or task-specific meaning. Every memory row remains an occurrence.
Independent sigmoid supports may select several rows, or none; there is no
softmax/argmax evidence choice and no pooled numeric timestamp.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class ReadAudit:
    """Scores [B,Q,N], gates [B,Q], and unchanged records [B,N,...].

    ``pool_weights`` only pool latent values. ``read_weights`` additionally
    include the null gate and supply the next hop's temporal context. Neither
    is a probability distribution or a replacement for occurrence addresses.
    Original records are returned by reference; callers must not edit them
    in-place while using the result. Invalid rows have zero scores/support.
    """

    content: Tensor
    source_ids: Tensor
    intervals: Tensor
    validity: Tensor
    content_logits: Tensor
    address_logits: Tensor
    temporal_logits: Tensor
    support_logits: Tensor
    support: Tensor
    pool_weights: Tensor
    read_weights: Tensor
    null_logits: Tensor
    predicted_null_support: Tensor
    null_support: Tensor
    null_gate: Tensor
    residual_gate: Tensor
    delta: Tensor


@dataclass
class ReaderOutput:
    output: Tensor  # [B,Q,D]: input queries plus residual
    residual: Tensor  # [B,Q,D]: sum of the two gated updates
    hops: Tuple[ReadAudit, ReadAudit]


class _ReadHop(nn.Module):
    def __init__(self, dim: int, rank: int, temporal_context: bool):
        super().__init__()
        self.content_query = nn.Linear(dim, rank, bias=False)
        self.content_key = nn.Linear(dim, rank, bias=False)
        self.content_value = nn.Linear(dim, rank, bias=False)
        self.address_query = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim, bias=False)
        self.null_head = nn.Linear(rank, 1)
        self.residual_head = nn.Linear(rank, 1)
        self.support_bias = nn.Parameter(torch.zeros(()))
        self.relation_query = (
            nn.Linear(rank, 3, bias=False) if temporal_context else None
        )


class AVBindReader(nn.Module):
    """Two generic successive low-rank reads over attached AV records.

    Inputs: queries [B,Q,D], memory_content [B,N,D], integer source_ids [B,N],
    original floating intervals [B,N,2] (start/end), boolean validity [B,N].
    Valid intervals must be finite and ordered. Padding may contain NaNs or
    out-of-range source IDs: it is sanitized for computation, while the audit
    preserves the original records. All tensors must share a device; content
    and queries must share a floating dtype compatible with this module.

    Semantic lookup uses content alone. A separate learned address score uses
    source embeddings and encoded (start, end, duration)/time_scale. The second
    read also scores signed-log pairwise endpoint relations, conditioned on the
    first read's non-null occurrence weights. Latent values bind content to
    each row's source/time encoding before pooling. Absolute address features
    can change under a time translation; the original metadata is never changed.

    ``null_support_override`` [B,Q,2] is an optional causal intervention in
    [0,1], replacing the learned null supports. A value of 1 disables that
    hop's residual and its temporal context exactly. All-invalid memory always
    forces null support 1, even when an override asks for 0.

    Pairwise temporal comparisons cost O(B*N*N) memory. This is an architecture
    building block, not training/evaluation code or a claim of learned accuracy.
    Default parameter count: 1,311,494 at D=2048, rank=64, num_sources=2.
    """

    def __init__(
        self, dim: int, rank: int = 64, num_sources: int = 2,
        time_scale: float = 1.0,
    ):
        super().__init__()
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
               for x in (dim, rank, num_sources)):
            raise ValueError("dim, rank and num_sources must be positive integers")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        self.dim = dim
        self.rank = rank
        self.num_sources = num_sources
        self.time_scale = time_scale
        self.source_embedding = nn.Embedding(num_sources, rank)
        self.time_encoding = nn.Linear(3, rank, bias=False)
        self.hops = nn.ModuleList([
            _ReadHop(dim, rank, temporal_context=False),
            _ReadHop(dim, rank, temporal_context=True),
        ])

    def _validate(
        self, queries: Tensor, content: Tensor, sources: Tensor,
        intervals: Tensor, validity: Tensor, override: Optional[Tensor],
    ) -> None:
        if queries.ndim != 3 or content.ndim != 3:
            raise ValueError("queries and memory_content must have three dimensions")
        b, q, d = queries.shape
        if d != self.dim or content.shape[0] != b or content.shape[2] != d:
            raise ValueError("content/query batch and feature dimensions must match")
        n = content.shape[1]
        if sources.shape != (b, n) or validity.shape != (b, n):
            raise ValueError("source_ids and validity must have shape [B,N]")
        if intervals.shape != (b, n, 2):
            raise ValueError("intervals must have shape [B,N,2]")
        if sources.dtype not in (torch.int32, torch.int64):
            raise ValueError("source_ids must be int32 or int64")
        if validity.dtype != torch.bool:
            raise ValueError("validity must be boolean")
        if not queries.is_floating_point() or content.dtype != queries.dtype:
            raise ValueError("queries/content must have the same floating dtype")
        if not intervals.is_floating_point():
            raise ValueError("intervals must be floating point")
        if any(t.device != queries.device for t in
               (content, sources, intervals, validity)):
            raise ValueError("all inputs must share a device")
        if not torch.isfinite(queries).all() or not torch.isfinite(content[validity]).all():
            raise ValueError("queries and valid content must be finite")
        if not torch.isfinite(intervals[validity]).all():
            raise ValueError("valid intervals must be finite")
        if (intervals[..., 1][validity] < intervals[..., 0][validity]).any():
            raise ValueError("valid interval end must not precede start")
        if ((sources[validity] < 0) | (sources[validity] >= self.num_sources)).any():
            raise ValueError("valid source IDs must be in the embedding vocabulary")
        if override is not None:
            if (override.shape != (b, q, 2) or override.device != queries.device
                    or not override.is_floating_point()):
                raise ValueError("null_support_override must be floating [B,Q,2] on input device")
            if not torch.isfinite(override).all() or ((override < 0) | (override > 1)).any():
                raise ValueError("null_support_override must be finite and in [0,1]")

    def forward(
        self, queries: Tensor, memory_content: Tensor, source_ids: Tensor,
        intervals: Tensor, validity: Tensor,
        null_support_override: Optional[Tensor] = None,
    ) -> ReaderOutput:
        self._validate(queries, memory_content, source_ids, intervals,
                       validity, null_support_override)
        mask = validity[:, None, :]
        has_valid = validity.any(dim=-1)[:, None]
        content = torch.where(validity[..., None], memory_content, 0)
        sources = torch.where(validity, source_ids, 0).long()
        # Casting is for latent computation only; audit addresses keep the exact
        # original dtype and numbers, including individual repeated occurrences.
        times = torch.where(validity[..., None], intervals, 0)
        times = times.to(dtype=queries.dtype) / self.time_scale
        start, end = times.unbind(dim=-1)
        time_features = torch.stack((start, end, end - start), dim=-1)
        record_features = self.source_embedding(sources) + self.time_encoding(time_features)
        record_features = torch.where(validity[..., None], record_features, 0)

        # [B,M,N,3]: candidate N versus previous occurrence M. These are
        # start difference, end difference and directed gap (N.start-M.end).
        # Compare exact endpoints BEFORE reducing relation features; never
        # compute a representative/averaged timestamp for either read.
        relations = torch.stack((
            start[:, None, :] - start[:, :, None],
            end[:, None, :] - end[:, :, None],
            start[:, None, :] - end[:, :, None],
        ), dim=-1)
        pair_valid = validity[:, :, None] & validity[:, None, :]
        relations = torch.where(pair_valid[..., None], relations, 0)
        # Encode each relation nonlinearly BEFORE pooling; this cannot be
        # reduced to comparing a candidate against an averaged timestamp.
        relations = relations.sign() * torch.log1p(relations.abs())

        residual = torch.zeros_like(queries)
        state = queries
        previous_weights = None
        audits = []
        scale = math.sqrt(self.rank)
        for index, hop in enumerate(self.hops):
            query_features = hop.content_query(state)
            address_query = hop.address_query(state)
            content_logits = torch.einsum(
                "bqr,bnr->bqn", query_features, hop.content_key(content)
            ) / scale
            address_logits = torch.einsum(
                "bqr,bnr->bqn", address_query, record_features
            ) / scale
            temporal_logits = torch.zeros_like(content_logits)
            if previous_weights is not None:
                contextual_relations = torch.einsum(
                    "bqm,bmnf->bqnf", previous_weights, relations
                )
                temporal_logits = torch.einsum(
                    "bqf,bqnf->bqn", hop.relation_query(address_query),
                    contextual_relations,
                ) / math.sqrt(3)
            content_logits = torch.where(mask, content_logits, 0)
            address_logits = torch.where(mask, address_logits, 0)
            temporal_logits = torch.where(mask, temporal_logits, 0)
            support_logits = torch.where(
                mask, content_logits + address_logits + temporal_logits + hop.support_bias, 0
            )
            support = torch.sigmoid(support_logits) * mask
            # A stable latent pooling rule, without normalizing sigmoid supports
            # into mutually exclusive evidence probabilities.
            pool_weights = support / support.sum(dim=-1, keepdim=True).clamp_min(1)
            bound_values = hop.content_value(content) + record_features
            read = torch.einsum("bqn,bnr->bqr", pool_weights, bound_values)
            null_logits = hop.null_head(query_features + read).squeeze(-1)
            predicted_null = torch.where(has_valid, torch.sigmoid(null_logits), 1)
            null_support = predicted_null
            if null_support_override is not None:
                null_support = torch.where(
                    has_valid, null_support_override[..., index].to(queries.dtype), 1
                )
            null_gate = 1 - null_support
            residual_gate = null_gate * torch.sigmoid(
                hop.residual_head(query_features + read).squeeze(-1)
            )
            delta = residual_gate[..., None] * hop.output(read)
            previous_weights = pool_weights * null_gate[..., None]
            residual = residual + delta
            state = queries + residual
            audits.append(ReadAudit(
                content=memory_content, source_ids=source_ids, intervals=intervals,
                validity=validity, content_logits=content_logits,
                address_logits=address_logits, temporal_logits=temporal_logits,
                support_logits=support_logits, support=support,
                pool_weights=pool_weights, read_weights=previous_weights,
                null_logits=null_logits, predicted_null_support=predicted_null,
                null_support=null_support, null_gate=null_gate,
                residual_gate=residual_gate, delta=delta,
            ))
        return ReaderOutput(state, residual, (audits[0], audits[1]))
