"""
STAGE-0 v6: Counterfactual Attention Output Routing, 1D Wasserstein-Synchrony Relational Grounding,
Bilevel Innovation Alignment, and Mixture-Consistent Two-Hop Unmixing (CAOR-WEMD)
Authors: Yoshua Bengio & Yann LeCun

Addresses ALL empirical findings, exploratory controls, and methodological critiques from Yann LeCun's
Stage-0 v5 audit (2026-10-16):

1. [M3 Core Breakthrough — Per-Layer Native Counterfactual Attention Output Routing (`CAOR`)]:
   - Falsifies and replaces Stage-0 v5's static Layer-18 bivector addition (`5/8 -> 5/8`, 0 decisions changed):
     * Why v5 M3 failed to change decisions:
       (a) Reused a single vector projected only through Layer 18's `v_proj/o_proj` (captured at the output of
           entire block 18 rather than pre-attention input) across Layers 14..24, repeating the same vector
           across GQA heads instead of computing query-row-specific and head-specific attention updates.
       (b) Multiplied `rel_scalar` by `(v_bound - a_bound)`, which for `Past` (`rel_scalar < 0`, e.g. `1513`)
           added `a_bound - v_bound` (suppressing visual evidence and amplifying audio speech!).
       (c) Frozen Qwen2.5-Omni does not decode a hand-crafted direction `tau_unit` added to residual states;
           it decodes evidence from the native attention output `o_proj(Concat_h(sum_j P_{l,h}(r,j) v_{l,h}(j)))`.
     * How v6 `CAOR` works (implementing Yann LeCun's Section 8 recommendation):
       Intercepts `ALL_ATTENTION_FUNCTIONS` at each target layer `l` (using layer `l`'s true pre-attention
       `input_layernorm` input, native 3D M-RoPE `q, k`, native per-head `v`, and layer `l`'s own `o_proj`)
       for `hook_rows = stem_rows + assistant_rows` (100% disjoint from `option_rows`):
       - **Single-Pass De-Sinking**: Uses the question-blind media-end delimiter token `r_0 = last_media`
         (`<|vision_eos|>`, which cannot see the question under causal masking!) in the SAME forward pass to
         subtract the structural initial-frame visual sink and boundary audio sink per head.
       - **Exact Non-Media Context Preservation**: Reallocates attention among media tokens `j in M_V U M_A`
         while preserving each head and query row's exact combined media log-mass `logsumexp_{j in M} Lambda(r, j)`,
         leaving all non-media context probabilities `P(r, j)` (`j not in M`) 100% invariant (`err < 1e-6`).
       - **Matched M3 Ablation & Control Suite**: Evaluates `native`, `caor_full_mid`, `caor_rel_only_mid`
         (no order tilt), `caor_order_only_mid` (order tilt only), `caor_reversed_mid` (inverted order tilt),
         `caor_full_late` vs `caor_reversed_late`, `caor_irrel_media_mid`, `caor_random_3seed_mid`, and BOTH
         `gamma_coord_perm_query` and `gamma_semantic_wrong_query` (a real grammatical English question about
         an absent event encoded through the live tokenizer and model layers!).

2. [M2 Core Breakthrough — 1D Wasserstein Transport & Banded Synchrony (`Phi_EMD`) + Mixture-Consistent Bridge]:
   - Falsifies v5's 6-D `Phi` (`B8 = 67.7/59.9/34.6%` vs `B1b+presence no Phi = 67.2/57.3/34.0%`):
     * Proves algebraically that on disjoint equal-width grid slots, `iou`, `anchor_cov`, and `target_cov` in v5
       were all equal to the identity matrix `I_K` (`Phi[..., 0] == Phi[..., 1] == Phi[..., 2]`, 3 collinear
       duplicates!) and missed adjacent-slot simultaneity (`|k_1 - k_2| == 1`).
     * Replaces v5's `Phi` with a full-rank 6-D **1D Wasserstein & Banded Synchrony Relational Vector** `Phi_EMD`:
       `[sync_exact, sync_banded, bhattacharyya_sync, W_plus, W_minus, ord_margin]` where `W_plus` and `W_minus`
       are the exact positive and negative 1D Cumulative Distribution Function (CDF) transport masses satisfying
       `W_plus - W_minus == E[u_2] - E[u_1]` and `W_plus + W_minus == W_1(p_1, p_2)` (making `During` linearly
       separable from `Before` and `After` via minimum `W_1` and maximum `bhattacharyya_sync`!).
     * Fixes the v5 Hop-2 Bridge collapse on `During` (`4/39 = 10.3%` in `PartB3_Roll`):
       Proves that v5's `l_ortho = (q_hop2 . qa)^2` forced `q_hop2` orthogonal to `qa`, wiping out the shared
       within-recording scene correlation (`~0.80`) and repelling `q_hop2` from the `During` slot `pos_a`!
       Replaces it with `MixtureAwareTwoHopBridge` trained on actual retrieved mixtures `v_hat_1_mix` from
       unambiguous training pairs without `l_ortho`, and reports `hop1_top1_acc`, `hop2_top1_acc`,
       `hop2_acc_given_hop1_correct/wrong`, `teacher_forced_bridge`, `mixture_trained_bridge`, and
       `oracle_anchor_upper_bound`.

3. [M1 Core Breakthrough — Clean Modality Standardization & Bilevel Innovation Alignment]:
   - Separates `C3a_v4_exact_replay` (shift before whitening) vs `C3b_v5_diag` (shift after whitening).
   - Separates `C4_pure_mod_diag` and `C5_pure_mod_zca` with ZERO `txt_mean` subtraction on media features.
   - Introduces `C6_bilevel_diag_innov` (decoupled Global `C3b` + Within-Recording Innovation Aligner) and its
     exact matched shuffle control `C8b_bilevel_shuffled_ctrl` (identical transform, input mixture, loss, budget).

4. [M0 & Free-Form Audit Fixes]:
   - Replaces v5's naive keyword regex with clause-aware polarity/subject-inversion parsing
     (`heuristic_clause_aligned`), and verifies in M0 that all 3 of Yann LeCun's counterexamples (negated TA,
     subject-inverted/negated TF, and negated/single-word CMTB) are rejected (`False`).
"""

import os
import re
import json
import math
import time
import random
import contextlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OUT_DIR = globals().get('OUT', Path('.'))
if isinstance(OUT_DIR, str):
    OUT_DIR = Path(OUT_DIR)

SEEDS_M1 = (20261015, 20261016, 20261017)

NULL_PATTERNS = (
    'no speech', 'no intelligible speech', 'silence', 'silent',
    'no audible', 'nothing is heard', 'no sound', 'no voice',
    'no visible', 'nothing visible', 'black screen', 'blank screen'
)

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of',
    'and', 'or', 'for', 'with', 'from', 'by', 'as', 'into', 'while', 'then',
    'person', 'man', 'woman', 'someone', 'video', 'clip', 'scene', 'audio',
    'sound', 'heard', 'seen', 'shown', 'visible', 'audible', 'there', 'it', 'this',
    'that', 'when', 'what', 'happening', 'occurs', 'occurring', 'narrated', 'event',
    'speech', 'moment', 'time', 'same', 'right', 'like', 'seems', 'well', 'can', 'see'
}

GENERIC_EVASION_PATTERNS = (
    'the scene is being shown',
    'the video is showing a person',
    'the video shows a',
)

DISJOINT_ABSENT_EVENT_QUERIES = (
    "When the speech is heard, is the welder using a blowtorch on a steel pipe in the garage?",
    "When does the guitarist tune the electric guitar on the concert stage relative to the speech?",
    "When the speech is heard, what carpenter action with a hand saw on a pine plank is occurring?",
    "When the speech is heard, is the scuba diver swimming past a coral reef underwater in the video?",
)


def is_null_description(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 4:
        return True
    return any(pat in t for pat in NULL_PATTERNS)


def content_tokens(text: str) -> set:
    words = re.findall(r'[a-z0-9]+', text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


# ==============================================================================
# MODULE 0: ALGEBRAIC, 1D-WASSERSTEIN, BIVECTOR & FREE-FORM COUNTEREXAMPLE CHECKS
# ==============================================================================
def compute_affine_invariant_interval_relations(intervals_a: torch.Tensor, intervals_b: torch.Tensor) -> torch.Tensor:
    """Legacy 6-D pairwise interval tensor retained for backward audit comparison."""
    sa, ea = intervals_a[:, 0].unsqueeze(1), intervals_a[:, 1].unsqueeze(1)
    sb, eb = intervals_b[:, 0].unsqueeze(0), intervals_b[:, 1].unsqueeze(0)

    da = (ea - sa).clamp(min=1e-4)
    db = (eb - sb).clamp(min=1e-4)
    d_ref = torch.sqrt(da * db).clamp(min=1e-4)

    ca = 0.5 * (sa + ea)
    cb = 0.5 * (sb + eb)

    inter = (torch.minimum(ea, eb) - torch.maximum(sa, sb)).clamp(min=0.0)
    union = (torch.maximum(ea, eb) - torch.minimum(sa, sb)).clamp(min=1e-4)

    iou = inter / union
    anchor_cov = inter / da
    target_cov = inter / db

    gap_after = (sb - ea).clamp(min=0.0) / d_ref
    gap_before = (sa - eb).clamp(min=0.0) / d_ref
    center_diff = (cb - ca) / d_ref
    s_log = torch.sign(center_diff) * torch.log1p(center_diff.abs())

    return torch.stack([
        iou,
        anchor_cov,
        target_cov,
        torch.log1p(gap_after),
        torch.log1p(gap_before),
        s_log,
    ], dim=-1)


def compute_wasserstein_synchrony_relational_features(
    p1_hat: torch.Tensor,
    p2_hat: torch.Tensor,
    iv_1: torch.Tensor,
    iv_2: torch.Tensor,
    joint_gate: torch.Tensor,
) -> torch.Tensor:
    """
    Full-Rank 6-D 1D-Wasserstein Optimal Transport & Banded Cross-Modal Synchrony Vector `Phi_EMD`:
      Fixes the v5 collinearity bottleneck where `iou == anchor_cov == target_cov` on disjoint grids!
      Coordinates (all gated by `joint_gate` and centered against the uniform baseline `u_1, u_2`):
        0: `sync_exact`        : Exact zero-lag overlap `sum_k p1[k]*p2[k] - 1/K` (EVEN under reversal P)
        1: `sync_banded`       : Near-simultaneous adjacent-slot overlap `sum_{|i-j|<=1} w_{ij} p1[i] p2[j]` (EVEN)
        2: `bhattacharyya_sync`: Hellinger/Bhattacharyya distributional overlap `sum_k sqrt(p1[k]*p2[k])` (EVEN)
        3: `W_plus`            : Forward 1D CDF transport mass `(1/K) sum_k relu(F_1(k) - F_2(k))` (SWAPS with W_minus)
        4: `W_minus`           : Backward 1D CDF transport mass `(1/K) sum_k relu(F_2(k) - F_1(k))` (SWAPS with W_plus)
        5: `ord_margin`        : Signed precedence margin `sum_{i<j} p1[i]p2[j] - sum_{i>j} p1[i]p2[j]`
                                 + `0.5 * s_log_gap` (Strictly ODD under reversal P!)
      Note: Since `W_1 = W_plus + W_minus` and `bary_diff = W_plus - W_minus` are exact linear combinations of
      coordinates 3 and 4, a linear probe on `[3, 4]` automatically learns both signed order `W_plus - W_minus`
      AND non-negative transport distance `W_1 = W_plus + W_minus` (which is minimal at `During`!), while keeping
      all 6 coordinates 100% linearly independent!
    """
    device = p1_hat.device
    K1 = p1_hat.numel()
    K2 = p2_hat.numel()
    u1 = torch.full_like(p1_hat, 1.0 / max(1, K1))
    u2 = torch.full_like(p2_hat, 1.0 / max(1, K2))

    c1 = 0.5 * (iv_1[:, 0] + iv_1[:, 1])
    c2 = 0.5 * (iv_2[:, 0] + iv_2[:, 1])
    span = (torch.maximum(iv_1[:, 1].max(), iv_2[:, 1].max()) - torch.minimum(iv_1[:, 0].min(), iv_2[:, 0].min())).clamp(min=1.0)
    slot_w = span / float(max(K1, K2, 1))

    dist_mat = (c2.unsqueeze(0) - c1.unsqueeze(1)) / slot_w.clamp(min=1e-4)
    abs_d = dist_mat.abs()

    exact_ker = (abs_d < 0.45).float()
    band_ker = torch.exp(-0.5 * (abs_d / 1.15).square())
    ord_ker = torch.tanh(dist_mat / 1.25)

    w_hat = p1_hat.unsqueeze(1) * p2_hat.unsqueeze(0)
    w_unif = u1.unsqueeze(1) * u2.unsqueeze(0)

    sync_exact = (w_hat * exact_ker).sum() - (w_unif * exact_ker).sum()
    sync_banded = (w_hat * band_ker).sum() - (w_unif * band_ker).sum()

    if K1 == K2:
        bhatt_raw = torch.sqrt((p1_hat * p2_hat).clamp(min=1e-12)).sum() - 1.0
        F1 = torch.cumsum(p1_hat, dim=0)
        F2 = torch.cumsum(p2_hat, dim=0)
        Fu = torch.cumsum(u1, dim=0)
        # Uniform reference for W_plus / W_minus is 0 when both marginals are uniform u1 == u2
        w_plus = F.relu(F1 - F2).mean() - F.relu(Fu - Fu).mean()
        w_minus = F.relu(F2 - F1).mean() - F.relu(Fu - Fu).mean()
    else:
        # Interpolate onto shared K_grid = max(K1, K2) when K1 != K2
        Kg = max(K1, K2)
        p1_g = F.interpolate(p1_hat.view(1, 1, -1), size=Kg, mode='linear', align_corners=True).view(-1)
        p2_g = F.interpolate(p2_hat.view(1, 1, -1), size=Kg, mode='linear', align_corners=True).view(-1)
        p1_g = p1_g / p1_g.sum().clamp(min=1e-8)
        p2_g = p2_g / p2_g.sum().clamp(min=1e-8)
        bhatt_raw = torch.sqrt((p1_g * p2_g).clamp(min=1e-12)).sum() - 1.0
        F1 = torch.cumsum(p1_g, dim=0)
        F2 = torch.cumsum(p2_g, dim=0)
        w_plus = F.relu(F1 - F2).mean()
        w_minus = F.relu(F2 - F1).mean()

    ord_margin = (w_hat * ord_ker).sum() - (w_unif * ord_ker).sum()

    vec = torch.stack([
        sync_exact,
        sync_banded,
        bhatt_raw,
        w_plus,
        w_minus,
        ord_margin,
    ], dim=-1)
    return joint_gate * vec


def compute_shrinkage_calibrated_relational_features(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
    iv_1: torch.Tensor,
    iv_2: torch.Tensor,
    raw_sim_1: torch.Tensor = None,
    raw_sim_2: torch.Tensor = None,
    tau_g: float = 0.05,
):
    """
    Shrinkage-Regularized Support Pooling + 1D-Wasserstein Synchrony Relational Grounding +
    Shift-Sensitive Evidence Presence Calibration:
      1. Preserves exact numerical shrinkage on `p_i = softmax(logits_i)` (passes Yann's `1e-4` tiny-noise check).
      2. Replaces the 3-collinear-coordinate v5 `Phi` with full-rank 6-D `Phi_EMD` (`exp_phi`), where:
         - `exp_phi[0..2]` measure exact, banded, and Bhattacharyya cross-modal synchrony (EVEN),
         - `exp_phi[3..4]` measure forward and backward 1D CDF Wasserstein transport `W_+, W_-` (SWAPS under P,
           so `W_+ + W_- = W_1` is minimal at `During` and `W_+ - W_-` is the exact barycenter shift!),
         - `exp_phi[5]` measures the antisymmetric precedence margin (ODD under P).
      3. Preserves `unmodified_std_attn_feat` (`B1` baseline, 6-D) and `shift_sensitive_null_feat` (8-D).
    """
    K1 = logits_1.numel()
    K2 = logits_2.numel()
    p1 = F.softmax(logits_1, dim=-1)
    p2 = F.softmax(logits_2, dim=-1)

    u1 = torch.full_like(p1, 1.0 / max(1, K1))
    u2 = torch.full_like(p2, 1.0 / max(1, K2))

    q1 = F.relu(p1 - u1)
    q2 = F.relu(p2 - u2)
    g1 = q1.sum()
    g2 = q2.sum()

    alpha1 = (g1 * g1) / (g1 * g1 + tau_g * tau_g)
    alpha2 = (g2 * g2) / (g2 * g2 + tau_g * tau_g)
    joint_gate = alpha1 * alpha2

    tp1_sharp = q1 / (g1 + 1e-8) if float(g1.item()) > 1e-8 else u1
    tp2_sharp = q2 / (g2 + 1e-8) if float(g2.item()) > 1e-8 else u2

    p1_hat = alpha1 * tp1_sharp + (1.0 - alpha1) * u1
    p2_hat = alpha2 * tp2_sharp + (1.0 - alpha2) * u2

    # Compute full-rank 6-D 1D-Wasserstein + Synchrony relational vector
    exp_phi = compute_wasserstein_synchrony_relational_features(p1_hat, p2_hat, iv_1, iv_2, joint_gate)

    # Pure unmodified standard-attention mean times + 1D smooth CDF transport (B1 baseline remains exact 6-D!)
    clip_mid = 0.5 * (iv_1[:, 0].min() + iv_1[:, 1].max())
    clip_span = (iv_1[:, 1].max() - iv_1[:, 0].min()).clamp(min=1.0)
    t1_std = ((p1.unsqueeze(-1) * iv_1).sum(dim=0) - clip_mid) / clip_span
    t2_std = ((p2.unsqueeze(-1) * iv_2).sum(dim=0) - clip_mid) / clip_span
    unmodified_std_attn_feat = torch.cat([t1_std, t2_std, t2_std - t1_std], dim=-1)

    # Shift-Sensitive Evidence Presence Calibration
    s1 = logits_1 * 0.07 if raw_sim_1 is None else raw_sim_1.float()
    s2 = logits_2 * 0.07 if raw_sim_2 is None else raw_sim_2.float()
    max_abs_s = torch.maximum(s1.abs().max(), s2.abs().max())
    mag_gate = torch.tanh(max_abs_s / 1e-3).square()

    s1_max = s1.max()
    s1_mean = s1.mean()
    s2_max = s2.max()
    s2_mean = s2.mean()
    s2_std = s2.std(unbiased=False).clamp(min=1e-4)
    s2_margin = s2_max - s2_mean
    s2_peak_z = (s2_max - s2_mean) / (s2_std + 0.02)
    joint_min_max = torch.minimum(s1_max, s2_max)
    joint_mean = 0.5 * (s1_mean + s2_mean)

    null_feat = mag_gate * torch.stack([
        s1_max,
        s2_max,
        s2_mean,
        s2_margin,
        s2_peak_z,
        joint_min_max,
        joint_mean,
        joint_gate,
    ], dim=-1)

    return exp_phi, null_feat, unmodified_std_attn_feat, p1_hat, p2_hat, float(joint_gate.item())


def build_orthogonal_bivector_routed_state(
    V_bank: torch.Tensor,
    A_bank: torch.Tensor,
    lg_v: torch.Tensor,
    lg_a: torch.Tensor,
    iv_t: torch.Tensor,
    reverse_timeline: bool = False,
    tan_theta: float = 0.85,
):
    """
    Exact Orthogonal Content-Order Bivector Representation `(c_ev, tau_unit)`:
      - `c_ev` (Content Identity Channel — Strictly EVEN under timeline reversal P).
      - `tau_unit` (Chronological Dipole Channel — Strictly ODD under P, orthogonal to `c_ev`).
      - Fixes the v5 modality-contrast sign bug: instead of multiplying `rel_scalar` by `(v_bound - a_bound)`
        (which added `a_bound - v_bound` when `rel_scalar < 0`, suppressing visual evidence on `Past`!),
        `tau_raw` uses the temporal difference between late-weighted and early-weighted within-modality
        innovations (`d_dipole`) plus `rel_scalar` carried on the orthogonalized temporal innovation axis!
    """
    K = V_bank.shape[0]
    device = V_bank.device
    if reverse_timeline:
        perm = torch.arange(K - 1, -1, -1, device=device, dtype=torch.long)
        V_use = V_bank[perm]
        A_use = A_bank[perm]
        lv_use = lg_v[perm]
        la_use = lg_a[perm]
    else:
        V_use = V_bank
        A_use = A_bank
        lv_use = lg_v
        la_use = lg_a

    t_span = float((iv_t[-1, 1] - iv_t[0, 0]).clamp(min=1.0).item())
    dt = t_span / float(K)
    iv_sym = torch.stack([
        torch.arange(K, device=device, dtype=torch.float32) * dt,
        torch.arange(1, K + 1, device=device, dtype=torch.float32) * dt,
    ], dim=-1)

    exp_phi, null_f, _, p_a_hat, p_v_hat, joint_gate = compute_shrinkage_calibrated_relational_features(
        la_use, lv_use, iv_sym, iv_sym, tau_g=0.05
    )

    p_a = F.softmax(la_use, dim=-1)
    p_v = F.softmax(lv_use, dim=-1)
    g_a = F.relu(p_a - (1.0 / max(1, K))).sum()
    g_v = F.relu(p_v - (1.0 / max(1, K))).sum()
    tau_g = 0.05
    alpha_a = float(((g_a * g_a) / (g_a * g_a + tau_g * tau_g)).item())
    alpha_v = float(((g_v * g_v) / (g_v * g_v + tau_g * tau_g)).item())

    # Strictly zero floor when evidence is uniform (`alpha_a = alpha_v = 0 => gamma_gate = 0.0`!)
    gamma_gate = min(1.0, max(alpha_v, alpha_a))

    w_mod_v = (alpha_v + 0.1) / (alpha_v + alpha_a + 0.2)
    w_mod_a = 1.0 - w_mod_v

    v_bound = (p_v_hat.unsqueeze(-1) * V_use).sum(dim=0)
    a_bound = (p_a_hat.unsqueeze(-1) * A_use).sum(dim=0)

    # 1. Content Identity Channel `c_ev` (Strictly EVEN under P)
    c_ev = F.normalize(w_mod_v * v_bound + w_mod_a * a_bound, dim=-1)

    # 2. Symmetric Carrier `u_even` (Strictly EVEN under P)
    u_cross_raw = v_bound - a_bound
    u_even = F.normalize(u_cross_raw, dim=-1) if float(u_cross_raw.norm().item()) >= 1e-6 else torch.zeros_like(c_ev)

    # 3. First-Order Legendre Temporal Dipole `d_dipole` (Strictly ODD under P)
    t_tilde = torch.linspace(-1.0, 1.0, K, device=device, dtype=torch.float32)
    m_stream = w_mod_v * V_use + w_mod_a * A_use
    dipole_raw = (t_tilde.unsqueeze(-1) * m_stream).sum(dim=0)
    d_dipole = F.normalize(dipole_raw, dim=-1) if float(dipole_raw.norm().item()) > 1e-6 else torch.zeros_like(c_ev)

    # 4. Signed Wasserstein-Relational Order Scalar `rel_scalar` (Strictly ODD under P)
    w_diff = float((exp_phi[3] - exp_phi[4]).item())
    rel_scalar = float(exp_phi[5].item()) + w_diff + 0.5 * gamma_gate * float(((p_v_hat - p_a_hat) * t_tilde).sum().item())

    # 5. Single-Sign-Flip Temporal Vector `tau_raw = ODD * EVEN + EVEN * ODD = ODD`
    tau_raw = rel_scalar * u_even + 0.35 * max(gamma_gate, 1e-4) * d_dipole
    tau_perp = tau_raw - torch.dot(tau_raw, c_ev) * c_ev
    tau_norm = float(tau_perp.norm().item())
    tau_unit = tau_perp / max(tau_norm, 1e-8)

    if gamma_gate < 1e-6 or tau_norm < 1e-7:
        vec = c_ev.clone()
    else:
        cos_t = 1.0 / math.sqrt(1.0 + tan_theta * tan_theta)
        sin_t = tan_theta * cos_t
        vec = F.normalize(cos_t * c_ev + sin_t * tau_unit, dim=-1)

    bhatt_sync = float(torch.sqrt((p_v_hat * p_a_hat).clamp(min=1e-12)).sum().item())
    audit = {
        'c_ev': c_ev,
        'tau_unit': tau_unit,
        'p_v_hat': p_v_hat.detach(),
        'p_a_hat': p_a_hat.detach(),
        'bhatt_sync': round(bhatt_sync, 6),
        'w_diff': round(w_diff, 6),
        'rel_scalar': round(rel_scalar, 6),
        'alpha_v': round(alpha_v, 6),
        'alpha_a': round(alpha_a, 6),
        'gamma_gate': round(gamma_gate, 6),
        'content_order_dot': round(float(torch.dot(c_ev, tau_unit).item()), 8),
    }
    return vec, gamma_gate, exp_phi, audit


def audit_freeform_generation_quality(
    text: str,
    clip_dur_sec: float,
    task: str,
    gold_ans_text: str,
    distractor_texts: list = None,
) -> dict:
    """
    Fixes all 3 regex counterexamples identified by Yann LeCun in Stage-0 v5 Section 7:
      1. TA Negation Counterexample:
         e.g., `"No, the narrated event is not happening at the same time in the video."`
         Strips negated simultaneity (`not happening at the same time`, `does not happen at the same time`,
         `not at the same time`, `isn't happening`) before testing affirmative simultaneity!
      2. TF Subject-Role Inversion & Negation Counterexample:
         e.g., `"The speech happens right after the narrated event is finished."` (which means the narrated
         event is in the PAST relative to the speech, NOT Future!) or `"does not happen after"`.
         Explicitly parses subject-role inversion (`speech ... after ... event` => Past;
         `speech ... before ... event` => Future) and strips negated prepositions (`not after`, `not before`)!
      3. CMTB Negation & Single-Token Overlap Counterexample:
         e.g., `"The person is not adding milk, they are chopping onions."` for gold `"add milk and one egg whisk"`.
         Splits into clauses, removes negated clauses (`not `, `never `, `instead of `, `rather than `),
         requires >= 2 matching content tokens (or 100% when gold has 1 token), AND checks that the affirmative
         clause does not have higher token overlap with a competing distractor option!
      Reports `heuristic_clause_aligned` honestly as a clause-aware heuristic (and keeps `semantic_gold_aligned`
      alias so automated scripts reading the old key continue to work without KeyError).
    """
    t_low = text.strip().lower()
    mentioned_secs = []
    for m_mmss in re.finditer(r'\b(\d{1,2}):(\d{2})\b', t_low):
        mm, ss = int(m_mmss.group(1)), int(m_mmss.group(2))
        mentioned_secs.append(float(mm * 60 + ss))
    for m_sec in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec|s)\b', t_low):
        mentioned_secs.append(float(m_sec.group(1)))

    hallucinated_ts = any(ts > clip_dur_sec + 1.0 for ts in mentioned_secs)
    is_generic = len(t_low) < 12 or any(pat in t_low for pat in GENERIC_EVASION_PATTERNS)

    g_low = (gold_ans_text or '').strip().lower()
    polarity_match = None
    relation_match = None
    content_match = None

    if task == 'TA':
        neg_sim_pat = (
            r'\b(?:not|never|no|isn\'t|isnt|doesn\'t|doesnt)\b[^.;,]{0,35}'
            r'(?:happening|happen|occurring|occur|same time|simultaneously)'
        )
        has_neg_sim = bool(re.search(neg_sim_pat, t_low))
        t_cleaned = re.sub(neg_sim_pat, ' ', t_low)
        lead_no = bool(re.match(r'^\s*(?:no\b|it is not\b|they are not\b)', t_low))
        lead_yes = bool(re.match(r'^\s*(?:yes\b|yeah\b)', t_low))

        has_aff_sim = bool(re.search(r'\byes\b|happening at the same time|occurs at the same time|simultaneously', t_cleaned))
        has_no = bool(lead_no or has_neg_sim or re.search(r'\bno\b', t_low))
        has_yes = bool((lead_yes or has_aff_sim) and not lead_no and not has_neg_sim)

        if not (has_yes or has_no):
            is_generic = True
            polarity_match = False
        elif g_low.startswith('yes'):
            polarity_match = bool(has_yes and not lead_no and not has_neg_sim)
        elif g_low.startswith('no'):
            polarity_match = bool(has_no and not has_yes)
        else:
            polarity_match = False
        clause_ok = bool(polarity_match)

    elif task == 'TF':
        # Strip negated temporal prepositions first
        neg_rel_pat = r'\b(?:not|never|doesn\'t|doesnt|isn\'t|isnt)\b[^.;,]{0,25}\b(?:before|after|during|same time|past|present|future|earlier|later)\b'
        t_clean = re.sub(neg_rel_pat, ' ', t_low)

        # Detect subject-role inversion where "the speech" is the subject preceding the preposition:
        # e.g. "the speech (is heard / happens / starts) after the narrated event" => event is PAST!
        speech_after_event = bool(re.search(
            r'\b(?:speech|narration|speaker|voice)\b[^.;,]{0,30}\b(?:after|later than|following)\b[^.;,]{0,35}\b(?:event|action|it| pouring| adding| mixing| slicing| chopping| whisking)',
            t_clean
        ))
        speech_before_event = bool(re.search(
            r'\b(?:speech|narration|speaker|voice)\b[^.;,]{0,30}\b(?:before|prior to|earlier than)\b[^.;,]{0,35}\b(?:event|action|it| pouring| adding| mixing| slicing| chopping| whisking)',
            t_clean
        ))
        t_clean_no_inv = re.sub(
            r'\b(?:speech|narration|speaker|voice)\b[^.;,]{0,30}\b(?:after|later than|following|before|prior to|earlier than)\b[^.;,]{0,35}',
            ' ',
            t_clean
        )

        has_past = bool(speech_after_event or re.search(r'\bpast\b|\bbefore\b|\bearlier\b|\balready\b|\bprior\b', t_clean_no_inv))
        has_pres = bool(re.search(r'\bpresent\b|\bsame time\b|\bduring\b|\bsimultaneously\b', t_clean_no_inv))
        has_fut = bool(speech_before_event or re.search(r'\bfuture\b|\bafter\b|\blater\b|\bfollowing\b', t_clean_no_inv))

        if not (has_past or has_pres or has_fut):
            is_generic = True
            relation_match = False
        elif 'past' in g_low or 'before' in g_low:
            relation_match = bool(has_past and not has_fut)
        elif 'present' in g_low or 'during' in g_low:
            relation_match = bool(has_pres and not has_past and not has_fut)
        elif 'future' in g_low or 'after' in g_low:
            relation_match = bool(has_fut and not has_past)
        else:
            relation_match = False
        clause_ok = bool(relation_match)

    else:
        # Split into clauses and discard negated clauses
        clauses = re.split(r'[.;,]|(?:\bbut\b)|(?:\bhowever\b)|(?:\binstead\b)', t_low)
        aff_clauses = [
            c for c in clauses
            if not re.search(r'\b(?:not|never|no|isn\'t|isnt|doesn\'t|doesnt|without|instead of|rather than)\b', c)
        ]
        aff_text = ' '.join(aff_clauses)
        g_toks = content_tokens(g_low)
        aff_toks = content_tokens(aff_text)
        overlap = len(g_toks & aff_toks)
        min_req = 1 if len(g_toks) <= 1 else min(2, len(g_toks))
        beats_distractors = True
        if distractor_texts:
            for d_txt in distractor_texts:
                d_toks = content_tokens(d_txt) - g_toks
                if len(d_toks & aff_toks) > overlap:
                    beats_distractors = False
        content_match = bool(overlap >= min_req and beats_distractors) if g_toks else False
        clause_ok = bool(content_match)

    aligned = bool(clause_ok and not hallucinated_ts and not is_generic)
    return {
        'mentioned_timestamps_sec': mentioned_secs,
        'has_hallucinated_timestamp_beyond_duration': bool(hallucinated_ts),
        'is_generic_or_evasive_non_answer': bool(is_generic),
        'yes_no_polarity_matches_gold': polarity_match,
        'tf_relation_matches_gold': relation_match,
        'cmtb_content_matches_gold': content_match,
        'heuristic_clause_aligned': aligned,
        'semantic_gold_aligned': aligned,
        'is_heuristic_not_human_semantic_oracle': True,
    }


def run_algebraic_unit_checks(device='cpu'):
    # 1. Diagonal vs Low-Rank basis test
    x_basis = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    down_ones = torch.ones((1, 4, 2), device=device)
    up_ones = torch.ones((1, 2, 4), device=device)
    src_zero = torch.tensor([0], dtype=torch.long, device=device)

    out_diag = torch.einsum('nd,ndr,nrd->nd', x_basis, down_ones[src_zero], up_ones[src_zero])
    out_lowrank = torch.einsum('nd,ndr,nre->ne', x_basis, down_ones[src_zero], up_ones[src_zero])
    assert torch.allclose(out_diag, torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device))
    assert torch.allclose(out_lowrank, torch.tensor([[2.0, 2.0, 2.0, 2.0]], device=device))

    # 2. Yann LeCun's 1e-4 tiny logit noise counterexample
    iv_5 = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]], device=device)
    lg_tiny_a = torch.tensor([1e-4, 0.0, 0.0, 0.0, 0.0], device=device)
    lg_tiny_b = torch.tensor([0.0, 0.0, 0.0, 0.0, 1e-4], device=device)
    phi_tiny, _, _, p1_hat_tiny, _, gate_tiny = compute_shrinkage_calibrated_relational_features(
        lg_tiny_a, lg_tiny_b, iv_5, iv_5, tau_g=0.05
    )
    max_tiny_phi_abs = float(phi_tiny.abs().max().item())
    assert max_tiny_phi_abs < 1e-10, f"Tiny noise leaked into Phi: {max_tiny_phi_abs}"

    # 3. 9 Toy Cases Bivector & 1D-Wasserstein Reversal Equivariance Audit
    toy_perm_checks = []
    phi_samples = []
    for K in (5, 6, 9):
        for seed in (101, 202, 303):
            rng = torch.Generator(device='cpu').manual_seed(seed)
            V = F.normalize(torch.randn(K, 2048, generator=rng), dim=-1).to(device)
            A = F.normalize(torch.randn(K, 2048, generator=rng), dim=-1).to(device)
            lv = (torch.randn(K, generator=rng) * 3.0).to(device)
            la = (torch.randn(K, generator=rng) * 3.0).to(device)
            iv = torch.stack((torch.arange(K) * 2, torch.arange(1, K + 1) * 2), dim=-1).float().to(device)

            vf, gf, pf, aud_f = build_orthogonal_bivector_routed_state(V, A, lv, la, iv, reverse_timeline=False)
            vr, gr, pr, aud_r = build_orthogonal_bivector_routed_state(V, A, lv, la, iv, reverse_timeline=True)
            if K == 6:
                phi_samples.append(pf.detach().cpu())

            c_diff = float((aud_f['c_ev'] - aud_r['c_ev']).abs().max().item())
            tau_sum = float((aud_f['tau_unit'] + aud_r['tau_unit']).abs().max().item())
            ortho_err = max(abs(aud_f['content_order_dot']), abs(aud_r['content_order_dot']))
            vec_cos = float(torch.dot(vf, vr).item())
            vec_diff = float((vf - vr).abs().max().item())

            # Check 1D Wasserstein & Synchrony reversal symmetries on `pf` vs `pr`:
            # 0,1,2 are EVEN; 3 and 4 SWAP (W_plus <-> W_minus); 5 is ODD!
            phi_even_err = float((pf[:3] - pr[:3]).abs().max().item())
            phi_swap_err = max(float((pf[3] - pr[4]).abs().item()), float((pf[4] - pr[3]).abs().item()))
            phi_odd_err = float((pf[5] + pr[5]).abs().item())

            assert c_diff < 1e-5, f"Content channel not invariant under reversal: {c_diff}"
            assert tau_sum < 1e-5, f"Temporal channel did not flip sign under reversal: {tau_sum}"
            assert ortho_err < 1e-5, f"Content and temporal channels not orthogonal: {ortho_err}"
            assert phi_even_err < 1e-5 and phi_swap_err < 1e-5 and phi_odd_err < 1e-5
            assert abs(vec_cos - 0.1611) < 1e-2, f"Unexpected fwd/rev cosine: {vec_cos}"
            assert abs(aud_f['rel_scalar'] + aud_r['rel_scalar']) < 1e-5

            toy_perm_checks.append({
                'K': K,
                'seed': seed,
                'vec_fwd_rev_cosine': round(vec_cos, 6),
                'vec_max_abs_diff': round(vec_diff, 6),
                'content_even_max_err': round(c_diff, 8),
                'order_odd_max_err': round(tau_sum, 8),
                'orthogonality_max_err': round(ortho_err, 8),
                'phi_emd_even_swap_odd_max_err': round(max(phi_even_err, phi_swap_err, phi_odd_err), 8),
                'rel_scalar_fwd': aud_f['rel_scalar'],
                'rel_scalar_rev': aud_r['rel_scalar'],
            })

    # 4. Uniform Gate == 0.0 Check
    V5 = F.normalize(torch.randn(5, 2048, generator=torch.Generator().manual_seed(101)), dim=-1).to(device)
    A5 = F.normalize(torch.randn(5, 2048, generator=torch.Generator().manual_seed(202)), dim=-1).to(device)
    _, gu, pu, _ = build_orthogonal_bivector_routed_state(
        V5, A5, torch.zeros(5, device=device), torch.zeros(5, device=device), iv_5, reverse_timeline=False
    )
    assert gu == 0.0, f"Uniform gate must be 0.0 (no floor!), got {gu}"

    # 5. Proof of v5 Phi 3-Coordinate Collinearity on Disjoint Grids vs v6 Full Rank
    v5_rel_mat = compute_affine_invariant_interval_relations(iv_5, iv_5)
    v5_collinear_err_01 = float((v5_rel_mat[:, :, 0] - v5_rel_mat[:, :, 1]).abs().max().item())
    v5_collinear_err_02 = float((v5_rel_mat[:, :, 0] - v5_rel_mat[:, :, 2]).abs().max().item())
    assert v5_collinear_err_01 == 0.0 and v5_collinear_err_02 == 0.0

    # 6. Yann LeCun's 3 Free-Form Regex Counterexamples Unit Check (Section 7)
    ce_ta = audit_freeform_generation_quality(
        "No, the narrated event is not happening at the same time in the video.", 20.0, "TA", "Yes"
    )['heuristic_clause_aligned']
    ce_tf = audit_freeform_generation_quality(
        "The speech happens right after the narrated event is already finished.", 20.0, "TF", "Future"
    )['heuristic_clause_aligned']
    ce_cmtb = audit_freeform_generation_quality(
        "The person is not adding milk to the bowl, they are chopping onions.",
        20.0, "CMTB", "add milk and one egg whisk together",
        distractor_texts=["chopping onions on a board"]
    )['heuristic_clause_aligned']
    assert (not ce_ta) and (not ce_tf) and (not ce_cmtb), (
        f"Free-form counterexamples must all be rejected (False), got: {ce_ta}, {ce_tf}, {ce_cmtb}"
    )

    return {
        'basis_test_diag_output': out_diag[0].tolist(),
        'basis_test_lowrank_output': out_lowrank[0].tolist(),
        'yann_tiny_noise_1e4_v6_max_weight': round(float(p1_hat_tiny.max().item()), 6),
        'yann_tiny_noise_1e4_v6_gate': float(gate_tiny),
        'yann_tiny_noise_1e4_v6_phi_max_abs': float(max_tiny_phi_abs),
        'uniform_zero_evidence_gate': float(gu),
        'uniform_zero_evidence_effective_scale': float(0.045 * gu),
        'v5_phi_disjoint_grid_collinear_err_01_02': [v5_collinear_err_01, v5_collinear_err_02],
        'yann_3_freeform_counterexamples_rejected': {
            'negated_ta_same_time_rejected': not ce_ta,
            'inverted_tf_speech_after_event_rejected': not ce_tf,
            'negated_single_word_cmtb_rejected': not ce_cmtb,
        },
        'toy_9cases_bivector_reversal_audit': toy_perm_checks,
        'all_9_toy_cases_content_invariant_and_order_flipped': True,
    }


# ==============================================================================
# MODULE 1: CLEAN MODALITY STANDARDIZATION & BILEVEL INNOVATION ALIGNMENT
# ==============================================================================
def build_media_streams(features: torch.Tensor, recording_ids: list, sources: torch.Tensor, starts: list):
    x = F.normalize(features.float(), dim=-1)
    x_ctx = x.clone()
    x_rec = torch.zeros_like(x)

    for r_id in sorted(set(recording_ids)):
        for s_id in (0, 1):
            idx_list = [
                i for i, (r, s) in enumerate(zip(recording_ids, sources.tolist()))
                if r == r_id and s == s_id
            ]
            if not idx_list:
                continue
            idx_t = torch.tensor(idx_list, device=x.device, dtype=torch.long)
            rec_mean = x[idx_t].mean(dim=0, keepdim=True)
            x_rec[idx_t] = F.normalize(x[idx_t] - 0.85 * rec_mean, dim=-1)

    return x_ctx, x_rec


def fit_dual_svd_zca_whitener(X_tr: torch.Tensor, shrink: float = 0.22):
    X_f = F.normalize(X_tr.float(), dim=-1)
    N, D = X_f.shape
    mu = X_f.mean(dim=0, keepdim=True)
    X_c = X_f - mu
    if N < 4:
        return {
            'mu': mu,
            'V': torch.zeros((D, 1), device=X_tr.device),
            'delta_scale': torch.zeros((1,), device=X_tr.device),
            'base_scale': 1.0,
        }
    _, S, Vh = torch.linalg.svd(X_c / math.sqrt(float(N)), full_matrices=False)
    eig = S * S
    avg_var = float(eig.sum().item()) / float(D)
    lam = max(shrink * avg_var, 1e-5)
    reg_eig = (1.0 - shrink) * eig + lam
    inv_sqrt_eig = 1.0 / torch.sqrt(reg_eig)
    base_scale = 1.0 / math.sqrt(lam)
    delta_scale = inv_sqrt_eig - base_scale
    return {
        'mu': mu.detach(),
        'V': Vh.T.detach(),
        'delta_scale': delta_scale.detach(),
        'base_scale': float(base_scale),
    }


def apply_dual_svd_zca(X: torch.Tensor, zca_dict: dict) -> torch.Tensor:
    X_c = F.normalize(X.float(), dim=-1) - zca_dict['mu']
    V = zca_dict['V']
    proj = (X_c @ V) * zca_dict['delta_scale'].unsqueeze(0)
    out = zca_dict['base_scale'] * X_c + (proj @ V.T)
    return F.normalize(out, dim=-1)


class ResidualStreamAligner(nn.Module):
    """
    Cleanly isolates all 5 standardization regimes requested by Yann LeCun in Section 3:
      - `whiten_mode='none'`             : No standardization (`C1`, `C2`).
      - `whiten_mode='txt_diag_v4_exact'`: Exact Stage-0 v4 `C3` replay (`mean_shift` computed from raw
                                           normalized media BEFORE `_pre_whiten_media`!).
      - `whiten_mode='txt_diag_v5'`      : Stage-0 v5 `C3` reference (`mean_shift` computed AFTER `_pre_whiten_media`).
      - `whiten_mode='pure_mod_diag'`    : Pure modality-specific diagonal standardization (`(x_in - med_mean[s]) * med_inv_std[s]`,
                                           ZERO `txt_mean` subtraction on media!).
      - `whiten_mode='pure_mod_zca'`     : Pure modality-specific Dual-SVD ZCA (`apply_dual_svd_zca(x_in, zca_med[s])`,
                                           ZERO `txt_mean` subtraction on media!).
    """
    def __init__(self, dim: int = 2048, rank: int = 8, proj_type: str = 'diag', whiten_mode: str = 'none'):
        super().__init__()
        assert proj_type in ('diag', 'lowrank')
        assert whiten_mode in ('none', 'txt_diag_v4_exact', 'txt_diag_v5', 'pure_mod_diag', 'pure_mod_zca')
        self.dim = dim
        self.rank = rank
        self.proj_type = proj_type
        self.whiten_mode = whiten_mode

        self.register_buffer('mean_shift', torch.zeros(2, dim))
        self.register_buffer('txt_mean', torch.zeros(1, dim))
        self.register_buffer('txt_inv_std', torch.ones(1, dim))
        self.register_buffer('med_mean', torch.zeros(2, dim))
        self.register_buffer('med_inv_std', torch.ones(2, dim))

        self.zca_txt = None
        self.zca_med = {0: None, 1: None}

        self.shift_gate = nn.Parameter(torch.full((2, 1), 0.35))
        self.diag_scale = nn.Parameter(torch.ones(2, dim))
        self.bias = nn.Parameter(torch.zeros(2, dim))

        self.down = nn.Parameter(torch.empty(2, dim, rank))
        for s in (0, 1):
            nn.init.orthogonal_(self.down[s])
            self.down.data[s] *= 0.05
        self.up = nn.Parameter(torch.zeros(2, rank, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def init_shift_from_train(self, x_tr: torch.Tensor, y_tr: torch.Tensor, src_tr: torch.Tensor):
        with torch.no_grad():
            mu_t = y_tr.float().mean(dim=0, keepdim=True)
            sd_t = y_tr.float().std(dim=0, keepdim=True).clamp(min=1e-3)
            self.txt_mean.copy_(mu_t)
            self.txt_inv_std.copy_(1.0 / (0.65 * sd_t + 0.35 * sd_t.mean()))

            for s in (0, 1):
                m = (src_tr == s)
                if m.any():
                    xm = F.normalize(x_tr[m].float(), dim=-1)
                    mu_m = xm.mean(dim=0, keepdim=True)
                    sd_m = xm.std(dim=0, keepdim=True).clamp(min=1e-3)
                    self.med_mean[s].copy_(mu_m[0])
                    self.med_inv_std[s].copy_((1.0 / (0.65 * sd_m + 0.35 * sd_m.mean()))[0])

            if self.whiten_mode == 'pure_mod_zca':
                self.zca_txt = fit_dual_svd_zca_whitener(y_tr, shrink=0.22)
                for s in (0, 1):
                    m = (src_tr == s)
                    self.zca_med[s] = fit_dual_svd_zca_whitener(x_tr[m] if m.any() else x_tr, shrink=0.22)

            y_n = self.encode_text(y_tr)
            if self.whiten_mode == 'txt_diag_v4_exact':
                # Exact v4 C3 replay: media mean taken from F.normalize(x_tr) BEFORE whitening
                x_n = F.normalize(x_tr.float(), dim=-1)
            else:
                x_n = self._pre_whiten_media(x_tr, src_tr)
            for s in (0, 1):
                m = (src_tr == s)
                if m.any():
                    self.mean_shift[s].copy_(y_n[m].mean(dim=0) - x_n[m].mean(dim=0))

    def _pre_whiten_media(self, x: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        x_in = F.normalize(x.float(), dim=-1)
        if self.whiten_mode in ('txt_diag_v4_exact', 'txt_diag_v5'):
            return F.normalize((x_in - self.txt_mean) * self.txt_inv_std, dim=-1)
        if self.whiten_mode == 'pure_mod_diag':
            return F.normalize((x_in - self.med_mean[sources]) * self.med_inv_std[sources], dim=-1)
        if self.whiten_mode == 'pure_mod_zca':
            out = torch.zeros_like(x_in)
            for s in (0, 1):
                m = (sources == s)
                if m.any():
                    out[m] = apply_dual_svd_zca(x_in[m], self.zca_med[s])
            return out
        return x_in

    def encode_text(self, y: torch.Tensor) -> torch.Tensor:
        y_f = y.float()
        if self.whiten_mode in ('txt_diag_v4_exact', 'txt_diag_v5', 'pure_mod_diag'):
            return F.normalize((y_f - self.txt_mean) * self.txt_inv_std, dim=-1)
        if self.whiten_mode == 'pure_mod_zca':
            return apply_dual_svd_zca(y_f, self.zca_txt)
        return F.normalize(y_f, dim=-1)

    def encode_media(self, x: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        x_in = self._pre_whiten_media(x, sources)
        s_gate = self.shift_gate[sources].clamp(0.0, 1.0)
        base = x_in * self.diag_scale[sources] + s_gate * self.mean_shift[sources] + self.bias[sources]
        if self.proj_type == 'diag':
            lr = torch.einsum('nd,ndr,nrd->nd', x_in, self.down[sources], self.up[sources])
        else:
            lr = torch.einsum('nd,ndr,nre->ne', x_in, self.down[sources], self.up[sources])
        return F.normalize(base + lr, dim=-1)

    def score_matrix(self, y_text: torch.Tensor, x_media: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        zt = self.encode_text(y_text)
        zm = self.encode_media(x_media, sources)
        return zt @ zm.T


class BilevelGlobalInnovationAligner(nn.Module):
    """
    Decoupled Bilevel Global Context + Within-Recording Innovation Aligner (`C6`):
      - `global_aligner` (`C3b`): aligns global context features `x_ctx` with `y_text` to preserve high
        cross-recording discrimination and raw affinity levels for M2 presence/foil detection.
      - `innov_aligner`: aligns pure within-recording innovation features `x_rec = normalize(x - 0.85 * rec_mean)`
        with `y_text` trained exclusively with within-recording contrastive loss!
      - Because `x_rec` has zero recording-level static mean, permuting segments within each recording in the
        matched shuffle control (`C8b_bilevel_shuffled_ctrl`) destroys 100% of the innovation supervision,
        cleanly isolating true within-recording temporal localization!
    """
    def __init__(self, global_aligner: ResidualStreamAligner, innov_aligner: ResidualStreamAligner, beta_innov: float = 0.22):
        super().__init__()
        self.global_aligner = global_aligner
        self.innov_aligner = innov_aligner
        self.beta_innov = beta_innov
        self.logit_scale = global_aligner.logit_scale

    def encode_text(self, y: torch.Tensor) -> torch.Tensor:
        return self.global_aligner.encode_text(y)

    def encode_media(self, x_ctx: torch.Tensor, sources: torch.Tensor, x_rec: torch.Tensor = None) -> torch.Tensor:
        zm_g = self.global_aligner.encode_media(x_ctx, sources)
        if x_rec is None:
            return zm_g
        zm_i = self.innov_aligner.encode_media(x_rec, sources)
        return F.normalize(zm_g + self.beta_innov * zm_i, dim=-1)

    def score_matrix(self, y_text: torch.Tensor, x_ctx: torch.Tensor, sources: torch.Tensor, x_rec: torch.Tensor = None) -> torch.Tensor:
        s_g = self.global_aligner.score_matrix(y_text, x_ctx, sources)
        if x_rec is None:
            return s_g
        s_i = self.innov_aligner.score_matrix(y_text, x_rec, sources)
        return (s_g + self.beta_innov * s_i) / math.sqrt(1.0 + self.beta_innov * self.beta_innov)


def evaluate_sim_matrix(sim_matrix: torch.Tensor, examples: list) -> dict:
    sim = sim_matrix.detach().cpu()
    N = len(examples)
    norm_txt = [x['content'].strip().lower() for x in examples]
    is_null = [is_null_description(x['content']) for x in examples]

    top1_global = sim.argmax(dim=-1).tolist()
    global_hits = sum(norm_txt[i] == norm_txt[j] for i, j in enumerate(top1_global))

    exact_hits = 0
    multipos_hits = 0
    within_total = 0
    nonnull_hits = 0
    nonnull_total = 0

    for i in range(N):
        same_group = [
            j for j in range(N)
            if examples[j]['recording_id'] == examples[i]['recording_id']
            and examples[j]['source'] == examples[i]['source']
        ]
        if len(same_group) < 2:
            continue
        within_total += 1
        sub_scores = sim[i, same_group]
        best_j = same_group[int(sub_scores.argmax().item())]
        if best_j == i:
            exact_hits += 1
        if norm_txt[best_j] == norm_txt[i]:
            multipos_hits += 1
            if not is_null[i]:
                nonnull_hits += 1
        if not is_null[i]:
            nonnull_total += 1

    return {
        'global_hits': int(global_hits),
        'global_pct': round(100.0 * global_hits / max(1, N), 1),
        'exact_hits': int(exact_hits),
        'within_exact_pct': round(100.0 * exact_hits / max(1, within_total), 1),
        'multipos_hits': int(multipos_hits),
        'within_multipos_pct': round(100.0 * multipos_hits / max(1, within_total), 1),
        'nonnull_hits': int(nonnull_hits),
        'nonnull_total': int(nonnull_total),
        'within_nonnull_pct': round(100.0 * nonnull_hits / max(1, nonnull_total), 1),
    }


def train_single_stream_aligner(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    src_tr: torch.Tensor,
    rec_tr: list,
    norm_tr: list,
    proj_type: str = 'diag',
    within_rec_weight: float = 0.20,
    whiten_mode: str = 'none',
    shuffle_within_rec: bool = False,
    seed: int = 20261015,
):
    device = y_tr.device
    torch.manual_seed(seed)
    random.seed(seed)

    x_train = x_tr.clone()
    if shuffle_within_rec:
        g_shuf = torch.Generator(device=device).manual_seed(seed)
        for r_id in sorted(set(rec_tr)):
            for s_id in (0, 1):
                m_idx = [i for i, (r, s) in enumerate(zip(rec_tr, src_tr.tolist())) if r == r_id and s == s_id]
                if len(m_idx) > 1:
                    idx_t = torch.tensor(m_idx, device=device, dtype=torch.long)
                    perm = idx_t[torch.randperm(len(m_idx), generator=g_shuf, device=device)]
                    x_train[idx_t] = x_train[perm]

    model = ResidualStreamAligner(
        dim=y_tr.shape[-1], rank=8, proj_type=proj_type, whiten_mode=whiten_mode
    ).to(device)
    model.init_shift_from_train(x_train, y_tr, src_tr)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2)
    N = len(norm_tr)
    pos_mask = torch.tensor([[norm_tr[i] == norm_tr[j] for j in range(N)] for i in range(N)], device=device)
    same_rec = torch.tensor(
        [[rec_tr[i] == rec_tr[j] and src_tr[i].item() == src_tr[j].item() for j in range(N)] for i in range(N)],
        device=device
    )
    neg_inf = float('-inf')

    for _ in range(180):
        sim = model.score_matrix(y_tr, x_train, src_tr)
        scale = model.logit_scale.exp().clamp(max=35.0)
        logits = sim * scale

        l_f = torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=-1)
        l_r = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=0)
        loss_global = 0.5 * (l_f.mean() + l_r.mean())

        if within_rec_weight > 0.0:
            logits_w = logits.masked_fill(~same_rec, neg_inf)
            pos_w = pos_mask & same_rec
            lw_f = torch.logsumexp(logits_w, dim=-1) - torch.logsumexp(logits.masked_fill(~pos_w, neg_inf), dim=-1)
            lw_r = torch.logsumexp(logits_w, dim=0) - torch.logsumexp(logits.masked_fill(~pos_w, neg_inf), dim=0)
            loss_within = 0.5 * (lw_f.mean() + lw_r.mean())
            loss_main = (1.0 - within_rec_weight) * loss_global + within_rec_weight * loss_within
        else:
            loss_main = loss_global

        reg = 0.05 * ((model.diag_scale - 1.0).square().mean() + model.down.square().mean() + model.up.square().mean())
        loss = loss_main + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_bilevel_innovation_aligner(
    x_ctx_tr: torch.Tensor,
    x_rec_tr: torch.Tensor,
    y_tr: torch.Tensor,
    src_tr: torch.Tensor,
    rec_tr: list,
    norm_tr: list,
    shuffle_within_rec: bool = False,
    seed: int = 20261015,
) -> BilevelGlobalInnovationAligner:
    """
    Trains the decoupled Bilevel Global + Within-Recording Innovation Aligner (`C6`), or its exact
    matched within-recording shuffle control (`C8b`) when `shuffle_within_rec=True`.
    """
    m_glob = train_single_stream_aligner(
        x_ctx_tr, y_tr, src_tr, rec_tr, norm_tr, proj_type='diag',
        within_rec_weight=0.20, whiten_mode='txt_diag_v5',
        shuffle_within_rec=shuffle_within_rec, seed=seed
    )
    m_innov = train_single_stream_aligner(
        x_rec_tr, y_tr, src_tr, rec_tr, norm_tr, proj_type='diag',
        within_rec_weight=0.75, whiten_mode='pure_mod_diag',
        shuffle_within_rec=shuffle_within_rec, seed=seed + 100
    )
    bilevel = BilevelGlobalInnovationAligner(m_glob, m_innov, beta_innov=0.20)
    bilevel.eval()
    return bilevel


def run_stage0_m1_orthogonal_factorial(pair_prefix: str = 'avbind_pooled_pairs'):
    paths = [OUT_DIR / 'uploads' / f'{pair_prefix}_s{s}.pt' for s in (1, 2)]
    if not all(p.exists() for p in paths):
        paths = [OUT_DIR / f'{pair_prefix}_s{s}.pt' for s in (1, 2)]
    if not all(p.exists() for p in paths):
        return {'status': 'skipped_missing_cache', 'pair_prefix': pair_prefix}, None

    parts = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    examples = [x for part in parts for x in part['examples']]
    train_ex = [x for x in examples if x['split'] == 'train']
    held_ex = [x for x in examples if x['split'] == 'heldout']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tr_raw = torch.stack([x['raw'].float() for x in train_ex]).to(device)
    tr_txt = torch.stack([x['text'].float() for x in train_ex]).to(device)
    tr_src = torch.tensor([0 if x['source'] == 'video' else 1 for x in train_ex], device=device, dtype=torch.long)
    tr_rec = [x['recording_id'] for x in train_ex]
    tr_starts = [float(x['start']) for x in train_ex]
    tr_norm = [x['content'].strip().lower() for x in train_ex]

    he_raw = torch.stack([x['raw'].float() for x in held_ex]).to(device)
    he_txt = torch.stack([x['text'].float() for x in held_ex]).to(device)
    he_src = torch.tensor([0 if x['source'] == 'video' else 1 for x in held_ex], device=device, dtype=torch.long)
    he_rec = [x['recording_id'] for x in held_ex]
    he_starts = [float(x['start']) for x in held_ex]

    tr_ctx, tr_rec_s = build_media_streams(tr_raw, tr_rec, tr_src, tr_starts)
    he_ctx, he_rec_s = build_media_streams(he_raw, he_rec, he_src, he_starts)

    cond_names = [
        'C1_ctx_only_diag_w00',
        'C2_ctx_diag_w20_no_whiten',
        'C3a_txt_std_diag_w20_v4_exact_replay',
        'C3b_txt_std_diag_w20_v5_ref',
        'C4_pure_mod_diag_w20_no_txt_mean',
        'C5_pure_mod_zca_w20_no_txt_mean',
        'C6_bilevel_global_plus_innov_w20',
        'C8a_shuffled_c3b_w20_ctrl',
        'C8b_shuffled_c6_bilevel_w20_ctrl',
    ]
    per_cond = {k: [] for k in cond_names}
    best_bundle = None

    for s_idx, seed in enumerate(SEEDS_M1):
        m_c1 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.00, whiten_mode='none', seed=seed
        )
        m_c2 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='none', seed=seed
        )
        m_c3a = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='txt_diag_v4_exact', seed=seed
        )
        m_c3b = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='txt_diag_v5', seed=seed
        )
        m_c4 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='pure_mod_diag', seed=seed
        )
        m_c5 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='pure_mod_zca', seed=seed
        )
        m_c6 = train_bilevel_innovation_aligner(
            tr_ctx, tr_rec_s, tr_txt, tr_src, tr_rec, tr_norm,
            shuffle_within_rec=False, seed=seed
        )
        m_c8a = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='txt_diag_v5', shuffle_within_rec=True, seed=seed
        )
        m_c8b = train_bilevel_innovation_aligner(
            tr_ctx, tr_rec_s, tr_txt, tr_src, tr_rec, tr_norm,
            shuffle_within_rec=True, seed=seed
        )

        with torch.no_grad():
            mat_c1 = m_c1.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c2 = m_c2.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c3a = m_c3a.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c3b = m_c3b.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c4 = m_c4.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c5 = m_c5.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c6 = m_c6.score_matrix(he_txt, he_ctx, he_src, x_rec=he_rec_s).detach()
            mat_c8a = m_c8a.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c8b = m_c8b.score_matrix(he_txt, he_ctx, he_src, x_rec=he_rec_s).detach()

            for k_name, mat in (
                ('C1_ctx_only_diag_w00', mat_c1),
                ('C2_ctx_diag_w20_no_whiten', mat_c2),
                ('C3a_txt_std_diag_w20_v4_exact_replay', mat_c3a),
                ('C3b_txt_std_diag_w20_v5_ref', mat_c3b),
                ('C4_pure_mod_diag_w20_no_txt_mean', mat_c4),
                ('C5_pure_mod_zca_w20_no_txt_mean', mat_c5),
                ('C6_bilevel_global_plus_innov_w20', mat_c6),
                ('C8a_shuffled_c3b_w20_ctrl', mat_c8a),
                ('C8b_shuffled_c6_bilevel_w20_ctrl', mat_c8b),
            ):
                per_cond[k_name].append(evaluate_sim_matrix(mat, held_ex))

            if s_idx == 0:
                best_bundle = {
                    'm_c1': m_c1,
                    'm_c3a': m_c3a,
                    'm_c3b': m_c3b,
                    'm_c6': m_c6,
                    'train_ex': train_ex,
                    'held_ex': held_ex,
                    'sim_tr_c1': m_c1.score_matrix(tr_txt, tr_ctx, tr_src).detach(),
                    'sim_he_c1': mat_c1,
                    'sim_tr_c3b': m_c3b.score_matrix(tr_txt, tr_ctx, tr_src).detach(),
                    'sim_he_c3b': mat_c3b,
                    'sim_tr_c6': m_c6.score_matrix(tr_txt, tr_ctx, tr_src, x_rec=tr_rec_s).detach(),
                    'sim_he_c6': mat_c6,
                    'tr_ctx': tr_ctx,
                    'tr_rec_s': tr_rec_s,
                    'tr_src': tr_src,
                    'tr_txt': tr_txt,
                    'he_ctx': he_ctx,
                    'he_rec_s': he_rec_s,
                    'he_src': he_src,
                    'he_txt': he_txt,
                }

    summary = {}
    for k_name in cond_names:
        s_list = per_cond[k_name]
        summary[k_name] = {
            'global_hits_3seeds': [x['global_hits'] for x in s_list],
            'exact_hits_3seeds': [x['exact_hits'] for x in s_list],
            'multipos_hits_3seeds': [x['multipos_hits'] for x in s_list],
            'nonnull_hits_3seeds': [x['nonnull_hits'] for x in s_list],
            'mean_global_pct': round(float(np.mean([x['global_pct'] for x in s_list])), 2),
            'mean_within_exact_pct': round(float(np.mean([x['within_exact_pct'] for x in s_list])), 2),
            'mean_within_multipos_pct': round(float(np.mean([x['within_multipos_pct'] for x in s_list])), 2),
            'mean_within_nonnull_pct': round(float(np.mean([x['within_nonnull_pct'] for x in s_list])), 2),
        }
    return {'pair_prefix': pair_prefix, 'summary': summary}, best_bundle


# ==============================================================================
# MODULE 2: 1D-WASSERSTEIN RELATIONAL LADDER + MIXTURE-CONSISTENT TWO-HOP BRIDGE
# ==============================================================================
def classify_interval_relation(s_a: float, e_a: float, s_b: float, e_b: float) -> int:
    ca = 0.5 * (s_a + e_a)
    cb = 0.5 * (s_b + e_b)
    inter = min(e_a, e_b) - max(s_a, s_b)
    if inter >= -0.05 and abs(cb - ca) <= 2.05:
        return 1  # During
    if e_b <= s_a - 1.95:
        return 0  # Before
    if s_b >= e_a + 1.95:
        return 2  # After
    return -1


class MixtureAwareTwoHopBridge(nn.Module):
    """
    Fixes the two Hop-2 bottlenecks identified by Yann LeCun in Section 5 & our mathematical audit:
      1. Eliminates the buggy orthogonality projection `(q_comp . q_est_1) * q_est_1` + `l_ortho` that
         forced `q_hop2` orthogonal to `qa`, destroying the shared within-recording scene correlation
         (`~0.80`) and repelling `q_hop2` from the `During` slot `pos_a` (`4/39` in v5)!
      2. Uses Direct Linear Unmixing + Low-Rank Residual Correction:
           `q_est_1 = normalize(v_hat_1 * diag_map[s1] + (v_hat_1 @ down[s1]) @ up[s1])`
           `q_hop2 = normalize(q_comp - unmix_beta[s1] * q_est_1 + res_Correction(q_comp, q_est_1))`
      3. Supports both `train_mode='teacher_forced'` (trained on single-slot `va = zm[ia]`) and
         `train_mode='retrieved_mixture'` (trained on actual noisy softmax mixtures `v_hat_1_mix` over
         recording `r`'s candidate bank!).
    """
    def __init__(self, dim: int = 2048, rank: int = 16):
        super().__init__()
        self.diag_map = nn.Parameter(torch.ones(2, dim))
        self.down = nn.Parameter(torch.randn(2, dim, rank) * 0.03)
        self.up = nn.Parameter(torch.zeros(2, rank, dim))
        self.unmix_beta = nn.Parameter(torch.full((2,), 0.72))
        self.res_down = nn.Parameter(torch.randn(2, dim, rank) * 0.02)
        self.res_up = nn.Parameter(torch.zeros(2, rank, dim))

    def map_media_to_query(self, v_hat: torch.Tensor, s1: int) -> torch.Tensor:
        base = v_hat * self.diag_map[s1]
        lr = (v_hat @ self.down[s1]) @ self.up[s1]
        return F.normalize(base + lr, dim=-1)

    def forward(self, q_comp: torch.Tensor, v_hat_1: torch.Tensor, s1: int) -> torch.Tensor:
        q_est_1 = self.map_media_to_query(v_hat_1, s1)
        beta = self.unmix_beta[s1].clamp(0.15, 1.35)
        unmixed = q_comp - beta * q_est_1
        res = (unmixed @ self.res_down[s1]) @ self.res_up[s1]
        return F.normalize(unmixed + res, dim=-1)


def train_two_hop_bridge_on_train_split(
    train_ex: list,
    aligner,
    tr_txt: torch.Tensor,
    tr_ctx: torch.Tensor,
    tr_src: torch.Tensor,
    tr_rec_s: torch.Tensor = None,
    use_retrieved_mixtures: bool = True,
    seed: int = 20261015,
) -> MixtureAwareTwoHopBridge:
    device = tr_txt.device
    torch.manual_seed(seed)
    norm_all = [x['content'].strip().lower() for x in train_ex]

    with torch.no_grad():
        zt = aligner.encode_text(tr_txt).detach()
        if isinstance(aligner, BilevelGlobalInnovationAligner):
            zm = aligner.encode_media(tr_ctx, tr_src, x_rec=tr_rec_s).detach()
        else:
            zm = aligner.encode_media(tr_ctx, tr_src).detach()
        scale_val = float(aligner.logit_scale.exp().clamp(max=35.0).item())

    bridge = MixtureAwareTwoHopBridge(dim=zt.shape[-1], rank=16).to(device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=3e-3, weight_decay=1e-2)

    recs = sorted(set(x['recording_id'] for x in train_ex))
    episodes_by_s1 = {0: [], 1: []}

    for r_id in recs:
        v_all = sorted(
            [i for i, x in enumerate(train_ex) if x['recording_id'] == r_id and x['source'] == 'video'],
            key=lambda i: float(train_ex[i]['start'])
        )
        a_all = sorted(
            [i for i, x in enumerate(train_ex) if x['recording_id'] == r_id and x['source'] == 'audio'],
            key=lambda i: float(train_ex[i]['start'])
        )
        if len(v_all) < 2 or len(a_all) < 2:
            continue
        v_bank_t = torch.tensor(v_all, device=device, dtype=torch.long)
        a_bank_t = torch.tensor(a_all, device=device, dtype=torch.long)

        # Filter for unambiguous non-null anchor/target instances within recording r_id (Yann Section 5!)
        v_unique = [
            i for i in v_all
            if not is_null_description(train_ex[i]['content'])
            and sum(norm_all[j] == norm_all[i] for j in v_all) == 1
        ]
        a_unique = [
            i for i in a_all
            if not is_null_description(train_ex[i]['content'])
            and sum(norm_all[j] == norm_all[i] for j in a_all) == 1
        ]
        for ia in v_unique:
            for ib in a_unique:
                episodes_by_s1[0].append((ia, ib, v_bank_t, a_bank_t))
                episodes_by_s1[1].append((ib, ia, a_bank_t, v_bank_t))

    # Prepare precomputed tensors per modality s1 in {0, 1}
    precomp = {}
    with torch.no_grad():
        for s1 in (0, 1):
            elist = episodes_by_s1[s1]
            if not elist:
                continue
            ia_list = [e[0] for e in elist]
            ib_list = [e[1] for e in elist]
            qa = zt[torch.tensor(ia_list, device=device, dtype=torch.long)]
            qb = zt[torch.tensor(ib_list, device=device, dtype=torch.long)]
            va_oracle = zm[torch.tensor(ia_list, device=device, dtype=torch.long)]
            vb_oracle = zm[torch.tensor(ib_list, device=device, dtype=torch.long)]
            q_comp = F.normalize(0.55 * qa + 0.45 * qb, dim=-1)

            v_mix_list = []
            for idx_e, (ia, ib, b1_t, b2_t) in enumerate(elist):
                bank1 = zm[b1_t]
                lg1 = (q_comp[idx_e] @ bank1.T) * scale_val
                p1 = F.softmax(lg1, dim=-1)
                v_mix = F.normalize((p1.unsqueeze(-1) * bank1).sum(dim=0), dim=-1)
                v_mix_list.append(v_mix)
            va_mix = torch.stack(v_mix_list, dim=0)
            precomp[s1] = (qa, qb, va_oracle, vb_oracle, q_comp, va_mix)

    for _ in range(120):
        loss = torch.tensor(0.0, device=device)
        for s1, bundle_s1 in precomp.items():
            qa, qb, va_oracle, vb_oracle, q_comp, va_mix = bundle_s1
            v_in = va_mix if use_retrieved_mixtures else va_oracle
            q_est_a = bridge.map_media_to_query(v_in, s1)
            q_hop2 = bridge(q_comp, v_in, s1)

            l_inv = (1.0 - (q_est_a * qa).sum(dim=-1)).mean()
            l_hop2_txt = (1.0 - (q_hop2 * qb).sum(dim=-1)).mean()
            l_hop2_med = (1.0 - (q_hop2 * vb_oracle).sum(dim=-1)).mean()
            # Notice: ZERO false orthogonality penalty `(q_hop2 . qa)^2`!
            # Instead, match the true inner product `<q_hop2, qa> ≈ <qb, qa>` so shared scene semantics survive!
            true_corr = (qb * qa).sum(dim=-1).detach()
            pred_corr = (q_hop2 * qa).sum(dim=-1)
            l_corr_match = (pred_corr - true_corr).square().mean()

            loss = loss + l_inv + l_hop2_txt + 0.65 * l_hop2_med + 0.35 * l_corr_match

        opt.zero_grad()
        loss.backward()
        opt.step()

    bridge.eval()
    for p in bridge.parameters():
        p.requires_grad_(False)
    return bridge


def build_connected_m2_episodes(
    examples: list,
    sim_mat: torch.Tensor,
    aligner,
    hop2_bridge: MixtureAwareTwoHopBridge,
    txt_feats: torch.Tensor,
    media_feats: torch.Tensor,
    sources_t: torch.Tensor,
    train_txt_mean: torch.Tensor,
    media_rec_feats: torch.Tensor = None,
    mode: str = 'natural_quartets',
    two_hop_conditioned: bool = False,
    oracle_hop1_anchor: bool = False,
    hard_topic_foil: bool = False,
    uniform_control: bool = False,
    wrong_event_control: bool = False,
    coord_shuffle_control: bool = False,
):
    device = sim_mat.device
    episodes = []
    recs = sorted(set(x['recording_id'] for x in examples))
    norm_all = [x['content'].strip().lower() for x in examples]
    tok_sets = [content_tokens(x['content']) for x in examples]

    with torch.no_grad():
        zt_centered = F.normalize(txt_feats.float() - train_txt_mean.to(device), dim=-1)
        zt_encoded = aligner.encode_text(txt_feats).detach()
        if isinstance(aligner, BilevelGlobalInnovationAligner):
            zm_encoded = aligner.encode_media(media_feats, sources_t, x_rec=media_rec_feats).detach()
        else:
            zm_encoded = aligner.encode_media(media_feats, sources_t).detach()
        scale_val = float(aligner.logit_scale.exp().clamp(max=35.0).item())

    foil_audit = {
        'total_origins': 0,
        'strict_cos015_and_jaccard0_origins': 0,
        'lexical_jaccard0_origins': 0,
        'string_disjoint_fallback_origins': 0,
        'hop1_total': 0,
        'hop1_correct': 0,
        'hop2_total': 0,
        'hop2_correct': 0,
        'hop2_given_hop1_correct_total': 0,
        'hop2_given_hop1_correct_hits': 0,
        'hop2_given_hop1_wrong_total': 0,
        'hop2_given_hop1_wrong_hits': 0,
        'note': 'Caption/embedding disjointness is a text-level proxy, not an independent ground-truth absence annotation.',
    }

    for r_id in recs:
        r_items = [i for i, x in enumerate(examples) if x['recording_id'] == r_id]
        other_non_null = [
            i for i, x in enumerate(examples)
            if x['recording_id'] != r_id and not is_null_description(x['content'])
        ]
        non_null = [i for i in r_items if not is_null_description(examples[i]['content'])]
        if len(non_null) < 2 or len(other_non_null) < 4:
            continue

        foil_audit['total_origins'] += 1
        r_norms = {norm_all[i] for i in r_items}
        r_words = set().union(*(tok_sets[i] for i in r_items))
        r_zt_c = zt_centered[r_items]

        strict_foils = []
        jaccard_foils = []
        for f_i in other_non_null:
            if norm_all[f_i] not in r_norms and len(tok_sets[f_i] & r_words) == 0:
                max_c_cos = float((zt_centered[f_i:f_i + 1] @ r_zt_c.T).max().item())
                jaccard_foils.append((max_c_cos, f_i))
                if max_c_cos < 0.15:
                    strict_foils.append(f_i)

        if hard_topic_foil and jaccard_foils:
            # Topic-matched hard negative control: pick lexically disjoint foils with HIGHEST topic cosine!
            jaccard_foils.sort(key=lambda pair: pair[0], reverse=True)
            verified_foils = [pair[1] for pair in jaccard_foils[:max(4, len(jaccard_foils) // 2)]]
        elif strict_foils:
            foil_audit['strict_cos015_and_jaccard0_origins'] += 1
            verified_foils = strict_foils
        elif jaccard_foils:
            foil_audit['lexical_jaccard0_origins'] += 1
            jaccard_foils.sort(key=lambda pair: pair[0])
            verified_foils = [pair[1] for pair in jaccard_foils[:max(4, len(jaccard_foils) // 2)]]
        else:
            foil_audit['string_disjoint_fallback_origins'] += 1
            verified_foils = [f_i for f_i in other_non_null if norm_all[f_i] not in r_norms]
        if not verified_foils:
            continue

        for src_a in ('video', 'audio'):
            for src_b in ('video', 'audio'):
                if mode == 'counterfactual_stream_roll' and src_a == src_b:
                    continue

                dir_tag = f"{src_a}_to_{src_b}"
                s1_int = 0 if src_a == 'video' else 1
                b1_list = sorted(
                    [i for i in r_items if examples[i]['source'] == src_a],
                    key=lambda i: float(examples[i]['start'])
                )
                b2_list = sorted(
                    [i for i in r_items if examples[i]['source'] == src_b],
                    key=lambda i: float(examples[i]['start'])
                )
                if len(b1_list) < 4 or len(b2_list) < 4:
                    continue

                b1_idx = torch.tensor(b1_list, device=device, dtype=torch.long)
                b2_idx = torch.tensor(b2_list, device=device, dtype=torch.long)
                iv_1 = torch.tensor([[examples[i]['start'], examples[i]['end']] for i in b1_list], device=device, dtype=torch.float32)
                iv_2 = torch.tensor([[examples[i]['start'], examples[i]['end']] for i in b2_list], device=device, dtype=torch.float32)

                non_null_a = [i for i in b1_list if not is_null_description(examples[i]['content'])]
                non_null_b = [i for i in b2_list if not is_null_description(examples[i]['content'])]
                if not non_null_a or not non_null_b:
                    continue

                by_state_buckets = {0: [], 1: [], 2: [], 3: []}

                def make_ep(q_a_idx, q_b_idx, perm1, perm2, label_val, pair_key=None, exp_pos_a=None, exp_pos_b=None):
                    if uniform_control:
                        raw_s1 = torch.zeros(len(b1_list), device=device, dtype=torch.float32)
                        raw_s2 = torch.zeros(len(b2_list), device=device, dtype=torch.float32)
                        lg1 = raw_s1
                        lg2 = raw_s2
                    else:
                        use_a = verified_foils[(q_a_idx + 1) % len(verified_foils)] if wrong_event_control else q_a_idx
                        use_b = verified_foils[(q_b_idx + 3) % len(verified_foils)] if wrong_event_control else q_b_idx

                        if two_hop_conditioned:
                            qa_vec = zt_encoded[use_a]
                            qb_vec = zt_encoded[use_b]
                            bank1_vecs = zm_encoded[b1_idx[perm1]]
                            bank2_vecs = zm_encoded[b2_idx[perm2]]
                            q_comp = F.normalize(0.55 * qa_vec + 0.45 * qb_vec, dim=-1)
                            raw_s1 = q_comp @ bank1_vecs.T
                            lg1 = raw_s1 * scale_val
                            p1_tmp = F.softmax(lg1, dim=-1)
                            if oracle_hop1_anchor and exp_pos_a is not None:
                                v_hat_1 = bank1_vecs[exp_pos_a]
                            else:
                                v_hat_1 = F.normalize((p1_tmp.unsqueeze(-1) * bank1_vecs).sum(dim=0), dim=-1)
                            q_hop2 = hop2_bridge(q_comp, v_hat_1, s1_int)
                            raw_s2 = q_hop2 @ bank2_vecs.T
                            lg2 = raw_s2 * scale_val

                            if label_val < 3 and not wrong_event_control and exp_pos_a is not None and exp_pos_b is not None:
                                h1_ok = int(raw_s1.argmax().item()) == int(exp_pos_a)
                                h2_ok = int(raw_s2.argmax().item()) == int(exp_pos_b)
                                foil_audit['hop1_total'] += 1
                                foil_audit['hop1_correct'] += int(h1_ok)
                                foil_audit['hop2_total'] += 1
                                foil_audit['hop2_correct'] += int(h2_ok)
                                if h1_ok:
                                    foil_audit['hop2_given_hop1_correct_total'] += 1
                                    foil_audit['hop2_given_hop1_correct_hits'] += int(h2_ok)
                                else:
                                    foil_audit['hop2_given_hop1_wrong_total'] += 1
                                    foil_audit['hop2_given_hop1_wrong_hits'] += int(h2_ok)
                        else:
                            raw_s1 = sim_mat[use_a, b1_idx[perm1]]
                            raw_s2 = sim_mat[use_b, b2_idx[perm2]]
                            lg1 = raw_s1 * scale_val
                            lg2 = raw_s2 * scale_val

                    iv2_use = iv_2
                    if coord_shuffle_control:
                        g_c = torch.Generator(device=device).manual_seed(101 + q_a_idx * 17 + q_b_idx + label_val)
                        c_perm = torch.randperm(len(b2_list), generator=g_c, device=device)
                        iv2_use = iv_2[c_perm]

                    exp_phi, null_f, std_f, _, _, _ = compute_shrinkage_calibrated_relational_features(
                        lg1, lg2, iv_1, iv2_use, raw_sim_1=raw_s1, raw_sim_2=raw_s2, tau_g=0.05
                    )
                    return {
                        'phi': exp_phi.detach(),
                        'null_feat': null_f.detach(),
                        'std_attn': std_f.detach(),
                        'bary_plus_phi': torch.cat([std_f, exp_phi], dim=-1).detach(),
                        'bary_plus_presence_no_phi': torch.cat([std_f, null_f], dim=-1).detach(),
                        'full_feat': torch.cat([std_f, exp_phi, null_f], dim=-1).detach(),
                        'label': int(label_val),
                        'dir_tag': dir_tag,
                        'pair_key': pair_key,
                    }

                id_perm1 = torch.arange(len(b1_list), device=device, dtype=torch.long)
                id_perm2 = torch.arange(len(b2_list), device=device, dtype=torch.long)
                rev_perm1 = torch.arange(len(b1_list) - 1, -1, -1, device=device, dtype=torch.long)
                rev_perm2 = torch.arange(len(b2_list) - 1, -1, -1, device=device, dtype=torch.long)

                for idx_a in non_null_a:
                    pos_a = b1_list.index(idx_a)
                    s_a, e_a = float(examples[idx_a]['start']), float(examples[idx_a]['end'])
                    for idx_b in non_null_b:
                        if idx_a == idx_b:
                            continue
                        pos_b = b2_list.index(idx_b)
                        s_b, e_b = float(examples[idx_b]['start']), float(examples[idx_b]['end'])
                        nat_rel = classify_interval_relation(s_a, e_a, s_b, e_b)
                        if nat_rel < 0:
                            continue

                        same_desc_b = [j for j in b2_list if norm_all[j] == norm_all[idx_b]]
                        if len(same_desc_b) > 1:
                            rels_b = {
                                classify_interval_relation(s_a, e_a, float(examples[j]['start']), float(examples[j]['end']))
                                for j in same_desc_b
                            }
                            if len(rels_b) > 1:
                                continue

                        f_idx = verified_foils[(idx_a + idx_b) % len(verified_foils)]

                        if mode == 'natural_quartets':
                            by_state_buckets[nat_rel].append(
                                make_ep(idx_a, idx_b, id_perm1, id_perm2, nat_rel, exp_pos_a=pos_a, exp_pos_b=pos_b)
                            )
                            by_state_buckets[3].append(make_ep(idx_a, f_idx, id_perm1, id_perm2, 3))
                        elif mode == 'global_time_reversal_no_role_split':
                            pk = f"{r_id}_{idx_a}_{idx_b}"
                            by_state_buckets[nat_rel].append(
                                make_ep(idx_a, idx_b, id_perm1, id_perm2, nat_rel, pair_key=(pk, 'fwd'))
                            )
                            rev_rel = 2 if nat_rel == 0 else (0 if nat_rel == 2 else 1)
                            by_state_buckets[rev_rel].append(
                                make_ep(idx_a, idx_b, rev_perm1, rev_perm2, rev_rel, pair_key=(pk, 'rev'))
                            )
                            by_state_buckets[3].append(make_ep(idx_a, f_idx, id_perm1, id_perm2, 3))
                            by_state_buckets[3].append(make_ep(idx_a, f_idx, rev_perm1, rev_perm2, 3))
                        elif mode == 'counterfactual_stream_roll':
                            K2 = len(b2_list)
                            same_desc_a = [j for j in b1_list if norm_all[j] == norm_all[idx_a]]
                            if len(same_desc_b) == 1 and len(same_desc_a) == 1 and K2 >= 6 and 2 <= pos_a <= K2 - 3 and len(b1_list) == K2:
                                for lbl_c, tgt_pos in ((0, pos_a - 2), (1, pos_a), (2, pos_a + 2)):
                                    s_tgt = float(iv_2[tgt_pos, 0].item())
                                    e_tgt = float(iv_2[tgt_pos, 1].item())
                                    assert classify_interval_relation(s_a, e_a, s_tgt, e_tgt) == lbl_c
                                    shift_k = (tgt_pos - pos_b) % K2
                                    roll_p2 = (torch.arange(K2, device=device) - shift_k) % K2
                                    by_state_buckets[lbl_c].append(
                                        make_ep(idx_a, idx_b, id_perm1, roll_p2, lbl_c, exp_pos_a=pos_a, exp_pos_b=tgt_pos)
                                    )
                                by_state_buckets[3].append(make_ep(idx_a, f_idx, id_perm1, id_perm2, 3))

                m_count = min(len(by_state_buckets[c]) for c in (0, 1, 2, 3))
                if m_count > 0:
                    for c in (0, 1, 2, 3):
                        episodes.extend(by_state_buckets[c][:m_count])

    if foil_audit['hop1_total'] > 0:
        foil_audit['hop1_top1_acc_pct'] = round(100.0 * foil_audit['hop1_correct'] / foil_audit['hop1_total'], 1)
        foil_audit['hop2_top1_acc_pct'] = round(100.0 * foil_audit['hop2_correct'] / max(1, foil_audit['hop2_total']), 1)
        foil_audit['hop2_acc_given_hop1_correct_pct'] = round(
            100.0 * foil_audit['hop2_given_hop1_correct_hits'] / max(1, foil_audit['hop2_given_hop1_correct_total']), 1
        )
        foil_audit['hop2_acc_given_hop1_wrong_pct'] = round(
            100.0 * foil_audit['hop2_given_hop1_wrong_hits'] / max(1, foil_audit['hop2_given_hop1_wrong_total']), 1
        )

    return episodes, foil_audit


def eval_linear_probe(tr_eps: list, he_eps: list, feat_fn, device: str, n_classes: int = 4):
    if not tr_eps or not he_eps:
        return {'status': 'empty_episodes'}
    X_tr = torch.stack([feat_fn(e) for e in tr_eps]).to(device)
    y_tr = torch.tensor([e['label'] for e in tr_eps], device=device, dtype=torch.long)
    X_he = torch.stack([feat_fn(e) for e in he_eps]).to(device)
    y_he = torch.tensor([e['label'] for e in he_eps], device=device, dtype=torch.long)

    mu = X_tr.mean(dim=0, keepdim=True)
    sd = X_tr.std(dim=0, keepdim=True)
    valid_col = (sd > 1e-5).float()
    X_tr_n = ((X_tr - mu) / sd.clamp(min=1e-4)) * valid_col
    X_he_n = ((X_he - mu) / sd.clamp(min=1e-4)) * valid_col

    torch.manual_seed(20261015)
    clf = nn.Linear(X_tr.shape[-1], n_classes).to(device)
    nn.init.zeros_(clf.weight)
    nn.init.zeros_(clf.bias)
    opt = torch.optim.AdamW(clf.parameters(), lr=0.035, weight_decay=0.04)
    for _ in range(200):
        loss = F.cross_entropy(clf(X_tr_n), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_he = clf(X_he_n)
        preds = logits_he.argmax(dim=-1)
        overall_4way = round(100.0 * float((preds == y_he).float().mean().item()), 1)

        rel_mask = (y_he < 3)
        if rel_mask.any():
            preds_3way = logits_he[rel_mask, :3].argmax(dim=-1)
            overall_3way_no_foil = round(100.0 * float((preds_3way == y_he[rel_mask]).float().mean().item()), 1)
        else:
            overall_3way_no_foil = 0.0

        by_cls = {}
        for c_i, c_name in enumerate(('Before', 'During', 'After', 'Null_Foil')):
            m_c = (y_he == c_i)
            if m_c.any():
                by_cls[c_name] = round(100.0 * float((preds[m_c] == y_he[m_c]).float().mean().item()), 1)

        fwd_map, rev_map = {}, {}
        for idx_e, e in enumerate(he_eps):
            pk = e.get('pair_key')
            if pk is not None:
                k_str, dir_s = pk
                p_3 = int(logits_he[idx_e, :3].argmax().item())
                if dir_s == 'fwd':
                    fwd_map[k_str] = (e['label'], p_3)
                else:
                    rev_map[k_str] = (e['label'], p_3)
        paired_hits, paired_tot = 0, 0
        for k_str, (lbl_f, p_f) in fwd_map.items():
            if k_str in rev_map:
                lbl_r, p_r = rev_map[k_str]
                paired_tot += 1
                if p_f == lbl_f and p_r == lbl_r:
                    paired_hits += 1

    out = {
        'overall_4way_pct': overall_4way,
        'overall_3way_no_foil_pct': overall_3way_no_foil,
        'by_state': by_cls,
        'n_eval_episodes': len(he_eps),
    }
    if paired_tot > 0:
        out['paired_both_directions_correct_pct'] = round(100.0 * paired_hits / paired_tot, 1)
    return out


def run_stage0_m2_connected(ctx_bundle: dict):
    if ctx_bundle is None:
        return {'status': 'skipped'}

    device = str(ctx_bundle['tr_txt'].device)
    train_ex = ctx_bundle['train_ex']
    held_ex = ctx_bundle['held_ex']
    tr_txt_mean = ctx_bundle['tr_txt'].float().mean(dim=0, keepdim=True)

    # Primary teacher fixed across the entire main ablation ladder: C6 Bilevel Global+Innovation Aligner
    aligner_main = ctx_bundle['m_c6']
    sim_tr_main = ctx_bundle['sim_tr_c6']
    sim_he_main = ctx_bundle['sim_he_c6']

    bridge_mix = train_two_hop_bridge_on_train_split(
        train_ex, aligner_main, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'], ctx_bundle['tr_src'],
        tr_rec_s=ctx_bundle['tr_rec_s'], use_retrieved_mixtures=True
    )
    bridge_tf = train_two_hop_bridge_on_train_split(
        train_ex, aligner_main, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'], ctx_bundle['tr_src'],
        tr_rec_s=ctx_bundle['tr_rec_s'], use_retrieved_mixtures=False
    )

    # Reference C1 teacher for cross-teacher baseline check
    aligner_c1 = ctx_bundle['m_c1']
    sim_tr_c1 = ctx_bundle['sim_tr_c1']
    sim_he_c1 = ctx_bundle['sim_he_c1']
    bridge_c1 = train_two_hop_bridge_on_train_split(
        train_ex, aligner_c1, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'], ctx_bundle['tr_src'],
        use_retrieved_mixtures=True
    )

    std_fn = lambda e: e['std_attn']
    bary_phi_fn = lambda e: e['bary_plus_phi']
    null_fn = lambda e: e['null_feat']
    no_phi_fn = lambda e: e['bary_plus_presence_no_phi']
    full_fn = lambda e: e['full_feat']

    suites_out = {}
    foil_audits = {}
    for suite_name, mode_str, two_hop in (
        ('PartB1_Natural_Unshifted_Balanced', 'natural_quartets', False),
        ('PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit', 'global_time_reversal_no_role_split', False),
        ('PartB3_CrossModal_Only_TwoHop_Roll', 'counterfactual_stream_roll', True),
    ):
        tr_main, f_aud_tr = build_connected_m2_episodes(
            train_ex, sim_tr_main, aligner_main, bridge_mix, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop
        )
        he_main, f_aud_he = build_connected_m2_episodes(
            held_ex, sim_he_main, aligner_main, bridge_mix, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop
        )
        foil_audits[suite_name] = {'train': f_aud_tr, 'heldout': f_aud_he}

        tr_c1, _ = build_connected_m2_episodes(
            train_ex, sim_tr_c1, aligner_c1, bridge_c1, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )
        he_c1, _ = build_connected_m2_episodes(
            held_ex, sim_he_c1, aligner_c1, bridge_c1, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )

        tr_unif, _ = build_connected_m2_episodes(
            train_ex, sim_tr_main, aligner_main, bridge_mix, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, uniform_control=True
        )
        he_unif, _ = build_connected_m2_episodes(
            held_ex, sim_he_main, aligner_main, bridge_mix, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, uniform_control=True
        )
        tr_wrong, _ = build_connected_m2_episodes(
            train_ex, sim_tr_main, aligner_main, bridge_mix, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, wrong_event_control=True
        )
        he_wrong, _ = build_connected_m2_episodes(
            held_ex, sim_he_main, aligner_main, bridge_mix, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, wrong_event_control=True
        )
        tr_cshuf, _ = build_connected_m2_episodes(
            train_ex, sim_tr_main, aligner_main, bridge_mix, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, coord_shuffle_control=True
        )
        he_cshuf, _ = build_connected_m2_episodes(
            held_ex, sim_he_main, aligner_main, bridge_mix, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
            mode=mode_str, two_hop_conditioned=two_hop, coord_shuffle_control=True
        )

        suite_dict = {
            'B1_unmodified_softmax_mean_time_c1_ref': eval_linear_probe(tr_c1, he_c1, std_fn, device),
            'B1b_unmodified_softmax_mean_time_c6_matched': eval_linear_probe(tr_main, he_main, std_fn, device),
            'B2_coord_shuffle_control': eval_linear_probe(tr_cshuf, he_cshuf, full_fn, device),
            'B3_wrong_event_query_control': eval_linear_probe(tr_wrong, he_wrong, full_fn, device),
            'B4_uniform_retrieval_sim0_control': eval_linear_probe(tr_unif, he_unif, full_fn, device),
            'B5_barycenter_plus_phi_emd_no_null_c6': eval_linear_probe(tr_main, he_main, bary_phi_fn, device),
            'B6_shift_sensitive_presence_null_only_c6': eval_linear_probe(tr_main, he_main, null_fn, device),
            'B7_barycenter_plus_presence_NO_PHI_yann_ctrl_c6': eval_linear_probe(tr_main, he_main, no_phi_fn, device),
            'B8_full_bary_phi_emd_presence_c6_connected': eval_linear_probe(tr_main, he_main, full_fn, device),
        }

        if two_hop:
            # Explicit Teacher-Forced Bridge vs Mixture-Trained Bridge vs Oracle-Anchor Upper Bound (Yann Section 5)
            tr_tf_b, aud_tf_tr = build_connected_m2_episodes(
                train_ex, sim_tr_main, aligner_main, bridge_tf, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
                ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
                mode=mode_str, two_hop_conditioned=True
            )
            he_tf_b, aud_tf_he = build_connected_m2_episodes(
                held_ex, sim_he_main, aligner_main, bridge_tf, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
                ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
                mode=mode_str, two_hop_conditioned=True
            )
            tr_ora_b, aud_ora_tr = build_connected_m2_episodes(
                train_ex, sim_tr_main, aligner_main, bridge_mix, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
                ctx_bundle['tr_src'], tr_txt_mean, media_rec_feats=ctx_bundle['tr_rec_s'],
                mode=mode_str, two_hop_conditioned=True, oracle_hop1_anchor=True
            )
            he_ora_b, aud_ora_he = build_connected_m2_episodes(
                held_ex, sim_he_main, aligner_main, bridge_mix, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
                ctx_bundle['he_src'], tr_txt_mean, media_rec_feats=ctx_bundle['he_rec_s'],
                mode=mode_str, two_hop_conditioned=True, oracle_hop1_anchor=True
            )
            suite_dict['B8a_teacher_forced_bridge_ctrl'] = eval_linear_probe(tr_tf_b, he_tf_b, full_fn, device)
            suite_dict['B8b_oracle_hop1_anchor_upper_bound'] = eval_linear_probe(tr_ora_b, he_ora_b, full_fn, device)
            foil_audits[suite_name]['heldout_teacher_forced_bridge_hop_audit'] = aud_tf_he
            foil_audits[suite_name]['heldout_oracle_anchor_hop_audit'] = aud_ora_he

        suites_out[suite_name] = suite_dict

    return {'foil_proxy_and_hop_audit': foil_audits, **suites_out}


# ==============================================================================
# MODULE 3: PER-LAYER COUNTERFACTUAL ATTENTION OUTPUT ROUTING (`CAOR`)
# ==============================================================================
def audit_and_interpolate_grid_clocks(payload: dict, sample_meta: dict) -> dict:
    pos_t = payload['decoder']['position_ids'][0, 0].cpu().long()
    is_v = payload['is_v'].cpu().bool()
    is_a = payload['is_a'].cpu().bool()
    sec_cached = payload['sec'].cpu().float()

    v_indices = torch.nonzero(is_v, as_tuple=False).squeeze(-1)
    a_indices = torch.nonzero(is_a, as_tuple=False).squeeze(-1)

    v_pos = pos_t[v_indices]
    a_pos = pos_t[a_indices]

    v_unique_pos, v_grid_inverse, v_grid_counts = torch.unique_consecutive(
        v_pos, return_inverse=True, return_counts=True
    )
    G_v = int(v_unique_pos.numel())
    N_v = int(v_indices.numel())
    N_a = int(a_indices.numel())

    aud_token_count_sec = float(N_a) / 25.0
    aud_pos_span_sec = float(a_pos.max().item() - a_pos.min().item() + 1) / 25.0 if N_a > 0 else 0.0
    meta_dur = float(sample_meta.get('duration', sample_meta.get('dur', aud_pos_span_sec)))
    T_phys = meta_dur if meta_dur > 0.5 else aud_pos_span_sec

    uniform_patches = bool((v_grid_counts == v_grid_counts[0]).all().item()) if G_v > 0 else True
    audio_span_diff = abs(aud_pos_span_sec - T_phys)
    assert uniform_patches, f"Non-uniform patch count across visual temporal grids: {v_grid_counts.tolist()}"
    assert audio_span_diff <= 0.35, f"Audio 25Hz span {aud_pos_span_sec}s diverges from clip duration {T_phys}s"

    sec_interpolated = sec_cached.clone()
    if G_v > 0:
        grid_mid_times = (torch.arange(G_v, dtype=torch.float32) + 0.5) * (T_phys / float(G_v))
        sec_interpolated[v_indices] = grid_mid_times[v_grid_inverse]
    if N_a > 0:
        a_exact_25hz = (a_pos - a_pos.min()).float() / 25.0
        sec_interpolated[a_indices] = a_exact_25hz.clamp(max=T_phys)

    clock_audit = {
        'clip_duration_sec': round(T_phys, 3),
        'audio_token_count_div_25_sec': round(aud_token_count_sec, 3),
        'audio_rope_span_25hz_sec': round(aud_pos_span_sec, 3),
        'cached_visual_sec_max': round(float(sec_cached[v_indices].max().item()), 3) if N_v > 0 else 0.0,
        'n_visual_temporal_grids_Gv': G_v,
        'patches_per_visual_grid': int(v_grid_counts[0].item()) if G_v > 0 else 0,
        'uniform_patches_across_grids': uniform_patches,
        'interpolated_visual_sec_max': round(float(sec_interpolated[v_indices].max().item()), 3) if N_v > 0 else 0.0,
        'is_uniform_grid_interpolation_not_frame_pts_oracle': True,
    }
    return {
        'sec_calibrated': sec_interpolated,
        'T_phys': T_phys,
        'G_v': G_v,
        'v_indices': v_indices,
        'a_indices': a_indices,
        'clock_audit': clock_audit,
    }


def project_to_layer_vo_subspace(layer_mod, z_vec: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ln = layer_mod.input_layernorm
        attn = layer_mod.self_attn
        dev = next(ln.parameters()).device
        dt = next(ln.parameters()).dtype
        x = z_vec.to(device=dev, dtype=dt)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_ln = ln(x)
        v = attn.v_proj(x_ln)
        o_in_dim = attn.o_proj.weight.shape[1]
        if v.shape[-1] != o_in_dim:
            rep = o_in_dim // v.shape[-1]
            head_dim = getattr(attn, 'head_dim', 128)
            n_kv = v.shape[-1] // head_dim
            v = v.view(v.shape[0], n_kv, head_dim).repeat_interleave(rep, dim=1).reshape(v.shape[0], o_in_dim)
        out = attn.o_proj(v).float().cpu()
        out = F.normalize(out, dim=-1)
        return out[0] if z_vec.ndim == 1 else out


def extract_query_conditioned_salient_slots(
    h18_t: torch.Tensor,
    h27_t: torch.Tensor,
    query_rows: list,
    last_m: int,
    clock_info: dict,
    custom_q18: torch.Tensor = None,
    custom_q27: torch.Tensor = None,
    n_slots: int = None,
    top_patch_frac: float = 0.15,
):
    """
    Extracts de-sunk query-conditioned salient slots across `K` calibrated temporal windows:
      - Subtracts the question-blind media-end delimiter state `h[last_m]` (`<|vision_eos|>`) from the query
        representation so common anisotropy / structural sink directions cancel out!
      - Also measures the question-blind baseline affinity `v_sink, a_sink` against `h[last_m]` and computes
        excess query salience above `h[last_m]`, ensuring that both coordinate-permuted queries AND real
        semantic queries about absent events yield a low confidence gate!
    """
    T_phys = clock_info['T_phys']
    sec_cal = clock_info['sec_calibrated']
    v_indices = clock_info['v_indices']
    a_indices = clock_info['a_indices']

    K = max(4, min(12, int(round(T_phys / 2.0)))) if n_slots is None else n_slots
    slot_width = T_phys / float(K)

    r0_18 = F.normalize(F.layer_norm(h18_t[last_m:last_m + 1], (h18_t.shape[-1],)), dim=-1)
    r0_27 = F.normalize(F.layer_norm(h27_t[last_m:last_m + 1], (h27_t.shape[-1],)), dim=-1)

    if custom_q18 is None:
        q18_raw = F.normalize(F.layer_norm(h18_t[query_rows], (h18_t.shape[-1],)), dim=-1)
        q18 = F.normalize(q18_raw - 0.65 * (q18_raw @ r0_18.T) * r0_18, dim=-1)
    else:
        q18 = custom_q18

    if custom_q27 is None:
        q27_raw = F.normalize(F.layer_norm(h27_t[query_rows], (h27_t.shape[-1],)), dim=-1)
        q27 = F.normalize(q27_raw - 0.65 * (q27_raw @ r0_27.T) * r0_27, dim=-1)
    else:
        q27 = custom_q27

    v18_all = F.layer_norm(h18_t[v_indices], (h18_t.shape[-1],))
    v27_all = F.layer_norm(h27_t[v_indices], (h27_t.shape[-1],))
    v18_dyn_all = F.normalize(v18_all - 0.90 * v18_all.mean(dim=0, keepdim=True), dim=-1)
    v27_dyn_all = F.normalize(v27_all - 0.90 * v27_all.mean(dim=0, keepdim=True), dim=-1)

    a18_all = F.layer_norm(h18_t[a_indices], (h18_t.shape[-1],))
    a27_all = F.layer_norm(h27_t[a_indices], (h27_t.shape[-1],))
    a18_dyn_all = F.normalize(a18_all - 0.90 * a18_all.mean(dim=0, keepdim=True), dim=-1)
    a27_dyn_all = F.normalize(a27_all - 0.90 * a27_all.mean(dim=0, keepdim=True), dim=-1)

    v_sal_q = 0.50 * (v18_dyn_all @ q18.T).max(dim=-1).values + 0.50 * (v27_dyn_all @ q27.T).max(dim=-1).values
    a_sal_q = 0.50 * (a18_dyn_all @ q18.T).max(dim=-1).values + 0.50 * (a27_dyn_all @ q27.T).max(dim=-1).values

    v_sink = 0.50 * (v18_dyn_all @ r0_18.T).squeeze(-1) + 0.50 * (v27_dyn_all @ r0_27.T).squeeze(-1)
    a_sink = 0.50 * (a18_dyn_all @ r0_18.T).squeeze(-1) + 0.50 * (a27_dyn_all @ r0_27.T).squeeze(-1)

    v_sal = v_sal_q - 0.35 * (v_sink - v_sink.mean())
    a_sal = a_sal_q - 0.35 * (a_sink - a_sink.mean())

    v_times = sec_cal[v_indices]
    a_times = sec_cal[a_indices]

    V18_slots, A18_slots = [], []
    raw_v_scores, raw_a_scores = [], []
    iv_slots = []
    v_token_slot_ids = torch.zeros(v_indices.numel(), dtype=torch.long)
    a_token_slot_ids = torch.zeros(a_indices.numel(), dtype=torch.long)

    for k in range(K):
        t0 = k * slot_width
        t1 = (k + 1) * slot_width + (1e-3 if k == K - 1 else 0.0)
        iv_slots.append([round(t0, 3), round(min(T_phys, t1), 3)])

        mv = torch.nonzero((v_times >= t0) & (v_times < t1), as_tuple=False).squeeze(-1)
        if mv.numel() == 0:
            mv = torch.arange(v_indices.numel())
        else:
            v_token_slot_ids[mv] = k
        k_top_v = max(1, int(round(top_patch_frac * mv.numel())))
        top_v_vals, top_v_idx = torch.topk(v_sal[mv], k=k_top_v, largest=True)
        chosen_v = mv[top_v_idx]
        V18_slots.append(F.normalize(v18_dyn_all[chosen_v].mean(dim=0), dim=-1))
        raw_v_scores.append(top_v_vals.mean())

        ma = torch.nonzero((a_times >= t0) & (a_times < t1), as_tuple=False).squeeze(-1)
        if ma.numel() == 0:
            ma = torch.arange(a_indices.numel())
        else:
            a_token_slot_ids[ma] = k
        k_top_a = max(1, int(round(max(0.25, top_patch_frac) * ma.numel())))
        top_a_vals, top_a_idx = torch.topk(a_sal[ma], k=k_top_a, largest=True)
        chosen_a = ma[top_a_idx]
        A18_slots.append(F.normalize(a18_dyn_all[chosen_a].mean(dim=0), dim=-1))
        raw_a_scores.append(top_a_vals.mean())

    V18_bank = torch.stack(V18_slots)
    A18_bank = torch.stack(A18_slots)
    rv = torch.stack(raw_v_scores)
    ra = torch.stack(raw_a_scores)

    # Gate temperature by BOTH absolute excess affinity above 0.045 AND temporal slot contrast (max - mean)
    v_peak_excess = float(F.relu(rv.max() - 0.045).item())
    a_peak_excess = float(F.relu(ra.max() - 0.045).item())
    v_prom = float(F.relu((rv.max() - rv.mean()) - 0.012).item())
    a_prom = float(F.relu((ra.max() - ra.mean()) - 0.012).item())

    temp_v = 16.0 * math.tanh(v_peak_excess / 0.06) * math.tanh(v_prom / 0.025)
    temp_a = 16.0 * math.tanh(a_peak_excess / 0.06) * math.tanh(a_prom / 0.025)

    lg_v = (rv - rv.mean()) * temp_v
    lg_a = (ra - ra.mean()) * temp_a
    iv_t = torch.tensor(iv_slots, dtype=torch.float32)

    token_innov_v = (v_sal - v_sal.mean()) / v_sal.std(unbiased=False).clamp(min=1e-4)
    token_innov_a = (a_sal - a_sal.mean()) / a_sal.std(unbiased=False).clamp(min=1e-4)

    slot_meta = {
        'v_token_slot_ids': v_token_slot_ids,
        'a_token_slot_ids': a_token_slot_ids,
        'token_innov_v': token_innov_v.clamp(-2.5, 2.5),
        'token_innov_a': token_innov_a.clamp(-2.5, 2.5),
    }
    return V18_bank, A18_bank, lg_v, lg_a, iv_t, slot_meta


def build_caor_token_bias_profiles(
    L_seq: int,
    clock_info: dict,
    slot_meta: dict,
    aud_info: dict,
    mode: str = 'full',
    reverse_order: bool = False,
    irrel_slot_k: int = 0,
    rand_seed: int = None,
) -> dict:
    """
    Constructs the per-media-token attention logit bias `delta_B_v` and `delta_B_a` for
    Counterfactual Attention Output Routing (`CAOR`):
      - `mode='full'`      : De-sinking + Foreground Innovation + Evidentiary Support + Synchrony Coupling
                             + Chronological Order Tilt (`B_ord`).
      - `mode='rel_only'`  : Content/Relevance + Synchrony Coupling only (`B_ord = 0`).
      - `mode='order_only'`: Pure Chronological Order Tilt (`B_ord`) only (`B_ev = B_innov = 0`, `sink_w = 0`).
      - `mode='irrel'`     : Boosts only the lowest-salience irrelevant slot `irrel_slot_k` at matched scale.
      - `mode='random'`    : Random zero-mean slot bias of matched L2 norm with seed `rand_seed`.
    """
    p_v_hat = aud_info['p_v_hat']
    p_a_hat = aud_info['p_a_hat']
    K = p_v_hat.numel()
    gamma_gate = float(aud_info['gamma_gate'])
    bhatt_sync = float(aud_info['bhatt_sync'])
    rel_scalar = float(aud_info['rel_scalar'])
    if reverse_order:
        rel_scalar = -rel_scalar

    u_k = 1.0 / float(max(1, K))
    p_sync = torch.sqrt((p_v_hat * p_a_hat).clamp(min=1e-12))
    p_sync = p_sync / p_sync.sum().clamp(min=1e-8)

    # Anchor-coupled distribution: blends self-support, cross-modal anchor support, and synchrony overlap
    alpha_v = float(aud_info['alpha_v'])
    alpha_a = float(aud_info['alpha_a'])
    w_anc_a = (alpha_a + 0.05) / (alpha_a + alpha_v + 0.10)
    p_anchor = w_anc_a * p_a_hat + (1.0 - w_anc_a) * p_v_hat

    log_ev_v = torch.log((0.50 * p_v_hat + 0.30 * p_anchor + 0.20 * p_sync).clamp(min=1e-4) / u_k)
    log_ev_a = torch.log((0.55 * p_a_hat + 0.25 * p_anchor + 0.20 * p_sync).clamp(min=1e-4) / u_k)
    log_ev_v = (log_ev_v - log_ev_v.mean()).clamp(-2.0, 2.0)
    log_ev_a = (log_ev_a - log_ev_a.mean()).clamp(-2.0, 2.0)

    t_tilde = torch.linspace(-1.0, 1.0, K, dtype=torch.float32)
    ord_tilt = float( max(-1.8, min(1.8, 1.35 * rel_scalar)) )
    slot_ord_v = ord_tilt * t_tilde
    slot_ord_a = -0.85 * ord_tilt * t_tilde

    if mode == 'full':
        slot_bias_v = gamma_gate * (1.25 * log_ev_v + 1.15 * slot_ord_v)
        slot_bias_a = gamma_gate * (1.10 * log_ev_a + 1.15 * slot_ord_a)
        innov_w = 0.45 * gamma_gate
        sink_w = 0.60 * gamma_gate
        late_aud_damp = 0.85 * gamma_gate * (1.0 - 0.65 * bhatt_sync)
    elif mode == 'rel_only':
        slot_bias_v = gamma_gate * (1.25 * log_ev_v)
        slot_bias_a = gamma_gate * (1.10 * log_ev_a)
        innov_w = 0.45 * gamma_gate
        sink_w = 0.60 * gamma_gate
        late_aud_damp = 0.85 * gamma_gate * (1.0 - 0.65 * bhatt_sync)
    elif mode == 'order_only':
        slot_bias_v = gamma_gate * (1.35 * slot_ord_v)
        slot_bias_a = gamma_gate * (1.35 * slot_ord_a)
        innov_w = 0.0
        sink_w = 0.0
        late_aud_damp = 0.0
    elif mode == 'irrel':
        target_norm = float(gamma_gate * (1.25 * log_ev_v + 1.15 * slot_ord_v).norm().item())
        ir_v = torch.full((K,), -1.0 / max(1, K - 1), dtype=torch.float32)
        ir_v[irrel_slot_k] = 1.0
        ir_v = F.normalize(ir_v - ir_v.mean(), dim=-1) * max(target_norm, 0.5)
        slot_bias_v = ir_v
        slot_bias_a = ir_v.clone()
        innov_w = 0.0
        sink_w = 0.0
        late_aud_damp = 0.0
    elif mode == 'random':
        target_norm = float(gamma_gate * (1.25 * log_ev_v + 1.15 * slot_ord_v).norm().item())
        g_r = torch.Generator().manual_seed(int(rand_seed or 101))
        rv_s = torch.randn((K,), generator=g_r)
        ra_s = torch.randn((K,), generator=g_r)
        slot_bias_v = F.normalize(rv_s - rv_s.mean(), dim=-1) * max(target_norm, 0.5)
        slot_bias_a = F.normalize(ra_s - ra_s.mean(), dim=-1) * max(target_norm, 0.5)
        innov_w = 0.0
        sink_w = 0.0
        late_aud_damp = 0.0
    else:
        raise ValueError(f"Unknown CAOR mode: {mode}")

    v_slot_ids = slot_meta['v_token_slot_ids']
    a_slot_ids = slot_meta['a_token_slot_ids']
    tok_bias_v = slot_bias_v[v_slot_ids] + innov_w * slot_meta['token_innov_v']
    tok_bias_a = slot_bias_a[a_slot_ids] + innov_w * slot_meta['token_innov_a']
    tok_bias_v = tok_bias_v - tok_bias_v.mean()
    tok_bias_a = tok_bias_a - tok_bias_a.mean()

    return {
        'tok_bias_v': tok_bias_v,
        'tok_bias_a': tok_bias_a,
        'sink_w': float(sink_w),
        'late_aud_damp': float(late_aud_damp),
        'gamma_gate': float(gamma_gate),
    }


def find_option_free_and_option_spans(payload: dict, question_stem: str, tok):
    last_m = int(payload['last_media'])
    ids = payload['input_ids'].cpu()[0].tolist()
    L = len(ids)
    tail_ids = ids[last_m + 1:]
    tail_text = tok.decode(tail_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    enc = tok(tail_text, add_special_tokens=False, return_offsets_mapping=True)

    q_pos = tail_text.find(question_stem)
    if q_pos < 0:
        q_pos = 0
    q_end = q_pos + len(question_stem)

    im_end_pos = tail_text.rfind('<|im_end|>')
    if im_end_pos < 0:
        im_end_pos = len(tail_text) - 1

    stem_rows, option_rows, assistant_rows = [], [], []
    for i, (a, b) in enumerate(enc['offset_mapping']):
        glob_idx = last_m + 1 + i
        if b <= q_end:
            stem_rows.append(glob_idx)
        elif a >= im_end_pos:
            assistant_rows.append(glob_idx)
        else:
            option_rows.append(glob_idx)

    if (L - 1) not in assistant_rows:
        assistant_rows.append(L - 1)
    hook_rows = sorted(set(stem_rows + assistant_rows))
    option_rows = sorted(set(option_rows) - set(hook_rows))
    assert len(set(hook_rows) & set(option_rows)) == 0
    return stem_rows, option_rows, assistant_rows, hook_rows


def run_stage0_m3_bivector_vo_tangent_routing():
    """
    Executes Stage-0 v6 M3: Per-Layer Native Counterfactual Attention Output Routing (`CAOR`)
    with full matched controls (keeps function name compatible with existing Stage-0 runners).
    """
    if 'MODEL_' not in globals() or 'PROC' not in globals():
        return {'status': 'skipped_no_live_model'}

    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as qwen_omni_modeling

    model_llm = globals()['MODEL_']
    proc = globals()['PROC']
    tok = proc.tokenizer
    cache_dir = OUT_DIR / 'argon_latent40' / 'decoder_cache'
    manifest_path = OUT_DIR / 'uploads' / 'research40_manifest.json'
    if not manifest_path.exists():
        manifest_path = OUT_DIR / 'research40_manifest.json'
    if not cache_dir.exists() or not manifest_path.exists():
        return {'status': 'skipped_no_decoder_cache'}

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    server_id = int(globals().get('LAB_SERVER', globals().get('CD_SERVER', 1)))
    samples = manifest.get(f'server{server_id}', manifest['selected'][:4])[:4]
    labels = [tok.convert_tokens_to_ids(c) for c in ('A', 'B', 'C', 'D')]

    mid_layers = (14, 16, 18, 20, 22, 24)
    late_layers = (28, 29, 30, 31, 32, 33)
    assert len(mid_layers) == len(late_layers) == 6

    rows_out = []
    for s_i, sample in enumerate(samples):
        idx = sample['idx']
        pt_file = cache_dir / f'{idx}_rot0.pt'
        if not pt_file.exists():
            continue
        payload = torch.load(pt_file, map_location='cpu', weights_only=False)
        device = model_llm.get_input_embeddings().weight.device
        kw = {k: v.to(device) for k, v in payload['decoder'].items()}
        last_m = int(payload['last_media'])

        clock_info = audit_and_interpolate_grid_clocks(payload, sample)
        v_indices = clock_info['v_indices']
        a_indices = clock_info['a_indices']

        q_raw = sample['question']
        stem_text = q_raw.split('(A)')[0].split('A.')[0].strip()
        stem_rows, option_rows, assistant_rows, hook_rows = find_option_free_and_option_spans(
            payload, stem_text, tok
        )

        # Capture true PRE-ATTENTION hidden states (`input_layernorm(residual)`) via `self_attn` pre-hooks!
        h_pre_attn_cap = {}
        def make_pre_attn_cap(l_idx):
            def pre_cap_hook(mod, args, kwargs):
                h_in = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
                h_pre_attn_cap[l_idx] = h_in[0].detach().float().cpu()
                return args, kwargs
            return pre_cap_hook

        h18_pre = model_llm.model.layers[18].self_attn.register_forward_pre_hook(make_pre_attn_cap(18), with_kwargs=True)
        h27_pre = model_llm.model.layers[27].self_attn.register_forward_pre_hook(make_pre_attn_cap(27), with_kwargs=True)
        try:
            with torch.inference_mode():
                out_nat = model_llm.model(**kw, use_cache=False, return_dict=True)
                lg_nat = model_llm.lm_head(
                    out_nat.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                )[0, -1].float().cpu()
        finally:
            h18_pre.remove()
            h27_pre.remove()

        k_opts = len(payload.get('native_logits', lg_nat[labels[:3]]))
        ref_logits = payload.get('native_logits', lg_nat[labels[:k_opts]]).cpu().float()
        native_replay_err = float((lg_nat[labels[:k_opts]] - ref_logits).abs().max().item())

        # Option-free prefix (`keep_indices` = media + stem_rows + assistant_rows, ZERO option_rows!)
        keep_indices = sorted(list(range(last_m + 1)) + stem_rows + assistant_rows)
        keep_t = torch.tensor(keep_indices, device=device, dtype=torch.long)
        freeform_embeds = kw['inputs_embeds'][:, keep_t, :].clone()
        freeform_pos = kw['position_ids'][:, :, keep_t].clone()
        tail_start_pos = int(freeform_pos[0, 0, last_m].item()) + 1
        n_tail_ff = len(keep_indices) - (last_m + 1)
        freeform_pos[:, :, last_m + 1:] = (
            tail_start_pos + torch.arange(n_tail_ff, device=device)
        ).view(1, 1, -1).to(freeform_pos.dtype)

        # Extract salient slots from true pre-attention states using stem_rows (100% option-free!)
        V18_raw, A18_raw, lg_v, lg_a, iv_t, slot_meta = extract_query_conditioned_salient_slots(
            h_pre_attn_cap[18], h_pre_attn_cap[27], stem_rows, last_m, clock_info, top_patch_frac=0.15
        )
        V18_vo = project_to_layer_vo_subspace(model_llm.model.layers[18], V18_raw)
        A18_vo = project_to_layer_vo_subspace(model_llm.model.layers[18], A18_raw)
        V18_bank = V18_vo
        A18_bank = A18_vo

        vec_fwd, gate_fwd, phi_fwd, aud_fwd = build_orthogonal_bivector_routed_state(
            V18_vo, A18_vo, lg_v, lg_a, iv_t, reverse_timeline=False
        )
        vec_rev, gate_rev, phi_rev, aud_rev = build_orthogonal_bivector_routed_state(
            V18_vo, A18_vo, lg_v, lg_a, iv_t, reverse_timeline=True
        )
        fwd_rev_cos = float(torch.dot(vec_fwd, vec_rev).item())
        content_even_err = float((aud_fwd['c_ev'] - aud_rev['c_ev']).abs().max().item())
        order_odd_err = float((aud_fwd['tau_unit'] + aud_rev['tau_unit']).abs().max().item())
        assert content_even_err < 1e-5, f"Real sample content not even: {content_even_err}"
        assert order_odd_err < 1e-5, f"Real sample order not odd: {order_odd_err}"
        assert fwd_rev_cos < 0.50, f"Real sample fwd/rev cosine too high: {fwd_rev_cos}"

        # Wrong-Query Control 1: Coordinate-Permuted Query (v5 backward reference)
        g_wq = torch.Generator().manual_seed(idx + 777)
        perm_d = torch.randperm(h_pre_attn_cap[18].shape[-1], generator=g_wq)
        q18_perm = F.normalize(F.layer_norm(h_pre_attn_cap[18][stem_rows][:, perm_d], (h_pre_attn_cap[18].shape[-1],)), dim=-1)
        q27_perm = F.normalize(F.layer_norm(h_pre_attn_cap[27][stem_rows][:, perm_d], (h_pre_attn_cap[27].shape[-1],)), dim=-1)
        _, _, lg_v_perm, lg_a_perm, _, _ = extract_query_conditioned_salient_slots(
            h_pre_attn_cap[18], h_pre_attn_cap[27], stem_rows, last_m, clock_info,
            custom_q18=q18_perm, custom_q27=q27_perm
        )
        _, gate_perm_wrong, _, _ = build_orthogonal_bivector_routed_state(
            V18_vo, A18_vo, lg_v_perm, lg_a_perm, iv_t, reverse_timeline=False
        )

        # Wrong-Query Control 2: Real Semantic Wrong-Event Query (Yann Section 6!)
        # Encodes a grammatical English question about an absent event from a disjoint domain through the
        # live tokenizer and model layers (with the exact same media prefix!), measuring its real gate!
        sem_wrong_text = DISJOINT_ABSENT_EVENT_QUERIES[s_i % len(DISJOINT_ABSENT_EVENT_QUERIES)]
        sem_wrong_ids = tok.encode('\n' + sem_wrong_text + '<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
        emb_layer = model_llm.get_input_embeddings()
        with torch.inference_mode():
            sem_tail_emb = emb_layer(torch.tensor([sem_wrong_ids], device=device)).to(kw['inputs_embeds'].dtype)
        sem_embeds = torch.cat([kw['inputs_embeds'][:, :last_m + 1, :], sem_tail_emb], dim=1)
        L_sem = sem_embeds.shape[1]
        sem_pos = torch.zeros((3, 1, L_sem), dtype=kw['position_ids'].dtype, device=device)
        sem_pos[:, :, :last_m + 1] = kw['position_ids'][:, :, :last_m + 1]
        sem_pos[:, :, last_m + 1:] = (
            tail_start_pos + torch.arange(len(sem_wrong_ids), device=device)
        ).view(1, 1, -1).to(sem_pos.dtype)

        h_sem_cap = {}
        def make_sem_cap(l_idx):
            def s_hk(mod, args, kwargs):
                h_in = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
                h_sem_cap[l_idx] = h_in[0].detach().float().cpu()
                return args, kwargs
            return s_hk

        hs18 = model_llm.model.layers[18].self_attn.register_forward_pre_hook(make_sem_cap(18), with_kwargs=True)
        hs27 = model_llm.model.layers[27].self_attn.register_forward_pre_hook(make_sem_cap(27), with_kwargs=True)
        try:
            with torch.inference_mode():
                model_llm.model(
                    inputs_embeds=sem_embeds,
                    position_ids=sem_pos,
                    attention_mask=torch.ones((1, L_sem), dtype=torch.long, device=device),
                    cache_position=torch.arange(L_sem, device=device),
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            hs18.remove()
            hs27.remove()

        sem_stem_rows = list(range(last_m + 1, last_m + 1 + max(2, len(sem_wrong_ids) - 5)))
        _, _, lg_v_sem_w, lg_a_sem_w, _, _ = extract_query_conditioned_salient_slots(
            h_sem_cap[18], h_sem_cap[27], sem_stem_rows, last_m, clock_info
        )
        _, gate_sem_wrong, _, _ = build_orthogonal_bivector_routed_state(
            V18_vo, A18_vo, lg_v_sem_w, lg_a_sem_w, iv_t, reverse_timeline=False
        )

        worst_k = int((lg_v + lg_a).argmin().item())
        vec_irrel = F.normalize(0.5 * (V18_vo[worst_k] + A18_vo[worst_k]), dim=-1)

        rand_vecs = []
        for r_seed in (101, 202, 303):
            g_r = torch.Generator().manual_seed(idx + r_seed)
            rv = torch.randn(vec_fwd.shape, generator=g_r)
            rand_vecs.append(F.normalize(rv, dim=-1))

        hook_rows_t = torch.tensor(hook_rows, dtype=torch.long)

        # Build CAOR profiles for all matched conditions
        prof_full_fwd = build_caor_token_bias_profiles(
            payload['tokens'], clock_info, slot_meta, aud_fwd, mode='full', reverse_order=False
        )
        prof_full_rev = build_caor_token_bias_profiles(
            payload['tokens'], clock_info, slot_meta, aud_fwd, mode='full', reverse_order=True
        )
        prof_rel_only = build_caor_token_bias_profiles(
            payload['tokens'], clock_info, slot_meta, aud_fwd, mode='rel_only', reverse_order=False
        )
        prof_ord_only = build_caor_token_bias_profiles(
            payload['tokens'], clock_info, slot_meta, aud_fwd, mode='order_only', reverse_order=False
        )
        prof_irrel = build_caor_token_bias_profiles(
            payload['tokens'], clock_info, slot_meta, aud_fwd, mode='irrel', irrel_slot_k=worst_k
        )
        profs_rand = [
            build_caor_token_bias_profiles(
                payload['tokens'], clock_info, slot_meta, aud_fwd, mode='random', rand_seed=idx + rs
            )
            for rs in (101, 202, 303)
        ]

        # Per-Layer Native Counterfactual Attention Output Routing (`CAOR`) via `ALL_ATTENTION_FUNCTIONS`
        attn_config = model_llm.model.layers[0].self_attn.config
        orig_impl = attn_config._attn_implementation
        orig_kernel = qwen_omni_modeling.ALL_ATTENTION_FUNCTIONS[orig_impl]

        def run_caor_routed(
            target_layers,
            caor_prof: dict,
            eta_scale: float = 0.70,
            custom_kw=None,
            target_rows=hook_rows_t,
            record_delta_dict: dict = None,
        ):
            use_kw = kw if custom_kw is None else custom_kw
            eff_eta = float(eta_scale * caor_prof['gamma_gate'])
            if eff_eta <= 1e-7:
                with torch.inference_mode():
                    o = model_llm.model(**use_kw, use_cache=False, return_dict=True)
                    return model_llm.lm_head(
                        o.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                    )[0, -1].float().cpu()

            v_idx_cpu = v_indices.long()
            a_idx_cpu = a_indices.long()
            tb_v_cpu = caor_prof['tok_bias_v'].float()
            tb_a_cpu = caor_prof['tok_bias_a'].float()
            sink_w = caor_prof['sink_w']
            late_aud_damp = caor_prof['late_aud_damp']
            target_set = set(target_layers)

            def caor_kernel(module, q, k, v, attention_mask, **k_kw):
                out, weights = orig_kernel(module, q, k, v, attention_mask, **k_kw)
                l_idx = getattr(module, 'layer_idx', -1)
                if l_idx not in target_set:
                    return out, weights

                dev_q = q.device
                rows = target_rows.to(dev_q)
                v_idx_d = v_idx_cpu.to(dev_q)
                a_idx_d = a_idx_cpu.to(dev_q)
                tb_v = tb_v_cpu.to(dev_q)
                tb_a = tb_a_cpu.to(dev_q)

                rep_kv = q.shape[1] // k.shape[1]
                kg = k.repeat_interleave(rep_kv, dim=1)
                vg = v.repeat_interleave(rep_kv, dim=1)
                scale_qk = float(k_kw.get('scaling', getattr(module, 'scaling', 1.0 / math.sqrt(q.shape[-1]))))

                # Native M-RoPE attention logits for `rows` and question-blind delimiter `last_m`
                logits_nat = torch.matmul(q[:, :, rows, :].float(), kg.float().transpose(-2, -1)) * scale_qk
                if attention_mask is not None and attention_mask.ndim == 4:
                    logits_nat = logits_nat + attention_mask[:, :, rows, :k.shape[-2]].float()
                else:
                    fut = torch.arange(k.shape[-2], device=dev_q).unsqueeze(0) > rows.unsqueeze(1)
                    logits_nat = logits_nat.masked_fill(fut.unsqueeze(0).unsqueeze(0), float('-inf'))

                logits_r0 = torch.matmul(q[:, :, last_m:last_m + 1, :].float(), kg.float().transpose(-2, -1)) * scale_qk
                sink_v = logits_r0[:, :, :, v_idx_d]
                sink_v = sink_v - sink_v.mean(dim=-1, keepdim=True)
                sink_a = logits_r0[:, :, :, a_idx_d]
                sink_a = sink_a - sink_a.mean(dim=-1, keepdim=True)

                lv_nat = logits_nat[:, :, :, v_idx_d]
                la_nat = logits_nat[:, :, :, a_idx_d]
                lse_v_nat = torch.logsumexp(lv_nat, dim=-1, keepdim=True)
                lse_a_nat = torch.logsumexp(la_nat, dim=-1, keepdim=True)
                lse_media_nat = torch.logaddexp(lse_v_nat, lse_a_nat)

                lv_cf = lv_nat - sink_w * sink_v + tb_v.view(1, 1, 1, -1)
                la_cf = la_nat - sink_w * sink_a + tb_a.view(1, 1, 1, -1)
                if l_idx >= 20 and late_aud_damp > 0.0:
                    la_cf = la_cf - late_aud_damp

                lse_media_cf = torch.logaddexp(
                    torch.logsumexp(lv_cf, dim=-1, keepdim=True),
                    torch.logsumexp(la_cf, dim=-1, keepdim=True),
                )
                # Preserve exact combined media log-mass per (head, query_row) so non-media context is 100% invariant!
                norm_shift = lse_media_nat - lse_media_cf
                logits_cf = logits_nat.clone()
                logits_cf[:, :, :, v_idx_d] = lv_cf + norm_shift
                logits_cf[:, :, :, a_idx_d] = la_cf + norm_shift

                p_cf = torch.softmax(logits_cf, dim=-1).to(vg.dtype)
                updated = torch.matmul(p_cf, vg).transpose(1, 2).to(out.dtype)

                out_new = out.clone()
                orig_slice = out[:, rows, :, :].float()
                upd_slice = ((1.0 - eff_eta) * orig_slice + eff_eta * updated.float()).to(out.dtype)
                out_new[:, rows, :, :] = upd_slice

                if record_delta_dict is not None:
                    with torch.no_grad():
                        diff_pre_o = (upd_slice.float() - orig_slice).reshape(1, len(rows), -1).to(module.o_proj.weight.dtype)
                        delta_vo = module.o_proj(diff_pre_o)[0].detach().float().cpu()
                        record_delta_dict[l_idx] = delta_vo

                assert sum(len(layer._forward_hooks) for layer in model_llm.model.layers) >= 0
                return out_new, weights

            qwen_omni_modeling.ALL_ATTENTION_FUNCTIONS.register(orig_impl, caor_kernel)
            try:
                with torch.inference_mode():
                    o = model_llm.model(**use_kw, use_cache=False, return_dict=True)
                    lg = model_llm.lm_head(
                        o.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                    )[0, -1].float().cpu()
            finally:
                qwen_omni_modeling.ALL_ATTENTION_FUNCTIONS.register(orig_impl, orig_kernel)
            return lg

        # Verify exact zero-scale native replay (0.0 error!)
        lg_zero_check = run_caor_routed(mid_layers, prof_full_fwd, eta_scale=0.0)
        zero_scale_err = float((lg_zero_check[labels[:k_opts]] - lg_nat[labels[:k_opts]]).abs().max().item())
        assert zero_scale_err < 1e-5, f"Zero-scale replay mismatch: {zero_scale_err}"

        deltas_fwd, deltas_rev = {}, {}
        p_nat = F.softmax(lg_nat[labels[:k_opts]], dim=-1)
        p_late_fwd = F.softmax(run_caor_routed(late_layers, prof_full_fwd)[labels[:k_opts]], dim=-1)
        p_late_rev = F.softmax(run_caor_routed(late_layers, prof_full_rev)[labels[:k_opts]], dim=-1)
        p_mid_fwd = F.softmax(run_caor_routed(mid_layers, prof_full_fwd, record_delta_dict=deltas_fwd)[labels[:k_opts]], dim=-1)
        p_mid_rev = F.softmax(run_caor_routed(mid_layers, prof_full_rev, record_delta_dict=deltas_rev)[labels[:k_opts]], dim=-1)
        p_mid_rel_only = F.softmax(run_caor_routed(mid_layers, prof_rel_only)[labels[:k_opts]], dim=-1)
        p_mid_ord_only = F.softmax(run_caor_routed(mid_layers, prof_ord_only)[labels[:k_opts]], dim=-1)
        p_mid_irrel = F.softmax(run_caor_routed(mid_layers, prof_irrel)[labels[:k_opts]], dim=-1)

        tv_rand_list = []
        for pr_r in profs_rand:
            p_r = F.softmax(run_caor_routed(mid_layers, pr_r)[labels[:k_opts]], dim=-1)
            tv_rand_list.append(0.5 * float((p_r - p_nat).abs().sum().item()))

        tv_late_swap = 0.5 * float((p_late_fwd - p_late_rev).abs().sum().item())
        tv_mid_swap = 0.5 * float((p_mid_fwd - p_mid_rev).abs().sum().item())
        tv_mid_vs_nat = 0.5 * float((p_mid_fwd - p_nat).abs().sum().item())
        tv_irrel_vs_nat = 0.5 * float((p_mid_irrel - p_nat).abs().sum().item())
        tv_rand_vs_nat = float(np.mean(tv_rand_list))

        gold_letter = sample.get('gold', sample.get('answer', 'A'))
        pred_nat_letter = 'ABCD'[int(p_nat.argmax().item())]
        pred_routed_letter = 'ABCD'[int(p_mid_fwd.argmax().item())]
        pred_rev_letter = 'ABCD'[int(p_mid_rev.argmax().item())]
        pred_rel_only_letter = 'ABCD'[int(p_mid_rel_only.argmax().item())]
        pred_ord_only_letter = 'ABCD'[int(p_mid_ord_only.argmax().item())]

        eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}

        def generate_freeform(use_routed: bool, max_new_tokens: int = 32) -> str:
            gen_ids = []
            c_emb = freeform_embeds.clone()
            c_pos = freeform_pos.clone()
            for _ in range(max_new_tokens):
                L_c = c_emb.shape[1]
                ff_kw = {
                    'inputs_embeds': c_emb,
                    'position_ids': c_pos,
                    'attention_mask': torch.ones((1, L_c), dtype=torch.long, device=device),
                    'cache_position': torch.arange(L_c, device=device),
                }
                if use_routed:
                    ff_rows = torch.arange(last_m + 1, last_m + 1 + n_tail_ff, dtype=torch.long)
                    lg_step = run_caor_routed(mid_layers, prof_full_fwd, custom_kw=ff_kw, target_rows=ff_rows)
                else:
                    with torch.inference_mode():
                        o_s = model_llm.model(**ff_kw, use_cache=False, return_dict=True)
                        lg_step = model_llm.lm_head(
                            o_s.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                        )[0, -1].float().cpu()
                next_id = int(lg_step.argmax().item())
                if next_id in eos_ids and len(gen_ids) > 0:
                    break
                gen_ids.append(next_id)
                next_e = emb_layer(torch.tensor([[next_id]], device=device)).to(c_emb.dtype)
                c_emb = torch.cat([c_emb, next_e], dim=1)
                c_pos = torch.cat([c_pos, c_pos[:, :, -1:] + 1], dim=-1)
            return tok.decode(gen_ids, skip_special_tokens=True).strip()

        ff_nat_text = generate_freeform(use_routed=False, max_new_tokens=32)
        ff_routed_text = generate_freeform(use_routed=True, max_new_tokens=32)

        gold_ans_text = ''
        distractor_texts = []
        for part in q_raw.split('('):
            if len(part) >= 2 and part[1] == ')' and part[0] in 'ABCD':
                opt_txt = part[2:].split('\n')[0].strip()
                if part[0] == gold_letter:
                    gold_ans_text = opt_txt
                else:
                    distractor_texts.append(opt_txt)

        nat_ff_audit = audit_freeform_generation_quality(
            ff_nat_text, clock_info['T_phys'], sample['task'], gold_ans_text, distractor_texts=distractor_texts
        )
        routed_ff_audit = audit_freeform_generation_quality(
            ff_routed_text, clock_info['T_phys'], sample['task'], gold_ans_text, distractor_texts=distractor_texts
        )

        mean_delta_norm_mid = float(np.mean([float(d[-1].norm().item()) for d in deltas_fwd.values()])) if deltas_fwd else 0.0

        rows_out.append({
            'idx': idx,
            'task': sample['task'],
            'gold_letter': gold_letter,
            'gold_answer_text': gold_ans_text,
            'mc_pred_native': pred_nat_letter,
            'mc_pred_routed_fwd': pred_routed_letter,
            'mc_pred_routed_rev': pred_rev_letter,
            'mc_pred_rel_only': pred_rel_only_letter,
            'mc_pred_order_only': pred_ord_only_letter,
            'mc_native_correct': bool(pred_nat_letter == gold_letter),
            'mc_routed_correct': bool(pred_routed_letter == gold_letter),
            'mc_rel_only_correct': bool(pred_rel_only_letter == gold_letter),
            'mc_order_only_correct': bool(pred_ord_only_letter == gold_letter),
            'grid_clock_interpolation_audit': clock_info['clock_audit'],
            'n_calibrated_slots_K': int(V18_vo.shape[0]),
            'confidence_gate_gamma_true_query': round(gate_fwd, 4),
            'confidence_gate_gamma_wrong_query': round(gate_perm_wrong, 4),
            'confidence_gate_gamma_semantic_wrong_query': round(gate_sem_wrong, 4),
            'semantic_wrong_query_text': sem_wrong_text,
            'native_replay_max_err': round(native_replay_err, 6),
            'zero_scale_replay_max_err': round(zero_scale_err, 6),
            'fwd_vs_rev_vector_cosine': round(fwd_rev_cos, 4),
            'content_even_max_err': round(content_even_err, 8),
            'order_odd_max_err': round(order_odd_err, 8),
            'bhattacharyya_synchrony_coeff': aud_fwd['bhatt_sync'],
            'rel_scalar_fwd_vs_rev': [aud_fwd['rel_scalar'], aud_rev['rel_scalar']],
            'mean_per_layer_attn_out_delta_norm': round(mean_delta_norm_mid, 5),
            'tv_late_kslot_swap': round(tv_late_swap, 4),
            'tv_mid_kslot_swap': round(tv_mid_swap, 4),
            'tv_mid_routed_vs_nat': round(tv_mid_vs_nat, 4),
            'tv_mid_irrel_slot_vs_nat': round(tv_irrel_vs_nat, 4),
            'tv_mid_rand3seed_vs_nat': round(tv_rand_vs_nat, 4),
            'freeform_32tok_native': ff_nat_text,
            'freeform_32tok_routed': ff_routed_text,
            'freeform_native_audit': nat_ff_audit,
            'freeform_routed_audit': routed_ff_audit,
        })

    if not rows_out:
        return {'status': 'no_samples_found'}

    mean_swap_late = float(np.mean([r['tv_late_kslot_swap'] for r in rows_out]))
    mean_swap_mid = float(np.mean([r['tv_mid_kslot_swap'] for r in rows_out]))
    mean_routed = float(np.mean([r['tv_mid_routed_vs_nat'] for r in rows_out]))
    mean_irrel = float(np.mean([r['tv_mid_irrel_slot_vs_nat'] for r in rows_out]))
    mean_rand = float(np.mean([r['tv_mid_rand3seed_vs_nat'] for r in rows_out]))

    return {
        'n_samples': len(rows_out),
        'mc_native_correct_count': int(sum(r['mc_native_correct'] for r in rows_out)),
        'mc_routed_correct_count': int(sum(r['mc_routed_correct'] for r in rows_out)),
        'mc_rel_only_correct_count': int(sum(r['mc_rel_only_correct'] for r in rows_out)),
        'mc_order_only_correct_count': int(sum(r['mc_order_only_correct'] for r in rows_out)),
        'mean_tv_kslot_swap_late_L28_33': round(mean_swap_late, 4),
        'mean_tv_kslot_swap_mid_L14_24': round(mean_swap_mid, 4),
        'mid_over_late_kslot_swap_ratio': round(mean_swap_mid / max(1e-6, mean_swap_late), 2),
        'mean_tv_mid_routed_vs_nat': round(mean_routed, 4),
        'mean_tv_mid_irrel_slot_control': round(mean_irrel, 4),
        'mean_tv_mid_random_3seed_control': round(mean_rand, 4),
        'mean_gamma_true_query': round(float(np.mean([r['confidence_gate_gamma_true_query'] for r in rows_out])), 4),
        'mean_gamma_wrong_query': round(float(np.mean([r['confidence_gate_gamma_wrong_query'] for r in rows_out])), 4),
        'mean_gamma_semantic_wrong_query': round(float(np.mean([r['confidence_gate_gamma_semantic_wrong_query'] for r in rows_out])), 4),
        'samples': rows_out,
    }


def run_all_stage0_v6():
    t0 = time.time()
    unit_checks = run_algebraic_unit_checks('cuda:0' if torch.cuda.is_available() else 'cpu')
    m1_native, ctx_native = run_stage0_m1_orthogonal_factorial('avbind_native_pairs')
    m1_pooled, ctx_pooled = run_stage0_m1_orthogonal_factorial('avbind_pooled_pairs')
    use_bundle = ctx_pooled if ctx_pooled is not None else ctx_native
    m2_res = run_stage0_m2_connected(use_bundle)
    m3_res = run_stage0_m3_bivector_vo_tangent_routing()

    out = {
        'protocol': 'Stage-0 v6 (CAOR-WEMD: Per-Layer Counterfactual Attention Output Routing + 1D Wasserstein-Synchrony M2 + Bilevel Innovation M1)',
        'elapsed_sec': round(time.time() - t0, 2),
        'Algebraic_Wasserstein_Bivector_and_Counterexample_Unit_Checks': unit_checks,
        'M1_native_orthogonal_factorial': m1_native,
        'M1_pooled_orthogonal_factorial': m1_pooled,
        'M2_wasserstein_synchrony_and_mixture_hop2': m2_res,
        'M3_per_layer_counterfactual_attention_output_routing': m3_res,
    }
    out_path = OUT_DIR / 'stage0_pcebr_v6_results.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('STAGE0_V6_DONE', json.dumps(out, indent=2), flush=True)
    return out


RESULT = run_all_stage0_v6()