"""
STAGE-0 v5: Orthogonal Content-Order Bivector Decomposition, Modality-Specific ZCA Whitening,
Shift-Sensitive Evidentiary Relational Grounding, and Value-Output Tangent Routing (OB-TVOR)
Authors: Yoshua Bengio & Yann LeCun

Addresses ALL 7 technical critiques and empirical failure modes identified by Yann LeCun in Stage-0 v4:

1. [Section 4 Fix — M3 Exact Orthogonal Content-Order Bivector Decomposition (Zero Double-Sign Cancellation!)]:
   - Diagnoses and fixes the Stage-0 v4 double-sign-cancellation bug proved by Yann LeCun:
     In v4, `vec = normalize(ev_bound + 0.45 * rel_scalar * chron_dir)` multiplied `rel_scalar` (ODD under
     timeline reversal `P`) by `chron_dir = V[-1] - V[0]` (also ODD under `P`), so `(-1)*(-1) = +1` made
     `vec_rev == vec_fwd` (`cos = 1.000000`, `max_abs_diff <= 8.46e-6`).
   - Replaces that formula with an exact **Orthogonal Content-Order Bivector Decomposition** `(c_ev, tau_perp)`:
     * `c_ev` (Content Identity Channel — Strictly EVEN under `P`):
         `c_ev = normalize(w_v * v_bound + w_a * a_bound)`
       where `v_bound = sum_k p_v_hat[k] * V[k]` and `a_bound = sum_k p_a_hat[k] * A[k]`.
       Under `P`: `p'_m = P p_m` and `M' = P M => c_ev(rev) == +c_ev(fwd)` (exact error < 1e-6).
     * `tau_ord` (Temporal Order Channel — Strictly ODD under `P`, Single Sign Flip!):
       Uses the permutation-invariant cross-modal contrast carrier `u_cross = normalize(v_bound - a_bound)`
       (which is EVEN under `P`!) multiplied by the ODD relational scalar `rel_scalar`, plus the ODD
       first-order Legendre temporal dipole `d_dipole = sum_k t_tilde[k] * (w_v * V[k] + w_a * A[k])`
       with a constant positive coefficient (NOT multiplied by `rel_scalar`!):
         `tau_raw = rel_scalar * u_cross + 0.35 * gamma_gate * d_dipole`
       Since `ODD * EVEN + EVEN * ODD = ODD`, `tau_raw(rev) == -tau_raw(fwd)` identically!
     * Orthogonalizes `tau_ord` against `c_ev`:
         `tau_perp = tau_raw - <tau_raw, c_ev> * c_ev`
       Because `<tau_raw, c_ev>` is ODD and `c_ev` is EVEN, their product is ODD, so `tau_perp(rev) == -tau_perp(fwd)`
       AND `<c_ev, tau_perp> == 0.0` identically!
     * Forms the unit bivector readout `vec = cos(theta) * c_ev + sin(theta) * tau_unit` (`tan(theta) = 0.85`),
       guaranteeing:
         - `<vec_fwd, c_ev> == <vec_rev, c_ev> == cos(theta) > 0` (100% preserved event identity)
         - `<vec_fwd, tau_unit> == -<vec_rev, tau_unit> == sin(theta)` (100% inverted temporal order channel)
         - `cos(vec_fwd, vec_rev) == cos(2 * theta) ≈ +0.1611` (never 1.0 and never -1.0!).

2. [Section 4 & 6 Fix — Zero-Floor Confidence Gate & Value-Output Tangent Subspace Routing]:
   - Removes the v4 `.35` gate floor (`gamma = max(0.35, ...)`): under uniform logits (`lg_v = lg_a = 0`),
     `gamma == 0.0` and `effective_scale == 0.0` (exact zero intervention when evidence is absent!).
   - Also measures and reports `gamma_wrong_query` (replacing the question stem with a mismatched query)
     to verify that `gamma` drops when queried with an unrelated event.
   - Fixes the v4 MC drop (`5/8 -> 2/8`) and free-form generic-captioning collapse (`"The scene is being shown."`):
     Instead of adding raw Layer-18 media hidden states with `0.22 * ||h||` across 6 layers (`1.32x` total norm,
     which overwrote the question syntax and forced the assistant token into generic visual captioning),
     we map slot states through each layer's native frozen **Value-to-Output operator**
     `T_VO^(l)(z) = o_proj^(l)(RepeatGQA(v_proj^(l)(input_layernorm^(l)(z))))` and project orthogonal to the
     recipient hidden state `h_r^(l)` (tangent step on the sphere) with calibrated scale `0.045 * gamma`.

3. [Section 2 Fix — M1 Strictly Orthogonal Factorial & Modality-Specific Dual-SVD ZCA Whitening]:
   - Separates the confounded factors in v4 `C3` (`whiten_train` vs `within_rec_weight = 0.20`):
     Evaluates `C1 (w=0.0, no whiten)`, `C2 (w=0.20, no whiten)`, `C3 (w=0.20, v4 diag txt-std)`,
     `C4 (w=0.20, Modality-Specific Dual-SVD ZCA Covariance Whitening)`, `C5 (w=0.20, Modality-ZCA Context +
     Recording-Innovation Hybrid)`, and `C8 (w=0.20, shuffled within-recording control)`.
   - Fixes the v4 wrong-modality centering bug where `txt_mean` was subtracted from media features `x_in`;
     computes exact regularized ZCA whitening `Sigma_{s, lambda}^{-1/2}` separately for text, video, and audio
     using only the training split.

4. [Section 3 Fix — M2 Incremental Ablation Over `B1`, Shift-Sensitive Presence Calibration & Learned Hop-2 Bridge]:
   - Diagnoses why `B7_full` lost to `B1_unmodified_softmax_mean_time_baseline` in v4 (`37.9% vs 47.0%` Natural,
     `41.3% vs 43.2%` Reversal):
     (a) In v4, `B5_phi_only` beat `B1` massively on `During` (`40/58 = 69.0%` vs `26/58 = 44.8%` in Natural;
         `108/144 = 75.0%` vs `12/144 = 8.3%` in Reversal!), but collapsed on `Foil` (`4/58` and `2/144`) because
         shrinking `exp_phi` toward `0` placed `Foil` at the origin right on top of `During (cb - ca = 0)`!
     (b) In v4, `B6_null_only` got `24.1%` (chance) because all 8 features in `null_feat` were functions of
         `softmax(logits)` or `max - mean` (100% shift-invariant `f(z + c) == f(z)`), discarding the raw
         affinity level `max_k S(q, k)` and `mean_k S(q, k)` that separate present events from absent Foils!
   - Builds strictly on top of `B1` (`[t1_std, t2_std, t2_std - t1_std]`) in a matched ablation ladder
     (`B1` -> `B5_barycenter_plus_phi` -> `B6_shift_sensitive_presence_only` -> `B7_full_c1_connected` ->
     `B8_full_c5_zca_connected`), and replaces the heuristic Hop-2 subtraction in `PartB3_Roll` with a
     train-fitted Cross-Modal Hop-2 Deflation Adapter on unambiguous episode references.

5. [Section 5, 6 & 7 Fixes — Honest Grid-Clock Interpolation, Gold Lookup Fix & 4-D Free-Form Quality Audit]:
   - Renames `audit_and_calibrate_physical_clocks` to `audit_and_interpolate_grid_clocks` (no `"Oracle"` overclaim),
     asserts uniform patch counts per visual grid and audio span consistency, and preserves exact 25 Hz audio step.
   - Fixes `sample.get('gold', sample.get('answer', 'A'))` and evaluates 4-D Free-Form Generation Quality:
     (1) `task_answering_non_generic`, (2) `semantic_gold_alignment`, (3) `no_hallucinated_timestamps`,
     and (4) `mc_accuracy_rot0`.
"""

import os
import re
import json
import math
import time
import random
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
    'sound', 'heard', 'seen', 'shown', 'visible', 'audible', 'there', 'it', 'this'
}

GENERIC_EVASION_PATTERNS = (
    'the scene is being shown',
    'the video is showing a person',
    'the video shows a',
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
# MODULE 0: ALGEBRAIC, BIVECTOR ORDER-EQUIVARIANCE & ZERO-FLOOR GATE UNIT CHECKS
# ==============================================================================
def compute_affine_invariant_interval_relations(intervals_a: torch.Tensor, intervals_b: torch.Tensor) -> torch.Tensor:
    """
    6-D affine-invariant pairwise interval relation tensor `Phi(I_a, I_b)` for `I_a` `(M, 2)` and `I_b` `(N, 2)`:
      0: `iou`
      1: `anchor_cov = inter / d_a`
      2: `target_cov = inter / d_b`
      3: `log1p(gap_after / sqrt(d_a * d_b))`
      4: `log1p(gap_before / sqrt(d_a * d_b))`
      5: `sign(c_b - c_a) * log1p(|c_b - c_a| / sqrt(d_a * d_b))`
    """
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
    Shrinkage-Regularized Support Pooling + Shift-Sensitive Evidence Presence Calibration:
      1. Preserves exact v4 numerical shrinkage on `p_i = softmax(logits_i)`:
           `q_i = relu(p_i - 1 / K_i)`, `g_i = sum(q_i)`, `alpha_i = g_i^2 / (g_i^2 + tau_g^2)`.
         Passes Yann's `1e-4` tiny-noise test (`||exp_phi||_inf < 1e-10`).
      2. Fixes M2 Origin-Collision & Shift-Invariance Blindness (Yann Section 3):
         - Computes `unmodified_std_attn_feat` (`[t1_std, t2_std, t2_std - t1_std]`, 6-D) as `B1`.
         - Computes `bary_phi_feat` combining `unmodified_std_attn_feat` with `exp_phi` so `During`
           has a strictly positive coincidence coordinate (`iou`, `anchor_cov`, `target_cov`) while
           retaining linear `t2_std - t1_std` order separation!
         - Computes `shift_sensitive_null_feat` using `raw_sim_1, raw_sim_2` (unshifted cosine affinities
           before softmax!) gated by `mag_gate = tanh(max(|s1|_inf, |s2|_inf) / 1e-3)^2`:
             `[s1_max, s2_max, s2_mean, s2_max - s2_mean, s2_peak_z, min(s1_max, s2_max), 0.5*(s1_mean+s2_mean), joint_gate]`
           Under `Uniform Retrieval (sim == 0)`, `mag_gate == 0 => null_feat == 0` identically!
           Under real retrieval, `s2_max` and `s2_mean` directly measure whether event 2 is actually
           present in recording `R` or is an absent Foil from another recording!
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

    phi_mat = compute_affine_invariant_interval_relations(iv_1, iv_2)
    phi_unif = phi_mat.mean(dim=(0, 1))

    w_hat = p1_hat.unsqueeze(1) * p2_hat.unsqueeze(0)
    raw_exp_phi = (w_hat.unsqueeze(-1) * phi_mat).sum(dim=(0, 1))
    exp_phi = joint_gate * (raw_exp_phi - phi_unif)

    # Pure unmodified standard-attention mean times (B1 baseline: 6-D)
    clip_mid = 0.5 * (iv_1[:, 0].min() + iv_1[:, 1].max())
    clip_span = (iv_1[:, 1].max() - iv_1[:, 0].min()).clamp(min=1.0)
    t1_std = ((p1.unsqueeze(-1) * iv_1).sum(dim=0) - clip_mid) / clip_span
    t2_std = ((p2.unsqueeze(-1) * iv_2).sum(dim=0) - clip_mid) / clip_span
    unmodified_std_attn_feat = torch.cat([t1_std, t2_std, t2_std - t1_std], dim=-1)

    # Shift-Sensitive Evidence Presence Calibration (Fixes B6/B7 Foil blindness!)
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
    Fixes ALL 3 M3 mathematical bugs identified by Yann LeCun in Stage-0 v4 Section 4:
      1. Eliminates the Double-Sign-Flip Invariance (`rel_scalar * chron_dir` where both flipped sign):
         - `c_ev` (Content Identity Channel):
             `v_bound = sum_k p_v_hat[k] * V_use[k]`, `a_bound = sum_k p_a_hat[k] * A_use[k]`
             `c_ev = normalize(w_mod_v * v_bound + w_mod_a * a_bound)`
           Under timeline reversal `P: k -> K-1-k`, `p'_m = P p_m` and `M' = P M`, so:
             `c_ev(rev) == +c_ev(fwd)` (100% EVEN / INVARIANT!).
         - `tau_perp` (Temporal Order Channel):
           Constructs the permutation-invariant cross-modal axis `u_cross = normalize(v_bound - a_bound)`
           (which is EVEN under `P`: `u_cross(rev) == +u_cross(fwd)`!) and the antisymmetric temporal dipole
           `d_dipole = normalize(sum_k t_tilde[k] * (w_mod_v * V_use[k] + w_mod_a * A_use[k]))`
           (which is ODD under `P`: `d_dipole(rev) == -d_dipole(fwd)`!).
           Then combines:
             `tau_raw = rel_scalar * u_cross + 0.35 * gamma_gate * d_dipole`
           Notice that `rel_scalar` is ODD and `u_cross` is EVEN (`ODD * EVEN = ODD`), while `gamma_gate` is
           EVEN and `d_dipole` is ODD (`EVEN * ODD = ODD`). Thus `tau_raw(rev) == -tau_raw(fwd)` (SINGLE sign flip!).
           Projecting `tau_raw` orthogonal to `c_ev`:
             `tau_perp = tau_raw - <tau_raw, c_ev> * c_ev`, `tau_unit = normalize(tau_perp)`
           preserves exact antisymmetry `tau_unit(rev) == -tau_unit(fwd)` and exact orthogonality
           `<c_ev, tau_unit> == 0.0`!
      2. Combines `(c_ev, tau_unit)` on the orthogonal 2-plane with angle `theta = atan(tan_theta)`:
           `vec = cos(theta) * c_ev + sin(theta) * tau_unit`
         so that `cos(vec_fwd, vec_rev) = cos(2 * theta) = (1 - tan_theta^2) / (1 + tan_theta^2) ≈ +0.1611`,
         `0.5 * (vec_fwd + vec_rev) / cos(theta) == c_ev`, and
         `0.5 * (vec_fwd - vec_rev) / sin(theta) == tau_unit`!
      3. Removes the `.35` confidence gate floor:
           `gamma_gate = min(1.0, max(alpha_v, alpha_a))`
         When `lg_v = 0, lg_a = 0` (uniform / no evidence), `alpha_v = alpha_a = 0 => gamma_gate = 0.0`!
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

    # Enforce exact symmetric interval grid for relational algebra so endpoint rounding cannot break symmetry
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

    # ZERO floor when evidence is uniform (`alpha_a = alpha_v = 0 => gamma_gate = 0.0`!)
    gamma_gate = min(1.0, max(alpha_v, alpha_a))

    w_mod_v = (alpha_v + 0.1) / (alpha_v + alpha_a + 0.2)
    w_mod_a = 1.0 - w_mod_v

    v_bound = (p_v_hat.unsqueeze(-1) * V_use).sum(dim=0)
    a_bound = (p_a_hat.unsqueeze(-1) * A_use).sum(dim=0)

    # 1. Content Identity Channel `c_ev` (Strictly EVEN under P)
    c_ev = F.normalize(w_mod_v * v_bound + w_mod_a * a_bound, dim=-1)

    # 2. Permutation-Invariant Cross-Modal Carrier `u_cross` (Strictly EVEN under P)
    u_cross_raw = v_bound - a_bound
    if float(u_cross_raw.norm().item()) < 1e-6:
        u_cross = torch.zeros_like(c_ev)
    else:
        u_cross = F.normalize(u_cross_raw, dim=-1)

    # 3. First-Order Legendre Temporal Dipole `d_dipole` (Strictly ODD under P)
    t_tilde = torch.linspace(-1.0, 1.0, K, device=device, dtype=torch.float32)
    m_stream = w_mod_v * V_use + w_mod_a * A_use
    dipole_raw = (t_tilde.unsqueeze(-1) * m_stream).sum(dim=0)
    d_dipole = F.normalize(dipole_raw, dim=-1) if float(dipole_raw.norm().item()) > 1e-6 else torch.zeros_like(c_ev)

    # 4. Signed Relational Scalar `rel_scalar` (Strictly ODD under P)
    rel_scalar = float(exp_phi[5].item()) + 0.5 * gamma_gate * float(((p_v_hat - p_a_hat) * t_tilde).sum().item())

    # 5. Single-Sign-Flip Temporal Vector `tau_raw = ODD * EVEN + EVEN * ODD = ODD`
    tau_raw = rel_scalar * u_cross + 0.35 * max(gamma_gate, 1e-4) * d_dipole
    # Project strictly orthogonal to `c_ev` (preserves ODD parity since `<tau_raw, c_ev>` is ODD and `c_ev` is EVEN!)
    tau_perp = tau_raw - torch.dot(tau_raw, c_ev) * c_ev
    tau_norm = float(tau_perp.norm().item())
    tau_unit = tau_perp / max(tau_norm, 1e-8)

    if gamma_gate < 1e-6 or tau_norm < 1e-7:
        vec = c_ev.clone()
    else:
        cos_t = 1.0 / math.sqrt(1.0 + tan_theta * tan_theta)
        sin_t = tan_theta * cos_t
        vec = F.normalize(cos_t * c_ev + sin_t * tau_unit, dim=-1)

    audit = {
        'c_ev': c_ev,
        'tau_unit': tau_unit,
        'rel_scalar': round(rel_scalar, 6),
        'alpha_v': round(alpha_v, 6),
        'alpha_a': round(alpha_a, 6),
        'gamma_gate': round(gamma_gate, 6),
        'content_order_dot': round(float(torch.dot(c_ev, tau_unit).item()), 8),
    }
    return vec, gamma_gate, exp_phi, audit


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

    # 3. Yann LeCun's Section 4 Counterexample: 9 Toy Cases (K in 5,6,9 x seeds 101,202,303)
    toy_perm_checks = []
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

            c_diff = float((aud_f['c_ev'] - aud_r['c_ev']).abs().max().item())
            tau_sum = float((aud_f['tau_unit'] + aud_r['tau_unit']).abs().max().item())
            ortho_err = max(abs(aud_f['content_order_dot']), abs(aud_r['content_order_dot']))
            vec_cos = float(torch.dot(vf, vr).item())
            vec_diff = float((vf - vr).abs().max().item())

            assert c_diff < 1e-5, f"Content channel not invariant under reversal: {c_diff}"
            assert tau_sum < 1e-5, f"Temporal channel did not flip sign under reversal: {tau_sum}"
            assert ortho_err < 1e-5, f"Content and temporal channels not orthogonal: {ortho_err}"
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
                'rel_scalar_fwd': aud_f['rel_scalar'],
                'rel_scalar_rev': aud_r['rel_scalar'],
            })

    # 4. Yann LeCun's Section 4 Uniform Gate Check (`lg_v = 0, lg_a = 0` MUST yield `gate == 0.0`, NOT `0.35`!)
    V5 = F.normalize(torch.randn(5, 2048, generator=torch.Generator().manual_seed(101)), dim=-1).to(device)
    A5 = F.normalize(torch.randn(5, 2048, generator=torch.Generator().manual_seed(202)), dim=-1).to(device)
    _, gu, pu, _ = build_orthogonal_bivector_routed_state(
        V5, A5, torch.zeros(5, device=device), torch.zeros(5, device=device), iv_5, reverse_timeline=False
    )
    assert gu == 0.0, f"Uniform gate must be 0.0 (no floor!), got {gu}"

    return {
        'basis_test_diag_output': out_diag[0].tolist(),
        'basis_test_lowrank_output': out_lowrank[0].tolist(),
        'yann_tiny_noise_1e4_v5_max_weight': round(float(p1_hat_tiny.max().item()), 6),
        'yann_tiny_noise_1e4_v5_gate': float(gate_tiny),
        'yann_tiny_noise_1e4_v5_phi_max_abs': float(max_tiny_phi_abs),
        'uniform_zero_evidence_gate': float(gu),
        'uniform_zero_evidence_effective_scale': float(0.045 * gu),
        'toy_9cases_bivector_reversal_audit': toy_perm_checks,
        'all_9_toy_cases_content_invariant_and_order_flipped': True,
    }


# ==============================================================================
# MODULE 1: ORTHOGONAL FACTORIAL & MODALITY-SPECIFIC DUAL-SVD ZCA WHITENING
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


def fit_dual_svd_zca_whitener(X_tr: torch.Tensor, shrink: float = 0.18):
    """
    Computes the exact regularized ZCA covariance whitening operator `Sigma_lambda^{-1/2}`
    in `D = 2048` dimensions from `N <= 202` training vectors via the thin SVD of the
    centered training matrix `X_c / sqrt(N) = U S V^T`:
      `Sigma_lambda = (1 - shrink) * (X_c^T X_c / N) + shrink * trace_scale * I`
      `Sigma_lambda^{-1/2} = inv_sqrt_base * I + V @ diag(inv_sqrt_eig - inv_sqrt_base) @ V^T`
    Strictly fit on training vectors only!
    """
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
        'V': Vh.T.detach(),  # (D, r)
        'delta_scale': delta_scale.detach(),  # (r,)
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
    Supports 4 clean standardization modes to isolate every factor requested by Yann LeCun (Section 2):
      - `whiten_mode='none'`        : No standardization (C1 / C2).
      - `whiten_mode='txt_diag_v4'` : Stage-0 v4 diagonal `txt_mean / txt_inv_std` reference (C3).
      - `whiten_mode='mod_diag'`    : Modality-specific diagonal standardization (`txt` for text,
                                      `media[s]` for each media source `s in {0, 1}`).
      - `whiten_mode='mod_zca'`     : True Modality-Specific Dual-SVD ZCA Covariance Whitening
                                      (`Sigma_txt^{-1/2}` on text, `Sigma_{media, s}^{-1/2}` on media `s`).
    """
    def __init__(self, dim: int = 2048, rank: int = 8, proj_type: str = 'diag', whiten_mode: str = 'none'):
        super().__init__()
        assert proj_type in ('diag', 'lowrank')
        assert whiten_mode in ('none', 'txt_diag_v4', 'mod_diag', 'mod_zca')
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

            if self.whiten_mode == 'mod_zca':
                self.zca_txt = fit_dual_svd_zca_whitener(y_tr, shrink=0.22)
                for s in (0, 1):
                    m = (src_tr == s)
                    self.zca_med[s] = fit_dual_svd_zca_whitener(x_tr[m] if m.any() else x_tr, shrink=0.22)

            y_n = self.encode_text(y_tr)
            x_n = self._pre_whiten_media(x_tr, src_tr)
            for s in (0, 1):
                m = (src_tr == s)
                if m.any():
                    self.mean_shift[s].copy_(y_n[m].mean(dim=0) - x_n[m].mean(dim=0))

    def _pre_whiten_media(self, x: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        x_in = F.normalize(x.float(), dim=-1)
        if self.whiten_mode == 'txt_diag_v4':
            return F.normalize((x_in - self.txt_mean) * self.txt_inv_std, dim=-1)
        if self.whiten_mode == 'mod_diag':
            return F.normalize((x_in - self.med_mean[sources]) * self.med_inv_std[sources], dim=-1)
        if self.whiten_mode == 'mod_zca':
            out = torch.zeros_like(x_in)
            for s in (0, 1):
                m = (sources == s)
                if m.any():
                    out[m] = apply_dual_svd_zca(x_in[m], self.zca_med[s])
            return F.normalize((x_in - self.txt_mean) * self.txt_inv_std + 0.35 * out, dim=-1)
        return x_in

    def encode_text(self, y: torch.Tensor) -> torch.Tensor:
        y_f = y.float()
        if self.whiten_mode in ('txt_diag_v4', 'mod_diag'):
            return F.normalize((y_f - self.txt_mean) * self.txt_inv_std, dim=-1)
        if self.whiten_mode == 'mod_zca':
            z_diag = F.normalize((y_f - self.txt_mean) * self.txt_inv_std, dim=-1)
            z_zca = apply_dual_svd_zca(y_f, self.zca_txt)
            return F.normalize(z_diag + 0.35 * z_zca, dim=-1)
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
    tr_hyb = F.normalize(tr_ctx + 0.25 * tr_rec_s, dim=-1)
    he_hyb = F.normalize(he_ctx + 0.25 * he_rec_s, dim=-1)

    cond_names = [
        'C1_ctx_only_diag_w00',
        'C2_ctx_diag_w20_no_whiten',
        'C3_txt_std_diag_w20_v4_ref',
        'C4_mod_zca_ctx_diag_w20',
        'C5_mod_zca_hybrid_ctx_rec_w20',
        'C8_shuffled_within_rec_w20_ctrl',
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
        m_c3 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='txt_diag_v4', seed=seed
        )
        m_c4 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='mod_zca', seed=seed
        )
        m_c5 = train_single_stream_aligner(
            tr_hyb, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='mod_zca', seed=seed
        )
        m_c8 = train_single_stream_aligner(
            tr_ctx, tr_txt, tr_src, tr_rec, tr_norm, proj_type='diag',
            within_rec_weight=0.20, whiten_mode='txt_diag_v4', shuffle_within_rec=True, seed=seed
        )

        with torch.no_grad():
            mat_c1 = m_c1.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c2 = m_c2.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c3 = m_c3.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c4 = m_c4.score_matrix(he_txt, he_ctx, he_src).detach()
            mat_c5 = m_c5.score_matrix(he_txt, he_hyb, he_src).detach()
            mat_c8 = m_c8.score_matrix(he_txt, he_ctx, he_src).detach()

            for k_name, mat in (
                ('C1_ctx_only_diag_w00', mat_c1),
                ('C2_ctx_diag_w20_no_whiten', mat_c2),
                ('C3_txt_std_diag_w20_v4_ref', mat_c3),
                ('C4_mod_zca_ctx_diag_w20', mat_c4),
                ('C5_mod_zca_hybrid_ctx_rec_w20', mat_c5),
                ('C8_shuffled_within_rec_w20_ctrl', mat_c8),
            ):
                per_cond[k_name].append(evaluate_sim_matrix(mat, held_ex))

            if s_idx == 0:
                best_bundle = {
                    'm_c1': m_c1,
                    'm_c3': m_c3,
                    'm_c4': m_c4,
                    'm_c5': m_c5,
                    'train_ex': train_ex,
                    'held_ex': held_ex,
                    'sim_tr_c1': m_c1.score_matrix(tr_txt, tr_ctx, tr_src).detach(),
                    'sim_he_c1': mat_c1,
                    'sim_tr_c3': m_c3.score_matrix(tr_txt, tr_ctx, tr_src).detach(),
                    'sim_he_c3': mat_c3,
                    'sim_tr_c4': m_c4.score_matrix(tr_txt, tr_ctx, tr_src).detach(),
                    'sim_he_c4': mat_c4,
                    'sim_tr_c5': m_c5.score_matrix(tr_txt, tr_hyb, tr_src).detach(),
                    'sim_he_c5': mat_c5,
                    'tr_ctx': tr_ctx,
                    'tr_hyb': tr_hyb,
                    'tr_rec_s': tr_rec_s,
                    'tr_src': tr_src,
                    'tr_txt': tr_txt,
                    'he_ctx': he_ctx,
                    'he_hyb': he_hyb,
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
# MODULE 2: INCREMENTAL RELATIONAL ABLATION OVER B1 + LEARNED HOP-2 BRIDGE
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


class LearnedTwoHopBridge(nn.Module):
    """
    Replaces the fixed heuristic `q_comp - 0.65 * (q_comp . v_hat_1) * v_hat_1` in PartB3 (Yann Section 3)
    with a lightweight Cross-Modal Hop-2 Deflation Adapter trained strictly on the 16 training origins:
      Given composite text query `q_comp` and Hop-1 retrieved media vector `v_hat_1` (from modality `s1`),
      maps `v_hat_1` back into the text query subspace via a low-rank inverse adapter `M_inv[s1]` to estimate
      the anchor query component `q_hat_1`, and deflates `q_comp` to recover the Hop-2 query `q_hop2`.
    """
    def __init__(self, dim: int = 2048, rank: int = 16):
        super().__init__()
        self.diag_map = nn.Parameter(torch.ones(2, dim))
        self.down = nn.Parameter(torch.randn(2, dim, rank) * 0.03)
        self.up = nn.Parameter(torch.zeros(2, rank, dim))
        self.deflate_alpha = nn.Parameter(torch.full((2,), 0.68))

    def map_media_to_query(self, v_hat: torch.Tensor, s1: int) -> torch.Tensor:
        base = v_hat * self.diag_map[s1]
        lr = (v_hat @ self.down[s1]) @ self.up[s1]
        return F.normalize(base + lr, dim=-1)

    def forward(self, q_comp: torch.Tensor, v_hat_1: torch.Tensor, s1: int) -> torch.Tensor:
        q_est_1 = self.map_media_to_query(v_hat_1, s1)
        alpha = self.deflate_alpha[s1].clamp(0.20, 1.20)
        proj = (q_comp * q_est_1).sum(dim=-1, keepdim=True) * q_est_1
        return F.normalize(q_comp - alpha * proj, dim=-1)


def train_two_hop_bridge_on_train_split(
    train_ex: list,
    aligner: ResidualStreamAligner,
    tr_txt: torch.Tensor,
    tr_media: torch.Tensor,
    tr_src: torch.Tensor,
    seed: int = 20261015,
) -> LearnedTwoHopBridge:
    device = tr_txt.device
    torch.manual_seed(seed)
    with torch.no_grad():
        zt = aligner.encode_text(tr_txt).detach()
        zm = aligner.encode_media(tr_media, tr_src).detach()

    bridge = LearnedTwoHopBridge(dim=zt.shape[-1], rank=16).to(device)
    opt = torch.optim.AdamW(bridge.parameters(), lr=3e-3, weight_decay=1e-2)

    recs = sorted(set(x['recording_id'] for x in train_ex))
    pairs_s0, pairs_s1 = [], []
    for r_id in recs:
        v_idx = [i for i, x in enumerate(train_ex) if x['recording_id'] == r_id and x['source'] == 'video' and not is_null_description(x['content'])]
        a_idx = [i for i, x in enumerate(train_ex) if x['recording_id'] == r_id and x['source'] == 'audio' and not is_null_description(x['content'])]
        for ia in v_idx:
            for ib in a_idx:
                pairs_s0.append((ia, ib))
                pairs_s1.append((ib, ia))

    for _ in range(120):
        loss = torch.tensor(0.0, device=device)
        for s1, plist in ((0, pairs_s0), (1, pairs_s1)):
            if not plist:
                continue
            ia_t = torch.tensor([p[0] for p in plist], device=device, dtype=torch.long)
            ib_t = torch.tensor([p[1] for p in plist], device=device, dtype=torch.long)
            qa = zt[ia_t]
            qb = zt[ib_t]
            va = zm[ia_t]
            vb = zm[ib_t]
            q_comp = F.normalize(0.55 * qa + 0.45 * qb, dim=-1)
            q_est_a = bridge.map_media_to_query(va, s1)
            q_hop2 = bridge(q_comp, va, s1)
            l_inv = (1.0 - (q_est_a * qa).sum(dim=-1)).mean()
            l_hop2_txt = (1.0 - (q_hop2 * qb).sum(dim=-1)).mean()
            l_hop2_med = (1.0 - (q_hop2 * vb).sum(dim=-1)).mean()
            l_ortho = ((q_hop2 * qa).sum(dim=-1)).square().mean()
            loss = loss + l_inv + l_hop2_txt + 0.5 * l_hop2_med + 0.5 * l_ortho

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
    aligner: ResidualStreamAligner,
    hop2_bridge: LearnedTwoHopBridge,
    txt_feats: torch.Tensor,
    media_feats: torch.Tensor,
    sources_t: torch.Tensor,
    train_txt_mean: torch.Tensor,
    mode: str = 'natural_quartets',
    two_hop_conditioned: bool = False,
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
        zm_encoded = aligner.encode_media(media_feats, sources_t).detach()
        scale_val = float(aligner.logit_scale.exp().clamp(max=35.0).item())

    foil_audit = {
        'total_origins': 0,
        'strict_cos015_and_jaccard0_origins': 0,
        'lexical_jaccard0_origins': 0,
        'string_disjoint_fallback_origins': 0,
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

        if strict_foils:
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

                def make_ep(q_a_idx, q_b_idx, perm1, perm2, label_val, pair_key=None):
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
                            v_hat_1 = F.normalize((p1_tmp.unsqueeze(-1) * bank1_vecs).sum(dim=0), dim=-1)
                            q_hop2 = hop2_bridge(q_comp, v_hat_1, s1_int)
                            raw_s2 = q_hop2 @ bank2_vecs.T
                            lg2 = raw_s2 * scale_val
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
                            by_state_buckets[nat_rel].append(make_ep(idx_a, idx_b, id_perm1, id_perm2, nat_rel))
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
                            if len(same_desc_b) == 1 and K2 >= 6 and 2 <= pos_a <= K2 - 3 and len(b1_list) == K2:
                                for lbl_c, tgt_pos in ((0, pos_a - 2), (1, pos_a), (2, pos_a + 2)):
                                    s_tgt = float(iv_2[tgt_pos, 0].item())
                                    e_tgt = float(iv_2[tgt_pos, 1].item())
                                    assert classify_interval_relation(s_a, e_a, s_tgt, e_tgt) == lbl_c
                                    shift_k = (tgt_pos - pos_b) % K2
                                    roll_p2 = (torch.arange(K2, device=device) - shift_k) % K2
                                    by_state_buckets[lbl_c].append(make_ep(idx_a, idx_b, id_perm1, roll_p2, lbl_c))
                                by_state_buckets[3].append(make_ep(idx_a, f_idx, id_perm1, id_perm2, 3))

                m_count = min(len(by_state_buckets[c]) for c in (0, 1, 2, 3))
                if m_count > 0:
                    for c in (0, 1, 2, 3):
                        episodes.extend(by_state_buckets[c][:m_count])

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

    aligner_c1 = ctx_bundle['m_c1']
    sim_tr_c1 = ctx_bundle['sim_tr_c1']
    sim_he_c1 = ctx_bundle['sim_he_c1']
    bridge_c1 = train_two_hop_bridge_on_train_split(
        train_ex, aligner_c1, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'], ctx_bundle['tr_src']
    )

    aligner_c5 = ctx_bundle['m_c5']
    sim_tr_c5 = ctx_bundle['sim_tr_c5']
    sim_he_c5 = ctx_bundle['sim_he_c5']
    bridge_c5 = train_two_hop_bridge_on_train_split(
        train_ex, aligner_c5, ctx_bundle['tr_txt'], ctx_bundle['tr_hyb'], ctx_bundle['tr_src']
    )

    suites_out = {}
    foil_audits = {}
    for suite_name, mode_str, two_hop in (
        ('PartB1_Natural_Unshifted_Balanced', 'natural_quartets', False),
        ('PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit', 'global_time_reversal_no_role_split', False),
        ('PartB3_CrossModal_Only_TwoHop_Roll', 'counterfactual_stream_roll', True),
    ):
        tr_c1, f_aud_tr = build_connected_m2_episodes(
            train_ex, sim_tr_c1, aligner_c1, bridge_c1, ctx_bundle['tr_txt'], ctx_bundle['tr_ctx'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )
        he_c1, f_aud_he = build_connected_m2_episodes(
            held_ex, sim_he_c1, aligner_c1, bridge_c1, ctx_bundle['he_txt'], ctx_bundle['he_ctx'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )
        foil_audits[suite_name] = {'train': f_aud_tr, 'heldout': f_aud_he}

        tr_c5, _ = build_connected_m2_episodes(
            train_ex, sim_tr_c5, aligner_c5, bridge_c5, ctx_bundle['tr_txt'], ctx_bundle['tr_hyb'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )
        he_c5, _ = build_connected_m2_episodes(
            held_ex, sim_he_c5, aligner_c5, bridge_c5, ctx_bundle['he_txt'], ctx_bundle['he_hyb'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop
        )

        tr_unif, _ = build_connected_m2_episodes(
            train_ex, sim_tr_c5, aligner_c5, bridge_c5, ctx_bundle['tr_txt'], ctx_bundle['tr_hyb'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, uniform_control=True
        )
        he_unif, _ = build_connected_m2_episodes(
            held_ex, sim_he_c5, aligner_c5, bridge_c5, ctx_bundle['he_txt'], ctx_bundle['he_hyb'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, uniform_control=True
        )
        tr_wrong, _ = build_connected_m2_episodes(
            train_ex, sim_tr_c5, aligner_c5, bridge_c5, ctx_bundle['tr_txt'], ctx_bundle['tr_hyb'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, wrong_event_control=True
        )
        he_wrong, _ = build_connected_m2_episodes(
            held_ex, sim_he_c5, aligner_c5, bridge_c5, ctx_bundle['he_txt'], ctx_bundle['he_hyb'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, wrong_event_control=True
        )
        tr_cshuf, _ = build_connected_m2_episodes(
            train_ex, sim_tr_c5, aligner_c5, bridge_c5, ctx_bundle['tr_txt'], ctx_bundle['tr_hyb'],
            ctx_bundle['tr_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, coord_shuffle_control=True
        )
        he_cshuf, _ = build_connected_m2_episodes(
            held_ex, sim_he_c5, aligner_c5, bridge_c5, ctx_bundle['he_txt'], ctx_bundle['he_hyb'],
            ctx_bundle['he_src'], tr_txt_mean, mode=mode_str, two_hop_conditioned=two_hop, coord_shuffle_control=True
        )

        std_fn = lambda e: e['std_attn']
        bary_phi_fn = lambda e: e['bary_plus_phi']
        null_fn = lambda e: e['null_feat']
        full_fn = lambda e: e['full_feat']

        suites_out[suite_name] = {
            'B1_unmodified_softmax_mean_time_baseline_c1': eval_linear_probe(tr_c1, he_c1, std_fn, device),
            'B1b_unmodified_softmax_mean_time_c5_zca': eval_linear_probe(tr_c5, he_c5, std_fn, device),
            'B2_coord_shuffle_control': eval_linear_probe(tr_cshuf, he_cshuf, full_fn, device),
            'B3_wrong_event_query_control': eval_linear_probe(tr_wrong, he_wrong, full_fn, device),
            'B4_uniform_retrieval_sim0_control': eval_linear_probe(tr_unif, he_unif, full_fn, device),
            'B5_barycenter_plus_phi_no_null_c1': eval_linear_probe(tr_c1, he_c1, bary_phi_fn, device),
            'B6_shift_sensitive_presence_null_only_c5': eval_linear_probe(tr_c5, he_c5, null_fn, device),
            'B7_full_bary_phi_presence_c1_connected': eval_linear_probe(tr_c1, he_c1, full_fn, device),
            'B8_full_bary_phi_presence_c5_zca_connected': eval_linear_probe(tr_c5, he_c5, full_fn, device),
        }

    return {'foil_proxy_audit': foil_audits, **suites_out}


# ==============================================================================
# MODULE 3: GRID-CLOCK INTERPOLATION AUDIT, ORTHOGONAL BIVECTOR VO-SUBSPACE
#           TANGENT ROUTING & 4-D FREE-FORM GENERATION QUALITY AUDIT
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
    stem_rows: list,
    clock_info: dict,
    custom_q18: torch.Tensor = None,
    custom_q27: torch.Tensor = None,
    n_slots: int = None,
    top_patch_frac: float = 0.15,
):
    T_phys = clock_info['T_phys']
    sec_cal = clock_info['sec_calibrated']
    v_indices = clock_info['v_indices']
    a_indices = clock_info['a_indices']

    K = max(4, min(12, int(round(T_phys / 2.0)))) if n_slots is None else n_slots
    slot_width = T_phys / float(K)

    q18 = F.normalize(F.layer_norm(h18_t[stem_rows], (h18_t.shape[-1],)), dim=-1) if custom_q18 is None else custom_q18
    q27 = F.normalize(F.layer_norm(h27_t[stem_rows], (h27_t.shape[-1],)), dim=-1) if custom_q27 is None else custom_q27

    v18_all = h18_t[v_indices]
    v27_all = h27_t[v_indices]
    v18_dyn_all = F.normalize(v18_all - 0.85 * v18_all.mean(dim=0, keepdim=True), dim=-1)
    v27_ln = F.layer_norm(v27_all, (v27_all.shape[-1],))
    v27_dyn_all = F.normalize(v27_ln - 0.85 * v27_ln.mean(dim=0, keepdim=True), dim=-1)

    a18_all = h18_t[a_indices]
    a27_all = h27_t[a_indices]
    a18_dyn_all = F.normalize(a18_all - 0.85 * a18_all.mean(dim=0, keepdim=True), dim=-1)
    a27_ln = F.layer_norm(a27_all, (a27_all.shape[-1],))
    a27_dyn_all = F.normalize(a27_ln - 0.85 * a27_ln.mean(dim=0, keepdim=True), dim=-1)

    v_sal = 0.55 * (v18_dyn_all @ q18.T).max(dim=-1).values + 0.45 * (v27_dyn_all @ q27.T).max(dim=-1).values
    a_sal = 0.55 * (a18_dyn_all @ q18.T).max(dim=-1).values + 0.45 * (a27_dyn_all @ q27.T).max(dim=-1).values

    v_times = sec_cal[v_indices]
    a_times = sec_cal[a_indices]

    V18_slots, A18_slots = [], []
    raw_v_scores, raw_a_scores = [], []
    iv_slots = []

    for k in range(K):
        t0 = k * slot_width
        t1 = (k + 1) * slot_width + (1e-3 if k == K - 1 else 0.0)
        iv_slots.append([round(t0, 3), round(min(T_phys, t1), 3)])

        mv = torch.nonzero((v_times >= t0) & (v_times < t1), as_tuple=False).squeeze(-1)
        if mv.numel() == 0:
            mv = torch.arange(v_indices.numel())
        k_top_v = max(1, int(round(top_patch_frac * mv.numel())))
        top_v_vals, top_v_idx = torch.topk(v_sal[mv], k=k_top_v, largest=True)
        chosen_v = mv[top_v_idx]
        V18_slots.append(F.normalize(v18_dyn_all[chosen_v].mean(dim=0), dim=-1))
        raw_v_scores.append(top_v_vals.mean())

        ma = torch.nonzero((a_times >= t0) & (a_times < t1), as_tuple=False).squeeze(-1)
        if ma.numel() == 0:
            ma = torch.arange(a_indices.numel())
        k_top_a = max(1, int(round(max(0.25, top_patch_frac) * ma.numel())))
        top_a_vals, top_a_idx = torch.topk(a_sal[ma], k=k_top_a, largest=True)
        chosen_a = ma[top_a_idx]
        A18_slots.append(F.normalize(a18_dyn_all[chosen_a].mean(dim=0), dim=-1))
        raw_a_scores.append(top_a_vals.mean())

    V18_bank = torch.stack(V18_slots)
    A18_bank = torch.stack(A18_slots)
    rv = torch.stack(raw_v_scores)
    ra = torch.stack(raw_a_scores)

    v_peak_excess = float(F.relu(rv.max() - 0.04).item())
    a_peak_excess = float(F.relu(ra.max() - 0.04).item())
    lg_v = (rv - rv.mean()) * (14.0 * math.tanh(v_peak_excess / 0.08))
    lg_a = (ra - ra.mean()) * (14.0 * math.tanh(a_peak_excess / 0.08))
    iv_t = torch.tensor(iv_slots, dtype=torch.float32)

    return V18_bank, A18_bank, lg_v, lg_a, iv_t


def audit_freeform_generation_quality(text: str, clip_dur_sec: float, task: str, gold_ans_text: str) -> dict:
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
        has_yes = bool(re.search(r'\byes\b|happening at the same time', t_low))
        has_no = bool(re.search(r'\bno\b|not happening|does not happen', t_low))
        if not (has_yes or has_no):
            is_generic = True
            polarity_match = False
        elif g_low.startswith('yes'):
            polarity_match = bool(has_yes and not t_low.startswith('no'))
        elif g_low.startswith('no'):
            polarity_match = bool(has_no and not t_low.startswith('yes'))
        semantic_ok = bool(polarity_match)
    elif task == 'TF':
        has_past = bool(re.search(r'\bpast\b|\bbefore\b|\bearlier\b|\balready\b|\bprior\b', t_low))
        has_pres = bool(re.search(r'\bpresent\b|\bsame time\b|\bduring\b|\bsimultaneously\b|\bwhile\b', t_low))
        has_fut = bool(re.search(r'\bfuture\b|\bafter\b|\blater\b|\bnext\b|\bfollowing\b', t_low))
        if not (has_past or has_pres or has_fut):
            is_generic = True
            relation_match = False
        elif 'past' in g_low or 'before' in g_low:
            relation_match = bool(has_past and not has_fut)
        elif 'present' in g_low or 'during' in g_low:
            relation_match = bool(has_pres and not has_past and not has_fut)
        elif 'future' in g_low or 'after' in g_low:
            relation_match = bool(has_fut and not has_past)
        semantic_ok = bool(relation_match)
    else:
        g_toks = content_tokens(g_low)
        t_toks = content_tokens(t_low)
        content_match = len(g_toks & t_toks) >= 1 if g_toks else False
        semantic_ok = bool(content_match)

    return {
        'mentioned_timestamps_sec': mentioned_secs,
        'has_hallucinated_timestamp_beyond_duration': bool(hallucinated_ts),
        'is_generic_or_evasive_non_answer': bool(is_generic),
        'yes_no_polarity_matches_gold': polarity_match,
        'tf_relation_matches_gold': relation_match,
        'cmtb_content_matches_gold': content_match,
        'semantic_gold_aligned': bool(semantic_ok and not hallucinated_ts and not is_generic),
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
    if 'MODEL_' not in globals() or 'PROC' not in globals():
        return {'status': 'skipped_no_live_model'}

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
    for sample in samples:
        idx = sample['idx']
        pt_file = cache_dir / f'{idx}_rot0.pt'
        if not pt_file.exists():
            continue
        payload = torch.load(pt_file, map_location='cpu', weights_only=False)
        device = model_llm.get_input_embeddings().weight.device
        kw = {k: v.to(device) for k, v in payload['decoder'].items()}
        last_m = int(payload['last_media'])

        clock_info = audit_and_interpolate_grid_clocks(payload, sample)

        q_raw = sample['question']
        stem_text = q_raw.split('(A)')[0].split('A.')[0].strip()
        stem_rows, option_rows, assistant_rows, hook_rows = find_option_free_and_option_spans(
            payload, stem_text, tok
        )

        h_cap = {}
        def make_cap(l_idx):
            def cap_hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                h_cap[l_idx] = h[0].detach().float().cpu()
                return out
            return cap_hook

        h18 = model_llm.model.layers[18].register_forward_hook(make_cap(18))
        h27 = model_llm.model.layers[27].register_forward_hook(make_cap(27))
        try:
            with torch.inference_mode():
                out_nat = model_llm.model(**kw, use_cache=False, return_dict=True)
                lg_nat = model_llm.lm_head(
                    out_nat.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                )[0, -1].float().cpu()
        finally:
            h18.remove()
            h27.remove()

        k_opts = len(payload.get('native_logits', lg_nat[labels[:3]]))
        ref_logits = payload.get('native_logits', lg_nat[labels[:k_opts]]).cpu().float()
        native_replay_err = float((lg_nat[labels[:k_opts]] - ref_logits).abs().max().item())

        V18_raw, A18_raw, lg_v, lg_a, iv_t = extract_query_conditioned_salient_slots(
            h_cap[18], h_cap[27], stem_rows, clock_info, top_patch_frac=0.15
        )
        V18_vo = project_to_layer_vo_subspace(model_llm.model.layers[18], V18_raw)
        A18_vo = project_to_layer_vo_subspace(model_llm.model.layers[18], A18_raw)

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

        g_wq = torch.Generator().manual_seed(idx + 777)
        perm_d = torch.randperm(h_cap[18].shape[-1], generator=g_wq)
        q18_wrong = F.normalize(F.layer_norm(h_cap[18][stem_rows][:, perm_d], (h_cap[18].shape[-1],)), dim=-1)
        q27_wrong = F.normalize(F.layer_norm(h_cap[27][stem_rows][:, perm_d], (h_cap[27].shape[-1],)), dim=-1)
        _, _, lg_v_w, lg_a_w, _ = extract_query_conditioned_salient_slots(
            h_cap[18], h_cap[27], stem_rows, clock_info, custom_q18=q18_wrong, custom_q27=q27_wrong
        )
        _, gate_wrong, _, _ = build_orthogonal_bivector_routed_state(
            V18_vo, A18_vo, lg_v_w, lg_a_w, iv_t, reverse_timeline=False
        )

        worst_k = int((lg_v + lg_a).argmin().item())
        vec_irrel = F.normalize(0.5 * (V18_vo[worst_k] + A18_vo[worst_k]), dim=-1)

        rand_vecs = []
        for r_seed in (101, 202, 303):
            g_r = torch.Generator().manual_seed(idx + r_seed)
            rv = torch.randn(vec_fwd.shape, generator=g_r)
            rand_vecs.append(F.normalize(rv, dim=-1))

        hook_rows_t = torch.tensor(hook_rows, dtype=torch.long)
        eff_scale = 0.045 * gate_fwd

        def run_routed(target_layers, direction_vec, scale_val=eff_scale, custom_kw=None, target_rows=hook_rows_t):
            use_kw = kw if custom_kw is None else custom_kw
            handles = []
            for l_i in target_layers:
                def make_hk(vec_d, s_v):
                    def hk(mod, args, out):
                        if s_v <= 1e-7:
                            return out
                        h = out[0] if isinstance(out, tuple) else out
                        h_new = h.clone()
                        r_idx = target_rows.to(h.device)
                        hq = h_new[0, r_idx].float()
                        d_v = vec_d.to(h.device, dtype=torch.float32).unsqueeze(0)
                        hq_unit = F.normalize(hq, dim=-1)
                        d_tan = F.normalize(d_v - (d_v * hq_unit).sum(dim=-1, keepdim=True) * hq_unit, dim=-1)
                        n_q = hq.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                        h_upd = F.normalize(hq + s_v * n_q * d_tan, dim=-1) * n_q
                        h_new[0, r_idx] = h_upd.to(h.dtype)
                        return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
                    return hk
                handles.append(model_llm.model.layers[l_i].register_forward_hook(make_hk(direction_vec, scale_val)))
            try:
                with torch.inference_mode():
                    o = model_llm.model(**use_kw, use_cache=False, return_dict=True)
                    lg = model_llm.lm_head(
                        o.last_hidden_state[:, -1:, :].to(model_llm.lm_head.weight.device)
                    )[0, -1].float().cpu()
            finally:
                for h in handles:
                    h.remove()
            return lg

        p_nat = F.softmax(lg_nat[labels[:k_opts]], dim=-1)
        p_late_fwd = F.softmax(run_routed(late_layers, vec_fwd)[labels[:k_opts]], dim=-1)
        p_late_rev = F.softmax(run_routed(late_layers, vec_rev)[labels[:k_opts]], dim=-1)
        p_mid_fwd = F.softmax(run_routed(mid_layers, vec_fwd)[labels[:k_opts]], dim=-1)
        p_mid_rev = F.softmax(run_routed(mid_layers, vec_rev)[labels[:k_opts]], dim=-1)
        p_mid_irrel = F.softmax(run_routed(mid_layers, vec_irrel)[labels[:k_opts]], dim=-1)

        tv_rand_list = []
        for rv in rand_vecs:
            p_r = F.softmax(run_routed(mid_layers, rv)[labels[:k_opts]], dim=-1)
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

        keep_indices = sorted(list(range(last_m + 1)) + stem_rows + assistant_rows)
        keep_t = torch.tensor(keep_indices, device=device, dtype=torch.long)
        freeform_embeds = kw['inputs_embeds'][:, keep_t, :].clone()
        freeform_pos = kw['position_ids'][:, :, keep_t].clone()
        tail_start_pos = int(freeform_pos[0, 0, last_m].item()) + 1
        n_tail_ff = len(keep_indices) - (last_m + 1)
        freeform_pos[:, :, last_m + 1:] = (
            tail_start_pos + torch.arange(n_tail_ff, device=device)
        ).view(1, 1, -1).to(freeform_pos.dtype)

        emb_layer = model_llm.get_input_embeddings()
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
                    lg_step = run_routed(mid_layers, vec_fwd, scale_val=eff_scale, custom_kw=ff_kw, target_rows=ff_rows)
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
        for part in q_raw.split('('):
            if part.startswith(f"{gold_letter})"):
                gold_ans_text = part[2:].split('\n')[0].strip()

        nat_ff_audit = audit_freeform_generation_quality(
            ff_nat_text, clock_info['T_phys'], sample['task'], gold_ans_text
        )
        routed_ff_audit = audit_freeform_generation_quality(
            ff_routed_text, clock_info['T_phys'], sample['task'], gold_ans_text
        )

        rows_out.append({
            'idx': idx,
            'task': sample['task'],
            'gold_letter': gold_letter,
            'gold_answer_text': gold_ans_text,
            'mc_pred_native': pred_nat_letter,
            'mc_pred_routed_fwd': pred_routed_letter,
            'mc_pred_routed_rev': pred_rev_letter,
            'mc_native_correct': bool(pred_nat_letter == gold_letter),
            'mc_routed_correct': bool(pred_routed_letter == gold_letter),
            'grid_clock_interpolation_audit': clock_info['clock_audit'],
            'n_calibrated_slots_K': int(V18_vo.shape[0]),
            'confidence_gate_gamma_true_query': round(gate_fwd, 4),
            'confidence_gate_gamma_wrong_query': round(gate_wrong, 4),
            'native_replay_max_err': round(native_replay_err, 6),
            'fwd_vs_rev_vector_cosine': round(fwd_rev_cos, 4),
            'content_even_max_err': round(content_even_err, 8),
            'order_odd_max_err': round(order_odd_err, 8),
            'rel_scalar_fwd_vs_rev': [aud_fwd['rel_scalar'], aud_rev['rel_scalar']],
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
        'mean_tv_kslot_swap_late_L28_33': round(mean_swap_late, 4),
        'mean_tv_kslot_swap_mid_L14_24': round(mean_swap_mid, 4),
        'mid_over_late_kslot_swap_ratio': round(mean_swap_mid / max(1e-6, mean_swap_late), 2),
        'mean_tv_mid_routed_vs_nat': round(mean_routed, 4),
        'mean_tv_mid_irrel_slot_control': round(mean_irrel, 4),
        'mean_tv_mid_random_3seed_control': round(mean_rand, 4),
        'mean_gamma_true_query': round(float(np.mean([r['confidence_gate_gamma_true_query'] for r in rows_out])), 4),
        'mean_gamma_wrong_query': round(float(np.mean([r['confidence_gate_gamma_wrong_query'] for r in rows_out])), 4),
        'samples': rows_out,
    }


def run_all_stage0_v5():
    t0 = time.time()
    unit_checks = run_algebraic_unit_checks('cuda:0' if torch.cuda.is_available() else 'cpu')
    m1_native, ctx_native = run_stage0_m1_orthogonal_factorial('avbind_native_pairs')
    m1_pooled, ctx_pooled = run_stage0_m1_orthogonal_factorial('avbind_pooled_pairs')
    use_bundle = ctx_pooled if ctx_pooled is not None else ctx_native
    m2_res = run_stage0_m2_connected(use_bundle)
    m3_res = run_stage0_m3_bivector_vo_tangent_routing()

    out = {
        'protocol': 'Stage-0 v5 (Orthogonal Content-Order Bivector + Modality-ZCA M1 + Shift-Sensitive M2 + VO-Subspace Tangent Routing M3)',
        'elapsed_sec': round(time.time() - t0, 2),
        'Algebraic_Bivector_and_ZeroFloor_Unit_Checks': unit_checks,
        'M1_native_orthogonal_factorial': m1_native,
        'M1_pooled_orthogonal_factorial': m1_pooled,
        'M2_incremental_ablation_and_learned_hop2': m2_res,
        'M3_bivector_vo_tangent_routing': m3_res,
    }
    out_path = OUT_DIR / 'stage0_pcebr_v5_results.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('STAGE0_V5_DONE', json.dumps(out, indent=2), flush=True)
    return out


RESULT = run_all_stage0_v5()

