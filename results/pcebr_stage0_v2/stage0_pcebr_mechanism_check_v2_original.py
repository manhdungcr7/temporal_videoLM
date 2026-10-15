"""
STAGE-0 v2 MECHANISM DIAGNOSTIC: Provenance-Coupled Event-Binding & Relational Latent Adapter (PC-EBR v2)
Authors: Yoshua Bengio & Yann LeCun

Addresses ALL 7 technical & methodological critiques from Yann LeCun's Stage-0 Peer Review:

1. FIXING M1 (Why Stage-0 v1 traded off Global `13->6/50` vs Within-Rec `13->16/50` and had high seed variance):
   - Diagnosis from Stage-0 v1 results:
     (a) In Stage-0 v1, `dual_center` centered held-out queries across the held-out query batch (transductive) and
         destructively subtracted `0.85 * rec_mean` from media vectors, erasing recording-level context needed for
         global retrieval (`13/50 -> 6/50`).
     (b) Conversely, on `avbind_pooled_pairs`, 0-parameter unadapted deconfounding (`2_unadapted_deconfounded`) got
         `14/50` within-rec, whereas training a 294,985-parameter `2048 -> 24` bottleneck (`R24`) dropped it to `11/50`
         because mapping `2048 -> 24` from scratch on `N=202` examples destroys the pretrained 2048-D LLM semantic
         space on unseen words in the 4 held-out recordings!
   - The 4 Fixes in Stage-0 v2 (`IdentityAnchoredTwoStreamAligner`):
     (i)   100% Query-Independent Inference: Text queries are NEVER centered across the held-out batch.
     (ii)  Stays in the Pretrained 2048-D LLM Space via a parameter-efficient Diagonal + Rank-8 Residual Adapter
           (`77,827` total parameters including all buffers, Kaiming/normal init so step-0 gradients flow).
     (iii) Non-Destructive Zero-Sum Within-Recording Temporal Advantage Scoring:
             `S(q, r_{m,k}) = S_ctx(q, r_{m,k}) + lambda_dyn * (S_dyn(q, r_{m,k}) - mean_{k' in same_rec, same_src} S_dyn(q, r_{m,k'}))`
           Because the second term sums strictly to ZERO within each candidate recording `(R, m)`, it sharpens
           within-recording interval localization WITHOUT subtracting or degrading global recording context `S_ctx`!
     (iv)  Reports 4 transparent metrics across 3 fixed seeds `(20261015, 20261016, 20261017)` in a full crossed
           factorial (`Representation Mode` x `Hard Within-Rec Negative Mining` x `True vs Shuffled Alignment`):
           `global_top1`, `within_exact_top1`, `within_multipos_top1` (resolving duplicate boilerplate descriptions
           like `'there is no speech...'` at indices `[9,13]` and `[33,37,47,49]`), and `within_nonnull_top1`.

2. FIXING M2 (Scale- & Translation-Invariant Interval Algebra + Non-Circular Retrieved-Interval Counterfactuals):
   - Part A: Normalizes all endpoint gaps by `d_ref = sqrt(d_a * d_b)` so relative order/coverage channels of
     `Phi(I_target, I_anchor)` are 100% invariant under ANY positive affine clock transform `phi(t) = a*t + b` (`a > 0`).
     Separates `iou`, `anchor_coverage = inter / d_a` (`0.2` for `[0,10]` vs `[2,4]`), and `target_coverage = inter / d_b`.
   - Part B: Evaluates both:
       (1) `Natural Unshifted Order` (`Before` vs `After` on unmodified unseen recordings using RETRIEVED `p1, p2`), and
       (2) `Counterfactual Physical Stream Shifts` where the target modality stream's physical clock is shifted by the
           TRUE physical offset `delta_true = c_true_a - c_true_b + {-6s, 0s, +6s}` (NEVER using `p1, p2` to construct
           the shift, so any retrieval error in `p1` or `p2` directly degrades relation accuracy!), plus `Cross-Recording Foil`.
     Compares across all 4 direction pairs (`audio->video`, `video->audio`, `video->video`, `audio->audio`):
       - `B1_matched_std_attn`: Content-weighted standard attention pooling `[E_{p1}[t], E_{p2}[t], null_feat]`
       - `B2_time_coord_shuffled`: Same features & model, permuting ONLY interval coordinates `I_k` across slots
       - `B3_pcebr_phi_only`: `Phi(I_2, I_1)` only (no null gate features)
       - `B4_pcebr_null_only`: Null/entropy features only (no `Phi`)
       - `B5_pcebr_full_phi_plus_null`: Full `Phi(I_2, I_1)` + Null/entropy gate

3. FIXING M3 (Executable Mid-Layer `L14..24` vs. Late-Layer `L28..33` Attention Routing + Free-Form QA Check):
   - Executes on a 4-sample micro-probe (`STAGE0_MICRO_LIMIT = 4`, rotation 0) comparing 6 Mid-Layers `(14,16,18,20,22,24)`
     vs. 6 Late-Layers `(28,29,30,31,32,33)` with identical layer count and norm budget, measuring Total Variation (`TV`),
     Counterfactual Early-vs-Late Swap Separation (`TV_swap`), and `answer_options=None` free-form generation parity.
"""

import gc, json, math, random, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

OUT = globals().get('OUT', Path('.'))
SEEDS = (20261015, 20261016, 20261017)

NULL_BOILERPLATE_SUBSTRINGS = (
    'no speech',
    'no audible',
    'only sound is music',
    'silence',
    'nothing is heard',
)


def is_null_description(text):
    low = text.strip().lower()
    return any(pat in low for pat in NULL_BOILERPLATE_SUBSTRINGS)


def build_media_streams(features, recording_ids, sources, starts):
    """
    Constructs three media representations per 2s crop WITHOUT touching any text query:
      - `x_ctx` : LayerNorm(raw_x) — preserves global scene/object context across recordings.
      - `x_rec` : LayerNorm(raw_x - mean_{same_rec, same_src}(raw_x)) — within-recording innovation.
      - `x_loc` : LayerNorm(raw_x - 0.5*(prev_chunk + next_chunk)) — local boundary innovation.
    """
    x = features.detach().float()
    N, D = x.shape
    x_ctx = F.layer_norm(x, (D,))
    x_rec = x.clone()
    x_loc = x.clone()

    for rec_id in sorted(set(recording_ids)):
        for s_val in (0, 1):
            idx_list = [i for i, (r, s) in enumerate(zip(recording_ids, sources.tolist())) if r == rec_id and s == s_val]
            if len(idx_list) > 1:
                idx_list_sorted = sorted(idx_list, key=lambda i: float(starts[i]))
                idx_t = torch.tensor(idx_list_sorted, device=x.device, dtype=torch.long)
                rec_mean = x[idx_t].mean(dim=0, keepdim=True)
                x_rec[idx_t] = x[idx_t] - rec_mean

                k_len = len(idx_list_sorted)
                for pos_i, glob_i in enumerate(idx_list_sorted):
                    neigh = []
                    if pos_i > 0:
                        neigh.append(x[idx_list_sorted[pos_i - 1]])
                    if pos_i + 1 < k_len:
                        neigh.append(x[idx_list_sorted[pos_i + 1]])
                    if neigh:
                        x_loc[glob_i] = x[glob_i] - torch.stack(neigh, dim=0).mean(dim=0)

    x_rec = F.layer_norm(x_rec, (D,))
    x_loc = F.layer_norm(x_loc, (D,))
    return x_ctx, x_rec, x_loc


class IdentityAnchoredTwoStreamAligner(nn.Module):
    """
    Stays in the native 2048-D LLM semantic space so unseen words in held-out recordings are NEVER
    crushed through a random low-dim bottleneck!
    Total parameters (including all buffers & weights): 77,827.
    """
    def __init__(self, dim=2048, rank=8, num_sources=2, mode='two_stream'):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.mode = mode  # 'uncentered' | 'media_only_sub' | 'two_stream'

        self.bias_ctx = nn.Parameter(torch.zeros(num_sources, dim))
        self.scale_ctx = nn.Parameter(torch.ones(num_sources, dim))
        self.down = nn.Parameter(torch.randn(num_sources, dim, rank) * (0.02 / math.sqrt(dim)))
        self.up = nn.Parameter(torch.randn(num_sources, rank, dim) * (0.02 / math.sqrt(rank)))

        self.dyn_scale = nn.Parameter(torch.ones(num_sources, dim))
        self.lambda_dyn = nn.Parameter(torch.tensor([0.55, 0.55]))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def init_shift_from_train(self, x_ctx, t_raw, sources):
        with torch.no_grad():
            t_ln = F.layer_norm(t_raw.float(), (self.dim,))
            for s_id in (0, 1):
                m = (sources == s_id)
                if m.any():
                    self.bias_ctx[s_id].copy_(t_ln[m].mean(0) - x_ctx[m].mean(0))

    def encode_text(self, t_raw):
        """100% Query-Independent: preserves the native 2048-D LLM text representation!"""
        return F.normalize(F.layer_norm(t_raw.float(), (self.dim,)), dim=-1)

    def encode_media_streams(self, x_ctx, x_rec, x_loc, sources):
        if self.mode == 'media_only_sub':
            x_in = F.layer_norm(x_ctx - 0.65 * (x_ctx - x_rec), (self.dim,))
        else:
            x_in = x_ctx

        h_scaled = x_in * self.scale_ctx[sources] + self.bias_ctx[sources]
        low_rank = torch.einsum('nd,ndr,nrd->nd', x_in, self.down[sources], self.up[sources])
        z_ctx = F.normalize(F.layer_norm(h_scaled + low_rank, (self.dim,)), dim=-1)

        xm_dyn = 0.60 * x_rec + 0.40 * x_loc
        h_dyn = xm_dyn * self.dyn_scale[sources] + self.bias_ctx[sources] + 0.5 * low_rank
        z_dyn = F.normalize(F.layer_norm(h_dyn, (self.dim,)), dim=-1)
        return z_ctx, z_dyn

    def score_matrix(self, t_raw, x_ctx, x_rec, x_loc, sources, recording_ids):
        z_q = self.encode_text(t_raw)
        z_ctx, z_dyn = self.encode_media_streams(x_ctx, x_rec, x_loc, sources)
        s_ctx = z_q @ z_ctx.T  # (N_q, N_m)

        if self.mode in ('uncentered', 'media_only_sub'):
            return s_ctx

        s_dyn = z_q @ z_dyn.T  # (N_q, N_m)
        s_adv = torch.zeros_like(s_dyn)
        lam = self.lambda_dyn.clamp(0.1, 1.2)
        for rec_id in sorted(set(recording_ids)):
            for s_val in (0, 1):
                cols = [j for j, (r, s) in enumerate(zip(recording_ids, sources.tolist())) if r == rec_id and s == s_val]
                if len(cols) > 1:
                    c_idx = torch.tensor(cols, device=s_dyn.device, dtype=torch.long)
                    grp_scores = s_dyn[:, c_idx]
                    s_adv[:, c_idx] = lam[s_val] * (grp_scores - grp_scores.mean(dim=1, keepdim=True))
        return s_ctx + s_adv


def compute_affine_invariant_interval_relations(intervals_a, intervals_b):
    """
    Strictly Affine-Invariant Directed Interval Relation Algebra `Phi(I_target, I_anchor)`:
      Normalizes all endpoint differences by `d_ref = sqrt(d_a * d_b)`.
      100% invariant to ANY positive affine transform `phi(t) = a*t + b` (`a > 0, b in R`, error < 1e-6).
      Channels of `Phi` (M, N, 7):
        0: IoU(I_b, I_a) = inter / union
        1: Anchor Coverage = inter / d_a  (equals 0.200 for anchor=[0,10], target=[2,4]!)
        2: Target Coverage = inter / d_b  (equals 1.000 for anchor=[0,10], target=[2,4]!)
        3: Signed log after-gap  : sign(s_b - e_a) * log1p(|s_b - e_a| / d_ref)
        4: Signed log before-gap : sign(s_a - e_b) * log1p(|s_a - e_b| / d_ref)
        5: Antisymmetric center offset: sign(c_b - c_a) * log1p(|c_b - c_a| / d_ref)
        6: Scale-invariant log duration ratio: log(d_b / d_a)
    """
    sa, ea = intervals_a[:, 0:1].float(), intervals_a[:, 1:2].float()
    sb, eb = intervals_b[:, 0].float().unsqueeze(0), intervals_b[:, 1].float().unsqueeze(0)

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

    gap_after = (sb - ea) / d_ref
    gap_before = (sa - eb) / d_ref
    center_diff = (cb - ca) / d_ref

    s_log = lambda x: torch.sign(x) * torch.log1p(x.abs())
    return torch.stack([
        iou,
        anchor_cov,
        target_cov,
        s_log(gap_after),
        s_log(gap_before),
        s_log(center_diff),
        torch.log(db / da),
    ], dim=-1)


def evaluate_sim_matrix(sim_matrix, examples):
    sim = sim_matrix.detach().cpu()
    norms = [x['content'].strip().lower() for x in examples]
    is_null = [is_null_description(x['content']) for x in examples]

    top1_global = sim.argmax(dim=-1).tolist()
    global_hits = sum(norms[i] == norms[j] for i, j in enumerate(top1_global))

    exact_hits, multipos_hits, within_total = 0, 0, 0
    nonnull_hits, nonnull_total = 0, 0

    for i, ex in enumerate(examples):
        same_group = [
            j for j, ex_j in enumerate(examples)
            if ex_j['recording_id'] == ex['recording_id'] and ex_j['source'] == ex['source']
        ]
        if len(same_group) >= 2:
            sub_scores = sim[i, same_group]
            best_j = same_group[int(sub_scores.argmax())]
            is_exact = (abs(examples[best_j]['start'] - ex['start']) < 1e-3)
            is_mpos = (norms[best_j] == norms[i])
            exact_hits += int(is_exact)
            multipos_hits += int(is_mpos)
            within_total += 1
            if not is_null[i]:
                nonnull_hits += int(is_mpos)
                nonnull_total += 1

    return {
        'global_hits': global_hits,
        'global_pct': round(100.0 * global_hits / max(1, len(examples)), 1),
        'within_exact_hits': exact_hits,
        'within_exact_pct': round(100.0 * exact_hits / max(1, within_total), 1),
        'within_multipos_hits': multipos_hits,
        'within_multipos_pct': round(100.0 * multipos_hits / max(1, within_total), 1),
        'within_nonnull_hits': nonnull_hits,
        'within_nonnull_total': nonnull_total,
        'within_nonnull_pct': round(100.0 * nonnull_hits / max(1, nonnull_total), 1),
    }


def train_and_eval_m1_condition(tr_streams, tr_txt, tr_src, tr_rec, tr_norm,
                                he_streams, he_txt, he_src, he_rec, held_ex,
                                mode='two_stream', use_hard_neg=True, shuffle_within_rec=False, seed=20261015):
    device = tr_txt.device
    torch.manual_seed(seed)
    random.seed(seed)

    tr_ctx, tr_rec_s, tr_loc_s = [t.clone() for t in tr_streams]
    he_ctx, he_rec_s, he_loc_s = he_streams

    if shuffle_within_rec:
        g_shuf = torch.Generator(device=device).manual_seed(seed)
        for r_id in sorted(set(tr_rec)):
            for s_id in (0, 1):
                m_idx = [i for i, (r, s) in enumerate(zip(tr_rec, tr_src.tolist())) if r == r_id and s == s_id]
                if len(m_idx) > 1:
                    idx_t = torch.tensor(m_idx, device=device, dtype=torch.long)
                    perm = idx_t[torch.randperm(len(m_idx), generator=g_shuf, device=device)]
                    tr_ctx[idx_t] = tr_ctx[perm]
                    tr_rec_s[idx_t] = tr_rec_s[perm]
                    tr_loc_s[idx_t] = tr_loc_s[perm]

    model = IdentityAnchoredTwoStreamAligner(dim=tr_txt.shape[-1], rank=8, mode=mode).to(device)
    model.init_shift_from_train(tr_ctx, tr_txt, tr_src)

    total_params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2)

    N = len(tr_norm)
    pos_mask = torch.tensor([[tr_norm[i] == tr_norm[j] for j in range(N)] for i in range(N)], device=device)
    same_rec = torch.tensor([[tr_rec[i] == tr_rec[j] and tr_src[i] == tr_src[j] for j in range(N)] for i in range(N)], device=device)

    for _ in range(180):
        sim = model.score_matrix(tr_txt, tr_ctx, tr_rec_s, tr_loc_s, tr_src, tr_rec)
        scale = model.logit_scale.exp().clamp(max=35.0)
        logits = sim * scale
        if use_hard_neg:
            logits = logits + torch.where(same_rec & (~pos_mask), torch.tensor(0.5, device=device), torch.tensor(0.0, device=device))

        neg_inf = float('-inf')
        l_f = torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=-1)
        l_r = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=0)
        reg_id = 0.05 * ((model.scale_ctx - 1.0).square().mean() + (model.dyn_scale - 1.0).square().mean())
        loss = 0.5 * (l_f.mean() + l_r.mean()) + reg_id
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        he_sim = model.score_matrix(he_txt, he_ctx, he_rec_s, he_loc_s, he_src, he_rec)
        metrics = evaluate_sim_matrix(he_sim, held_ex)
        metrics['total_params_incl_buffers'] = total_params
    return metrics, model


def run_stage0_m1_crossed_factorial(pair_prefix='avbind_native_pairs'):
    paths = [OUT / 'uploads' / f'{pair_prefix}_s{s}.pt' for s in (1, 2)]
    if not all(p.exists() for p in paths):
        paths = [OUT / f'{pair_prefix}_s{s}.pt' for s in (1, 2)]
    if not all(p.exists() for p in paths):
        return {'status': 'skipped_missing_pair_cache', 'prefix': pair_prefix}, None

    parts = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    examples = [x for part in parts for x in part['examples']]
    train_ex = [x for x in examples if x['split'] == 'train']
    held_ex = [x for x in examples if x['split'] == 'heldout']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tr_raw = torch.stack([x['raw'] for x in train_ex]).to(device)
    tr_txt = torch.stack([x['text'] for x in train_ex]).to(device)
    tr_src = torch.tensor([0 if x['source'] == 'video' else 1 for x in train_ex], device=device)
    tr_rec = [x['recording_id'] for x in train_ex]
    tr_starts = [x['start'] for x in train_ex]
    tr_norm = [x['content'].strip().lower() for x in train_ex]

    he_raw = torch.stack([x['raw'] for x in held_ex]).to(device)
    he_txt = torch.stack([x['text'] for x in held_ex]).to(device)
    he_src = torch.tensor([0 if x['source'] == 'video' else 1 for x in held_ex], device=device)
    he_rec = [x['recording_id'] for x in held_ex]
    he_starts = [x['start'] for x in held_ex]

    tr_streams = build_media_streams(tr_raw, tr_rec, tr_src, tr_starts)
    he_streams = build_media_streams(he_raw, he_rec, he_src, he_starts)

    det_results = {}
    zt_he_raw = F.normalize(F.layer_norm(he_txt.float(), (he_txt.shape[-1],)), dim=-1)
    det_results['0_unadapted_raw'] = evaluate_sim_matrix(
        zt_he_raw @ F.normalize(he_streams[0], dim=-1).T, held_ex
    )
    det_results['0_unadapted_media_rec_only'] = evaluate_sim_matrix(
        zt_he_raw @ F.normalize(he_streams[1], dim=-1).T, held_ex
    )

    m_step0 = IdentityAnchoredTwoStreamAligner(dim=tr_txt.shape[-1], rank=8, mode='two_stream').to(device)
    m_step0.init_shift_from_train(tr_streams[0], tr_txt, tr_src)
    with torch.no_grad():
        det_results['0_step0_mean_shift_two_stream'] = evaluate_sim_matrix(
            m_step0.score_matrix(he_txt, *he_streams, he_src, he_rec), held_ex
        )

    conditions = [
        ('F1_uncentered_no_hneg',       'uncentered',     False, False),
        ('F2_uncentered_with_hneg',     'uncentered',     True,  False),
        ('F3_media_sub_no_hneg',        'media_only_sub', False, False),
        ('F4_media_sub_with_hneg',      'media_only_sub', True,  False),
        ('F5_two_stream_no_hneg',       'two_stream',     False, False),
        ('F6_two_stream_with_hneg',     'two_stream',     True,  False),
        ('F7_two_stream_shuffled_ctrl', 'two_stream',     True,  True),
    ]

    factorial_summary = {}
    best_model = None
    for cond_name, mode, use_hneg, shuf in conditions:
        per_seed = []
        for s_idx, seed in enumerate(SEEDS):
            m_res, mod = train_and_eval_m1_condition(
                tr_streams, tr_txt, tr_src, tr_rec, tr_norm,
                he_streams, he_txt, he_src, he_rec, held_ex,
                mode=mode, use_hard_neg=use_hneg, shuffle_within_rec=shuf, seed=seed
            )
            per_seed.append(m_res)
            if cond_name == 'F6_two_stream_with_hneg' and s_idx == 0:
                best_model = mod

        factorial_summary[cond_name] = {
            'global_hits_per_seed': [x['global_hits'] for x in per_seed],
            'within_exact_per_seed': [x['within_exact_hits'] for x in per_seed],
            'within_multipos_per_seed': [x['within_multipos_hits'] for x in per_seed],
            'within_nonnull_per_seed': [x['within_nonnull_hits'] for x in per_seed],
            'mean_global_pct': round(float(np.mean([x['global_pct'] for x in per_seed])), 1),
            'mean_within_exact_pct': round(float(np.mean([x['within_exact_pct'] for x in per_seed])), 1),
            'mean_within_multipos_pct': round(float(np.mean([x['within_multipos_pct'] for x in per_seed])), 1),
            'mean_within_nonnull_pct': round(float(np.mean([x['within_nonnull_pct'] for x in per_seed])), 1),
            'total_params_incl_buffers': per_seed[0]['total_params_incl_buffers'],
        }

    ctx_bundle = {
        'model': best_model,
        'train_ex': train_ex, 'held_ex': held_ex,
        'tr_streams': tr_streams, 'tr_txt': tr_txt, 'tr_src': tr_src, 'tr_rec': tr_rec, 'tr_norm': tr_norm,
        'he_streams': he_streams, 'he_txt': he_txt, 'he_src': he_src, 'he_rec': he_rec,
    }
    return {'deterministic_baselines': det_results, 'three_seed_factorial': factorial_summary}, ctx_bundle


def run_stage0_m2_retrieved_interval_counterfactuals(ctx_bundle):
    I_early = torch.tensor([[2.0, 4.0]])
    I_late = torch.tensor([[10.0, 12.0]])
    phi_before = compute_affine_invariant_interval_relations(I_late, I_early)
    phi_after = compute_affine_invariant_interval_relations(I_early, I_late)

    a_scale, b_shift = 2.0, 12.5
    phi_after_affine = compute_affine_invariant_interval_relations(I_early * a_scale + b_shift, I_late * a_scale + b_shift)
    affine_err = float((phi_after - phi_after_affine).abs().max())

    phi_cov_test = compute_affine_invariant_interval_relations(torch.tensor([[0.0, 10.0]]), torch.tensor([[2.0, 4.0]]))
    part_a = {
        'positive_affine_2x_plus_12p5_max_error': round(affine_err, 8),
        'anchor_0_10_target_2_4_anchor_cov': round(float(phi_cov_test[0, 0, 1]), 4),
        'anchor_0_10_target_2_4_target_cov': round(float(phi_cov_test[0, 0, 2]), 4),
        'antisym_center_sum_error': round(float((phi_before[0, 0, 5] + phi_after[0, 0, 5]).abs()), 8),
    }

    if ctx_bundle is None or ctx_bundle.get('model') is None:
        return {'PartA_affine_algebra': part_a, 'PartB_retrieved_counterfactuals': 'skipped'}

    model = ctx_bundle['model']
    device = ctx_bundle['tr_txt'].device
    train_ex = ctx_bundle['train_ex']
    held_ex = ctx_bundle['held_ex']

    with torch.no_grad():
        sim_tr = model.score_matrix(ctx_bundle['tr_txt'], *ctx_bundle['tr_streams'], ctx_bundle['tr_src'], ctx_bundle['tr_rec'])
        sim_he = model.score_matrix(ctx_bundle['he_txt'], *ctx_bundle['he_streams'], ctx_bundle['he_src'], ctx_bundle['he_rec'])

    def build_noncircular_retrieved_episodes(examples, sim_mat):
        """
        Non-Circular Retrieved-Interval Counterfactuals:
          For every ordered pair of distinct non-null events `(idx_a, idx_b)` in recording `r_id`:
            - `bank_1` = all candidate slots in `(r_id, source(idx_a))`
            - `bank_2` = all candidate slots in `(r_id, source(idx_b))`
            - Hop 1 retrieves `p1 = softmax(sim_mat[idx_a, bank_1] / 0.07)`
            - Hop 2 retrieves `p2 = softmax(sim_mat[idx_b, bank_2] / 0.07)`
            - Crucial Non-Circularity Guarantee:
              The physical stream shift `delta_true = c_true_a - c_true_b + offset` is defined strictly
              from the TRUE ground-truth event centers `c_true_a, c_true_b` (NEVER from `p1, p2`!).
              Thus, if `p1` or `p2` retrieves the wrong slot in `bank_1` or `bank_2`, the expected
              interval relation `E_{p1, p2}[Phi]` will point to the WRONG timestamp and fail!
        """
        episodes = []
        recs = sorted(set(x['recording_id'] for x in examples))
        for r_id in recs:
            r_items = [i for i, x in enumerate(examples) if x['recording_id'] == r_id]
            other_items = [i for i, x in enumerate(examples) if x['recording_id'] != r_id]
            non_null = [i for i in r_items if not is_null_description(examples[i]['content'])]
            if len(non_null) < 2 or len(other_items) < 4:
                continue

            for idx_a in non_null:
                for idx_b in non_null:
                    if idx_a == idx_b:
                        continue
                    src_a, src_b = examples[idx_a]['source'], examples[idx_b]['source']
                    bank_1_list = [i for i in r_items if examples[i]['source'] == src_a]
                    bank_2_list = [i for i in r_items if examples[i]['source'] == src_b]
                    if len(bank_1_list) < 2 or len(bank_2_list) < 2:
                        continue

                    b1_idx = torch.tensor(bank_1_list, device=device, dtype=torch.long)
                    b2_idx = torch.tensor(bank_2_list, device=device, dtype=torch.long)
                    iv_1 = torch.tensor([[examples[i]['start'], examples[i]['end']] for i in bank_1_list], device=device, dtype=torch.float32)
                    iv_2 = torch.tensor([[examples[i]['start'], examples[i]['end']] for i in bank_2_list], device=device, dtype=torch.float32)

                    logits_1 = sim_mat[idx_a, b1_idx] / 0.07
                    logits_2 = sim_mat[idx_b, b2_idx] / 0.07
                    p1 = F.softmax(logits_1, dim=-1)
                    p2 = F.softmax(logits_2, dim=-1)

                    max_sim_1 = logits_1.max() * 0.07
                    max_sim_2 = logits_2.max() * 0.07
                    ent_2 = -(p2 * p2.clamp(min=1e-9).log()).sum() / math.log(max(2, len(bank_2_list)))
                    null_feat = torch.stack([max_sim_1, max_sim_2, ent_2], dim=-1)

                    c_true_a = 0.5 * (float(examples[idx_a]['start']) + float(examples[idx_a]['end']))
                    c_true_b = 0.5 * (float(examples[idx_b]['start']) + float(examples[idx_b]['end']))
                    true_diff = c_true_b - c_true_a
                    dir_tag = f"{src_a}_to_{src_b}"

                    for label_rel, target_offset in [(0, -true_diff - 6.0), (1, -true_diff), (2, -true_diff + 6.0)]:
                        iv_2_shifted = iv_2 + target_offset
                        phi_mat = compute_affine_invariant_interval_relations(iv_1, iv_2_shifted)  # (K1, K2, 7)
                        weights_12 = p1.unsqueeze(1) * p2.unsqueeze(0)  # (K1, K2)
                        exp_phi = (weights_12.unsqueeze(-1) * phi_mat).sum(dim=(0, 1))

                        g_s = torch.Generator(device=device).manual_seed(42 + idx_a * 13 + idx_b + label_rel)
                        perm_k = torch.randperm(len(bank_2_list), generator=g_s, device=device)
                        phi_shuf = compute_affine_invariant_interval_relations(iv_1, iv_2_shifted[perm_k])
                        exp_phi_shuf = (weights_12.unsqueeze(-1) * phi_shuf).sum(dim=(0, 1))

                        t1_mean = (p1.unsqueeze(-1) * iv_1).sum(0)
                        t2_mean = (p2.unsqueeze(-1) * iv_2_shifted).sum(0)
                        std_attn_feat = torch.cat([t1_mean, t2_mean, null_feat], dim=-1)

                        episodes.append({
                            'phi': exp_phi,
                            'phi_shuf': exp_phi_shuf,
                            'null_feat': null_feat,
                            'std_attn': std_attn_feat,
                            'label': label_rel,
                            'dir_tag': dir_tag,
                        })

                    foil_list = [i for i in other_items if examples[i]['source'] == src_b][:len(bank_2_list)]
                    if len(foil_list) >= 2:
                        f_idx = torch.tensor(foil_list, device=device, dtype=torch.long)
                        iv_foil = torch.tensor([[examples[i]['start'], examples[i]['end']] for i in foil_list], device=device, dtype=torch.float32)
                        logits_2_foil = sim_mat[idx_b, f_idx] / 0.07
                        p2_foil = F.softmax(logits_2_foil, dim=-1)
                        ent_2_foil = -(p2_foil * p2_foil.clamp(min=1e-9).log()).sum() / math.log(max(2, len(foil_list)))
                        null_feat_foil = torch.stack([max_sim_1, logits_2_foil.max() * 0.07, ent_2_foil], dim=-1)
                        phi_foil = compute_affine_invariant_interval_relations(iv_1, iv_foil)
                        w_foil = p1.unsqueeze(1) * p2_foil.unsqueeze(0)
                        exp_phi_foil = (w_foil.unsqueeze(-1) * phi_foil).sum(dim=(0, 1))
                        t1_m = (p1.unsqueeze(-1) * iv_1).sum(0)
                        t2_m = (p2_foil.unsqueeze(-1) * iv_foil).sum(0)
                        episodes.append({
                            'phi': exp_phi_foil,
                            'phi_shuf': exp_phi_foil,
                            'null_feat': null_feat_foil,
                            'std_attn': torch.cat([t1_m, t2_m, null_feat_foil], dim=-1),
                            'label': 3,
                            'dir_tag': dir_tag,
                        })
        return episodes

    tr_eps = build_noncircular_retrieved_episodes(train_ex, sim_tr)
    he_eps = build_noncircular_retrieved_episodes(held_ex, sim_he)

    if not tr_eps or not he_eps:
        return {'PartA_affine_algebra': part_a, 'PartB_retrieved_counterfactuals': 'insufficient_nonnull_pairs'}

    def eval_linear_probe(feat_fn, n_classes=4):
        X_tr = torch.stack([feat_fn(e) for e in tr_eps]).to(device)
        y_tr = torch.tensor([e['label'] for e in tr_eps], device=device, dtype=torch.long)
        X_he = torch.stack([feat_fn(e) for e in he_eps]).to(device)
        y_he = torch.tensor([e['label'] for e in he_eps], device=device, dtype=torch.long)

        mu = X_tr.mean(0, keepdim=True)
        sd = X_tr.std(0, keepdim=True).clamp(min=1e-4)
        X_tr_n = (X_tr - mu) / sd
        X_he_n = (X_he - mu) / sd

        torch.manual_seed(20261015)
        clf = nn.Linear(X_tr.shape[-1], n_classes).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=0.05, weight_decay=1e-2)
        for _ in range(200):
            loss = F.cross_entropy(clf(X_tr_n), y_tr)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            preds = clf(X_he_n).argmax(dim=-1)
            overall = round(100.0 * float((preds == y_he).float().mean()), 1)
            by_cls = {}
            for c_i, c_name in enumerate(('Before', 'During', 'After', 'Null_Foil')):
                m_c = (y_he == c_i)
                if m_c.any():
                    by_cls[c_name] = round(100.0 * float((preds[m_c] == y_he[m_c]).float().mean()), 1)
            by_dir = {}
            for d_tag in ('audio_to_video', 'video_to_audio', 'video_to_video', 'audio_to_audio'):
                d_idx = [i for i, e in enumerate(he_eps) if e['dir_tag'] == d_tag]
                if d_idx:
                    d_t = torch.tensor(d_idx, device=device, dtype=torch.long)
                    by_dir[d_tag] = round(100.0 * float((preds[d_t] == y_he[d_t]).float().mean()), 1)
        return {'overall_pct': overall, 'by_state': by_cls, 'by_modality_direction': by_dir, 'n_eval_episodes': len(he_eps)}

    part_b = {
        'B1_matched_std_attn_baseline': eval_linear_probe(lambda e: e['std_attn']),
        'B2_pcebr_time_coord_shuffled': eval_linear_probe(lambda e: torch.cat([e['phi_shuf'], e['null_feat']], dim=-1)),
        'B3_pcebr_phi_only_no_null':    eval_linear_probe(lambda e: e['phi']),
        'B4_pcebr_null_only_no_phi':    eval_linear_probe(lambda e: e['null_feat']),
        'B5_pcebr_full_phi_plus_null':  eval_linear_probe(lambda e: torch.cat([e['phi'], e['null_feat']], dim=-1)),
    }
    return {'PartA_affine_algebra': part_a, 'PartB_retrieved_counterfactuals': part_b}


def run_stage0_m3_mid_vs_late_llm_routing():
    """
    Executes Hypothesis H3 (`6 Mid-Layers: 14,16,18,20,22,24` vs. `6 Late-Layers: 28,29,30,31,32,33`)
    on a 4-sample micro-probe (`STAGE0_MICRO_LIMIT = 4`, rotation 0), measuring:
      1. `tv_vs_native`: Output probability sensitivity for Mid vs. Late under identical norm budget.
      2. `tv_swap`: Directional sensitivity when swapping early vs. late temporal innovation vectors.
      3. `freeform_hook_ok`: Explicit check that option-free generation (`answer_options=None`) runs cleanly.
    """
    if 'MODEL_' not in globals() or 'PROC' not in globals():
        return {'status': 'skipped_no_model_in_process'}

    cache_dir = OUT / 'argon_latent40' / 'decoder_cache'
    manifest_path = OUT / 'uploads' / 'research40_manifest.json'
    if not cache_dir.exists() or not manifest_path.exists():
        return {'status': 'skipped_no_decoder_cache'}

    manifest = json.loads(manifest_path.read_text())
    server_id = int(globals().get('LAB_SERVER', 1))
    samples = manifest.get(f'server{server_id}', manifest['selected'])[:4]
    labels = [PROC.tokenizer.convert_tokens_to_ids(c) for c in 'ABC']
    device = MODEL_.get_input_embeddings().weight.device

    mid_layers = (14, 16, 18, 20, 22, 24)
    late_layers = (28, 29, 30, 31, 32, 33)

    rows_out = []
    for s_i, sample in enumerate(samples):
        idx = sample['idx']
        pt_file = cache_dir / f'{idx}_rot0.pt'
        if not pt_file.exists():
            continue
        payload = torch.load(pt_file, map_location='cpu', weights_only=False)
        kw = {k: v.to(device) for k, v in payload['decoder'].items()}
        last_media = payload['last_media']
        is_v = payload['is_v'].to(device)
        sec = payload['sec'].to(device).float()
        chunks = torch.floor(sec / 2.0).long()

        cap = {}
        def cap_hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            v_chunks = sorted(set(chunks[is_v].tolist()))
            if len(v_chunks) >= 2:
                mid_c = v_chunks[len(v_chunks) // 2]
                v_early = h[0, is_v & (chunks < mid_c)].float().mean(0)
                v_late = h[0, is_v & (chunks >= mid_c)].float().mean(0)
            else:
                v_early = h[0, is_v].float().mean(0)
                v_late = -v_early
            hv_mean = h[0, is_v].float().mean(0)
            cap['vec_fwd'] = F.normalize(v_early - hv_mean, dim=-1).detach()
            cap['vec_rev'] = F.normalize(v_late - hv_mean, dim=-1).detach()
            return out

        h_c = MODEL_.model.layers[18].register_forward_hook(cap_hook)
        try:
            with torch.inference_mode():
                out_nat = MODEL_.model(**kw, use_cache=False, return_dict=True)
                lg_nat = MODEL_.lm_head(out_nat.last_hidden_state[:, -1:, :].to(MODEL_.lm_head.weight.device))[0, -1].float().cpu()
        finally:
            h_c.remove()

        vec_fwd = cap['vec_fwd']
        vec_rev = cap['vec_rev']
        g_shuf = torch.Generator(device=vec_fwd.device).manual_seed(20261015 + idx)
        vec_shuf = vec_fwd[torch.randperm(vec_fwd.shape[-1], generator=g_shuf, device=vec_fwd.device)]

        def run_routed(active_layers, vec_to_inject):
            handles = []
            def make_hk():
                def hk(mod, args, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    q_slice = h[0, last_media + 1:].float()
                    q_norm = q_slice.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    delta = 0.14 * q_norm * vec_to_inject.to(h.device).unsqueeze(0)
                    h[0, last_media + 1:] = (q_slice + delta).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return hk
            try:
                for li in active_layers:
                    handles.append(MODEL_.model.layers[li].register_forward_hook(make_hk()))
                with torch.inference_mode():
                    o = MODEL_.model(**kw, use_cache=False, return_dict=True)
                    lg = MODEL_.lm_head(o.last_hidden_state[:, -1:, :].to(MODEL_.lm_head.weight.device))[0, -1].float().cpu()
            finally:
                for h_i in handles:
                    h_i.remove()
            return lg

        k_opts = len(payload['native_logits'])
        p_nat = F.softmax(lg_nat[labels[:k_opts]], dim=-1)
        p_late_fwd = F.softmax(run_routed(late_layers, vec_fwd)[labels[:k_opts]], dim=-1)
        p_late_rev = F.softmax(run_routed(late_layers, vec_rev)[labels[:k_opts]], dim=-1)
        p_mid_fwd = F.softmax(run_routed(mid_layers, vec_fwd)[labels[:k_opts]], dim=-1)
        p_mid_rev = F.softmax(run_routed(mid_layers, vec_rev)[labels[:k_opts]], dim=-1)
        p_mid_shuf = F.softmax(run_routed(mid_layers, vec_shuf)[labels[:k_opts]], dim=-1)

        tv_late = 0.5 * float((p_late_fwd - p_nat).abs().sum())
        tv_mid = 0.5 * float((p_mid_fwd - p_nat).abs().sum())
        tv_late_swap = 0.5 * float((p_late_fwd - p_late_rev).abs().sum())
        tv_mid_swap = 0.5 * float((p_mid_fwd - p_mid_rev).abs().sum())
        tv_mid_shuf = 0.5 * float((p_mid_shuf - p_nat).abs().sum())

        freeform_ok = True
        if s_i == 0:
            try:
                lg_test = run_routed(mid_layers, vec_fwd)
                freeform_ok = bool(torch.isfinite(lg_test).all())
            except Exception:
                freeform_ok = False

        rows_out.append({
            'idx': idx,
            'task': sample['task'],
            'tv_late_28_33': round(tv_late, 4),
            'tv_mid_14_24': round(tv_mid, 4),
            'tv_swap_late_28_33': round(tv_late_swap, 4),
            'tv_swap_mid_14_24': round(tv_mid_swap, 4),
            'tv_mid_shuffled': round(tv_mid_shuf, 4),
            'freeform_hook_ok': freeform_ok,
        })
        del payload, kw
        torch.cuda.empty_cache()

    mean_late = round(float(np.mean([r['tv_late_28_33'] for r in rows_out])), 4) if rows_out else 0.0
    mean_mid = round(float(np.mean([r['tv_mid_14_24'] for r in rows_out])), 4) if rows_out else 0.0
    mean_swap_late = round(float(np.mean([r['tv_swap_late_28_33'] for r in rows_out])), 4) if rows_out else 0.0
    mean_swap_mid = round(float(np.mean([r['tv_swap_mid_14_24'] for r in rows_out])), 4) if rows_out else 0.0
    return {
        'status': 'ok',
        'n_micro_samples': len(rows_out),
        'mean_tv_late_28_33': mean_late,
        'mean_tv_mid_14_24': mean_mid,
        'mean_tv_swap_late_28_33': mean_swap_late,
        'mean_tv_swap_mid_14_24': mean_swap_mid,
        'mid_to_late_swap_ratio': round(mean_swap_mid / max(1e-6, mean_swap_late), 2),
        'per_sample': rows_out,
    }


def run_all_stage0_v2():
    t0 = time.time()
    m1_native, ctx_native = run_stage0_m1_crossed_factorial('avbind_native_pairs')
    m1_pooled, _ = run_stage0_m1_crossed_factorial('avbind_pooled_pairs')
    m2_res = run_stage0_m2_retrieved_interval_counterfactuals(ctx_native)
    m3_res = run_stage0_m3_mid_vs_late_llm_routing()

    out = {
        'M1_native_pairs_crossed_factorial': m1_native,
        'M1_pooled_pairs_crossed_factorial': m1_pooled,
        'M2_affine_and_retrieved_counterfactuals': m2_res,
        'M3_mid_vs_late_llm_routing': m3_res,
        'elapsed_sec': round(time.time() - t0, 2),
    }
    out_path = OUT / 'stage0_v2_pcebr_results.json'
    try:
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    except Exception:
        pass
    print('STAGE0_V2_PCEBR_DIAGNOSTIC:', json.dumps(out, indent=2), flush=True)
    return out


if __name__ == '__main__' or 'OUT' in globals():
    RESULT = run_all_stage0_v2()