# JEP-CLC / PC-EBR Stage-0 v7 — kết quả và kiểm toán, 2026-10-16

**Micro8: native 5/8; CAOR-only 5/8; v7 full 5/8. Cứu 0, làm hỏng 0. M0 gốc thất bại assertion dòng827 trên cả hai server.**

**Audit cơ chế: audio salience/gate bằng0 trên cả8 mẫu, kéo joint_gate và toàn bộ14D Psi về0. Phần joint event-pair mới chưa hoạt động trên micro8; outer routing/CLC vẫn có thể tác động. Khi real mass≈1e−14 bị chia cho denominator clamp1e−12, J_cond có tổng≈.01 thay vì1. Xem [deep_mechanism_audit.json](deep_mechanism_audit.json).**

M1/M2 dùng cùng252 pseudo pairs (202 train/16 origins;50 held-out/4 origins đã xem) trên hai runtimes2xT4. Đây là replicas, không tăng số recordings độc lập. Native cache là post-media last-token readout block27; pooled là mean media states cùng block. Source gốc giữ nguyên SHA; M3 gọi trained C6 + M2 probe + trained six-scalar bridge. Chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.

Xem [protocol.md](protocol.md), [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md) và [math_note_augmented_dustbin.md](math_note_augmented_dustbin.md).

## M0 — lỗi của bản gốc và diagnostic

Bản gốc dừng trước M1 vì assert `bimod_std > .30`. Feature13 là std_J−std_uniform, không phải raw std. Ví dụ bimodal có raw std=.4 bằng baseline nên feature=0; synchronous≈−.4. W+=W−≈.175 và max feature difference=1, nên vector14D vẫn phân biệt hai ví dụ. Không đổi ngưỡng để tuyên bố pass. Diagnostic riêng quan sát mọi assert và gọi các pha M1–M3 nguyên bản sau đó; logs nguồn lỗi được giữ ở original_m0_failure và local job_run_s*.json.

Original M0 diagnostic:

```json
{
  "basis_test_diag_output": [
    2.0,
    0.0,
    0.0,
    0.0
  ],
  "basis_test_lowrank_output": [
    2.0,
    2.0,
    2.0,
    2.0
  ],
  "yann_tiny_noise_1e4_v7_psi_max_abs": 3.764451586823342e-15,
  "bimodal_zero_mean_lag_counterexample": {
    "signed_w_diff_bimodal": 0.0,
    "unsigned_w_l1_bimodal": 0.35,
    "lag_std_bimodal": 0.0,
    "psi_14d_sync_vs_bimodal_max_diff": 1.0
  },
  "prescribed_marginal_sinkhorn_and_augmented_dustbin_check": {
    "j_aug_row_marginal_max_err": 6e-08,
    "j_aug_col_marginal_max_err": 3e-08,
    "j_cond_row_marginal_max_err": 3e-08,
    "j_cond_col_marginal_max_err": 3e-08,
    "real_block_vs_active_times_jcond_max_err": 1.5e-08,
    "union_dustbin_mass_consistency_err": 6e-08,
    "shared_dustbin_helper_used_in_train_and_infer": true,
    "factorized_spurious_cross_pair_mass": 0.496302,
    "sinkhorn_joint_spurious_cross_pair_mass": 0.010906
  },
  "uniform_zero_evidence_gate": 0.0,
  "freeform_evaluator_checks": {
    "all_5_counterexamples_and_1929_non_speech_ref_rejected": true,
    "id2150_and_id1821_explicit_after_speech_recovered": true
  },
  "toy_9cases_bivector_reversal_audit": [
    {
      "K": 5,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 1.2e-07,
      "pair_omega_nonadditive_max_abs": 0.050761,
      "rel_scalar_fwd": 1.46726,
      "rel_scalar_rev": -1.46726
    },
    {
      "K": 5,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 6e-08,
      "pair_omega_nonadditive_max_abs": 0.056435,
      "rel_scalar_fwd": -0.043321,
      "rel_scalar_rev": 0.043321
    },
    {
      "K": 5,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 1e-08,
      "pair_omega_nonadditive_max_abs": 0.042131,
      "rel_scalar_fwd": -2.253389,
      "rel_scalar_rev": 2.253389
    },
    {
      "K": 6,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 6e-08,
      "pair_omega_nonadditive_max_abs": 0.06204,
      "rel_scalar_fwd": 2.011294,
      "rel_scalar_rev": -2.011294
    },
    {
      "K": 6,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 6e-08,
      "pair_omega_nonadditive_max_abs": 0.064187,
      "rel_scalar_fwd": 0.149853,
      "rel_scalar_rev": -0.149853
    },
    {
      "K": 6,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 0.0,
      "order_odd_max_err": 0.0,
      "psi_14d_symmetry_max_err": 3e-08,
      "pair_omega_nonadditive_max_abs": 0.059178,
      "rel_scalar_fwd": -2.718007,
      "rel_scalar_rev": 2.718007
    },
    {
      "K": 9,
      "seed": 101,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 6e-08,
      "pair_omega_nonadditive_max_abs": 0.078902,
      "rel_scalar_fwd": 2.974573,
      "rel_scalar_rev": -2.974573
    },
    {
      "K": 9,
      "seed": 202,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 6e-08,
      "pair_omega_nonadditive_max_abs": 0.084905,
      "rel_scalar_fwd": -3.027298,
      "rel_scalar_rev": 3.027298
    },
    {
      "K": 9,
      "seed": 303,
      "vec_fwd_rev_cosine": 0.161103,
      "content_even_max_err": 1e-08,
      "order_odd_max_err": 1e-08,
      "psi_14d_symmetry_max_err": 1.2e-07,
      "pair_omega_nonadditive_max_abs": 0.070555,
      "rel_scalar_fwd": -0.988974,
      "rel_scalar_rev": 0.988974
    }
  ]
}
```

Các kiểm tra toy không xác nhận event matching, null calibration hoặc QA correctness.

## M1 — từng seed

Global/Exact/Multipos có mẫu số50; Nonnull31 theo substring filter. Global chấm normalized caption content, không yêu cầu đúng recording/interval. Exact là đúng index trong cùng recording/modality; Multipos chấp nhận mô tả trùng trong group. Số sai = mẫu số−số đúng. Ba seeds không tạo thêm held-out origins.

### M1_native

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.0/32.0/42.0/32.3 |
| C2_ctx_diag_w20_no_whiten | [11, 11, 11] | [16, 16, 16] | [21, 21, 21] | [10, 10, 10] | 22.0/32.0/42.0/32.3 |
| C3a_txt_std_diag_w20_v4_exact_replay | [15, 15, 15] | [20, 20, 20] | [26, 26, 26] | [12, 12, 12] | 30.0/40.0/52.0/38.7 |
| C3b_txt_std_diag_w20_v5_ref | [15, 12, 13] | [20, 18, 19] | [26, 24, 25] | [12, 10, 11] | 26.67/38.0/50.0/35.5 |
| C4_pure_mod_diag_w20_no_txt_mean | [18, 18, 18] | [18, 18, 18] | [23, 23, 23] | [9, 9, 9] | 36.0/36.0/46.0/29.0 |
| C5_pure_mod_zca_w20_no_txt_mean | [8, 8, 8] | [15, 15, 15] | [18, 18, 18] | [8, 8, 8] | 16.0/30.0/36.0/25.8 |
| C6_bilevel_global_plus_innov_w20 | [18, 17, 17] | [19, 18, 18] | [25, 24, 24] | [10, 9, 9] | 34.67/36.67/48.67/30.1 |
| C8a_shuffled_c3b_w20_ctrl | [13, 11, 15] | [19, 18, 17] | [21, 21, 18] | [13, 10, 13] | 26.0/36.0/40.0/38.7 |
| C8b_innov_only_shuffled_c6_ctrl | [11, 11, 10] | [15, 13, 16] | [19, 14, 21] | [10, 9, 9] | 21.33/29.33/36.0/30.1 |
| C8c_both_branches_shuffled_c6_ctrl | [8, 10, 9] | [14, 13, 15] | [18, 14, 20] | [9, 9, 8] | 18.0/28.0/34.67/27.93 |

### M1_pooled

| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |
|---|---|---|---|---|---|
| C1_ctx_only_diag_w00 | [16, 16, 16] | [19, 19, 19] | [23, 23, 23] | [14, 14, 14] | 32.0/38.0/46.0/45.2 |
| C2_ctx_diag_w20_no_whiten | [16, 16, 16] | [20, 20, 20] | [24, 24, 24] | [14, 14, 14] | 32.0/40.0/48.0/45.2 |
| C3a_txt_std_diag_w20_v4_exact_replay | [22, 21, 21] | [21, 20, 20] | [24, 23, 23] | [15, 14, 14] | 42.67/40.67/46.67/46.27 |
| C3b_txt_std_diag_w20_v5_ref | [23, 23, 23] | [23, 23, 23] | [26, 26, 26] | [16, 16, 16] | 46.0/46.0/52.0/51.6 |
| C4_pure_mod_diag_w20_no_txt_mean | [23, 23, 23] | [22, 22, 22] | [25, 25, 25] | [16, 16, 16] | 46.0/44.0/50.0/51.6 |
| C5_pure_mod_zca_w20_no_txt_mean | [16, 16, 16] | [19, 19, 19] | [19, 19, 19] | [15, 15, 15] | 32.0/38.0/38.0/48.4 |
| C6_bilevel_global_plus_innov_w20 | [22, 21, 22] | [22, 21, 22] | [25, 24, 25] | [15, 14, 15] | 43.33/43.33/49.33/47.33 |
| C8a_shuffled_c3b_w20_ctrl | [23, 23, 23] | [22, 21, 22] | [25, 24, 25] | [16, 15, 16] | 46.0/43.33/49.33/50.53 |
| C8b_innov_only_shuffled_c6_ctrl | [17, 18, 18] | [20, 22, 21] | [23, 26, 21] | [15, 15, 15] | 35.33/42.0/46.67/48.4 |
| C8c_both_branches_shuffled_c6_ctrl | [17, 18, 18] | [20, 22, 21] | [23, 26, 21] | [15, 15, 15] | 35.33/42.0/46.67/48.4 |

## M2 — số đếm từ raw logits

Pool held-out có4 origins; bank sizes8/3/5/9 mỗi modality. Điều kiện bank>=4 loại p-gN4cbmunQ khỏi target contexts, còn3 recording contexts đủ điều kiện; origin bị loại vẫn có thể cung cấp foil queries. Foil là text proxy chưa được annotation absence. Reversal/Roll đổi coordinates/bank assignments, không dịch waveform hoặc đảo frames. Acc3 dùng argmax riêng3 relation logits trên nonfoil subset. Paired chỉ xét các cặp còn đủ thuận/đảo sau balancing.

### PartB1_Natural_Unshifted_Balanced

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 123/109/232 | 53.0 | 64.4 | 41/58 | 22/58 | 32/58 | 28/58 | — |
| B1b_unmodified_softmax_mean_time_c6_matched | 119/113/232 | 51.3 | 59.8 | 35/58 | 28/58 | 28/58 | 28/58 | — |
| B2_coord_shuffle_control | 116/116/232 | 50.0 | 40.8 | 21/58 | 25/58 | 20/58 | 50/58 | — |
| B3_wrong_event_query_control | 47/185/232 | 20.3 | 37.9 | 13/58 | 11/58 | 12/58 | 11/58 | — |
| B4_uniform_retrieval_sim0_control | 58/174/232 | 25.0 | 33.3 | 12/58 | 0/58 | 46/58 | 0/58 | — |
| B5_barycenter_plus_psi_joint_no_null_c6 | 121/111/232 | 52.2 | 56.3 | 28/58 | 33/58 | 28/58 | 32/58 | — |
| B6_shift_sensitive_presence_null_only_c6 | 105/127/232 | 45.3 | 33.3 | 7/58 | 31/58 | 13/58 | 54/58 | — |
| B7_barycenter_plus_presence_NO_PSI_yann_ctrl_c6 | 148/84/232 | 63.8 | 59.2 | 28/58 | 37/58 | 33/58 | 50/58 | — |
| B8_full_bary_psi_joint_presence_c6_connected | 143/89/232 | 61.6 | 58.6 | 30/58 | 38/58 | 29/58 | 46/58 | — |

### PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 248/328/576 | 43.1 | 60.0 | 95/144 | 32/144 | 81/144 | 40/144 | 125/216 |
| B1b_unmodified_softmax_mean_time_c6_matched | 222/354/576 | 38.5 | 54.6 | 78/144 | 33/144 | 66/144 | 45/144 | 113/216 |
| B2_coord_shuffle_control | 311/265/576 | 54.0 | 43.5 | 54/144 | 52/144 | 71/144 | 134/144 | 56/216 |
| B3_wrong_event_query_control | 176/400/576 | 30.6 | 42.1 | 54/144 | 37/144 | 55/144 | 30/144 | 88/216 |
| B4_uniform_retrieval_sim0_control | 144/432/576 | 25.0 | 33.3 | 0/144 | 120/144 | 24/144 | 0/144 | 60/216 |
| B5_barycenter_plus_psi_joint_no_null_c6 | 263/313/576 | 45.7 | 55.8 | 54/144 | 84/144 | 49/144 | 76/144 | 120/216 |
| B6_shift_sensitive_presence_null_only_c6 | 274/302/576 | 47.6 | 37.0 | 43/144 | 82/144 | 13/144 | 136/144 | 44/216 |
| B7_barycenter_plus_presence_NO_PSI_yann_ctrl_c6 | 349/227/576 | 60.6 | 53.2 | 67/144 | 85/144 | 67/144 | 130/144 | 114/216 |
| B8_full_bary_psi_joint_presence_c6_connected | 350/226/576 | 60.8 | 55.8 | 67/144 | 92/144 | 66/144 | 125/144 | 120/216 |

### PartB3_CrossModal_Only_TwoHop_Roll

| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |
|---|---|---:|---:|---|---|---|---|---|
| B1_unmodified_softmax_mean_time_c1_ref | 32/124/156 | 20.5 | 33.3 | 8/39 | 6/39 | 2/39 | 16/39 | — |
| B1b_unmodified_softmax_mean_time_c6_matched | 36/120/156 | 23.1 | 40.2 | 17/39 | 6/39 | 9/39 | 4/39 | — |
| B2_coord_shuffle_control | 55/101/156 | 35.3 | 38.5 | 14/39 | 7/39 | 13/39 | 21/39 | — |
| B3_wrong_event_query_control | 37/119/156 | 23.7 | 34.2 | 14/39 | 4/39 | 13/39 | 6/39 | — |
| B4_uniform_retrieval_sim0_control | 39/117/156 | 25.0 | 33.3 | 0/39 | 0/39 | 39/39 | 0/39 | — |
| B5_barycenter_plus_psi_joint_no_null_c6 | 46/110/156 | 29.5 | 41.0 | 9/39 | 8/39 | 13/39 | 16/39 | — |
| B6_shift_sensitive_presence_null_only_c6 | 49/107/156 | 31.4 | 32.5 | 18/39 | 2/39 | 4/39 | 25/39 | — |
| B7_barycenter_plus_presence_NO_PSI_yann_ctrl_c6 | 61/95/156 | 39.1 | 39.3 | 17/39 | 8/39 | 8/39 | 28/39 | — |
| B8_full_bary_psi_joint_presence_c6_connected | 57/99/156 | 36.5 | 41.0 | 9/39 | 11/39 | 16/39 | 21/39 | — |
| B8a_teacher_forced_bridge_ctrl | 56/100/156 | 35.9 | 40.2 | 10/39 | 11/39 | 15/39 | 20/39 | — |
| B8b_full_oracle_hop1_upper_bound | 105/51/156 | 67.3 | 58.1 | 26/39 | 20/39 | 21/39 | 38/39 | — |
| B8c_subspace_unmixer_NO_bg_subtract_ctrl | 60/96/156 | 38.5 | 38.5 | 8/39 | 11/39 | 15/39 | 26/39 | — |
| B8d_factorized_outer_product_NO_sinkhorn_ctrl | 57/99/156 | 36.5 | 41.0 | 9/39 | 11/39 | 16/39 | 21/39 | — |
| B8e_legacy_6d_phi_emd_matched_ctrl | 58/98/156 | 37.2 | 41.9 | 11/39 | 8/39 | 13/39 | 26/39 | — |

### Foil/hop audit

```json
{
  "PartB1_Natural_Unshifted_Balanced": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations."
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations."
    }
  },
  "PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations."
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations."
    }
  },
  "PartB3_CrossModal_Only_TwoHop_Roll": {
    "train": {
      "total_origins": 16,
      "strict_cos015_and_jaccard0_origins": 5,
      "lexical_jaccard0_origins": 10,
      "string_disjoint_fallback_origins": 1,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations.",
      "retained_post_balancing_hop_diagnostics": {
        "n_nonfoil_episodes": 333,
        "hop1_correct": 192,
        "hop1_top1_acc_pct": 57.7,
        "hop2_correct": 186,
        "hop2_top1_acc_pct": 55.9,
        "hop2_given_hop1_correct_total": 192,
        "hop2_given_hop1_correct_hits": 111,
        "hop2_acc_given_hop1_correct_pct": 57.8,
        "hop2_given_hop1_wrong_total": 141,
        "hop2_given_hop1_wrong_hits": 75,
        "hop2_acc_given_hop1_wrong_pct": 53.2
      },
      "pre_balancing_candidate_pool_hop_diagnostics": {
        "n_nonfoil_episodes": 333,
        "hop1_correct": 192,
        "hop1_top1_acc_pct": 57.7,
        "hop2_correct": 186,
        "hop2_top1_acc_pct": 55.9,
        "hop2_given_hop1_correct_total": 192,
        "hop2_given_hop1_correct_hits": 111,
        "hop2_acc_given_hop1_correct_pct": 57.8,
        "hop2_given_hop1_wrong_total": 141,
        "hop2_given_hop1_wrong_hits": 75,
        "hop2_acc_given_hop1_wrong_pct": 53.2
      }
    },
    "heldout": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations.",
      "retained_post_balancing_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 45,
        "hop1_top1_acc_pct": 38.5,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 45,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 20.0,
        "hop2_given_hop1_wrong_total": 72,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 33.3
      },
      "pre_balancing_candidate_pool_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 45,
        "hop1_top1_acc_pct": 38.5,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 45,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 20.0,
        "hop2_given_hop1_wrong_total": 72,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 33.3
      }
    },
    "heldout_teacher_forced_bridge_hop_audit": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations.",
      "retained_post_balancing_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 45,
        "hop1_top1_acc_pct": 38.5,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 45,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 20.0,
        "hop2_given_hop1_wrong_total": 72,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 33.3
      },
      "pre_balancing_candidate_pool_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 45,
        "hop1_top1_acc_pct": 38.5,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 45,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 20.0,
        "hop2_given_hop1_wrong_total": 72,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 33.3
      }
    },
    "heldout_full_oracle_hop1_audit": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations.",
      "retained_post_balancing_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 117,
        "hop1_top1_acc_pct": 100.0,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 117,
        "hop2_given_hop1_correct_hits": 33,
        "hop2_acc_given_hop1_correct_pct": 28.2,
        "hop2_given_hop1_wrong_total": 0,
        "hop2_given_hop1_wrong_hits": 0,
        "hop2_acc_given_hop1_wrong_pct": 0.0
      },
      "pre_balancing_candidate_pool_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 117,
        "hop1_top1_acc_pct": 100.0,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 117,
        "hop2_given_hop1_correct_hits": 33,
        "hop2_acc_given_hop1_correct_pct": 28.2,
        "hop2_given_hop1_wrong_total": 0,
        "hop2_given_hop1_wrong_hits": 0,
        "hop2_acc_given_hop1_wrong_pct": 0.0
      }
    },
    "heldout_no_bg_subtract_hop_audit": {
      "total_origins": 4,
      "strict_cos015_and_jaccard0_origins": 0,
      "lexical_jaccard0_origins": 2,
      "string_disjoint_fallback_origins": 2,
      "uniqueness_filter_type": "string_unique_nonnull_proxy",
      "note": "Caption/embedding disjointness and string uniqueness are text-level proxies, not independent ground-truth absence or event-instance annotations.",
      "retained_post_balancing_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 48,
        "hop1_top1_acc_pct": 41.0,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 48,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 18.8,
        "hop2_given_hop1_wrong_total": 69,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 34.8
      },
      "pre_balancing_candidate_pool_hop_diagnostics": {
        "n_nonfoil_episodes": 117,
        "hop1_correct": 48,
        "hop1_top1_acc_pct": 41.0,
        "hop2_correct": 33,
        "hop2_top1_acc_pct": 28.2,
        "hop2_given_hop1_correct_total": 48,
        "hop2_given_hop1_correct_hits": 9,
        "hop2_acc_given_hop1_correct_pct": 18.8,
        "hop2_given_hop1_wrong_total": 69,
        "hop2_given_hop1_wrong_hits": 24,
        "hop2_acc_given_hop1_wrong_pct": 34.8
      }
    }
  }
}
```

## M3 — cố định8 mẫu rotation0

| ID | Task/gold | Native | CAOR-only | V7 full | True/wrong semantic gate | TV full/native |
|---|---|---|---|---|---|---|
| 1449 | TA/A | A | A | A | 0.8228/0.8908 | 0.0115 |
| 2150 | TF/C | A | A | A | 0.1216/0.9811 | 0.0075 |
| 1929 | TF/C | C | C | C | 0.7818/0.9742 | 0.0534 |
| 2054 | TA/B | A | A | A | 0.0844/0.9831 | 0.0038 |
| 1956 | CMTB/A | A | A | A | 0.0274/0.9516 | 0.0009 |
| 1513 | TF/A | B | B | B | 0.6233/0.9769 | 0.0153 |
| 2264 | CMTB/B | B | B | B | 0.0001/0.9783 | 0.0 |
| 1821 | TF/C | C | C | C | 0.7642/0.9759 | 0.0077 |

| Task/gold | N | Native đúng/sai | CAOR-only đúng/sai | Full đúng/sai |
|---|---:|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo |

All source MC conditions correct/8: `{"mc_pred_native": 5, "mc_pred_legacy_v5_vo_tangent": 5, "mc_pred_caor_attn_only": 5, "mc_pred_routed_fwd": 5, "mc_pred_no_desink": 5, "mc_pred_order_sign_rev": 5, "mc_pred_slot_profile_rev": 5, "mc_pred_rel_only": 5, "mc_pred_order_only": 5}`.
Means: `{"confidence_gate_gamma_true_query": 0.4032, "confidence_gate_gamma_legacy_v5": 0.9464875, "confidence_gate_gamma_wrong_query": 0.0, "confidence_gate_gamma_semantic_wrong_query": 0.9639875, "tv_late_kslot_swap": 0.000975, "tv_mid_kslot_swap": 0.0066375, "tv_mid_slot_profile_swap": 0.0083125, "tv_mid_routed_vs_nat": 0.012512500000000001, "tv_mid_irrel_slot_vs_nat": 0.0058249999999999994, "tv_mid_rand3seed_vs_nat": 0.00675}`.

Gamma measures concentration/excess with fixed thresholds, not an independently calibrated event-presence probability. Source 2x2 gate diagnostics also vary excess cap and therefore do not isolate a true factorial design. Mid/late TV sensitivity is not directional accuracy; full mid-vs-reversed includes CLC while late is attention-only.

### Original-control accuracy and exploratory CLC controls

| Condition | Đúng/sai/8 |
|---|---|
| original_irrelevant_slot_attn_only | 5/3/8 |
| original_random_profile_101_attn_only | 5/3/8 |
| original_random_profile_202_attn_only | 5/3/8 |
| original_random_profile_303_attn_only | 5/3/8 |
| expl_full_caor_random_clc_101 | 5/3/8 |
| expl_full_caor_random_clc_202 | 5/3/8 |
| expl_full_caor_random_clc_303 | 5/3/8 |
| expl_full_caor_sink_matched_event_clc | 5/3/8 |
| expl_optionfree_prepass_clc | 5/3/8 |

Original random/irrelevant controls are attention-only. Exploratory random-CLC seeds101/202/303 keep full attention profiles and CLC strength fixed; sink-matched contrast changes only the foil sink correction before reconstructing the latent direction. Option-free-prepass control recomputes support/foil/reverse L22 states with choices absent. These are separate post-mainrun controls, not mainrun changes or tuned variants.

The original generated freeform prompt excludes choices but reuses a CLC direction built from the MC prompt with choices. Therefore original freeform is not fully option-independent.

### Actual-input coupling audit

```json
[
  {
    "file": "m3_1449.pt",
    "K": 10,
    "joint_gate": 0.0,
    "dustbin_score": -2.160937786102295,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 5.960464477539063e-08,
    "J_aug_col_error": 5.960464477539063e-08,
    "J_cond_row_error": 0.0989999994635582,
    "J_cond_col_error": 0.4670923054218292,
    "cond_distance_from_outer_L1": 0.9900000095367432,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_1929.pt",
    "K": 12,
    "joint_gate": 0.0,
    "dustbin_score": -0.7786769866943359,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 5.960464477539063e-08,
    "J_aug_col_error": 1.2434497875801753e-14,
    "J_cond_row_error": 0.08250000327825546,
    "J_cond_col_error": 0.18281123042106628,
    "cond_distance_from_outer_L1": 0.9900000095367432,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_2054.pt",
    "K": 12,
    "joint_gate": 0.0,
    "dustbin_score": -0.8945080041885376,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 2.6645352591003757e-15,
    "J_aug_col_error": 1.4210854715202004e-14,
    "J_cond_row_error": 0.08250000327825546,
    "J_cond_col_error": 0.23443645238876343,
    "cond_distance_from_outer_L1": 0.9899998903274536,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_2150.pt",
    "K": 6,
    "joint_gate": 0.0,
    "dustbin_score": -0.2724851369857788,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 5.960464477539063e-08,
    "J_aug_col_error": 5.960464477539063e-08,
    "J_cond_row_error": 0.16500000655651093,
    "J_cond_col_error": 0.2636430263519287,
    "cond_distance_from_outer_L1": 0.9900001287460327,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_1513.pt",
    "K": 7,
    "joint_gate": 0.0,
    "dustbin_score": -0.6272560358047485,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 1.4210854715202004e-14,
    "J_aug_col_error": 5.960464477539063e-08,
    "J_cond_row_error": 0.14142857491970062,
    "J_cond_col_error": 0.36513447761535645,
    "cond_distance_from_outer_L1": 0.9899998903274536,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_1821.pt",
    "K": 9,
    "joint_gate": 0.0,
    "dustbin_score": -1.1213654279708862,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 1.1920928955078125e-07,
    "J_aug_col_error": 5.960464477539063e-08,
    "J_cond_row_error": 0.10999999940395355,
    "J_cond_col_error": 0.37402623891830444,
    "cond_distance_from_outer_L1": 0.9900000095367432,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_1956.pt",
    "K": 11,
    "joint_gate": 0.0,
    "dustbin_score": 0.293304443359375,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 1.1920928955078125e-07,
    "J_aug_col_error": 5.960464477539063e-08,
    "J_cond_row_error": 0.09000000357627869,
    "J_cond_col_error": 0.09561619162559509,
    "cond_distance_from_outer_L1": 0.9900001287460327,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  },
  {
    "file": "m3_2264.pt",
    "K": 12,
    "joint_gate": 0.0,
    "dustbin_score": 0.5021774768829346,
    "act_target": 1.0000000116860974e-07,
    "P_null": 1.0,
    "analytic_one_minus_act_squared": 1.0,
    "analytic_pnull_error": 0.0,
    "P_null_zero_pair_omega": 1.0,
    "pnull_change_with_fixed_inputs_zero_omega": 0.0,
    "J_aug_row_error": 5.960464477539063e-08,
    "J_aug_col_error": 7.105427357601002e-15,
    "J_cond_row_error": 0.08250000327825546,
    "J_cond_col_error": 0.0825188010931015,
    "cond_distance_from_outer_L1": 0.9900000095367432,
    "marginal_only_phi_channels_9_10_11_12_max_change": 0.0
  }
]
```

Important: all8 actual-input audits have joint_gate0. The stored Psi vectors must be read with deep_mechanism_audit.json: low real mass≈1e−14 is divided by clamp(min=1e−12), returning J_cond total≈.01 and breaking its normalized/marginal claims. Multiplication by joint_gate0 masks the resulting Phi channels. The outer routing gamma can remain nonzero because it uses max(alpha_a,alpha_v) plus an excess cap, while the joint gate uses their product. Thus a non-additive omega does not establish active pair binding.

Under converged positive-mass dual-calibrated construction, real-event mass=a² and P_null=1−a²; at fixed marginals/score/gate pair compatibility supplies no extra null evidence. The independent CPU checks validate this with nonzero gate; the8 QA inputs all use the zero-gate regime and therefore do not independently test positive-gate coupling. Wasserstein/Bhattacharyya/mean-lag channels depend only on prescribed marginals. The lag RBF/synchrony/dispersion channels can still use joint interactions. See the derivation and numerical residuals; finite iteration can produce departures.

## Generated text and heuristic audit

Original heuristic counts: native 3/8, routed 3/8. This rule has been revised after earlier samples were observed; it is not independently annotated semantic accuracy.

### 1449 — TA/A

Native: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can see the chickpeas being added to

V7: Yes, the narrated event of adding chickpeas is happening at the same time in the video. You can see the chickpeas being added to the

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": true, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`

### 2150 — TF/C

Native: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

V7: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`

### 1929 — TF/C

Native: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

V7: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": true, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

### 2054 — TA/B

Native: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

V7: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": false, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

### 1956 — CMTB/A

Native: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

V7: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

### 1513 — TF/A

Native: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it's right

V7: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it's right

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": false, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

### 2264 — CMTB/B

Native: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

V7: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": null, "cmtb_content_matches_gold": false, "heuristic_clause_aligned": false, "semantic_gold_aligned": false, "is_heuristic_not_human_semantic_oracle": true}`

### 1821 — TF/C

Native: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

V7: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

Native audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`
V7 audit: `{"mentioned_timestamps_sec": [], "has_hallucinated_timestamp_beyond_duration": false, "is_generic_or_evasive_non_answer": false, "yes_no_polarity_matches_gold": null, "tf_relation_matches_gold": true, "cmtb_content_matches_gold": null, "heuristic_clause_aligned": true, "semantic_gold_aligned": true, "is_heuristic_not_human_semantic_oracle": true}`

## Provenance

Source LF SHA256 `d27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80`; all archive entries verified locally without extraction.

Native replay and active zero-noop results are in m3_audit*.json; same-runtime archived-v5 replay is checked from raw probabilities. Parameter version counters remained unchanged and hooks/CUDA health checks completed. Fixed preprocessing and sample IDs preserved; historical exact logits need not match fresh-runtime caches.

Archive metadata: `[{"server": 1, "file": "pcebr_stage0_v7/raw_artifacts.tar.gz", "bytes": 173203231, "sha256": "e4eba01e838d6a70b881d1fc32c9890c4615e26abc457a3709d8cb39e8641416", "raw_files": 215, "retrieval_queries": 3000, "probes": 32, "verified_every_entry": true}, {"server": 2, "file": "pcebr_stage0_v7/raw_artifacts.tar.gz", "bytes": 172159245, "sha256": "0acd6ce0f391db8e90ce1cbc144d9b21ac2c75eb3ec8ea195962e208cda643f7", "raw_files": 215, "retrieval_queries": 3000, "probes": 32, "verified_every_entry": true}]`.
