# Gửi Yoshua Bengio — kết quả và phản biện JEP-CLC v7, 2026-10-16

Yoshua, tôi đã chạy bản khóa cuối trên cả hai runtime2xT4. Tên Yoshua/Yann trong trao đổi là quy ước vai trò. Source LF SHA256 d27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80 khớp attachment và giữ nguyên trong artifacts.

**MC micro8: native 5/8; CAOR-only 5/8; JEP-CLC full 5/8. Cứu 0, làm hỏng 0; đổi đáp án IDs=[].**

**Phát hiện cơ chế quyết định: cả8 mẫu có audio salience logits bằng0, alpha_audio0, joint_gate0 và toàn bộ14D Psi bằng0. Pair_omega vẫn phi cộng tính nhưng effective interaction=joint_gate*omega bằng0. Như vậy phần joint reasoning mới chưa thực sự hoạt động trên micro8. Xem [deep_mechanism_audit.json](deep_mechanism_audit.json).**

Bản gốc không chạy hết tự động: M0 assertion dòng827 thất bại trên cả hai server. Sau diagnostic riêng, tôi gọi nguyên các hàm M1/M2/M3 theo thứ tự và ngân sách gốc; không sửa feature/loss/seed theo QA. Vì thế không được viết “original M0 qua100%”. Xem [report.md](report.md), [protocol.md](protocol.md) và raw logs.

## 1. Những gì đã thực sự đo

Hai replicas dùng cùng252 pseudo pairs:202 train/16 origins,50 held-out/four origins đã xem. M2 yêu cầu target banks>=4 slots; một origin held-out chỉ có3 slots/modality nên chỉ3 recording target contexts đủ điều kiện, origin thứ tư có thể cung cấp foil. Hàng trăm episodes, ba seeds và hai replicas không làm tăng số recording độc lập. M3 giữ đúng first4/shard,rotation0; TF/B và CMTB/C chưa đo. Chưa chạy lại QA40 đầy đủ, A0–A7 hoặc reserve400.

| Task/gold | N | Native đúng/sai | CAOR-only đúng/sai | V7 full đúng/sai |
|---|---:|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo |

## 2. M1 và M2: điểm nào có tín hiệu, điểm nào chưa có

M1 tái lập các số đếm v6. Pooled C6 mean Global/Exact=43.33/43.33%, dưới C3b46/46%; native C6 tăng Global nhưng không tăng localization nhất quán. Đây vẫn là trade-off giữa caption-level retrieval và định vị, không phải cải thiện đã xác lập. Global chấm normalized caption content, không đòi đúng recording/interval.

| Suite | Full đúng/tổng | Full Acc3 | Bỏ Psi đúng/tổng | C1 Acc3 |
|---|---|---:|---|---:|
| PartB1_Natural_Unshifted_Balanced | 143/232 (61.6%) | 58.6% | 148/232 (63.8%) | 64.4% |
| PartB2_Global_Physical_Stream_TimeReversal_NoRoleSplit | 350/576 (60.8%) | 55.8% | 349/576 (60.6%) | 60.0% |
| PartB3_CrossModal_Only_TwoHop_Roll | 57/156 (36.5%) | 41.0% | 61/156 (39.1%) | 33.3% |

Roll ablations:

| Variant | Đúng/sai/tổng | Acc3 |
|---|---|---:|
| B8_full_bary_psi_joint_presence_c6_connected | 57/99/156 | 41.0% |
| B2_coord_shuffle_control | 55/101/156 | 38.5% |
| B7_barycenter_plus_presence_NO_PSI_yann_ctrl_c6 | 61/95/156 | 39.3% |
| B8a_teacher_forced_bridge_ctrl | 56/100/156 | 40.2% |
| B8b_full_oracle_hop1_upper_bound | 105/51/156 | 58.1% |
| B8c_subspace_unmixer_NO_bg_subtract_ctrl | 60/96/156 | 38.5% |
| B8d_factorized_outer_product_NO_sinkhorn_ctrl | 57/99/156 | 41.0% |
| B8e_legacy_6d_phi_emd_matched_ctrl | 58/98/156 | 41.9% |

Full và control factorized trên Roll có cùng dự đoán ở toàn bộ156 episodes, dù logits khác (max≈.0537); bỏ Psi và legacy6D không yếu hơn full về các điểm tổng tương ứng. Đây là kết quả bất lợi cho implementation hiện tại của joint/14D, không bác bỏ mọi mô hình joint event-pair. Raw per-query comparisons ở archive_audit_s1.json.

Hop-2 v7 train186/333=55.9%, held-out33/117=28.2%, so v6 held-out21/117=17.9%. Train/test gap nhỏ hơn nhưng chưa biến mất. Hop-1 held-out45/117=38.5%; Hop-2 khi Hop-1 đúng9/45=20.0%. No-bg, teacher-forced và oracle đều có tổng Hop-2=33/117; xem [hop_condition_comparison_s1.json](hop_condition_comparison_s1.json) để phân biệt sự bằng nhau về số đếm và về từng query. Chưa xác nhận phép trừ nền hoặc anchor deflation là nguyên nhân của mức tăng so v6.

Oracle-label intervention tăng phân loại Roll lên67.3%, nhưng Hop-2 vẫn28.2%. Nó cung cấp tọa độ/anchor thật, còn boosting raw anchor affinity làm đổi cả null evidence; classifier được fit lại trên oracle features. Không được diễn giải là đã sửa event retrieval. Calibrated p1_hat còn shrinkage uniform floor, nên tên “true mathematical upper bound” quá mạnh; đây là một oracle intervention diagnostic.

Natural/Reversal gain Acc4 cần đọc cùng từng lớp và Acc3; phần foil proxy có thể giải thích nhiều gain mà quan hệ không tăng. Foils vẫn chưa được annotation rằng event thật sự vắng mặt; coordinate/bank reversal/roll chưa phải chỉnh waveform/frames.

## 3. Gate mới chưa được xác nhận là semantic relevance

Mean gate true=0.403200; semantic-wrong=0.963987.

| ID | True gate | Semantic-wrong gate |
|---|---:|---:|
| 1449 | 0.8228 | 0.8908 |
| 2150 | 0.1216 | 0.9811 |
| 1929 | 0.7818 | 0.9742 |
| 2054 | 0.0844 | 0.9831 |
| 1956 | 0.0274 | 0.9516 |
| 1513 | 0.6233 | 0.9769 |
| 2264 | 0.0001 | 0.9783 |
| 1821 | 0.7642 | 0.9759 |

2150:0.1216 true vs0.9811 wrong;1956:0.0274 vs0.9516. Các phản ví dụ này không phù hợp với kỳ vọng “wrong content=>excess≈0=>gate≈0”. Không được đổi threshold theo8 mẫu để làm đẹp gate. Một điểm vượt scaffold/concentration chưa phải xác suất hiện diện được hiệu chuẩn.

Audit hiện không phải factorial2x2 sạch: cell cuối thêm excess cap, các cell khác dùng sharp_gate_from_scores với cách chuẩn hóa/ngưỡng khác; tất cả còn được chấm trên patches chọn bởi v7 salience. Muốn kết luận tác động từng yếu tố, cần cùng pipeline/ngưỡng/cap cho cả4 cells và nói rõ giữ hoặc chọn lại patches.

Tổ hợp .65 mean+.35 max không yêu cầu mọi entity đều khớp. Ví dụ similarities[.9,-.1,-.1,-.1] cho.4125, cao hơn [.4,.4,.4,.4] cho.4. Với câu hỏi về hai sự kiện khác nhau ở hai thời điểm, ép toàn bộ content words khớp cùng một patch cũng không phải định nghĩa đúng của temporal binding. Nên học/đo riêng hai endpoint query roles rồi mới ghép cặp.

## 4. Augmented dustbin đã nối thật nhưng có dạng suy biến dễ kiểm tra

Ngoài lỗi M0, audit actual QA inputs phát hiện lỗi chuẩn hóa ở low-mass regime. act_target bị clamp tối thiểu1e−7; real mass≈a²≈1e−14, nhưng M_act lại clamp tối thiểu1e−12. J_cond=J_real/M_act có tổng≈.01, không phải1, và sai số marginal thực tế tới khoảng.467. Các kiểm tra M0 với gate1 không bắt được case này. Cần tính conditional normalization trong log-space hoặc fallback được định nghĩa rõ ở negligible mass, rồi test riêng sum/marginal ở gate0/tiny. Đây là sửa số học có thể làm; không tuyên bố nó tự khôi phục temporal QA vì Psi vẫn bị joint_gate0 triệt tiêu. Mainrun hiện giữ nguyên và báo lỗi này.

M3 nay truyền pair_omega và trained dustbin_theta; J_cond/P_null được lấy từ J_aug, nên lỗi source trước đã sửa. Tuy nhiên dual-calibrated borders tạo nghiệm hội tụ J_real=a² B, dustbin corner=(1−a)², P_null=1−a², với a=joint_gate*sigmoid(−z) và B là conditional coupling khớp biên. Tại fixed marginals/score/gate, pair compatibility không cung cấp thêm null evidence. Đây là giới hạn của công thức, không chỉ là vấn đề naming.

Actual-input maximum residuals: `{"analytic_pnull_error": 0.0, "pnull_change_with_fixed_inputs_zero_omega": 0.0, "J_aug_row_error": 1.1920928955078125e-07, "J_aug_col_error": 5.960464477539063e-08, "J_cond_row_error": 0.16500000655651093, "J_cond_col_error": 0.4670923054218292, "marginal_only_phi_channels_9_10_11_12_max_change": 0.0}`.

Xem [math_note_augmented_dustbin.md](math_note_augmented_dustbin.md), [independent_cpu_math.json](independent_cpu_math.json) và coupling_actual_input_audit_s*.json. W+/W−, Bhattacharyya và mean lag chỉ phụ thuộc marginals; joint coupling chỉ thay được những kênh histogram/synchrony/dispersion liên quan.14 số không bảo toàn đầy đủ một phân phối joint tùy ý; bỏ claim “giữ trọn vẹn hình dạng”. Nonadditive residual>0 chỉ chứng minh ma trận phi cộng tính, không chứng minh ghép đúng instance hoặc coupling khác đáng kể outer product.

M0 gốc nhầm raw std với excess std. Đã lưu [suggested_m0_test_fix.diff](suggested_m0_test_fix.diff), chỉ gợi ý sửa assertion để so chênh lệch feature giữa hai toy; bản sửa đó không được dùng để đổi mainrun.

## 5. CLC, controls và phạm vi option-free

Original MC counts: `{"mc_pred_native": 5, "mc_pred_legacy_v5_vo_tangent": 5, "mc_pred_caor_attn_only": 5, "mc_pred_routed_fwd": 5, "mc_pred_no_desink": 5, "mc_pred_order_sign_rev": 5, "mc_pred_slot_profile_rev": 5, "mc_pred_rel_only": 5, "mc_pred_order_only": 5}`.

| Control/extension | Đúng/sai/8 |
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

Cosine CLC direction có/không options: mean=0.311678. Các xác suất/predictions từng mẫu ở control_accuracy_s*.json. Không chọn best seed hoặc cường độ từ các số này.

CAOR-only vs full kiểm tra được bước cộng CLC. Nhưng random/irrel/no-desink/rel-only/order-only/slot-profile-reversed trong source là attention-only, còn full cóCLC; chúng không cô lập từng thành phần của full. TV mid order-swap cóCLC, TV late không cóCLC; ratio không chứng minh tầng giữa chính xác temporal hơn. Match norm của bias chưa match norm của actual attention-output correction.

Foil-Suppressed đặt kappa_sink0 thay vì.60 ởSupport; delta_ev trộn đảo bias với tắt sink correction. Tôi đã thêm exploratory sink-matched contrast cùng cường độ, báo riêng. Cả delta_ev và delta_ord vẫn là hướng do can thiệp định tuyến tạo ra, chưa được chứng minh “thuần bằng chứng” hoặc “khử sạch language prior”. Không có định lý cho phép suy tính thuần đó từ phép trừ hai hidden states trong decoder phi tuyến.

Slot extraction tránh MC options, nhưng source lấy h22_supp/foil/rev từ MC prompt chứa options, rồi tái dùng CLC direction đó cho generate_freeform đã bỏ options. Vì thế original freeform còn nhận thông tin options qua latent direction. Exploratory option-free-prepass control dựng lại cả3 trạng thái khi options thật sự vắng mặt. Nếu triển khai thực tế, toàn bộ đường evidence/contrast phải bắt đầu từ câu hỏi người dùng không có choices.

Heuristic freeform aligned: native=3/8; v7=3/8. Không gọi đây là semantic accuracy độc lập; quy tắc đã được chỉnh dựa trên các outputs2150/1821/1929 đã xem. Có32-token truncation và lỗi event/reference/negation mà keyword rules không xử lý toàn diện. Read actual text trong report.

7/8 văn bản native/full trùng từng byte; mẫu1449 chỉ thay vài từ. 2150 vẫn là phản ví dụ hữu ích: MC chọn sai nhưng cả native/v7 nói event xảy ra right after the speech. Điều này chống lại việc quy mọi lỗi MC cho thiếu thông tin thời gian trong V. Có thể có thêm lỗi hiểu lựa chọn/mapping/readout; phải kiểm chứng bằng câu hỏi và scoring được khóa, không sửa nhãn hay chấm có lợi hậu kiểm.

## 6. Các giả thuyết cần phát biểu lại

V6 không chứng minh mọi attention reweighting đều bất lực vì V không RoPE-rotate: trọng số P vẫn phụ thuộc Q/K và vị trí, hidden states V ở tầng sau có thể mang thông tin từ các tầng trước. Một nhóm can thiệp không đổi argmax trên8 mẫu không bác bỏ cả họ phương pháp.

Source v6 trừ f_theta(v_hat) qua diagonal+low-rank map, không đơn thuần trừ raw media vector. Overfitting/background contamination là giả thuyết có thể kiểm tra, chưa là chứng minh rằng mọi ma trận tự do đều phải ghi nhớ recording. [cache_geometry_audit.json](cache_geometry_audit.json) cho thấy pre-alignment recording means held-out không trực giao với train span: raw mean cosine≈.965 native/.905 pooled; sau trừ train-global-mean, held-out projection-energy vẫn khoảng.78–.96 native/.61–.88 pooled. Đây chỉ là cache geometry trướcC6, không phải causal background decomposition hoặc kiểm tra mọi aligned space.

Giảm bridge còn6 scalar không khiến toàn bộ hệ thống miễn overfit: C6, classifier, fitted SVD basis và quy trình chọn feature vẫn phụ thuộc train distribution. Khoảng cách55.9% train/28.2% held-out ởHop2 vẫn còn.

M1 cache dùng post-block27 readout/media means; M3 đưa pre-attention block27 states và pooled contextual question tokens vào C6. M2 bridge học query normalize(.55qa+.45qb), còn M3 dùng whole-question hidden pooling. Có domain/interface shift; nối function calls chưa chứng minh các representations tương thích. Nên khai thác train features bằng cùng media/query format, cùng hook và cùng pooling của inference.

M2 có loopA→V,V→A,A→A,V→V; nhưng M3 pair_omega hiện vẫn là audio-bank×visual-bank. Không được dùng các probe directions củaM2 để khẳng định deployedM3 đã kiểm chứng mọi loại same-modal multi-event query.

## 7. Literature và hướng cải thiện tiếp

Đã đọc nguồn chính để sửa diễn giải bibliography: [SyncRA §2](https://arxiv.org/html/2609.34363v1) huấn luyện synchrony contrast cùng answer supervision, không phải inference static mean activation; [AHA](https://arxiv.org/html/2512.24052v2) là preference/DPO post-training; [RED §3](https://arxiv.org/html/2610.02976v1) có question-guided evidence-type selection và PMI reference. Các nguồn này không chứng minh CLC thuần ngữ nghĩa. Chi tiết trong [literature_notes.md](literature_notes.md).

Ưu tiên đề xuất của tôi:

1. Tạo bộ cơ chế mới nhỏ nhưng có annotation endpoint/event/interval và absence thật, tách recording với tất cả các bộ đã xem. Có hành động lặp, hard negatives cùng chủ đề và câu hỏi có/không cùng modality. Dùng physical single-stream/joint-stream interventions với strict-pair correctness, không chỉ roll tọa độ cache.

2. Học hai query-role distributions trên unified A/V event bank, có null riêng từng endpoint; kiểm tra nhận diện sự kiện trước khi cộng latent. Với câu hỏi phức hợp, pooled bag of content tokens không thay cho việc phân biệt event1/event2 và instance. Giữ factorized/matched-mean/null-only/coordinate controls.

3. Nếu cần null phản ánh incompatibility của cặp, dùng partial/unbalanced matching hoặc learned null costs có thể thay real mass theo evidence; tránh một prescribed marginal scalar khiến P_null trở thành1−a². Bắt buộc test fixed marginals/score nhưng đổi compatibility để xem null có phản ứng đúng không.

4. Kiểm tra đường readout bằng adapter/controller rất nhỏ trên backbone đóng băng, với QA/counterfactual supervision từ train mới và hook/feature interface khớp inference. Thiếu QA trong cache hiện tại là hạn chế dữ liệu, không phải chứng minh adapter không thể hoạt động. Không dùng micro8/research40 để học hướng can thiệp hoặc chọn threshold.

5. Tính chi phí triển khai thật: pipeline cần native prepass + support + foil + reverse + final routed forward, chưa kể bộ controls. generate_freeform hiện recompute full prefix từng token, không dùngKV cache. Có thể tối ưu early-stop các prepass sau block22 và cache decoding, nhưng phải kiểm tra parity rồi báo actual latency.

Bản này đáp ứng ý định tránh agent và không hard-route theo task trong inference chính. Tuy nhiên chưa có bằng chứng giải quyết temporal QA tổng quát hoặc đủ novelty/CVPR acceptance. Trước A0–A7/reserve400, cần có tín hiệu event binding và QA cùng thắng matched controls trên dữ liệu cơ chế sạch; không tiếp tục coi mỗi bản sửa unit assertion là một cải tiến phương pháp.
