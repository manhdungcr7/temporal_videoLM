Tổng hợp thực nghiệm và phản hồi cho Yoshua Bengio — 2026-10-15

Tên Yoshua Bengio dưới đây là cách gọi bên đề xuất theo yêu cầu trong cuộc trao đổi. Báo cáo không xác nhận người nghiên cứu ngoài đời là tác giả hoặc đã bảo chứng các ý tưởng này. Tên thư mục/method gốc được giữ để đối chiếu artifact.

Kết quả cao nhất là D1/D2: 27 đúng, 13 sai; nhưng cả hai dự đoán Present cho toàn bộ 15 câu TF. Trong các đề xuất CCLD, C2/C7 cao nhất: 26 đúng, 14 sai. Chưa có phương pháp đồng thời tốt ở mọi ô và đáp ứng đầy đủ yêu cầu triển khai nhiều ngữ cảnh.

Quy ước: mọi ô là ĐÚNG/SAI, cộng lại bằng 5; cột tổng cộng lại bằng 40. TA:A=Yes,B=No. TF:A=Past,B=Present,C=Future. CMTB:A/B/C là nội dung lựa chọn riêng của từng câu. Cột cuối là SỐ LẦN DỰ ĐOÁN, không phải số câu đúng.

Sử dụng đúng 40 ID, 5 mẫu/ô, video và decoded audio <=45s; mọi cyclic option rotations, tổng 110 rotations/suite. Nhãn dùng cho phân tầng và đánh giá, không router chọn phương pháp theo nhãn. Đây là development đã tái sử dụng, không phải held-out. Không trộn các báo cáo cũ 28/29/32 mẫu <=50s vào những bảng này.

Sửa lỗi trình bày ở câu trả lời trước: native TA/B đúng 3, sai 2; không phải 0 đúng/5 sai. Native tổng 18 đúng, 22 sai. Báo cáo và CSV gốc đã ghi đúng; bản tổng hợp này đếm lại từ dự đoán.

**1. Đề xuất ban đầu của Yoshua Bengio: M0–M9**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R3_M0_base|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R3_M1_true_text_pmi|5/0|1/4|0/5|1/4|4/1|3/2|2/3|0/5|16/24|0/3/12|
|R3_M2_true_gate_audio|3/2|5/0|0/5|5/0|0/5|5/0|4/1|4/1|26/14|0/15/0|
|R3_M3_recurrent_L22_to_L10|3/2|2/3|0/5|3/2|0/5|5/0|4/1|4/1|21/19|0/12/3|
|R3_M4_manifold_cont_thought|5/0|1/4|0/5|4/1|3/2|4/1|3/2|0/5|20/20|0/11/4|
|R3_M5_latent_handoff|3/2|1/4|0/5|5/0|0/5|5/0|4/1|4/1|22/18|0/14/1|
|R3_M6_latent_chrono_unified|2/3|1/4|0/5|4/1|0/5|5/0|4/1|4/1|20/20|0/13/2|
|R3_M7_clock_mask_pmi_control|5/0|1/4|0/5|2/3|4/1|4/1|3/2|0/5|19/21|0/4/11|
|R3_M8_recurrent_only|3/2|5/0|0/5|3/2|0/5|5/0|4/1|4/1|24/16|5/10/0|
|R3_M9_thought_only|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|6/4/5|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|M0 Native|Đối chứng 18/40; TA8/10, TF5/15, CMTB5/15. Là mốc cùng phiên, không thay bằng baseline lịch sử 19/40.|
|M1 Text-only PMI|16/40, giảm 2 câu. Prior correction không gắn riêng một task, nhưng cấu hình hiện tại không đem lại lợi ích; prior văn bản không mặc nhiên là bias thuần. Giữ làm đối chứng, không coi là cải tiến.|
|M2 Gate+clock+foreground+PMI|26/40, CMTB13/15 nhưng TF5/15 và Present15/15. Chưa phù hợp: hard gate/cửa sổ làm hẹp evidence và clock dùng từ nhãn. Tín hiệu đáng giữ là cải thiện binding, không phải đã sửa order.|
|M3 Recurrent L22 về tầng sớm|21/40, CMTB13/15 nhưng TA5/10 và TF3/15. Feedback là giả thuyết dùng rộng được, pipeline hiện còn clock/mask/PMI và thiếu mapping giữa tầng. Chưa đạt.|
|M4 Continuous thought kèm clock/mask/PMI|20/40, TA/B1/5 và CMTB/C0/5. Thought liên tục có thể dùng tổng quát, nhưng kết quả còn lẫn nhiều thành phần; chưa chứng minh alignment AV hoặc temporal grounding.|
|M5 Latent handoff|22/40, TA4/10, TF5/15, Present14/15. Chưa phù hợp với nhiều hướng reasoning vì giảm quyền đọc audio theo tầng cố định; không mặc định mọi câu hỏi đều nên chuyển từ audio sang visual.|
|M6 Unified blend|20/40, thấp hơn M2 sáu câu. Heuristic blend theo skew prior chưa là độ tin cậy evidence. Chưa đạt; không nên tiếp tục ghép thành phần chỉ vì từng nhánh cứu một số câu.|
|M7 Không inject latent trực tiếp|19/40; là control clock/mask/foreground/PMI, không phải bằng chứng latent mới. Còn thành phần phụ thuộc cửa sổ/clock và chưa đạt hiệu quả.|
|M8 Recurrent injection control|24/40, CMTB13/15, TF3/15. Không phải recurrence thuần: delta vẫn hình thành từ hard/anti có clock. Chưa đáp ứng yêu cầu; cần tách nguồn delta khỏi gate/clock trước khi quy gain cho recurrence.|
|M9 Thought injection control|19/40, TF8/15 và phân bố TF6/4/5. Đáng chú ý vì hồi phục Past3/5, nhưng CMTB4/15 và còn thought theo cửa sổ. Đây là giả thuyết nên tách ra kiểm chứng, không phải phương pháp tổng quát đã thành công.|

Lưu ý triển khai: thành phần mang tên Value-Clock ở lượt này cộng vào residual layer output, không trực tiếp sửa v_proj. Không dùng tên đó để suy ra đã hiện thực value-space clock injection đúng nghĩa. Shared vocabulary coordinates cũng không bảo đảm zero modality gap.

**2. Đề xuất CCLD tiếp nối của Yoshua Bengio: C0–C7**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R4_C0_baseline|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R4_C1_m9_thought_ctrl|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|6/4/5|
|R4_C2_m8_noclk_nopmi|1/4|5/0|4/1|1/4|0/5|5/0|5/0|5/0|26/14|11/4/0|
|R4_C3_causal_vis_rec|5/0|3/2|0/5|5/0|0/5|3/2|2/3|3/2|21/19|0/15/0|
|R4_C4_shuffled_rec_ctrl|5/0|0/5|0/5|1/4|3/2|3/2|2/3|0/5|14/26|4/1/10|
|R4_C5_m9_plus_causal_rec|5/0|3/2|0/5|5/0|0/5|3/2|1/4|4/1|21/19|0/15/0|
|R4_C6_ccld_unmasked|3/2|4/1|2/3|4/1|0/5|4/1|3/2|5/0|25/15|5/10/0|
|R4_C7_ccld_late_handoff|3/2|5/0|2/3|4/1|0/5|4/1|3/2|5/0|26/14|5/10/0|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|C1 M9 control|19/40, tái lập từng dự đoán M9. Control hợp lệ; còn thought theo cửa sổ. Chưa là cải tiến hoặc bằng chứng TMRoPE tự hiểu mọi quan hệ thời gian.|
|C2 Bỏ clock/PMI khỏi M8|26/40, CMTB15/15 nhưng TA/A1/5, TF/Future0/5; TF đoán11/4/0. Chưa phù hợp: vẫn hard-sync theo cửa sổ. Cũng bỏ foreground/desmear nên không phải ablation chỉ bỏ clock. Giữ làm đối chứng quan trọng cho đánh đổi giữa binding và order.|
|C3 Visual recurrence|21/40, TF Present15/15. Không speech-window/từ đáp án/task router: phù hợp hơn về thiết kế. Nhưng chỉ khuếch đại visual delta, feedback khác tầng, chưa chứng minh nhiều hướng reasoning; hiệu quả chưa đạt.|
|C4 Shuffled delta|14/40. Đây là negative control, không phải candidate triển khai. C3 hơn 7 câu cho thấy hướng vector có tác dụng trong đối chứng này; một seed chưa đủ chứng minh cơ chế ngữ nghĩa/nhân quả.|
|C5 Global M9+visual recurrence|21/40, TF Present15/15, không hơn C3. Không dùng speech-window trong biến thể global này; phù hợp hơn về cấu trúc nhưng chưa đạt hiệu quả và vẫn bất đối xứng visual. Không kết luận M9 bổ sung được temporal order.|
|C6 CCLD unmasked|25/40, Future0/5; TF đoán5/10/0. Pass cuối unmasked nhưng 45% delta lấy từ hard-sync theo cửa sổ: toàn thuật toán chưa window-free. Chưa đáp ứng yêu cầu.|
|C7 C6+late audio damping|26/40, Future0/5. Chỉ sửa một câu TA/B idx2054 so C6, không đổi câu TF. Vẫn window và giảm audio cố định từ block16: chưa phù hợp mọi câu cần audio; chưa chứng minh late handoff hữu ích tổng quát.|

Trong CCLD, H_Q gồm stem, choices và assistant format. Vì thế thay đổi lựa chọn/format có thể góp vào gain. Phép trừ hidden states của transformer phi tuyến không được coi là chứng minh đã khử sạch prior hoặc giữ nguyên mọi evidence.

C2 so native cứu17 câu, làm sai9; C7 cứu16, làm sai8. Exact McNemar p lần lượt 0.1686 và 0.1516; Holm p của 7 so sánh C1–C7 đều 1.0. N nhỏ nên chưa chứng minh gain ổn định, cũng không kết luận phương pháp hoàn toàn vô ích. Hiệu chỉnh lượt này không xóa selection bias của cả lịch sử thử nghiệm.

**3. Các phép thử của tôi: bỏ clock, mở lại query và local encoding**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|D1_M2_no_clock|4/1|5/0|0/5|5/0|0/5|5/0|4/1|4/1|27/13|0/15/0|
|D2_M2_reopen24|5/0|4/1|0/5|5/0|0/5|5/0|4/1|4/1|27/13|0/15/0|
|D3_M2_global_query|5/0|1/4|0/5|2/3|5/0|4/1|3/2|0/5|20/20|0/4/11|
|G1_local_encode_global_read|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/3/11|
|G2_local_encode_reopen24|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/3/11|
|G3_latent_source_reflection|4/1|4/1|0/5|1/4|5/0|3/2|3/2|1/4|21/19|1/2/12|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|D1 Bỏ clock khỏi M2|27/40 nhưng Present15/15. Bác giả thuyết clock là điều kiện cần cho collapse trong cấu hình này; không xác định riêng tác dụng clock ở M8. Vẫn gate/window nên chưa đạt yêu cầu tổng quát.|
|D2 Mở native attention từ block24|27/40 nhưng Present15/15. Mở muộn không sửa collapse; còn pipeline M2 nên chưa phù hợp.|
|D3 Query đọc global|20/40, TF0/4/11. Chuyển thiên lệch sang Future; thay cả quyền đọc audio và thời gian nên không tách được hai tác dụng. Chưa đạt.|
|G1 Source-local encoding, global read|19/40. Không speech-window/từ đáp án/router; query giữ fullAV. Phù hợp hơn về cấu trúc, nhưng ngưỡng local3s là giả định cần kiểm tra và hiệu quả chưa đạt.|
|G2 G1+mở lại từ24|19/40, cùng dự đoán G1 trên lượt này. Không có bằng chứng lợi ích bổ sung.|
|G3 Source-local reflection|21/40, Past0/5 và TF dự đoán1/2/12. Phản hồi cùng tầng không đủ; thiên Future còn mạnh. Phù hợp hơn về thiết kế nhưng chưa đạt hiệu quả.|

**4. Phản hồi đối xứng cùng tầng: S0–S6**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|S0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|S1_visual_same_layer|0/5|5/0|2/3|0/5|4/1|3/2|2/3|3/2|19/21|4/1/10|
|S2_joint_same_layer|1/4|4/1|3/2|0/5|4/1|4/1|2/3|4/1|22/18|5/0/10|
|S3_orthogonal_joint|2/3|3/2|1/4|1/4|4/1|3/2|2/3|2/3|18/22|1/3/11|
|S4_two_layer_orthogonal_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|2/3|19/21|1/3/11|
|S5_question_stem_joint|3/2|3/2|0/5|1/4|4/1|3/2|2/3|0/5|16/24|1/5/9|
|S6_question_stem_argon_delta|0/5|5/0|0/5|2/3|3/2|3/2|3/2|0/5|16/24|0/7/8|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|S1 Visual delta cùng tầng|19/40, TA/A0/5. Không window nhưng vẫn ưu tiên visual; chỉ là đối chứng cho lỗi truyền khác tầng. Chưa đạt.|
|S2 Joint AV interaction|22/40, TF dự đoán5/0/10; Present0/5, TA/A1/5. Có cấu trúc phù hợp nhất với mục tiêu jointAV trong nhóm này: không gate theo speech, không ưu tiên chiều audio/visual cố định. Nhưng kết quả chưa đạt; chỉ cứu9/làm sai5 so native, p=0.424.|
|S3 Chiếu loại phần đơn modality|18/40. Không chứng minh phần còn lại chính là suy luận jointAV; phép chiếu có thể bỏ thông tin hữu ích. Phù hợp ý định, không cải thiện.|
|S4 S3 trên hai tầng|19/40. Thêm tầng chưa đem lại cải thiện rõ. Chưa có căn cứ tăng số tầng tiếp theo cùng hướng.|
|S5 Joint feedback chỉ vào stem|16/40. Giảm can thiệp trực tiếp lên choices/format, hợp mục tiêu tổng quát hơn; kết quả giảm. Không chứng minh gain cũ chỉ do choices, vì can thiệp còn khác về phạm vi token.|
|S6 Visual delta kiểu C3 chỉ vào stem|16/40. Đối chứng cho lựa chọn vị trí can thiệp; chưa cải thiện, vẫn visual-asymmetric.|

S v1 có lỗi replay vượt ngưỡng, được giữ riêng implementation_v1 và không trộn vào bảng chính. S v2 giữ ngưỡng, dùng full fallback; không thay mẫu hay công thức để bỏ qua lỗi.

**5. Thought điều kiện theo câu hỏi: Q0–Q5**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|Q0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|Q1_global_prototype|5/0|2/3|0/5|2/3|3/2|3/2|2/3|0/5|17/23|1/7/7|
|Q2_question_prototype|3/2|3/2|1/4|3/2|3/2|3/2|2/3|0/5|18/22|2/8/5|
|Q3_question_evidence_contrast|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|
|Q4_contrast_plus_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|0/5|17/23|2/4/9|
|Q5_shuffled_thought|2/3|3/2|1/4|2/3|3/2|3/2|2/3|1/4|17/23|1/6/8|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|Q1 Global prototype|17/40. Không window/router và chỉ phản hồi vào stem, phù hợp hơn về cách áp dụng; không đem lại lợi ích.|
|Q2 Question prototype|18/40, TF7/15 nhưng TA6/10; đổi nơi sai chứ không tăng tổng. Không giả định hướng modality cố định, nhưng chưa đạt hiệu quả.|
|Q3 Evidence contrast|17/40. Không hơn Q2 hoặc control Q5; chưa chứng minh phần chênh là evidence hữu ích.|
|Q4 Contrast+joint delta|17/40. Không có bằng chứng hai thành phần bổ sung lẫn nhau.|
|Q5 Shuffled thought|17/40. Negative control, không phải candidate triển khai; kết quả ngang Q3/Q4 làm yếu lập luận rằng hướng thought đang cung cấp grounding có ích.|

**6. Tối ưu latent memory theo video: L0–L3**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|L0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L1_unoptimized_memory|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|
|L2_optimized_memory|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|L3_time_shuffled_target|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|0/4/11|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|L1 Memory chưa tối ưu|19/40. Control cho hiệu ứng thêm memory; không chứng minh latent optimization.|
|L2 Memory được tối ưu|18/40, giữ nguyên argmax native trên40/40. Không task router/speech-window/handoff, phù hợp hơn về thiết kế. Tuy nhiên loss tái dựng teacher nội bộ chưa sửa temporal QA; chưa đạt.|
|L3 Target thời gian bị xáo|17/40. Negative control; L2 chỉ hơn một câu và không hơn native/L1, chưa chứng minh temporal fitting đem lại gain downstream.|

L2 tăng video self-match113/527 lên458/527 và audio257/514 lên440/514. Các số này đo khớp index nội bộ cùng model, không phải event grounding có annotation độc lập. Mean TV phân phối đáp án=0.0463: xác suất có đổi dù argmax không đổi. Không suy ra model bỏ qua memory tại mọi tầng hoặc mọi latent optimization đều thất bại.

**7. Giữ nguyên slots, đổi tầng đọc: E0–E5**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|E0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E1_unoptimized_read15|4/1|3/2|0/5|2/3|2/3|3/2|2/3|0/5|16/24|1/7/7|
|E2_optimized_read15|4/1|3/2|0/5|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E3_shuffled_read15|5/0|1/4|1/4|1/4|4/1|2/3|2/3|1/4|17/23|2/3/10|
|E4_optimized_read23_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|E5_unoptimized_read23_reference|4/1|3/2|0/5|2/3|4/1|3/2|2/3|1/4|19/21|0/6/9|

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|E1 Memory chưa tối ưu, read15|16/40, thấp hơn read23 ba câu. Mở sớm gây hại trong cấu hình này.|
|E2 Memory tối ưu, read15|17/40, thấp hơn read23 một câu; so E4 cứu1/làm sai2. Phù hợp về cách áp dụng nhưng không sửa lỗi chỉ bằng thời điểm đọc.|
|E3 Target xáo, read15|17/40, ngang E2. Chưa có lợi ích downstream rõ của đúng thứ tự target.|
|E4 Optimized read23|18/40, điểm tham chiếu L2 được sao chép, không phải candidate mới được fit/chạy lại toàn bộ.|
|E5 Unoptimized read23|19/40, điểm tham chiếu L1 được sao chép, không phải một cải tiến độc lập mới.|

Read23 parity kiểm tra lại bằng0.0. Slots giữ nguyên từng giá trị; so E2/E4 tách riêng thời điểm đọc, không lẫn việc fit lại.

**8. Đối chứng clock đầu vào: P0–P2**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|P0_original_native_reference|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|P1_pyav_frame_index|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/4/10|
|P2_pyav_presentation_time|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/4/10|

P1 lấy frame theo index đạt19/40; P2 theo presentation timestamp đạt18/40. Đây là input fidelity control, không phải đóng góp latent reasoning hay định vị sự kiện bằng oracle. Cả hai giữ audio/budget; P1 so native còn thay decoder và số frame khả dụng, P2 so P1 mới tách quy tắc chọn frame. Chưa sửa temporal hallucination. P0 là điểm native tham chiếu được sao chép.

P đủ40 aggregate records và110 rotation audit entries. Do tunnel server1 mất ở cuối lượt, raw per-rotation scores của19 mẫu server1 chưa archive được; đúng ID thiếu1898 được hoàn tất trên server2, không thay mẫu. Không gọi lượt P là đã lưu đủ raw110 scores. Các suite M/C/DG/S/Q/L/E có artifacts riêng và không bị thiếu raw theo sự cố P.

**Góp ý gửi Yoshua Bengio**

1. Giữ hai quan sát thực nghiệm: synchrony-style intervention giúp CMTB mạnh; thought-only M9 hồi phục một phần TF/Past. Không suy ra đã sửa joint temporal reasoning hoặc đã chứng minh cơ chế nội tại.
2. Không dùng điểm26–27/40 làm tiêu chí chọn duy nhất. D1/D2 TF chỉ5/15 bằng native dù tổng tăng9 câu; C2 tăng CMTB lên15/15 nhưng Future0/5, TA/A1/5. Báo đủ8 ô, phân phối dự đoán, rescue/regression và tính bất định.
3. Bỏ clock dùng từ nhãn và cửa sổ lời nói làm mặc định cho toàn bài toán. Bảo toàn full timeline, provenance audio/video; không cố định trình tự tìm audio rồi khớp visual hoặc giảm audio ở tầng muộn.
4. Đừng đề xuất lại đúng same-layer delta, orthogonal interaction, stem-only thought hay self-reconstruction slots chỉ bằng cách đổi gain/layer: các nhánh này đã được chạy và chưa đạt. Nếu quay lại phải có giả thuyết mới và bằng chứng xác định nút thắt.
5. Ablation cần trực giao: cùng prompt/thought/scale/mask, chỉ đổi một thành phần; đối chứng nhiều random direction/seed, cùng norm và compute. Unmasked pass cuối không làm nguồn delta tự hết phụ thuộc window.
6. Kiểm chứng quan hệ thời gian bằng biến đổi audio/video giữ nguyên nội dung: dịch tương đối, đổi thứ tự, nhiều sự kiện, kiểm tra câu cần audio và câu cần visual. Với biến đổi nhãn không xác định được chắc chắn, phải gán nhãn/kiểm chứng lại; không tự tạo gold sai.
7. Hướng mới có căn cứ hơn là latent reasoner/adapter được giám sát bằng event–source–time trên train pool tách video/nguồn khỏi40 dev, kết hợp temporal hard negatives có kiểm chứng. Kiểm tra grounding và đáp án cùng nhau. Đây là đề xuất chưa chạy, không có điểm kết quả.
8. Thiết kế không có task router mới chỉ thỏa một phần yêu cầu. Để đánh giá triển khai cần held-out/free-form, audio-led/visual-led, reverse lookup, nhiều sự kiện, order/count/duration/causal; đo latency/VRAM từng method, không dùng thời gian cả suite làm runtime triển khai.

Về mục tiêu CVPR: chủ đề phù hợp hướng nghiên cứu, nhưng hiện chưa có phương pháp được chứng minh đủ mạnh/tổng quát hoặc có novelty đã xác lập. Mốc30/40 và >=3/5 mỗi ô chỉ là tiêu chí phát triển nội bộ, không phải tiêu chí nhận bài. Hiện không phương pháp nào đạt cả hai. Các primitive đã được đối chiếu tài liệu ở báo cáo trước; bản này không thực hiện một literature review mới hay xác nhận quyền ưu tiên.

Artifacts gốc: report.md, summary.json, per_sample.csv/JSONL, protocol.md trong các thư mục ở từng bảng. Bản tổng hợp được tạo từ các dự đoán đã lưu, không chạy inference mới hoặc thay kết quả cũ. CSV kèm theo chứa đủ task/gold/prediction để liệt kê chính xác ID đúng/sai mỗi method. Summary JSON nguồn được đối chiếu với số đếm tái tính cho mọi ô.