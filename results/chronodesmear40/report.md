Yoshua Bengio: ChronoDeSmear Round5

Ngày2026-10-15. Hoàn tất40/40 ID; 5 mỗi task/gold, duration<=45s, mọi110 cyclic rotations.

Đây là development đã tái sử dụng, không phải held-out. Mọi ô là **đúng/sai**; TA:A=Yes/B=No, TF:A=Past/B=Present/C=Future; CMTB là lựa chọn riêng mỗi câu.

Điểm cao nhất: **21/40**, cấu hình R5_C2_stratified_dyn_cd. Cấu hình đạt mục tiêu nội bộ30/40 và>=3/5 mọi ô: không có. Mục tiêu này không phải luật nhận CVPR.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R5_C0_baseline|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R5_C1_causal_desmear_1p|3/2|2/3|0/5|4/1|0/5|3/2|3/2|1/4|16/24|0/13/2|
|R5_C2_stratified_dyn_cd|4/1|3/2|1/4|2/3|1/4|4/1|2/3|4/1|21/19|8/6/1|
|R5_C3_balanced_modal_cd|2/3|5/0|0/5|2/3|0/5|3/2|3/2|4/1|19/21|6/8/1|
|R5_C4_stem_plus_media|4/1|2/3|0/5|3/2|0/5|4/1|2/3|1/4|16/24|1/13/1|
|R5_C5_chronodesmear_full|5/0|3/2|1/4|1/4|0/5|2/3|1/4|5/0|18/22|5/7/3|
|R5_C6_multiseed_shuffled|1/4|4/1|5/0|2/3|1/4|3/2|2/3|0/5|18/22|12/2/1|
|R5_C6_seed_42|0/5|4/1|2/3|2/3|1/4|3/2|2/3|0/5|14/26|8/5/2|
|R5_C6_seed_1337|4/1|1/4|2/3|4/1|2/3|3/2|3/2|0/5|19/21|2/9/4|
|R5_C6_seed_2027|1/4|5/0|4/1|2/3|0/5|2/3|2/3|2/3|18/22|11/3/1|


**Nhận xét từng ý tưởng**

|Method|Phù hợp về thiết kế|Đánh giá từ kết quả và cơ chế|
|---|---|---|
|R5_C0_baseline|Đối chứng|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. Native cùng phiên; so từng ID với baseline cũ.|
|R5_C1_causal_desmear_1p|Thiết kế không task/window|16 đúng/24 sai; ô<3/5: TA/B, TF/A, TF/C, CMTB/C. Trừ history có thể mất context hữu ích; điểm không xác nhận tự nó đã xóa causal echo.|
|R5_C2_stratified_dyn_cd|Chưa đầy đủ: không window nhưng cần choices|21 đúng/19 sai; ô<3/5: TF/A, TF/B, TF/C, CMTB/B. Delta thực tế dùng H_ds và V-only; không chỉ base-static. Static selector không phải event oracle; option-centering chưa có định nghĩa free-form.|
|R5_C3_balanced_modal_cd|Chưa phù hợp đầy đủ|19 đúng/21 sai; ô<3/5: TA/A, TF/A, TF/B, TF/C. Còn speech-window; nhiều yếu tố đổi cùng lúc. Không diễn giải gain là đã bù riêng V_out.|
|R5_C4_stem_plus_media|Thiết kế không task/window|16 đúng/24 sai; ô<3/5: TA/B, TF/A, TF/C, CMTB/B, CMTB/C. Không chạm choices trực tiếp; gain khác C2 nên chưa ablation riêng phạm vi token.|
|R5_C5_chronodesmear_full|Chưa đầy đủ: fullAV/no window nhưng cần choices|18 đúng/22 sai; ô<3/5: TF/A, TF/B, TF/C, CMTB/A, CMTB/B. Static delta vẫn ưu tiên visual; option-centering phụ thuộc MC format. Không suy ra event grounding hoặc novelty từ tên full.|
|R5_C6_multiseed_shuffled|Đối chứng cho C2|18 đúng/22 sai; ô<3/5: TA/A, TF/B, TF/C, CMTB/B, CMTB/C. Ba forward/seed và ensemble log-prob; không là shuffled full C5. Phải xem từng seed và compute.|
|R5_C6_seed_42|Đối chứng random seed|14 đúng/26 sai; ô<3/5: TA/A, TF/A, TF/B, TF/C, CMTB/B, CMTB/C. Xem riêng seed, không chỉ ensemble.|
|R5_C6_seed_1337|Đối chứng random seed|19 đúng/21 sai; ô<3/5: TA/B, TF/A, TF/C, CMTB/C. Xem riêng seed, không chỉ ensemble.|
|R5_C6_seed_2027|Đối chứng random seed|18 đúng/22 sai; ô<3/5: TA/A, TF/B, TF/C, CMTB/A, CMTB/B, CMTB/C. Xem riêng seed, không chỉ ensemble.|

**So ghép cặp với native cùng lượt**

|Method|Cứu câu native sai|Làm sai câu native đúng|Exact paired p|Holm p trong suite|
|---|---|---|---|---|
|R5_C1_causal_desmear_1p|7|9|0.8036|1.0000|
|R5_C2_stratified_dyn_cd|10|7|0.6291|1.0000|
|R5_C3_balanced_modal_cd|11|10|1.0000|1.0000|
|R5_C4_stem_plus_media|6|8|0.7905|1.0000|
|R5_C5_chronodesmear_full|8|8|1.0000|1.0000|
|R5_C6_multiseed_shuffled|7|7|1.0000|1.0000|
|R5_C6_seed_42|5|9|0.4240|1.0000|
|R5_C6_seed_1337|6|5|1.0000|1.0000|
|R5_C6_seed_2027|9|9|1.0000|1.0000|

Paired exact McNemar hai phía; Holm chỉ hiệu chỉnh nội bộ suite, không xử lý mọi lần chọn phương pháp trong lịch sử. N nhỏ và tái dùng dev giới hạn suy luận gain ổn định.

Baseline khác lịch sử trên0/40 mẫu: [].

**Giới hạn và artifacts**

Không có task router, gold-dependent selection hay ghép oracle. Không suy ra CVPR acceptance/novelty hoặc broad deployment từ điểm40 này. Các bước mask/steering/soft thoughts và scalar gating cần đối chiếu prior art; kết quả local không tự xác lập novelty.

Protocol trước chạy: [protocol.md](protocol.md). Raw sample scores: results_s1/s2.jsonl. Chi tiết từng câu: per_sample.csv. Mọi per-rotation scores và archive audit được lưu riêng; kiểm tra trạng thái audit trước khi gọi raw archive hoàn tất.

Raw archives đã audit: 2/2 server; tổng rotations=110.

Các argmax có margin<1e-6 (tie chọn A theo np.argmax, không sửa hậu kiểm): {'R5_C3_balanced_modal_cd': [1449], 'R5_C0_baseline': [1898], 'R5_C1_causal_desmear_1p': [1997]}.

**Độ nhạy quyết định**

|Method|Đổi đáp án/40|Mean TV|
|---|---|---|
|R5_C0_baseline|0|0.0000|
|R5_C1_causal_desmear_1p|17|0.2248|
|R5_C2_stratified_dyn_cd|22|0.2251|
|R5_C3_balanced_modal_cd|29|0.3404|
|R5_C4_stem_plus_media|16|0.2333|
|R5_C5_chronodesmear_full|21|0.2896|
|R5_C6_multiseed_shuffled|17|0.1763|

TV dùng mean log-prob qua rotations rồi normalize trên choices. Không đổi argmax không đồng nghĩa mọi hidden state/path bị bỏ qua; đây không phải event-grounding metric.

Bộ40 có38 nguồn gốc. McNemar theo clip chỉ thăm dò khi có cùng nguồn lặp. Paired sign-flip theo source origin (chưa hiệu chỉnh multiple tests): {'R5_C0_baseline': 1.0, 'R5_C1_causal_desmear_1p': 0.803619384765625, 'R5_C2_stratified_dyn_cd': 0.629058837890625, 'R5_C3_balanced_modal_cd': 1.0, 'R5_C4_stem_plus_media': 0.7744140625, 'R5_C5_chronodesmear_full': 1.0, 'R5_C6_multiseed_shuffled': 1.0}. Cả hai loại kiểm định đều chưa xử lý selection bias do tái dùng development.

**Góp ý gửi Yoshua Bengio**

Giữ fullAV/no task routing là tiến bộ về thiết kế. Tuy nhiên C2 không phải delta nền tĩnh thuần, C3 vẫn dùng cửa sổ và không kiểm chứng riêng việc bù V_out; C6 là control C2. Không gọi matrix này orthogonal factorial.

Center options không bảo đảm prior bất biến: hệ số.55 bị per-row norm rescaling triệt tiêu về magnitude. Trong toy linearization của đề xuất, thay H_base bằng H_Vall không bù V_out mà còn trừ thêm audio. Xem [mechanism_review.md](mechanism_review.md).

Cosine resonance ba vùng không phải shift test hoặc event grounding. Không có annotation independently verified thì không claim đã xác nhận temporal link. Causal accumulation có thể mang context hữu ích, không mặc định toàn bộ là echo gây lỗi.

Option-centering trong C2/C3/C5 cần choices, không chỉ là scoring wrapper. Chưa có định nghĩa hoặc đánh giá bản không choices, vì thế chưa đáp ứng đầy đủ yêu cầu QA tự do/mọi ngữ cảnh. Không task router là điều kiện khác với độc lập định dạng câu hỏi.

[Causality Matters](https://ojs.aaai.org/index.php/AAAI/article/view/37856) phân tích temporal pathway qua inter-frame và query trên các VideoLM; [Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1) mô tả2s interleaving/TMRoPE. Đây là căn cứ đặt giả thuyết/controls, không là bằng chứng hiệu quả của ChronoDeSmear.
