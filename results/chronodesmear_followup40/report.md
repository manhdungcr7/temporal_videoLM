Tiếp nối: factor ladder và source-symmetric feedback

Ngày2026-10-15. Hoàn tất40/40 ID; 5 mỗi task/gold, duration<=45s, mọi110 cyclic rotations.

Đây là development đã tái sử dụng, không phải held-out. Mọi ô là **đúng/sai**; TA:A=Yes/B=No, TF:A=Past/B=Present/C=Future; CMTB là lựa chọn riêng mỗi câu.

Điểm cao nhất: **20/40**, cấu hình N4_centered_plus_desmear. Cấu hình đạt mục tiêu nội bộ30/40 và>=3/5 mọi ô: không có. Mục tiêu này không phải luật nhận CVPR.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|N0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|N1_desmear_reference|3/2|2/3|0/5|4/1|0/5|3/2|3/2|1/4|16/24|0/13/2|
|N2_pure_delta|5/0|0/5|2/3|1/4|0/5|3/2|2/3|3/2|16/24|9/5/1|
|N3_pure_delta_centered|3/2|3/2|2/3|1/4|0/5|3/2|2/3|3/2|17/23|9/5/1|
|N4_centered_plus_desmear|3/2|4/1|3/2|0/5|0/5|4/1|2/3|4/1|20/20|13/2/0|
|N5_centered_desmear_thought|3/2|1/4|2/3|1/4|0/5|4/1|1/4|5/0|17/23|9/5/1|
|N6_random_stratified|2/3|5/0|1/4|1/4|0/5|2/3|3/2|2/3|16/24|5/7/3|
|N7_source_symmetric|4/1|3/2|1/4|0/5|4/1|2/3|2/3|2/3|18/22|3/3/9|
|N8_symmetric_same_layer_stem|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|N9_fixed_depth_ensemble|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|


**Nhận xét từng ý tưởng**

|Method|Phù hợp về thiết kế|Đánh giá từ kết quả và cơ chế|
|---|---|---|
|N0_native|Đối chứng|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. Native rerun, kiểm tra parity với Round5.|
|N1_desmear_reference|Tham chiếu sao chép C1|16 đúng/24 sai; ô<3/5: TA/B, TF/A, TF/C, CMTB/C. Không tính là inference mới độc lập.|
|N2_pure_delta|Không task/window|16 đúng/24 sai; ô<3/5: TA/B, TF/A, TF/B, TF/C, CMTB/B. Pure H_AV-H_staticA; giữ recurrence gain cố định để bỏ confound H_ds/V-only trong delta.|
|N3_pure_delta_centered|Không task/window; cần choices|17 đúng/23 sai; ô<3/5: TF/A, TF/B, TF/C, CMTB/B. So N2 tách riêng mean-centering; hệ số.55 bị norm-rescaling triệt tiêu về magnitude. Chưa có bản free-form.|
|N4_centered_plus_desmear|Không task/window; cần choices|20 đúng/20 sai; ô<3/5: TF/B, TF/C, CMTB/B. So N3 thêm de-smear final, giữ delta/gain cố định. Chưa có bản free-form.|
|N5_centered_desmear_thought|Không task/window; cần choices|17 đúng/23 sai; ô<3/5: TA/B, TF/A, TF/B, TF/C, CMTB/B. So N4 thêm thought; weighted vocabulary projection không mặc nhiên semantic alignment. Chưa có bản free-form.|
|N6_random_stratified|Negative selection control|16 đúng/24 sai; ô<3/5: TA/A, TF/A, TF/B, TF/C, CMTB/A, CMTB/C. Cùng số token mỗi bin nhưN3; một seed2027, không phải event-negative có nhãn.|
|N7_source_symmetric|JointAV tốt hơn về cấu trúc, còn cần choices|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/A, CMTB/B, CMTB/C. Đối xứng phép xây delta cho cả hai nguồn, giữ full timeline; center vẫn phụ thuộc options. Không claim causal decomposition.|
|N8_symmetric_same_layer_stem|Phù hợp hơn về jointAV|18 đúng/22 sai; ô<3/5: TF/A, TF/B, CMTB/B, CMTB/C. Same-layer22, norm cap15%, lexical stem. Đổi nhiều yếu tố soN7 nên là candidate riêng.|
|N9_fixed_depth_ensemble|Diagnostic/readout baseline|17 đúng/23 sai; ô<3/5: TA/B, TF/A, TF/B, CMTB/B, CMTB/C. Layer22/26/30/34/35 cố định, không chọn bằng gold. Không claim novelty ensemble.|

**So ghép cặp với native cùng lượt**

|Method|Cứu câu native sai|Làm sai câu native đúng|Exact paired p|Holm p trong suite|
|---|---|---|---|---|
|N1_desmear_reference|7|9|0.8036|1.0000|
|N2_pure_delta|6|8|0.7905|1.0000|
|N3_pure_delta_centered|7|8|1.0000|1.0000|
|N4_centered_plus_desmear|9|7|0.8036|1.0000|
|N5_centered_desmear_thought|9|10|1.0000|1.0000|
|N6_random_stratified|7|9|0.8036|1.0000|
|N7_source_symmetric|5|5|1.0000|1.0000|
|N8_symmetric_same_layer_stem|0|0|1.0000|1.0000|
|N9_fixed_depth_ensemble|0|1|1.0000|1.0000|

Paired exact McNemar hai phía; Holm chỉ hiệu chỉnh nội bộ suite, không xử lý mọi lần chọn phương pháp trong lịch sử. N nhỏ và tái dùng dev giới hạn suy luận gain ổn định.

Baseline khác lịch sử trên0/40 mẫu: [].

**Giới hạn và artifacts**

Không có task router, gold-dependent selection hay ghép oracle. Không suy ra CVPR acceptance/novelty hoặc broad deployment từ điểm40 này. Các bước mask/steering/soft thoughts và scalar gating cần đối chiếu prior art; kết quả local không tự xác lập novelty.

Protocol trước chạy: [protocol.md](protocol.md). Raw sample scores: results_s1/s2.jsonl. Chi tiết từng câu: per_sample.csv. Mọi per-rotation scores và archive audit được lưu riêng; kiểm tra trạng thái audit trước khi gọi raw archive hoàn tất.

Raw archives đã audit: 2/2 server; tổng rotations=110.

Các argmax có margin<1e-6 (tie chọn A theo np.argmax, không sửa hậu kiểm): {'N5_centered_desmear_thought': [1457], 'N0_native': [1898], 'N1_desmear_reference': [1997]}.

**Độ nhạy quyết định**

|Method|Đổi đáp án/40|Mean TV|
|---|---|---|
|N0_native|0|0.0000|
|N1_desmear_reference|17|0.2248|
|N2_pure_delta|19|0.2385|
|N3_pure_delta_centered|20|0.2368|
|N4_centered_plus_desmear|23|0.3043|
|N5_centered_desmear_thought|25|0.3068|
|N6_random_stratified|22|0.2596|
|N7_source_symmetric|10|0.1447|
|N8_symmetric_same_layer_stem|0|0.0147|
|N9_fixed_depth_ensemble|1|0.0286|

TV dùng mean log-prob qua rotations rồi normalize trên choices. Không đổi argmax không đồng nghĩa mọi hidden state/path bị bỏ qua; đây không phải event-grounding metric.

Bộ40 có38 nguồn gốc. McNemar theo clip chỉ thăm dò khi có cùng nguồn lặp. Paired sign-flip theo source origin (chưa hiệu chỉnh multiple tests): {'N0_native': 1.0, 'N1_desmear_reference': 0.803619384765625, 'N2_pure_delta': 0.79052734375, 'N3_pure_delta_centered': 1.0, 'N4_centered_plus_desmear': 0.79052734375, 'N5_centered_desmear_thought': 1.0, 'N6_random_stratified': 0.803619384765625, 'N7_source_symmetric': 1.0, 'N8_symmetric_same_layer_stem': 1.0, 'N9_fixed_depth_ensemble': 1.0}. Cả hai loại kiểm định đều chưa xử lý selection bias do tái dùng development.

**Các thay đổi từng yếu tố**

|So sánh A với B|A cứu/B sai|A sai/B đúng|Chênh số đúng|
|---|---|---|---|
|N3_pure_delta_centered vs N2_pure_delta|3|2|+1|
|N4_centered_plus_desmear vs N3_pure_delta_centered|6|3|+3|
|N5_centered_desmear_thought vs N4_centered_plus_desmear|2|5|-3|
|N3_pure_delta_centered vs N6_random_stratified|4|3|+1|
|N7_source_symmetric vs N3_pure_delta_centered|6|5|+1|
|N8_symmetric_same_layer_stem vs N7_source_symmetric|5|5|+0|

**Logit lens native theo tầng cố định**

|Block index|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|Đúng tầng này/sai cuối|Sai tầng này/đúng cuối|
|---|---|---|---|---|---|---|---|---|---|---|---|
|4|0/5|5/0|0/5|0/5|5/0|0/5|2/3|0/5|12/28|4|10|
|8|4/1|1/4|0/5|5/0|0/5|2/3|2/3|1/4|15/25|9|12|
|12|5/0|0/5|0/5|5/0|0/5|2/3|2/3|1/4|15/25|6|9|
|16|0/5|5/0|1/4|3/2|0/5|3/2|3/2|1/4|16/24|8|10|
|18|5/0|0/5|0/5|5/0|0/5|3/2|2/3|0/5|15/25|5|8|
|22|5/0|0/5|5/0|0/5|0/5|3/2|1/4|2/3|16/24|8|10|
|26|5/0|0/5|5/0|0/5|0/5|2/3|1/4|1/4|14/26|7|11|
|30|5/0|2/3|0/5|2/3|4/1|3/2|2/3|0/5|18/22|1|1|
|34|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|0|0|
|35|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|0|0|

Logit lens chưa calibrated và có thể đọc prior chữ cái; không dùng gold để chọn layer cho N9. Correct-at-middle/wrong-final là quan sát readout, chưa đủ chứng minh causal late overwrite.

N8 đổi cả layer, norm cap và token span so N7 nên không gán chênh lệch riêng cho một yếu tố. Low-innovation/static và mean norm ratio chỉ là proxy representation, không event annotation.
