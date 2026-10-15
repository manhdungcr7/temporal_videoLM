# Đánh giá Argon Round4 CCLD trên đúng40 mẫu

Ngày2026-10-15. Đã chạy đủ8 cấu hình trên hai server Kaggle, mỗi server20 mẫu.

Mô hình Qwen2.5-Omni-3B FP16. Mỗi ô có5 video đã lọc<=45s trước khi lấy mẫu; không thay mẫu hoặc bỏ lỗi.

Mỗi câu dùng mọi cyclic answer rotations, tổng110 rotations. Đây là development đã dùng nhiều lần.

**Trong bảng: đúng/sai, không phải đúng/tổng.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động riêng của từng câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|R4_C0_baseline|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R4_C1_m9_thought_ctrl|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|6/4/5|
|R4_C2_m8_noclk_nopmi|1/4|5/0|4/1|1/4|0/5|5/0|5/0|5/0|26/14|11/4/0|
|R4_C3_causal_vis_rec|5/0|3/2|0/5|5/0|0/5|3/2|2/3|3/2|21/19|0/15/0|
|R4_C4_shuffled_rec_ctrl|5/0|0/5|0/5|1/4|3/2|3/2|2/3|0/5|14/26|4/1/10|
|R4_C5_m9_plus_causal_rec|5/0|3/2|0/5|5/0|0/5|3/2|1/4|4/1|21/19|0/15/0|
|R4_C6_ccld_unmasked|3/2|4/1|2/3|4/1|0/5|4/1|3/2|5/0|25/15|5/10/0|
|R4_C7_ccld_late_handoff|3/2|5/0|2/3|4/1|0/5|4/1|3/2|5/0|26/14|5/10/0|

Cấu hình có số đúng cao nhất trong lượt này: **R4_C2_m8_noclk_nopmi = 26/40 (65.0%)**.

**Không cấu hình nào đạt tiêu chí development30/40 và ít nhất3/5 ở mọi ô.**

## So ghép cặp với baseline cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p|Holm p (7 phép so)|
|---|---:|---:|---:|---:|---:|
|R4_C1_m9_thought_ctrl|4|3|+1|1.0000|1.0000|
|R4_C2_m8_noclk_nopmi|17|9|+8|0.1686|1.0000|
|R4_C3_causal_vis_rec|10|7|+3|0.6291|1.0000|
|R4_C4_shuffled_rec_ctrl|1|5|-4|0.2188|1.0000|
|R4_C5_m9_plus_causal_rec|9|6|+3|0.6072|1.0000|
|R4_C6_ccld_unmasked|15|8|+7|0.2100|1.0000|
|R4_C7_ccld_late_handoff|16|8|+8|0.1516|1.0000|

Các p-value hai phía dùng exact McNemar. Holm chỉ hiệu chỉnh7 so sánh nội bộ lượt này; không xóa selection bias từ toàn bộ lịch sử thử nghiệm.

## Nhận xét từng cấu hình

- **R4_C1_m9_thought_ctrl:** 19 đúng, 21 sai. Thought M9 dùng cửa sổ âm thanh; đây là đối chứng tái lập, không phải nhánh không phụ thuộc cửa sổ. Dự đoánTF A/B/C=[6, 4, 5]; không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.

- **R4_C2_m8_noclk_nopmi:** 26 đúng, 14 sai. Bỏ clock/PMI nhưng vẫn dùng sync/anti và cũng bỏ foreground/desmear cũ. Không phải thí nghiệm chỉ thay clock. Dự đoánTF A/B/C=[11, 4, 0]; không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.

- **R4_C3_causal_vis_rec:** 21 đúng, 19 sai. Không dùng cửa sổ hoặc từ chỉ đáp án. Tuy nhiên chỉ khuếch đại phần chênh do visual và truyền vector từ block22 sang các block sớm. Dự đoánTF A/B/C=[0, 15, 0]; collapse vào một nhãn.

- **R4_C4_shuffled_rec_ctrl:** 14 đúng, 26 sai. Đối chứng giữ norm theo từng query row, xáo chiều với một seed cố định; không phải nhiều hướng ngẫu nhiên độc lập. Dự đoánTF A/B/C=[4, 1, 10]; không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.

- **R4_C5_m9_plus_causal_rec:** 21 đúng, 19 sai. Global thought và visual delta không cần cửa sổ; đáp ứng ràng buộc hình thức tốt hơn, nhưng vẫn chưa kiểm chứng câu hỏi thiên về audio hoặc nhiều bước khác. Dự đoánTF A/B/C=[0, 15, 0]; collapse vào một nhãn.

- **R4_C6_ccld_unmasked:** 25 đúng, 15 sai. Pass2 không mask, nhưng delta vẫn lấy45% từ hard_sync theo cửa sổ. Vì vậy toàn phương pháp còn phụ thuộc cửa sổ. Dự đoánTF A/B/C=[5, 10, 0]; không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.

- **R4_C7_ccld_late_handoff:** 26 đúng, 14 sai. Còn hard_sync và giảm audio cố định từ block16. Chính sách này chưa có căn cứ áp dụng cho mọi câu hỏi cần audio. Dự đoánTF A/B/C=[5, 10, 0]; không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.

## Kiểm chứng các khẳng định của đề xuất

C3 hơn shuffled trên12 câu và thua trên5 câu. Chênh đúng=+7; cần xem đối chứng này trước khi gán lợi ích cho hướng ngữ nghĩa của delta.

C7 khác C6 trên1/40 dự đoán: [{'idx': 2054, 'task': 'TA', 'gold': 'B', 'before': 'A', 'after': 'B'}]. Không có thay đổi dự đoán nào ở TF; lợi ích của late audio damping trong lượt này chỉ nằm ở một câu TA/B, chưa chứng minh khả năng tổng quát.

Không có chứng minh phép trừ hidden state triệt tiêu sạch prior/thì/nhầm nguồn. Attention và MLP phi tuyến, nên hiệu trạng thái cũng có thể làm mất tín hiệu hữu ích hoặc đưa can thiệp ra khỏi phân phối quen thuộc.

M9 cải thiện TF không chứng minh TMRoPE tự so được trước/trong/sau. Muốn kiểm chứng cần đổi thời gian tương đối khi giữ nội dung, kiểm tra mọi nhãn và loại bỏ lối tắt vị trí lời nói.

Các cấu hình không hoàn toàn trực giao: C2 khác M8 cũ ở foreground/desmear; C1 dùng thought theo cửa sổ với scale.22, C5 dùng global thought với scale.20; C6/C7 vừa đổi delta vừa giữ nhánh sync.

Trong code này, H_Q được lấy từ mọi token sau media: có cả stem, lựa chọn và assistant-format tokens. Vì vậy can thiệp có thể đổi đường đọc lựa chọn/format, chưa được tách khỏi grounding. Hướng riêng S5/S6 kiểm tra giới hạn can thiệp vào đúng stem câu hỏi.

## Phù hợp với yêu cầu tổng quát và mục tiêu CVPR

C3/C5 phù hợp hơn với yêu cầu không router theo task, không chèn từ chỉ đáp án và không chọn cửa sổ bằng VAD. Chúng giữ nguyên fullAV cho pass cuối. Tuy nhiên, không phụ thuộc câu chữ trong code không đồng nghĩa đã tổng quát về hiệu quả.

C1/C2/C6/C7 còn phụ thuộc cửa sổ; C7 còn mặc định giảm audio. Chưa thể coi đây là phương pháp triển khai cho mọi kiểu câu hỏi hay mọi hướng reasoning. Cần audio-only/visual-only, reverse lookup, nhiều sự kiện, nhiều khoảng lời nói, count/duration/causal và free-form QA ngoài SVHalluc.

Chưa có đủ bằng chứng để gọi là đóng góp mới đủ mạnh cho CVPR. Counterfactual contrasts và activation steering đã có tiền lệ; cần novelty cụ thể, ablation sạch, cải thiện ổn định, held-out và kiểm chứng bằng can thiệp thời gian.

[RED](https://arxiv.org/abs/2610.02976) phân rã hỗ trợ audio/video/interaction ở decoding; [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/) đã can thiệp hidden states; [LatentOmni](https://arxiv.org/abs/2605.22012) dùng huấn luyện cho latentAV. Các cơ chế không giống hoàn toàn CCLD, nhưng chặn các claim novelty quá rộng.

## Kiểm tra triển khai và khả năng tái lập

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`. Native/decoder replay max abs error=0.00000000; shuffled row-norm max error=0.00000572.

Baseline khác lịch sử trên0/40 câu: []. M9 khác lịch sử trên0/40 câu: []. Mọi so ghép cặp chính dùng C0 của cùng lượt.

Thời gian trung bình cho toàn suite8 cấu hình và tất cả rotations: 58.1s/mẫu; đây không phải latency của một cấu hình triển khai.

Các thay đổi cần thiết so code gửi được ghi trong protocol.md: giữ đúng40 và<=45s, không reencode full-cut, giữ prompt baseline, decoder replay, mask theo device, bool triangle, không xóa output và không silently skip.

## Góp ý tiếp theo

Giữ C3/C5 như giả thuyết cần kiểm chứng, không mặc định trộn thêm nhiều thành phần. Tách cùng một thought/scale để đo delta; thử same-layer và không khuếch đại delta nhỏ; bổ sung đối chứng phản hồi audio đối xứng và several random directions.

Hướng riêng đã chuẩn bị: S1–S6 trong results/symmetric_latent40, được xếp sau Argon trên mỗi server. Hướng này kiểm tra phản hồi đúng tầng, phần tương tác hai nguồn, giới hạn norm và stem-only controls; không dùng cửa sổ, clock từ vựng hoặc policy giảm cố định một modality. S1/S6 là đối chứng visual-asymmetric, không phải ứng viên đối xứng.

Audit đồng hồ đầu vào được lưu riêng trong results/presentation_clock_audit. TimestampAPI Decord có thể không khớp pixel; chưa dùng nó làm oracle để thay đổi đầu vào CCLD.

[Tổng hợp các lượt tiếp nối đã chạy](../argon_and_latent_research_20261015.md): same-layer symmetric feedback, query-conditioned thoughts, fitted source/time latent memory và đối chứng thời điểm đọc.

Artifacts: proposal_original.txt/.py, protocol.md, results_s1/s2.jsonl, summary.json, per_sample.csv, table.md và job_s1/s2.json.
