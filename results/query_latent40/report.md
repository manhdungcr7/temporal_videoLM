# Thought có điều kiện câu hỏi và đối chứng scaffold:40/40

Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.

**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Correct/wrong|TF predictions A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|Q0_native|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|Q1_global_prototype|5/0|2/3|0/5|2/3|3/2|3/2|2/3|0/5|17/23|1/7/7|
|Q2_question_prototype|3/2|3/2|1/4|3/2|3/2|3/2|2/3|0/5|18/22|2/8/5|
|Q3_question_evidence_contrast|5/0|2/3|0/5|1/4|4/1|3/2|2/3|0/5|17/23|1/5/9|
|Q4_contrast_plus_joint|3/2|3/2|1/4|1/4|4/1|3/2|2/3|0/5|17/23|2/4/9|
|Q5_shuffled_thought|2/3|3/2|1/4|2/3|3/2|3/2|2/3|1/4|17/23|1/6/8|

Cấu hình có số đúng cao nhất trong lượt này: **Q2_question_prototype: 18/40**.

**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**

## So ghép cặp với native cùng lượt

|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|
|---|---:|---:|---:|---:|
|Q0_native|0|0|+0|1.0000|
|Q1_global_prototype|1|2|-1|1.0000|
|Q2_question_prototype|3|3|+0|1.0000|
|Q3_question_evidence_contrast|0|1|-1|1.0000|
|Q4_contrast_plus_joint|1|2|-1|1.0000|
|Q5_shuffled_thought|3|4|-1|1.0000|

## Nhận xét và phạm vi

Q1 dùng global prototype, Q2 dùng prototype của stem câu hỏi đã đọcAV, Q3 lấy chênh prototype với trường hợp không có nội dung AV, Q4 thêm tương tác cùng tầng, Q5 xáo chiều thought củaQ3 với cùng norm.

Mọi thought đi vào stem trước choices, không vào trực tiếp đáp án/format. Không có cửa sổ lời nói hoặc thứ tự modality cố định. Đây là thay đổi ở nguồn thought và chỗ phản hồi, không phải phát minh mới weighted vocabulary embeddings.

Q3 cần hơn cảQ2 và shuffled Q5 trước khi gán lợi ích cho phần chênh theo bằng chứng. Chuẩn hóa một chênh lệch nhỏ có thể khuếch đại nhiễu; prototype_difference_norm được lưu để kiểm tra.

Cache chỉ là tối ưu tính toán: native prefix check<=.03, full fallback khi không qua, và kiểm tra mọi điều kiện ở rotation cache-eligible đầu của mỗi worker. Baseline dùng full native pass.

Không có mã theo dạng câu hỏi hoặc nhãn, nhưng điều đó chưa chứng minh khả năng triển khai đa ngữ cảnh. Cần held-out, free-form, reverse reasoning, nhiều sự kiện và kiểm tra thực sự theo can thiệp thời gian.

Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.

## Kiểm tra triển khai

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`. Native replay max=0.00000000.

Native prefix fallback ở1/110 rotations. Số condition fallback ở các parity checks đầu worker=0.

Thời gian trung bình của toàn suite và mọi rotations: 27.0s/mẫu; không phải latency của một method.

Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.
