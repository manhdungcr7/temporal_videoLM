# Argon LatentChrono-CoT: kết quả thực nghiệm40 mẫu

Đã hoàn tất40/40, 5 mẫu mỗi task/nhãn; cùng raw AV và mọi cyclic answer rotations.
Bảng dưới ghi **đúng/sai**. A/B/C là nhãn ground-truth, không phải tên điều kiện.

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|Accuracy|
|---|---|---|---|---|---|---|---|---|---|---|
|R3_M0_base|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|45.0%|
|R3_M1_true_text_pmi|5/0|1/4|0/5|1/4|4/1|3/2|2/3|0/5|16/24|40.0%|
|R3_M2_true_gate_audio|3/2|5/0|0/5|5/0|0/5|5/0|4/1|4/1|26/14|65.0%|
|R3_M3_recurrent_L22_to_L10|3/2|2/3|0/5|3/2|0/5|5/0|4/1|4/1|21/19|52.5%|
|R3_M4_manifold_cont_thought|5/0|1/4|0/5|4/1|3/2|4/1|3/2|0/5|20/20|50.0%|
|R3_M5_latent_handoff|3/2|1/4|0/5|5/0|0/5|5/0|4/1|4/1|22/18|55.0%|
|R3_M6_latent_chrono_unified|2/3|1/4|0/5|4/1|0/5|5/0|4/1|4/1|20/20|50.0%|
|R3_M7_clock_mask_pmi_control|5/0|1/4|0/5|2/3|4/1|4/1|3/2|0/5|19/21|47.5%|
|R3_M8_recurrent_only|3/2|5/0|0/5|3/2|0/5|5/0|4/1|4/1|24/16|60.0%|
|R3_M9_thought_only|4/1|3/2|3/2|2/3|3/2|2/3|2/3|0/5|19/21|47.5%|

## So sánh paired với M0

|Method|Cứu câu sai|Làm hỏng câu đúng|Δ số đúng|Exact paired p (chưa điều chỉnh multiple tests)|
|---|---|---|---|---|
|R3_M0_base|0|0|+0|1.0000|
|R3_M1_true_text_pmi|0|2|-2|0.5000|
|R3_M2_true_gate_audio|14|6|+8|0.1153|
|R3_M3_recurrent_L22_to_L10|12|9|+3|0.6636|
|R3_M4_manifold_cont_thought|5|3|+2|0.7266|
|R3_M5_latent_handoff|13|9|+4|0.5235|
|R3_M6_latent_chrono_unified|12|10|+2|0.8318|
|R3_M7_clock_mask_pmi_control|4|3|+1|1.0000|
|R3_M8_recurrent_only|13|7|+6|0.2632|
|R3_M9_thought_only|4|3|+1|1.0000|

## Kiểm tra kỹ thuật

Manifest `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`. Max sequence=7113; dense fp16 mask=96.5MiB/device; max native/decoder replay error=0.00000.
Baseline lịch sử đúng 19/40; khác dự đoán 1/40 mẫu so với baseline rerun phiên này.
Dữ liệu raw: results_s1.jsonl, results_s2.jsonl; dự đoán từng câu: per_sample.csv; số đếm đầy đủ: summary.json.
Chi tiết đề xuất và thay đổi triển khai: protocol.md; nguyên văn: proposal_original.txt và proposal_original.py.

## Phân tích và lời khuyên

M2 `true_gate_audio` đứng đầu với26/40, hơn M0 là8 câu. Nó cứu14 câu M0 sai và làm sai6 câu M0 đúng; paired exact McNemar p=0.115, chưa điều chỉnh cho nhiều cấu hình. Trên N=40 đây là tín hiệu đáng kiểm tra tiếp, chưa phải bằng chứng thống kê thuyết phục. M8 đạt24/40; so ghép cặp, M2 cứu3 câu M8 sai và làm sai1 câu M8 đúng (p=0.625). M2 hơn M8 hai câu, nhưng chưa tách được tác dụng gate/clock/foreground/PMI khỏi latent loop.

M2 cải thiện CMTB lên13/15 từ5/15 của M0, nhất là CMTB/C từ0/5 lên4/5; TF/B tăng từ1/5 lên5/5. Đổi lại TA/A giảm từ5/5 xuống3/5 và M2 vẫn0/5 ở TF/A lẫn TF/C. Có vẻ cấu hình đổi prior/decision boundary theo hướng lợi cho vài cell temporal binding, trong khi làm sụp hai nhãn thời gian. Các latent method M3–M6 chỉ đạt50–55%; unified M6 đạt50%, không hơn rõ thành phần đơn. Text-only PMI làm baseline giảm18 xuống16.

M2 đoán Present cho15/15 câu TF: TF/B5/5 đến từ collapse, không chứng minh hiểu đồng thời. M8 tên “recurrent only” là đối chứng gần đúng: delta vẫn lấy từ hard/anti và clock. M2 và M7 cũng dùng salience/clock từ phép chiếu vocabulary chung, dù không inject continuous thought hay delta trực tiếp. M4 đạt20/40 nhưng gồm cả clock/mask/PMI; M9 thought-only đạt19/40 và TF8/15 so với5/15 của M0. Chưa có factorial ablation đủ sạch để quy đóng góp cho top-48 projection, clock, mask, hay recurrence.

Mỗi cell chỉ có5 mẫu. Không method nào đạt development criterion30/40 hoặc ít nhất3/5 ở mọi cell. M2 được chọn sau10 cấu hình trên chính bộ development này nên có selection bias; đây không phải held-out test. Baseline rerun đạt18/40 và khác dự đoán baseline lịch sử1 mẫu. So sánh nội bộ vẫn dùng cùng raw AV, cyclic rotations và replay khớp chính xác.

Lời khuyên Argon: giữ latent feedback như giả thuyết, nhưng chưa gọi là đã tái hiện Coconut/recurrent-depth hoặc chứng minh “zero modality gap”. Coconut hồi tiếp last hidden state làm next input embedding; M4 ở đây qua top-k logit-lens, tính kỳ vọng token embeddings rồi cộng residual vào block0–2, là cơ chế khác ([Coconut](https://arxiv.org/abs/2412.06769)). Recurrent Depth dùng recurrent block được huấn luyện, khác nối block22 về block10 tùy ý ([Recurrent Depth](https://arxiv.org/abs/2502.05171)). Unsilencing Visual Latents tối ưu latent với contrastive alignment và confidence progression; top-k expectation chưa triển khai objective đó ([Unsilencing Visual Latents](https://arxiv.org/abs/2605.02735)).

Bước tiếp theo: preregister ablation trên held-out mới, giữ cố định mask/gate/clock, so M0, M2, M8, M4 và thiết kế factorial tách từng thành phần. Đo per-sample, dùng McNemar và hiệu chỉnh nhiều phép thử; tập trung xem TA/A và TF/A,C có tiếp tục sụp không. Chỉ mở rộng nếu lợi thế M2 lặp lại ngoài development40.

Job hoàn tất40/40, không OOM; native/decoder replay sai số logit tuyệt đối tối đa0.0. Thời gian trung bình khoảng70 giây/mẫu với10 điều kiện và cyclic rotations.

