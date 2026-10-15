# Lượt mở lại bằng chứng: kết quả40/40

Ngày2026-10-15. Tất cả ô có5 mẫu<=45s; mọi cyclic rotations; cùng manifest0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459.

**Bảng ghi đúng/sai.**

|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|TF dự đoán A/B/C|
|---|---|---|---|---|---|---|---|---|---|---|
|D1_M2_no_clock|4/1|5/0|0/5|5/0|0/5|5/0|4/1|4/1|27/13|0/15/0|
|D2_M2_reopen24|5/0|4/1|0/5|5/0|0/5|5/0|4/1|4/1|27/13|0/15/0|
|D3_M2_global_query|5/0|1/4|0/5|2/3|5/0|4/1|3/2|0/5|20/20|0/4/11|
|G1_local_encode_global_read|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/3/11|
|G2_local_encode_reopen24|5/0|3/2|0/5|1/4|4/1|3/2|2/3|1/4|19/21|1/3/11|
|G3_latent_source_reflection|4/1|4/1|0/5|1/4|5/0|3/2|3/2|1/4|21/19|1/2/12|
|R3_M0_base|5/0|3/2|0/5|1/4|4/1|3/2|2/3|0/5|18/22|1/5/9|
|R3_M2_true_gate_audio|3/2|5/0|0/5|5/0|0/5|5/0|4/1|4/1|26/14|0/15/0|

## Diễn giải

D1 bỏ clock nhưng giữ gate/foreground/PMI. Nếu vẫn collapse Present, clock không phải điều kiện cần cho collapse ở M2; điều này không tự xác định tác dụng của clock trong delta M8.
D2 mở lại native attention từ block24; D3 mở quyền đọc toàn bộ AV của query trong mọi block nhưng giữ can thiệp media cũ. D3 đổi cả quyền đọc thời gian lẫn audio, nên không tách riêng hai yếu tố.
G1/G2 chỉ hạn chế các cạnh media khác nguồn cách nhau>3s, giữ query đọc toàn bộ AV; G3 phản hồi chênh trạng thái về cùng block18/22. Đây là ứng viên không dùng speech window, clock từ đáp án, PMI hoặc task router.
Đánh giá cải thiện phải dựa trên cả8 ô. Cứu Future bằng cách chuyển thiên lệch sang Future, trong khi Past/Present thấp, chưa giải quyết temporal reasoning.
Hướng này chưa đạt bằng chứng về một phương pháp tổng quát đủ mạnh cho CVPR. Không mở rộng sang held-out chỉ vì một tổng điểm tăng nhẹ.

## Triển khai

Server1 dùng lại decoder caches; server2 là phiên mới nên mã hóa lại đúng20 ID, không đổi mẫu. Replay và baseline được lưu cùng lượt; so ghép cặp không dựa vào một baseline khác phiên.
Bốn CPU contract tests đã qua: causal edges, query đọc fullAV, zero-gain identity/media unchanged và hook cleanup. Các kiểm tra này không chứng minh QA correctness.
Kết quả raw: results_s1/s2.jsonl; bảng số: summary.json; giả thuyết/đối chứng: protocol.md. CCLD của Argon được xếp ngay sau lượt này trên từng server.
