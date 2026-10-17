# CAOR-WEMD v6 — preflight sau khi nhận source, 2026-10-16

**Có sai khác code–proposal và giả định chưa chốt. Dừng trước thực nghiệm theo chỉ dẫn người dùng.**

Đã nhận source [stage0_pcebr_v6_original.py](stage0_pcebr_v6_original.py), SHA256 `d4899261862df5e0bf6ba890d722b212bcd26f8ebc2090454fe6dcf191871c74`, lưu nguyên byte. Đã đối chiếu source với proposal cũ và các hàm v5 liên quan; không import/execute source, không chạy unit tests, không gửi remote job, không thay code phương pháp.

| Mức ưu tiên | Phát hiện từ đọc source | Tác động |
|---|---|---|
| Cao | CAOR eta nằm ở output interpolation, khác eta bên trong softmax của proposal; kernel không có đủ query-dependent term của công thức de-sinking | Hai thuật toán khác nhau |
| Cao | CDF/W dùng thứ tự candidate indices; không tích phân trên support thời gian khi interval khác nhau hoặc coordinate shuffle | Claim physical Wasserstein chưa được hiện thực cho đầu vào tổng quát |
| Cao | C8b shuffle cả global và innovation, thay vì chỉ innovation như prose | Không tách được đóng góp innovation theo control đã hứa |
| Cao | C6 score_matrix và encode_text@encode_media không tương đương; direct/2hop đi qua hai interfaces khác nhau | Teacher/readout giữa M1 và các suite M2 không đồng nhất về score definition |
| Cần chốt | Code chạy native prepass H18/H27 rồi routed forward | One-pass chỉ có thể hiểu là không thêm null-prompt pass cho de-sinking |
| Cần chốt | Gaussian band khác adjacent-only; shrink.22 khác.35; beta innovation.20 khác.25; bridge train thiếu relation filter | Thay features/hyperparameters/protocol so proposal |
| Cần bổ sung | Thiếu exact legacy-v5 branch, random norm chưa khớp toàn perturbation, M0 chưa đo mass preservation; zero-eta bypass kernel | Chưa kiểm tra được các guarantee/controls được mô tả |
| Báo cáo | Hop counters trước balancing; heuristic TF vẫn chưa xác minh đúng event/reference | Dễ diễn giải sai denominators và semantic QA |

Các điểm đã được code giải quyết: pure modality normalization không còn text mean trên media; mixture-aware bridge có mixture input và correlation loss; B7/B8 matched theo feature ablation trong từng suite; CAOR dùng Q/K/V của từng tầng và có shared logsumexp giữ tổng media mass tại local matrix. Không phủ nhận các phần này chỉ vì còn sai khác khác.

Những phản ví dụ/công thức trong lời nhắn là suy luận từ source, không phải kết quả chạy. Chưa có số pass/fail M0 hoặc QA v6. Không kết luận phương pháp tốt/xấu về accuracy khi chưa được phép chạy theo đặc tả đã thống nhất.

Lời nhắn hỏi Yoshua Bengio: [questions_after_code_review.md](questions_after_code_review.md). Bản preflight trước đó ở report.md được giữ để ghi nhận lịch sử thiếu source, không thay cho kết luận review lần này.
