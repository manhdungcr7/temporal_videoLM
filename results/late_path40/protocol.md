# Late-path intervention40 — protocol trước thực nghiệm, 2026-10-15

Mục đích: phân biệt intermediate readout pattern với tác dụng nhân quả của
attention/MLP muộn. Chạy sau ChronoDeSmear và factor ladder, đúng40/110 rotations.
Không speech-window, clock, chọn layer theo task/gold, sửa media hoặc backbone.
Đây là diagnostic và candidate đơn giản, không claim novelty scalar gating.

P0 native reference sao chép Round5. P1 giảm output MLP tại last decoder row
x.5 ở blocks28–35. P2 tương tự attention output x.5. P3 tăng MLP x1.5. P4 giảm
MLP x.5 chỉ exact lexical stem, không choices/format. P5 giảm cả attention/MLP
x.5 tại last row. P6 thêm random direction vào output MLP last row, norm bằng
.5*norm current MLP output, fixed seed2027+layer. Tất cả giữ native raw AV,
causal mask, RoPE và media hidden computation; chỉ tác động các query rows.

Last-row policy có thể áp dụng mỗi bước generation bất kể câu hỏi hoặc đáp án;
không cố định token Yes/No/Past/Present/Future. Stem matching chỉ là định vị
token trong harness, không phân loại dạng câu hỏi.

Với P6, norm khớp local current MLP update, không bảo đảm toàn trajectory khớp
P1 vì hidden states diverge sau can thiệp. Chỉ một direction seed, là control
yếu hơn multiseed. Không chọn gain sau khi thấy đúng/sai. Hook cleanup kiểm tra.

Native parity spotcheck ở rotation đầu mỗi worker, so cached input và Round5.
Mỗi condition lưu rawscore/duration/update norms; per-cell/confusion và paired
rescue/regression theo P0. Đừng quy gain cho grounding chỉ vì một nhánh tăng:
attenuation có thể đổi prior. Không kết luận model không có temporal link từ
một null result hoặc logit lens uncalibrated. Nếu muốn chứng minh causal late
overwrite cần thêm source/time perturbations có annotation độc lập.

Tối ưu triển khai trước khi suite bắt đầu: cache native media-prefix KV vì mọi
can thiệp chỉ ở query rows và không thay prefix dưới causal attention. Native
prefix parity<=.03 và cùng argmax mỗi rotation; fallback full khi không qua.
Mỗi condition ở rotation đầu worker so full/cached, nếu error>.03 hoặc khác
argmax thì dùng full và ghi method full-only cho worker. Không nới ngưỡng.
Native replay so Round5 mỗi rotation<=.03. Source/module/scale giữ nguyên,
chỉ tái dùng tính toán; record prefix/error/fallback trong raw rotations.
Runtime trong parity rotation có cả full reference, không phải latency triển khai.

Các nhánh source isolation, AV attention boost, query refresh, slot memory đã
thử trong lịch sử không chạy lại. Lượt này hỏi về computational paths trong
decoder muộn, khác native attention bias/memory readout đã thất bại.
