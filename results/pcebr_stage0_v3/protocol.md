# PC-EBR Stage-0 v3 — trước thực nghiệm

2026-10-15. Lưu nguyên attachment và trích đúng một code block Python v3. Chạy cùng thuật toán M1/M2 trên cả hai server để kiểm tra replica, không gộp thành thêm samples. Các ngưỡng gate lấy từ phần lời nhắn trước code của Yoshua, không đổi hậu kiểm.

- Cùng4 pair caches,202 train/50 held-out pairs,16/4 origins. Không chọn mẫu/candidate/seed theo đáp án; held-out đã được xem qua các vòng nên vẫn là development validation.
- Hai runtime mới. Dựng4 decoder caches/shard, rotation0, đúng manifest research40 và raw AV/prompt/pixel budget lịch sử; bổ sung input_ids để v3 tách span. Tổng8mẫu, không chạy bảng40/A0–A7/reserve400.
- M1 giữ đúng180steps,3seeds,diag/lowrank/context/localization/shuffle. Wrapper lưu frozen context/local weights, mỗi score matrix và audit trước khi chạy tiếp.
- Gate M1: group-max error0, winning-group agreement100%; C4/C7 localization Exact/Multipos/Nonnull hơn context-only và C8 trên cả native/pooled. Chỉ đúng identity toán học không đủ thông qua gate hiệu năng.
- Gate M2 trên mỗi B1/B2/B3: uniform ≤28%4-way,≤35%3-way; full hơn cả std/coord-shuffle/wrong-event/uniform ít nhất10pp; During recall≥35%. Không gọi chance dưới sim=0 là tự chứng minh không còn shortcut khác.
- Gate M3: replay≤1e-5 ở8/8; option rows không bị hook trực tiếp; mid/late swap≥2; M1+M2 sensitivity hơn random và irrelevant; free-form12-token path thành công8/8. Phân biệt sensitivity và temporal correctness.
- Chạy nguyên function bodies trước. Nếu lỗi autograd/device/cache schema, lưu traceback và chỉ sửa kỹ thuật cần thiết, giữ riêng source và không lặp train M1 đã xong. Không tự thay các công thức hay labels.
- Các quan sát bổ sung về token spans, uniform positive-part, foil selection, exact gate values và semantic limitations là audit riêng, không sửa mainrun.
