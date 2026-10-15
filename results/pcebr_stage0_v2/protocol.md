# Stage-0 v2 — protocol và phạm vi trước chạy

2026-10-15. Chạy đúng attachment v2; lưu riêng khỏi v1. Token chỉ ở ignored config.

- Dùng nguyên4 pair caches,202 train/50 held-out, nguồn/split không đổi. Hai server chạy cùng M1/M2 để kiểm tra replica; không tăng N bằng cách gộp replicas hoặc3seeds.
- Script M3 lấy4 mẫu đầu của mỗi shard research40, rotation0. Dựng đúng8 decoder caches (4/server), raw full AV và letter-only prompt/budget lịch sử; không thay sample hoặc chọn theo đáp án. Không chạy bảng40 hay A0–A7.
- Mỗi server chạy nguyên function bodies của attachment. Wrapper lưu checkpoint theo condition, similarity matrices và kết quả từng phase; không đổi architecture/loss/seeds/training steps/conditions/episode labels. Nếu lỗi runtime, lưu nguyên lỗi và chỉ sửa tương thích kỹ thuật có giải trình.
- M1 có7 conditions×3seeds×2representations. Không coi7 conditions là full3×2×2 factorial: shuffle chỉ có two-stream+hard-neg.
- M2 PartB là synthetic interval-coordinate episode classifier với query pseudo-description, không phải raw-media counterfactual QA. Báo baseline/shuffle/null-only đầy đủ. Natural-unshifted evaluation chỉ được ghi nếu code thực sự tạo episodes này.
- M3 so6 mid với6 late layers, gain.14. Báo TV là sensitivity; không xem finite logits trên MC input là free-form generation. Phân biệt micro4/server với tổng8.
- H1 cũ+15pp không được âm thầm hạ sau kết quả. Nếu v2 không nêu gate mới định lượng thì báo chênh so matched controls, không tự tuyên bố pass toàn Stage-0.
- Sau mainrun có thể kiểm tra zero-sum/global-rank counterexample, coordinate-only/no-retrieval controls và query-batch invariance; giữ tên exploratory riêng, không thay số mainrun.
