# Đọc thêm trong lúc Round5 chạy — 2026-10-15

Chỉ dùng nguồn paper/project chính chủ. Không dùng điểm paper làm kết quả
thực nghiệm của Qwen2.5-Omni-3B. Đây không phải tìm kiếm exhaustive.

|Nguồn đã đọc|Nội dung liên quan|Hệ quả cho nghiên cứu hiện tại|
|---|---|---|
|[Qwen2.5-Omni technical report, mục2.2](https://arxiv.org/html/2503.20215v1)|TMRoPE và interleaving AV theo chunk2s.|Cần phân biệt causal token order với physical time. Positional alignment không tự bảo đảm event reasoning.|
|[Causality Matters, AAAI2026, mục4–5](https://ojs.aaai.org/index.php/AAAI/article/view/37856/41818)|Dùng PE/order perturbation và attention knockout để phân tích đường truyền temporal qua video rồi query.|Accumulation có thể là context hữu ích. Suy luận của chúng tôi: cần ablate trước khi đồng nhất accumulation với smearing. Bài nghiên cứu VideoLM, không xác nhận cơ chế QwenOmni3B.|
|[EventHallusion/TCD, mụcTemporal Contrastive Decoding](https://arxiv.org/html/2409.16597v6)|Tạo đầu vào temporal cues mơ hồ nhưng giữ thông tin không gian rồi contrast ở decoding.|Contrast factual/background để giảm event hallucination đã có tiền lệ. ChronoDeSmear không thể claim primitive temporal contrastive decoding mới.|
|[SEASON, mục3.2–3.3](https://arxiv.org/html/2512.04643v1)|Temporal homogenization từng tầng vision encoder; spatial/temporal negatives; frame-attention divergence điều khiển contrastive logits từng token.|Rất gần động cơ background-vs-dynamics nhưng khác nơi can thiệp và công thức. Cần so trực tiếp nếu xây contribution inference-time temporal contrast; không chỉ so native. Không gọi ChronoDeSmear là tái lập SEASON.|
|[AIR/Look Carefully, mụcmethod](https://arxiv.org/html/2602.24041v1)|Prototype reduction và OT-guided patch reinforcement vào feed-forward layers.|Can thiệp FFN để tăng visual grounding không mới về ý tưởng rộng. Late-path suite hiện chỉ diagnostic scalar gating, không claim contribution cùng AIR.|
|[ACPO, paper và project](https://arxiv.org/html/2604.14129v2)|Input-contrastive audio swapping và output-contrastive preference để giảm visual shortcuts khi cần nghe.|Source sensitivity/attribution supervision là hướng có căn cứ. Nhưng audio grounding không đồng nghĩa jointAV temporal reasoning; không lấy gain của ACPO làm gain trên40.|

Đọc này củng cố lựa chọn không đi theo agent workflow. Nó cũng cho thấy chỉ
đổi tên recurrence/contrast hoặc gộp primitive không đủ xác lập novelty. Khoảng
cần kiểm chứng là event-instance/source/time provenance và việc các can thiệp
bảo toàn evidence khi hỏi nhiều hướng, đi kèm mechanism controls và held-out.

Những gì chưa làm: chưa chạy checkpoint/official implementation của các paper,
chưa chuyển SEASON sang Omni hoặc tái lập ACPO training, chưa có event grounding
annotations độc lập cho40. Không trình bày các nhận xét này như benchmark
comparison với prior art hoặc chứng minh khả năng được nhận CVPR.
