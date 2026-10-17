# Kiểm tra tài liệu trong đề xuất OB-TVOR v5 — 2026-10-16

Đã kiểm tra các trang nguồn chính thức dưới đây; mở HTML của RED, DTR, SEASON và LatentOmni để đối chiếu phạm vi phương pháp. Đây là kiểm tra nguồn và cách lập luận, không phải tái lập các paper. Những nguồn khác trong proposal chưa được kiểm toán toàn văn trong lượt này.

| Tài liệu | Nguồn chính thức | Ý nghĩa đối với thí nghiệm hiện tại |
|---|---|---|
| Activation Steering Decoding | [ACL 2025.acl-long.634](https://aclanthology.org/2025.acl-long.634/) | Có can thiệp activations để giảm hallucination, dùng directions từ calibration set và contrastive decoding. Không chứng minh mọi vector media bắt buộc qua VO hoặc phép tangent bảo toàn cú pháp 100%. |
| RED | [Relevant Evidence Decoding](https://arxiv.org/html/2610.02976v1) | Dùng question-required evidence và PMI decomposition audio/video/interaction. Đây là động cơ cho đánh giá có thay đổi bằng chứng liên quan/không liên quan, không phải bằng chứng riêng cho giải thích hình học VO của v5. |
| DTR | [Relaxing Anchor-Frame Dominance](https://arxiv.org/html/2604.12582v1) | Đúng tên paper về anchor-frame dominance và video hallucinations. Có thể liên quan đến thiết kế đối chứng temporal; chưa xác nhận chẩn đoán 8 micro Qwen ở đây. |
| SEASON | [Self-Diagnostic Contrastive Decoding](https://arxiv.org/html/2512.04643v1) | Đúng paper temporal hallucination. Cần so sánh protocol và chi phí inference nếu phát triển OB-TVOR thành một phương pháp. |
| Qwen2.5-Omni | [Technical Report](https://arxiv.org/abs/2503.20215) | Mô tả audio/video interleaving và TMRoPE. Decoder block source của chính runtime được lưu trong controls để đối chiếu vị trí lấy states và can thiệp. |
| AVTrace | [arXiv:2609.19991](https://arxiv.org/abs/2609.19991) | Là diagnostic suite temporal audio/video; bản abstract nhấn mạnh semantic text overlap không thay thế temporal localization. Hợp với yêu cầu chấm localization độc lập thay keyword overlap. |
| SyncRA | [arXiv:2609.34363](https://arxiv.org/abs/2609.34363) | Định danh khớp “Learning Temporal Correspondence in Omni-Modal Models”. Không coi việc dẫn paper là xác nhận cơ chế v5. |
| OmniHalluc-L | [arXiv:2606.03614](https://arxiv.org/abs/2606.03614) | Định danh khớp benchmark/calibration long-form omni hallucination; chưa chạy trong lượt này. |
| ReGuLaR | [arXiv:2605.30587](https://arxiv.org/abs/2605.30587) | Dùng supervision về objects/relations để grounding latent reasoning. Hỗ trợ cân nhắc labels relational độc lập; khác hướng vector thủ công không supervision của M3 hiện tại. |
| DIPE | [arXiv:2603.10863](https://arxiv.org/abs/2603.10863) | Định danh khớp positional encoding tách tương tác nội/chéo modality; không xác nhận uniform grid interpolation là frame PTS. |
| Revisiting MPE | [arXiv:2510.23095](https://arxiv.org/abs/2510.23095) | Định danh khớp. Cần phân biệt temporal coordinate của model và timestamps của frame thật. |
| PAS | [arXiv:2511.10979](https://arxiv.org/abs/2511.10979) | Định danh khớp training-free temporal encoding stabilizer. Chưa được tái lập ở lượt này. |
| LatentOmni | [arXiv:2605.22012v1](https://arxiv.org/html/2605.22012v1) | Đúng paper unified audio-visual latent reasoning; khuyến khích so sánh latent training và grounding, không chứng minh vector even/odd thủ công sẽ được frozen Qwen đọc đúng. |
| Unsilencing Visual Latents | [arXiv:2605.02735](https://arxiv.org/abs/2605.02735) | Định danh khớp “Visual Latents Know More Than They Say”. Cần kiểm tra đường chuyển thông tin từ probe sang answer generation. |
| AHA | [arXiv:2512.24052](https://arxiv.org/abs/2512.24052) | Định danh khớp audio-language hallucination và counterfactual hard negatives; khác việc dùng caption khác origin làm absence truth. |

Hai định danh sai của v4 không còn được dùng làm nguồn hỗ trợ trong bảng này. Định danh đúng chưa đồng nghĩa kết luận cơ chế trong proposal đã được paper chứng minh.

Ba suy diễn cần giữ ở mức giả thuyết:

1. Cộng residual lớn có thể làm hỏng câu trả lời; quan sát đó chưa chứng minh lệch “pre/post attention subspace” là nguyên nhân duy nhất. Các layer dùng residual dimension chung; phải kiểm soát cường độ, vị trí và cách chiếu.
2. Chiếu tangent và giữ norm chỉ ràng buộc hình học từng state. Nó không bảo toàn cú pháp/ngữ nghĩa hoặc logits 100%.
3. Cosine thuận/đảo bằng công thức đặt sẵn chỉ xác nhận parity của representation. Cần directional correctness bằng nhãn và can thiệp media độc lập để chứng minh temporal reasoning.
