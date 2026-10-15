# Kiểm tra các nguồn được dẫn trong đề xuất v4 — 2026-10-15

Đã mở trang chính thức của năm tài liệu được dẫn và HTML toàn văn của VideoRoPE/HoPE. Bảng này kiểm tra định danh và phạm vi lập luận; không phải tái lập kết quả các paper.

| Tên trong đề xuất | Nguồn được kiểm tra | Kết quả |
|---|---|---|
| VideoRoPE | [arXiv:2502.05173v3](https://arxiv.org/abs/2502.05173v3), [HTML](https://arxiv.org/html/2502.05173v3) | Đúng paper về thiết kế RoPE cho video: phân bổ tần số thời gian, bố trí và khoảng cách temporal/spatial. Đây là động cơ nghiên cứu positional encoding. Không tự chứng minh nguyên nhân thất bại của CQ-TSPR trên Qwen2.5-Omni-3B. |
| HoPE | [arXiv:2505.20444v2](https://arxiv.org/abs/2505.20444v2), [HTML](https://arxiv.org/html/2505.20444v2) | Đúng paper về positional embedding cho long-context VLM; có hybrid frequency allocation/dynamic temporal scaling. Chưa phải bằng chứng can thiệp nhân quả cho các clip micro 12–37 giây của lượt này. |
| SynchOnly | [arXiv:2509.23142](https://arxiv.org/abs/2509.23142) | **Sai định danh**: URL là “Exploring Magnetic Phases in Dual-Species Mott insulating Spinor Lattice Gases”, một paper quantum gases. Không dùng URL này làm bằng chứng cho AV temporal reasoning. |
| AV-Haystack / LART | [arXiv:2410.18514](https://arxiv.org/abs/2410.18514) | **Sai định danh**: URL là “Scaling up Masked Diffusion Models on Text”. Không phải nguồn AV-Haystack/LART được mô tả. |
| Ethayarajh 2019 | [ACL D19-1006](https://aclanthology.org/D19-1006/) | Đúng paper “How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings”. Liên quan anisotropy; không xác nhận rằng cosine/Jaccard thấp chứng minh event vắng mặt trong một recording. |

Tìm kiếm bổ sung theo tên không xác minh được cặp tên/link SynchOnly và AV-Haystack/LART như đề xuất. Có [repo chính thức MAGNET](https://github.com/schowdhury671/magnet) dùng tên AV-Haystacks; điều đó không đủ để tự thay thế tên hoặc kết luận của tài liệu Yoshua muốn dẫn. Cần cung cấp đúng title/authors/URL trước khi đưa vào related work.

Các kết luận cơ chế như “90% patch là background”, “SNR < −11 dB”, “pooling đã xóa sạch thông tin”, “đã chứng minh audio shortcut do M-RoPE” cần phép đo riêng trên mô hình và dữ liệu đang chạy. Cosine .9231 và thất bại của một vài projector chỉ hỗ trợ giả thuyết về representation; không định lượng tỷ lệ patch nền, không chứng minh mutual information bằng 0. Data Processing Inequality giới hạn thông tin qua phép biến đổi, không cấm projector học khai thác phần thông tin hữu ích còn lại.
