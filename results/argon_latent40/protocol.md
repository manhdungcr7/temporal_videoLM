# Argon LatentChrono-CoT — development40, 2026-10-15

Giả thuyết: logit-lens top-48 kỳ vọng embedding giúp so khớp audio/video;
vector trạng thái câu hỏi tầng index 22 được hồi tiếp vào các tầng 10/12/14/16;
continuous thought được cộng vào residual ở tầng 0/1/2. Kết hợp clock,
foreground, additive attention mask, audio handoff tầng 14 và text-only PMI.
Đây là can thiệp training-free vào mô hình đóng băng, chưa chứng minh tái hiện
Coconut, recurrent-depth đã huấn luyện, hay xóa khoảng cách giữa modality.

Dùng đúng `results/research40_manifest.json`, SHA256
`0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
40 mẫu, 5 mỗi TA/A,B; TF/A,B,C; CMTB/A,B,C. Server1/2 mỗi20.
Không đổi mẫu khi lỗi; không chọn hyperparameter theo nhãn. Tất cả cyclic rotations.
Đây là tập development đã thử nhiều phương pháp, không phải held-out test.

M0 native; M1 native-.75 text prior; M2 hard gate+clock+foreground-.45 prior;
M3 recurrent+clock+foreground+desmear-.75 prior;
M4 thought+clock+foreground+desmear-.75 prior;
M5 tất cả+soft gate+handoff-.70 prior; M6 hỗn hợp thích ứng theo skew prior.
Đối chứng thêm: M7 clock+foreground+desmear-.75 prior không latent;
M8 chỉ inject recurrent ở pass cuối, nhưng delta vẫn lấy từ hard/anti có clock;
M9 chỉ inject thought ở pass cuối, không clock/mask/PMI trực tiếp.
M2 và M7 dùng vocabulary projection chung trong salience/clock, dù không
inject thought hay recurrent delta trực tiếp. Các control chưa độc lập hoàn toàn.

Thay đổi kỹ thuật công khai so với code gửi: manifest40 thay random32; raw AV
không cắt/reencode; prompt letter-only giống research40 baseline; truyền đúng
last_media ở mọi pass; decoder-only replay từ native encoded inputs; chỉ tính
lm_head token cuối; use_cache=False; mask đặt riêng theo device/layer thay mutation;
triangle bool thay int64 index pairs; explicit KV repeat cho SDPA T4.
Lưu từng rotation và từng sample; không xóa kết quả hay silently skip.
Mỗi rotation kiểm tra replay native/decoder max abs logit error<=.12 và logits hữu hạn.
Giữ nguyên sự nới causal V->A gần đồng bộ do code gốc có chủ ý.

Giới hạn cần đánh giá: “Value-Clock” ở đây thực chất cộng residual layer output,
không sửa v_proj. Clock dùng các từ Yes/No/Past/Present/Future, nên đặc thù benchmark.
`speech_window` dùng biên năng lượng âm thanh, không phải VAD xác nhận lời nói.
Shared embedding coordinates không bảo đảm aligned semantics hoặc zero modality gap.
Layer numbers là index0; index22 tương ứng decoder block thứ23, không layer21 probe.
Các câu về83.3% linear probe không chứng minh lm_head sẵn có đọc được hidden media.

Đối chiếu các bài gốc (đã đọc abstract ngày2026-10-15):
[Coconut](https://arxiv.org/abs/2412.06769) hồi tiếp last hidden state làm next
input embedding; tên tác giả là Hao và cộng sự, không có LeCun trong danh sách.
[Recurrent Depth](https://arxiv.org/abs/2502.05171) là mô hình với recurrent block
được huấn luyện; không suy ra hồi tiếp tùy ý giữa tầng của Qwen sẽ hiệu quả.
[Unsilencing Visual Latents](https://arxiv.org/abs/2605.02735) tối ưu latent tại
inference bằng contrastive alignment và confidence-progression; chỉ lấy top-k
embedding expectation chưa triển khai các bước tối ưu đó. Kết luận ở đây là
phân biệt cơ chế, không phủ định toàn bộ hướng latent reasoning.

Đầu ra: raw JSONL hai server, summary JSON, per-sample CSV, report tiếng Việt.
Đếm đúng/sai trên đủ40; paired rescue/regression, đối chiếu baseline lịch sử,
theo từng task/nhãn và từng method. Lỗi kỹ thuật được báo riêng, không tính thành sai.
