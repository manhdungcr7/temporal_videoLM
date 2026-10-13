# Tổng quan nghiên cứu liên quan: lỗi thời gian (temporal) của video/omni LM và SVHalluc

Cập nhật 11/10/2026. Nguồn đọc: trang abstract và bản HTML trên arXiv. Các con số lấy từ paper (tự báo cáo, chưa có tái lập độc lập). Mục "Mức đọc" ghi rõ đã đọc đến đâu.

## 1. Các paper đã chạy trên chính SVHalluc

| Paper | Loại | Kết quả temporal | Mức đọc |
|---|---|---|---|
| [SVHalluc (CVPR 2026)](https://arxiv.org/abs/2606.02642) | Benchmark | Mọi model mở gần mức ngẫu nhiên ở TA/TF. Gemini 2.5 Pro: TA 85,17, TF 53,89, CMTB 69,25. Thêm transcript không giúp TA/TF và làm CMTB tệ hơn. | Toàn văn (bản local) |
| [RED (2610.02976)](https://arxiv.org/abs/2610.02976) | Decoding không huấn luyện | Qwen2.5-Omni-3B: TA 48,1→50,0; TF 21,4→29,0; CMTB 49,0→53,4. Cả 8 phương pháp decoding (VCD, ICD, CODE, SID, AVCD, ASD, MAD, RED) đều quanh mức ngẫu nhiên. | Bảng phụ lục đầy đủ |

**Kết luận:** chưa tìm thấy phương pháp nào cải thiện rõ phần temporal của SVHalluc. Đây vẫn là khoảng trống.

## 2. Đồng bộ thời gian nghe–nhìn trong model omni

| Paper | Nội dung | Huấn luyện? | Ghi chú cho ta |
|---|---|---|---|
| [SyncRA (2609.34363)](https://arxiv.org/abs/2609.34363) | Model omni không gắn được "nghe gì" với "thấy gì cùng lúc". Phép thử đảo cặp audio–video (AllFour) cho model gốc: Qwen2.5-Omni-7B **2,25%**, Qwen3-Omni 13,25%, MiniCPM-o 4.5 35,5%. Cách sửa là một hàm mất mát tương phản trên biểu diễn trung gian, căn theo mốc thời gian sẵn có. Sau huấn luyện: AllFour Qwen2.5-Omni 11→20,75 (so với SFT thường); trung bình 5 benchmark 52,65→54,92. Thời gian trôi qua giải mã tuyến tính được từ trạng thái ẩn (sai số ~5,7 s với video). | **Có** (10K câu, 2 GPU H200) | Gần nhất với vấn đề của SVHalluc nhưng **không chạy SVHalluc**, không phân tích positional encoding. Phép thử đảo cặp là công cụ đánh giá nên dùng. |
| [AVTrace (2609.19991)](https://arxiv.org/abs/2609.19991) | Benchmark chẩn đoán: định vị khởi điểm/khoảng, đồng bộ, thứ tự sự kiện. Cả 5 model omni **thấp hơn baseline nhãn đa số (0,556)** ở kiểm tra đồng bộ. | Có thử hậu huấn luyện | Thêm bằng chứng lỗi là chung; benchmark để kiểm tra tổng quát. |
| [Daily-Omni (2505.17862)](https://arxiv.org/html/2505.17862) | Benchmark suy luận nghe–nhìn có căn thời gian; kèm agent định vị + LLM. | Không (agent) | Benchmark để kiểm tra tổng quát. |
| [Qwen3.5-Omni report (2604.15804)](https://arxiv.org/html/2604.15804v1) | Thừa nhận vị trí theo thời gian tuyệt đối của TMRoPE quá thưa với đầu vào dài, làm yếu mô hình hoá thời gian dài. Đổi cách đánh số khi huấn luyện. | Có | Hỗ trợ giả thuyết "positional encoding là một nguồn lỗi". |

## 3. Positional encoding và thiên lệch vị trí trong video LM

| Paper | Phát hiện | Cách sửa | Huấn luyện? | Số liệu chính |
|---|---|---|---|---|
| [VideoRoPE (ICML 2025 oral)](https://arxiv.org/html/2502.05173v1) | Chia tần số cho chiều thời gian kém nên dễ bị nhiễu bởi đoạn giống nhau. Nêu 4 tính chất RoPE video tốt. | RoPE mới | **Có** (~704 giờ A100) | V-NIAH-D 74,67→87,11; VideoHallucer 34,3→46,2 |
| [PAS (CVPR 2026)](https://arxiv.org/pdf/2511.10979) | RoPE thời gian tạo "gợn sóng" theo khoảng cách nên một số khung bị giảm trọng số. | Lệch pha nhỏ, ngược chiều ở query giữa các nhóm head | **Không** | Qwen2.5-VL-7B: MVBench 67,2→69,5; TempCompass 71,5→73,3 |
| [Revisiting MPE (ICLR 2026)](https://arxiv.org/html/2510.23095v1) | MRoPE dồn thời gian vào kênh tần số cao. **VideoRoPE/HoPE làm tụt DocVQA, InfoVQA, ChartQA.** | MHRoPE, MRoPE-I | Có | Bài học: sửa vị trí phải kiểm tra cả task không phải video |
| [VRoPE (EMNLP 2025)](https://arxiv.org/abs/2502.11664) | Thiên lệch vị trí trong chú ý; đứt gãy ở chỗ nối video–văn bản | RoPE mới | Chưa xác nhận | — |
| [Video-LevelGauge (2508.19650)](https://arxiv.org/pdf/2508.19650) | 27 LVLM **ưu tiên phần đầu hoặc phần lân cận** của ngữ cảnh | Không đề xuất | — | — |
| **[DTR — Relaxing Anchor-Frame Dominance (2604.12582)](https://arxiv.org/html/2604.12582)** | Decoder dồn chú ý vào một "khung neo" **gần như độc lập nội dung** (khung đen vẫn giữ vị trí neo), thường ở **đầu và cuối** (Qwen2.5-VL, InternVL3.5); gắn với hallucination | Cân bằng lại chú ý theo khung ở các lớp giữa–cuối, khung bị bỏ quên được tăng điểm | **Không** | Qwen2.5-VL-7B: EventHallusion 68,22→73,84; VideoHallucer 75,50→77,75; MVBench 58,90→59,25. Video-LLaVA: VideoHallucer 16,9→31,78. Có subset bị tụt (EventHallusion-Mix). |
| [LLaVA-MLB (2503.11205)](https://arxiv.org/abs/2503.11205) | Image-LLM dùng cho video có **thiên lệch về khung cuối** | Gộp chú ý theo lưới | Không | — |
| [Video sycophancy (2506.07180)](https://arxiv.org/pdf/2506.07180) | Chú ý dồn vào **khung đầu** bất kể nội dung (chênh 2,11 lần) | Chọn khung chính | — | — |

**Điểm chung:** thiên lệch vị trí là hiện tượng đã được ghi nhận, chiều (đầu hay cuối) phụ thuộc model. Chưa có bài nào trong danh sách này làm trên **model omni nghe–nhìn** hoặc trên **SVHalluc**.

## 4. Giảm hallucination thời gian bằng decoding (video LM)

| Paper | Cách làm | Số liệu |
|---|---|---|
| [SEASON (CVPR 2026)](https://arxiv.org/abs/2512.04643) | Tự chẩn đoán từng token, contrastive decoding với mẫu âm thời gian/không gian | LLaVA-Video-7B: VidHalluc 63,8→78,7; TempCompass 69,7→73,7 |
| [EventHallusion / TCD (2409.16597)](https://arxiv.org/abs/2409.16597) | Contrastive với video bị làm mờ tín hiệu thời gian | Baseline của SEASON, DTR |
| [SmartSight (AAAI 2026)](https://arxiv.org/pdf/2512.18671) | Sinh nhiều câu trả lời, chọn theo độ "sụp" chú ý thời gian | — |

## 5. Định vị thời gian từ bên trong model

| Paper | Phát hiện / cách làm |
|---|---|
| [MLLMs Know When Before Speaking (2605.21954)](https://arxiv.org/abs/2605.21954) | Một số head ở prefill đã chú ý đúng đoạn; lúc sinh thì trôi đi. Lấy đoạn từ attention, cắt video, chạy lại. Chỉ cho xuất timestamp; model chỉ hình. |
| [Your VLM Already Knows When (2608.08315)](https://arxiv.org/pdf/2608.08315) | Hỏi Yes/No theo từng đoạn, đọc điểm từ logits để định vị, không huấn luyện. |

## 6. Bằng chứng của chính chúng ta (Qwen2.5-Omni-3B)

| Thí nghiệm | Kết quả | Mức tin cậy |
|---|---|---|
| Cấu trúc benchmark (397 clip) | Nửa không lời nói là im lặng tuyệt đối; chỉ vị trí lời nói đã cho TF 66,7% không cần video | Chắc (đo toàn bộ) |
| Cắt đúng đoạn lời nói (Kaggle, CMTB, n=55) | 52,7% → **96,4%**; đoạn im lặng 5,5%; 24/26 câu sai trùng lựa chọn của đoạn im lặng | Khá (một model) |
| Vị trí của hành động đúng (cùng dữ liệu) | Chỉ hình: nửa đầu 61,5% so với nửa sau 24,1% → **thiên về đầu clip** | Gợi ý, chưa nhân quả (đang chạy phép đảo thứ tự) |
| Head nội bộ có tự biết đoạn lời nói? (24 clip) | Không thấy head ổn định | Âm tính, mẫu nhỏ |
| Chặn chú ý theo khoảng lời nói (log Kaggle, 80/200) | CMTB 48→62%, TA 41→71%, TF không đổi | Sơ bộ, cần recall theo nhãn |

## 7. Hệ quả cho định vị đóng góp

1. **Không thể claim phát hiện thiên lệch vị trí là mới.** DTR, Video-LevelGauge và bài sycophancy đã ghi nhận; ta chỉ có thể chỉ ra nó **xuất hiện trong model omni và giải thích được lỗi SVHalluc**.
2. **Khoảng trống còn lại:**
   - chưa ai cải thiện temporal của SVHalluc;
   - DTR/PAS là cân bằng **không điều kiện** (không biết câu hỏi hướng tới lúc nào), chỉ làm trên model hình;
   - SyncRA giải quyết đúng vấn đề nghe–nhìn nhưng **cần huấn luyện** và không phân tích cơ chế vị trí.
3. **Hướng phân biệt khả dĩ:** *điều hướng chú ý theo thời gian có điều kiện theo tín hiệu* (lời nói hoặc âm thanh chỉ ra lúc nào cần nhìn), không huấn luyện, cho model nghe–nhìn. So trực tiếp với DTR và PAS (cân bằng không điều kiện), RED (decoding), SyncRA (huấn luyện).
4. **Đánh giá tổng quát bắt buộc:**
   - **Phơi ra vấn đề:** SVHalluc temporal; phép đảo cặp kiểu SyncRA; AVTrace.
   - **Tổng quát:** WorldSense, Daily-Omni, OmniVideoBench (omni); MVBench, TempCompass, EventHallusion, VideoHallucer (video).
   - **Không làm hỏng:** một benchmark ảnh/tài liệu (DocVQA/ChartQA).
   - **Model:** Qwen2.5-Omni 3B/7B, MiniCPM-o, và model chỉ hình (Qwen2.5-VL) cho bản neo bằng chữ.

## 8. Việc cần đọc toàn văn trước khi chốt claim

- DTR (2604.12582): công thức đầy đủ và cách chọn lớp, để tái hiện làm baseline.
- SyncRA (2609.34363): phần còn lại (phân tích lỗi, dữ liệu phép đảo cặp có công bố không).
- PAS: chi tiết phép lệch pha, để tái hiện trên Qwen-Omni.
- VRoPE: có cần huấn luyện không.
