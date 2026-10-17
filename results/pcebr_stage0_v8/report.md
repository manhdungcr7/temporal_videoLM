# CLRV v8 — thí nghiệm micro8 trên hai server, 2026-10-17

**Kết quả mở rộng40: native 23/40, CAOR-only 20/40, CLC-only 19/40, CLRV full-direct 16/40.** Xem [bảng đúng/sai từng task/gold](report40.md) và [phản hồi gửi Yoshua](feedback_to_yoshua_bengio.md). Full rollout bridge chỉ chạy trên micro8.

Source gốc `fc0a9643973893f7362288161a685a0356395df65d6a8d6c57c359a8e5e9b895`. Bộ 8 ID cố định: 1449, 2150, 1929, 2054 / 1956, 1513, 2264, 1821; rotation 0, mỗi mẫu dưới 45 giây. Mỗi server chạy một shard 4 ID; đây không phải hai replicas của QA8. Không huấn luyện, không chỉnh ngưỡng theo đáp án.

M0 gốc tái lập qua trên CPU ở cả hai server. M0 CUDA gốc lỗi index GPU vào buffer CPU; wrapper chuẩn hóa đầu vào extraction về CPU như đường M3 thực tế. Các kết quả CPU do Argon gửi được giữ riêng (`argon_cpu_results.json`, M3 `skipped_no_live_model`); chúng không phải kết quả QA.

Nhánh gốc giữ công thức của source. Sửa v1 bổ sung ánh xạ endpoint→physical modality, gate foreground theo modality endpoint, phân loại whole-word qua offsets của tokenizer để không coi mảnh BPE là entity, ưu tiên connector quan hệ hơn `and` trong tên sự kiện, và sửa nhánh gate0 bỏ qua KV cache. Sửa v2 bổ sung tách mệnh đề mở đầu Relative/When theo dấu câu, và nhận diện các tính từ modality chung visual/auditory/acoustic/audiovisual. Các source/hash và thay đổi runtime lưu riêng; không đổi các hệ số gate/CLRV hay dùng nhãn QA để chọn tham số. Đây là sửa giao diện sau chẩn đoán trên development micro8, không phải đánh giá held-out độc lập.

Đối chứng bổ sung: native rollout rồi cùng bridge scorer; bridge khi không có rollout; permutation đã ánh xạ về nhãn gốc; order-reversed latent routing; trace-only không media. Order-reversed routing không bảo đảm văn bản sinh thực sự đảo quan hệ, nên phải đọc văn bản control trước khi diễn giải TV.

## Gốc: 8/8 mẫu đã lưu

| Cấu hình | Đúng | Sai |
|---|---:|---:|
| Native | 5/8 | 3 |
| CAOR attention only | 5/8 | 3 |
| CLRV direct | 5/8 | 3 |
| CLRV full | 5/8 | 3 |
| CLRV bridge only | 3/8 | 5 |
| Native rollout → bridge | 3/8 | 5 |
| Empty rollout → bridge | 4/8 | 4 |

| ID | Task/gold | Native | Direct | Bridge | Full | Native→bridge | Gate |
|---|---|---|---|---|---|---|---:|
| 1449 | TA/A | A | A | A | A | A | 0.0 |
| 1513 | TF/A | B | B | B | B | B | 0.0 |
| 1821 | TF/C | C | C | B | C | B | 0.0 |
| 1929 | TF/C | C | C | B | C | B | 0.0 |
| 1956 | CMTB/A | A | A | A | A | A | 0.0 |
| 2054 | TA/B | A | A | A | A | A | 0.0 |
| 2150 | TF/C | A | A | B | A | B | 0.0 |
| 2264 | CMTB/B | B | B | B | B | B | 0.0 |

Cứu/làm hỏng so native: 0/0. Gate >0 trên 0/8 mẫu.

## Sửa lỗi interface v1: 8/8 mẫu đã lưu

| Cấu hình | Đúng | Sai |
|---|---:|---:|
| Native | 5/8 | 3 |
| CAOR attention only | 5/8 | 3 |
| CLRV direct | 3/8 | 5 |
| CLRV full | 3/8 | 5 |
| CLRV bridge only | 3/8 | 5 |
| Native rollout → bridge | 3/8 | 5 |
| Empty rollout → bridge | 4/8 | 4 |

| ID | Task/gold | Native | Direct | Bridge | Full | Native→bridge | Gate |
|---|---|---|---|---|---|---|---:|
| 1449 | TA/A | A | B | A | B | A | 0.9924 |
| 1513 | TF/A | B | B | B | B | B | 0.9823 |
| 1821 | TF/C | C | C | B | C | B | 0.9925 |
| 1929 | TF/C | C | A | B | B | B | 0.9932 |
| 1956 | CMTB/A | A | A | A | A | A | 0.0 |
| 2054 | TA/B | A | A | A | A | A | 0.992 |
| 2150 | TF/C | A | B | B | B | B | 0.9915 |
| 2264 | CMTB/B | B | B | B | B | B | 0.0 |

Cứu/làm hỏng so native: 0/2. Gate >0 trên 6/8 mẫu.

## Sửa đầy đủ interface v2: 8/8 mẫu đã lưu

| Cấu hình | Đúng | Sai |
|---|---:|---:|
| Native | 5/8 | 3 |
| CAOR attention only | 5/8 | 3 |
| CLRV direct | 3/8 | 5 |
| CLRV full | 3/8 | 5 |
| CLRV bridge only | 3/8 | 5 |
| Native rollout → bridge | 3/8 | 5 |
| Empty rollout → bridge | 4/8 | 4 |

| ID | Task/gold | Native | Direct | Bridge | Full | Native→bridge | Gate |
|---|---|---|---|---|---|---|---:|
| 1449 | TA/A | A | B | A | B | A | 0.9924 |
| 1513 | TF/A | B | B | B | B | B | 0.9928 |
| 1821 | TF/C | C | C | B | C | B | 0.9928 |
| 1929 | TF/C | C | A | B | B | B | 0.9933 |
| 1956 | CMTB/A | A | A | A | A | A | 0.9928 |
| 2054 | TA/B | A | A | A | A | A | 0.992 |
| 2150 | TF/C | A | B | B | B | B | 0.9909 |
| 2264 | CMTB/B | B | B | B | B | B | 0.9886 |

Cứu/làm hỏng so native: 0/2. Gate >0 trên 8/8 mẫu.

## Theo task/ground truth

| Task/gold | N | Native đúng/sai | Gốc đúng/sai | Sửa v1 đúng/sai | Sửa v2 đúng/sai |
|---|---:|---|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 0/1 | 0/1 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 1/2 | 1/2 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

## Ablation trên bản sửa đầy đủ (không rollout)

| Cấu hình cuối decoder | Đúng/tổng |
|---|---:|
| CAOR_only | 5/8 |
| CAOR_CLC_no_CT | 5/8 |
| CAOR_CT_no_CLC | 4/8 |
| CT_only | 5/8 |
| CLC_only | 6/8 |
| CAOR_CLC_CT_direct_replay | 3/8 |

Replay combined direct khớp xác suất đã lưu: max error=0.0. CLC-only vẫn dùng các counterfactual CAOR để xây hướng CLC; nó chỉ bỏ CAOR và CT ở lượt decode cuối. Kết quả ablation được đọc exploratory, không chọn lại hệ số.

## Giới hạn và phản biện cơ chế

- Micro8 đã được xem qua nhiều vòng; không có TF/B hoặc CMTB/C. Không coi đây là đánh giá 40 mẫu, held-out mới hoặc bằng chứng tổng quát hóa.
- M0 dùng unit tokenizer theo nguyên từ và activations nhân tạo; quá trình thực thi phát hiện sai khác quan trọng khi chuyển sang tokenizer BPE/activations Qwen thật.
- `P_null` là heuristic. Independent CPU audit cho thấy có thể đổi coupling L1=1.282 mà null gần như không đổi dưới biên uniform; mean lag phụ thuộc các marginal; gate “conjunctive” có thể mở 0.45 khi hai từ chỉ được khớp ở hai slots khác nhau. Xem `independent_math_audit.json`.
- Null features được ghi ra audit nhưng gamma routing gốc được tính từ sharpness và excess gate; không được mặc định mọi cải thiện null score đều đã đi vào QA.
- Điểm freeform từ regex được giữ như diagnostic, không dùng làm semantic oracle. Cần chấm nghĩa độc lập và dữ liệu event/interval/absence có annotation cho bài toán thực tế.
- Lag sử dụng lưới nội suy; không coi là timestamp vật lý đã xác nhận. Giữ nguyên dữ liệu và kích thước media của baseline frozen40.
