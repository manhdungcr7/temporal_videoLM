# 5 notebook chẩn đoán nguyên nhân gốc (mỗi tài khoản Kaggle một notebook)

Mỗi tài khoản: mở notebook → **Session options: GPU T4 ×2, Internet = On** → **Save Version → Save & Run All** → tải file `.zip` ở cell cuối và gửi lại.
Mỗi notebook tự chạy thử 4 mẫu trước (dừng ngay nếu có lỗi thật; clip quá dài không tính là lỗi), rồi chạy đầy đủ và tự dừng trước hạn 12 giờ, giữ kết quả đã có.

| Tài khoản | Notebook | Câu hỏi | Thời gian |
|---|---|---|---|
| 1 | `nb1_j1_event_perception.ipynb` | Mô hình không *thấy* hành động, hay thấy mà không *gắn* với thời điểm lời nói? | 4–5 h |
| 2 | `nb2_j2_position_causal.ipynb` | Dùng thời gian hay chỉ dùng vị trí/im lặng? (đổi chỗ nửa clip, đảo TMRoPE, thêm nhiễu) | ~4 h |
| 3 | `nb3_j3_prior_lens.ipynb` | Thiên kiến "No" (TA) và "không bao giờ Past" (TF) đến từ đâu, ở lớp nào? (logit lens) | 3–4 h |
| 4 | `nb4_j4_postdiction.ipynb` | Mask nhân quả có chặn việc "hiểu lại" cảnh đã thấy sau khi nghe lời nói không (postdiction)? | ~6 h |
| 5 | `nb5_j5_internal_localization.ipynb` | Mô hình tự biết lời nói ở đâu không (attention theo head), kể cả khi có nhiễu? | ~4 h |

Nguồn: `jobs/` (mã thật của từng job), `build_notebooks.py` (sinh lại notebook: `python build_notebooks.py`).
Không có nhãn nào được đưa vào mô hình. Whisper (J1) chỉ để chấm. Notebook không chứa token hay URL nào.

`jobs/j4_official_protocol.py` (tái lập baseline toàn bộ benchmark) được giữ lại nhưng KHÔNG chạy bây giờ: chỉ chạy khi đã có phương pháp tổng quát được kiểm chứng trên mẫu nhỏ.

## Đợt 2 (2026-10-13): phương pháp can thiệp attention, ưu tiên chạy theo thứ tự

| Ưu tiên | Notebook | Nội dung | Thời gian |
|---|---|---|---|
| 1 | `nb8_j8_acar_pilot.ipynb` | ACAR: mốc từ attention câu hỏi → audio, phân bổ lại attention video (bộ pilot) | 2,5–3 h |
| 2 | `nb10_j9_nonspeech.ipynb` | Mốc từ attention với âm thanh không phải lời nói (chuông, bíp, vỗ tay, huýt sáo) | 1–1,5 h |
| 3 | `nb6_j7_cq_pilot.ipynb` | Câu tóm tắt 2 s của nhóm + bản vị trí song song (chạy lại, kết quả server bị mất) | 3–4 h |
| 4 | `nb9_j8_acar_heldout.ipynb` | ACAR trên tập held-out | 4–5 h |
| 5 | `nb7_j7_cq_heldout.ipynb` | Câu tóm tắt 2 s trên tập held-out | 5–6 h |
