# Tổng hợp phương pháp và kết quả J14–J25 (cập nhật 2026-10-16)

Mô hình: Qwen2.5-Omni-3B (Thinker), đóng băng. Testbed: SVHalluc temporal (TA, TF, CMTB) và semantic (GSA, FGSA, CSMB), thêm Daily-Omni.
Mọi kết quả báo theo từng (task, đáp án đúng), dạng `đúng/sai`. "bal" là độ chính xác cân bằng giữa các ô.

## 1. Mục tiêu và nguyên tắc

- **Mục tiêu:** Video LM thật sự hiểu thời gian, gồm: sự kiện xảy ra khi nào, trước hay sau, và hiểu đúng dù đổi cách hỏi.
- **Phương pháp phải nằm bên trong mô hình** (attention, residual, latent). Không dùng công cụ ngoài, agent, ASR hay cắt clip.
- **Phương pháp phải tổng quát:** không định tuyến theo tên task, và không được gây hại cho tác vụ khác.
- **Quy trình:**
  - Thí nghiệm tìm nguyên nhân chạy trên server.
  - Thí nghiệm dài hơn 1 giờ hoặc có huấn luyện thì chạy bằng notebook Kaggle.
  - Phải chứng minh hiệu quả trước khi huấn luyện.
  - Phép thử xác nhận phải khoá protocol trước khi chạy.

## 2. Tóm tắt ngắn

1. **Mô hình trả lời câu hỏi thời gian chủ yếu bằng TIẾNG tại KHOẢNH KHẮC CÓ LỜI NÓI.** Bằng chứng này được đọc xong trước layer 24 (J20).
2. **Mô hình tự biết khoảnh khắc lời nói nằm ở đâu.** Vùng neo lấy từ attention của chính nó trùng với đoạn có tiếng nói 100% (J21).
3. **Phương pháp không huấn luyện `gated_anchor` đã vượt qua phép thử đăng ký trước** trên 143 câu mới: 12 câu sửa đúng / 0 câu làm sai, p = 0,0005, semantic không bị hại (J23). Nhưng nó chỉ giúp CMTB (49% → 73%); TA và TF không đổi.
4. **Chia head thành nhóm trước/trong/sau không tạo ra hiểu biết về thứ tự** (J21).
5. **Thông tin thời gian CÓ trong token hình.** Probe tuyến tính đạt r ≈ 0,86 từ layer 6 và 0,91 ở layer 20 (J25). Giả thuyết "token hình không có đồng hồ" không đúng ở mức biểu diễn.
6. **Với mốc thời gian cho sẵn bằng chữ, mô hình suy luận được:**
   - thứ tự hai hành động 60/60;
   - Before/After 30/30 và 30/30;
   - nhưng với cách hỏi Past/Present/Future thì luôn chọn "Past" (90/90).
   Mô hình không bền khi đổi cách hỏi (J25).
7. **Trên video thật, TF dồn về "Present"** (20/30 câu), không phải "Past". Vậy trên video còn một lỗi nữa: không gắn được sự kiện với thời điểm của nó.
8. **Không riêng SVHalluc:** trên Daily-Omni, đảo ngược video chỉ làm đổi 35% đáp án Event Sequence (T2).

**Kết luận làm việc:** thông tin thời gian có trong biểu diễn, nhưng đường trả lời không dùng nó. Phần còn phải giải là gắn nội dung sự kiện với thời điểm trên video thật, rồi đọc ra đúng quan hệ.

## 3. Kết quả theo thí nghiệm

### J14–J16: probe token theo chunk
- **J14:** đối chiếu từ vựng ở hop 2 đúng 72–80%; độ liên quan PMI trên probe đúng 94–97%.
- **J15:** việc đáp án dồn về một lựa chọn (TA → Yes, TF → Present) là do câu hỏi đọc chính phần chữ của probe.
- **J16:**
  - probe tiếng chỉ đúng chunk có lời nói khoảng 100% (giống VAD, ngay từ layer 1);
  - probe hình chỉ đúng chunk có sự kiện ở mức ngẫu nhiên.
  - Nghĩa là mô hình không định vị được sự kiện HÌNH theo thời gian.

### J17: đặt câu hỏi lên trước video
- Gần như không có tác dụng: qpre 17/32, qpre_blind 14/32, p = 0,25.

### J18: id thời gian của TMRoPE
- Làm hằng số hoặc xáo trộn id thời gian của token hình chỉ đổi 3–9% đáp án. Mô hình gần như không dùng id thời gian của hình.
- Làm hằng số id thời gian của tiếng: 19/32, p = 0,06.

### J19: logit lens qua 36 layer
- Layer 8–30 chọn chữ A khoảng 100%, tức chỉ theo vị trí.
- Biên nội dung tăng ở layer 26–30. Lựa chọn chữ cái chỉ phụ thuộc nội dung từ layer 32–35.
- Tính theo từng token, layer 8–30 đọc tiếng nhiều gấp 3–7 lần đọc hình; các layer cuối chuyển sang đọc hình.

### J20: bản đồ chặn attention (độ sâu × nguồn × thời điểm)
28 câu (temporal 16, semantic 12). Mỗi ô là % câu đổi đáp án khi hàng câu hỏi bị chặn từ layer L0 tới cuối.

| Temporal | L0=8 | 16 | 24 | 30 |
|---|---|---|---|---|
| chặn tiếng (noA) | 31% | 25% | 12% | 0% |
| chặn hình (noV) | 6% | 12% | 6% | 0% |
| chặn khoảnh khắc lời nói (noSpeech) | 44% | 38% | 12% | 0% |
| chặn phần còn lại (noRest) | 12% | 12% | 0% | 0% |

- Base trùng đáp án với input chỉ có tiếng ở 94% số câu. Với CMTB, chặn hình làm đổi 0/6 câu.
- Semantic thì khác: chặn hình đổi 50/42/0/0%, tức semantic có dùng hình.
- Mọi việc đọc bằng chứng xong trước layer 24.
- Báo cáo đầy đủ: `results/j20/report.txt`.

### SEM-CHRONO: ChronoDeSmear của bạn, chạy trên semantic, 60 câu
- Mọi biến thể đều thấp hơn mô hình gốc: gốc 40/60 (bal 66,7); tốt nhất C5 35, C2 32; đối chứng xáo trộn 31.
- Kết quả âm. Chi tiết: `results/semantic_chrono/`.

### J21: định tuyến attention nội tại (bản sửa lỗi; bản v1 KHÔNG hợp lệ)
- **Vì sao v1 hỏng:** khi SDPA nhận `attention_mask=None`, phần tính lại attention bị thiếu mask nhân quả.
- **Bản sửa có kiểm tra tương đương:** đường "null" (không thêm bias) phải cho kết quả bằng base, đạt 55/55 câu, chênh lệch tối đa 0,031.
- **Vùng neo:** lấy khối lượng attention của câu hỏi vào từng chunk tiếng ở layer 12–35, chia cho câu hỏi trung tính theo kiểu PMI. Không dùng VAD trong phương pháp; VAD chỉ dùng để phân tích.

| | TA/A | TA/B | TF/A | TF/B | TF/C | CMTB/A | CMTB/B | CMTB/C | GSA/A | GSA/B | FGSA/A | FGSA/B | CSMB/A | CSMB/B | bal T | bal S | sửa/làm sai |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2/2 | 3/1 | 0/4 | 4/0 | 3/1 | 3/1 | 1/3 | 1/2 | 3/1 | 3/1 | 4/0 | 3/1 | 4/0 | 3/1 | 54,2 | 83,3 | – |
| block_audio | 0/4 | 4/0 | 0/4 | 4/0 | 0/4 | 2/2 | 2/2 | 1/2 | 0/4 | 3/1 | 4/0 | 2/2 | 4/0 | 4/0 | 41,7 | 70,8 | 4/11 |
| anchor_video | 0/4 | 4/0 | 1/3 | 2/2 | 0/4 | 3/1 | 4/0 | 3/0 | 0/4 | 4/0 | 3/1 | 3/1 | 4/0 | 4/0 | 56,2 | 75,0 | 10/12 |
| split (head trước/trong/sau) | 0/4 | 3/1 | 0/4 | 3/1 | 0/4 | 2/2 | 2/2 | 2/1 | 1/3 | 4/0 | 4/0 | 2/2 | 4/0 | 3/1 | 39,6 | 75,0 | 4/11 |
| split_random | 0/4 | 2/2 | 0/4 | 4/0 | 0/4 | 1/3 | 1/3 | 1/2 | 1/3 | 4/0 | 4/0 | 2/2 | 4/0 | 3/1 | 29,2 | 75,0 | 2/12 |
| gated_anchor (ghép sau khi nhìn số) | 2/2 | 3/1 | 1/3 | 4/0 | 3/1 | 3/1 | 3/1 | 2/1 | 3/1 | 3/1 | 4/0 | 3/1 | 4/0 | 3/1 | 67,7 | 83,3 | 4/0 |

- `anchor_video` sửa được CMTB, nhưng nếu bật cho mọi câu thì TA và GSA dồn về "No", và TF/C mất hết.
- `split` không giúp TF.
- Cổng entropy (< 0,8) mở chủ yếu cho CMTB.
- Báo cáo đầy đủ: `results/j21/report.txt`.

### J23: xác nhận `gated_anchor` theo protocol khoá trước
- 143 câu mới (seed 2030), loại toàn bộ câu của J20/J21. Protocol: `results/j23/protocol.md`.
- Kiểm tra tương đương null == base: 143/143.

| | TA/A | TA/B | TF/A | TF/B | TF/C | CMTB/A | CMTB/B | CMTB/C | GSA/A | GSA/B | FGSA/A | FGSA/B | CSMB/A | CSMB/B | bal T | bal S | sửa/làm sai |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 2/8 | 6/4 | 0/10 | 5/5 | 3/7 | 8/7 | 8/7 | 6/9 | 6/2 | 4/4 | 6/2 | 2/6 | 8/0 | 3/5 | 38,3 | 60,4 | – |
| **gated_anchor** | 2/8 | 6/4 | 0/10 | 5/5 | 3/7 | **13/2** | **11/4** | **9/6** | 6/2 | 4/4 | 6/2 | 2/6 | 8/0 | 4/4 | **47,5** | **62,5** | **12/0, p = 0,0005** |
| anchor_video (không cổng, tham khảo) | 1/9 | 10/0 | 4/6 | 9/1 | 0/10 | 14/1 | 13/2 | 12/3 | 0/8 | 8/0 | 6/2 | 3/5 | 8/0 | 7/1 | 62,5 | 66,7 | 39/11 |

- **ĐẠT** cả tiêu chí chính (sửa/làm sai, p < 0,05) lẫn tiêu chí phụ (semantic không giảm quá 2 điểm).
- Số câu cổng mở: CMTB 33/45, CSMB 4/16, các task khác 0.
- `anchor_video` không cổng làm đáp án dồn về một lựa chọn: TA 19/20 chọn No, GSA 16/16 chọn No, TF không bao giờ chọn Future. Con số 62,5 của nó KHÔNG dùng được.
- Lưu ý: zip Kaggle có thư mục `smoke/` chứa câu trùng. Chỉ phân tích file jsonl chính.

### J25: hai kiểm tra trước khi huấn luyện
- **Kiểm tra 1, chỉ dùng chữ:** đưa mốc thời gian bằng chữ rồi hỏi đúng câu của SVHalluc.

| Câu hỏi | Kết quả |
|---|---|
| Thứ tự hai hành động | 60/60 (không phụ thuộc thứ tự nhắc tên) |
| TA | bal 78,3% |
| TF Past/Present/Future | 90/90 chọn "Past" (giống hệt khi không có mốc) |
| TF Before/During/After | Before 30/30, After 30/30, During 5/30 |

  Ô During không đáng tin vì mẫu dựng sai: không mẫu nào có sự kiện nằm trọn trong lời nói. Mô hình trả lời theo đầu mút của sự kiện, và cách đọc đó hợp lý.

- **Kiểm tra 2, probe tuyến tính:** 38 clip, 462 chunk; ridge, kiểm chứng chéo theo clip; đích là chỉ số chunk đã trừ trung bình trong clip. Mức nhãn xáo trộn luôn xấp xỉ 0.

| Sau layer | 0 | 1 | 6 | 13 | 20 | 28 | 36 |
|---|---|---|---|---|---|---|---|
| r (hình) | 0,08 | 0,63 | 0,86 | 0,86 | 0,91 | 0,91 | 0,90 |
| r (tiếng) | 0,39 | 0,77 | 0,87 | 0,88 | 0,90 | 0,88 | 0,87 |

  Lưu ý: chỉ số chunk trùng với thứ tự trong chuỗi, và attention nhân quả có thể tự mã hoá vị trí. Vì vậy kết quả này chưa chứng minh mô hình dùng thời gian của TMRoPE. Báo cáo: `results/j25/report.txt`.

### T2: Daily-Omni, đảo video (81 câu, chạy dở)
- **Event Sequence:** base 24/48; đảo ngược video 20/48, chỉ đổi 17/48 đáp án.
- **AV Event Alignment:** base 17/33; đảo ngược 16/33, đổi 10/33.

## 4. Phương pháp hiện có: `gated_anchor` (không huấn luyện)

1. **Vùng neo:** đo khối lượng attention của hàng câu hỏi vào từng chunk tiếng 2 giây, trung bình qua layer 12–35, cho câu hỏi thật và cho câu hỏi trung tính ("What is happening in the video?"). Tính rel = p·(p/p0)², chuẩn hoá.
2. **Cổng:** entropy chuẩn hoá của rel < 0,8 (cố định, không chỉnh).
3. **Khi cổng mở:** ở layer 13–24, chỉ các hàng câu hỏi/đáp án bị chặn đọc token tiếng, và được cộng bias 4·log(rel/max rel) vào token hình của từng chunk. Khi cổng đóng, mô hình giữ nguyên.
4. **Mã nguồn:** `repo/kaggle/jobs/j21_temporal_split.py` (hàm `anchor`, `make_bias`, `route_attention`) và `j23_gated_confirm.py`.

## 5. Kế hoạch tiếp theo (chưa chạy)

**J22: huấn luyện nhẹ phần đọc thời gian.**
- **Mô-đun:** LoRA hạng 8 trên q, k, v, o ở layer 13–35. Đồng hồ hình f(t) chỉ còn là nhánh đối chứng, vì J25 cho thấy thông tin thời gian đã có sẵn.
- **Dữ liệu tự ghép** (không dùng SVHalluc), hai loại với tỉ lệ ngang nhau:
  - **Hình–tiếng:** bộ ba X trước / cùng lúc / sau lời nói Y. "Cùng lúc" phải chồng lấn ít nhất 80%; "trước/sau" cách nhau ít nhất 1,5 s.
  - **Hình–hình:** X₁ trước X₂ và ngược lại.
  - Nhiều cách diễn đạt (Past/Present/Future, Before/During/After, earlier/later...). Chỉ giữ lại câu dẫn của SVHalluc, không giữ lại từ vựng.
  - Khoảng 30% câu không liên quan thời gian, có loss KL với mô hình gốc.
- **Đối chứng:** nhãn thời gian xáo trộn, f(t) + LoRA, chỉ LoRA.
- **Đánh giá:**
  - SVHalluc TA/TF/CMTB theo từng (task, đáp án), kèm ma trận nhầm lẫn;
  - Daily-Omni (Event Sequence, AV Event Alignment, kèm đảo video);
  - semantic;
  - chạy lại kiểm tra 1 của J25;
  - tập ghép giữ lại, đo tỉ lệ đổi đáp án đúng khi đổi thứ tự.
- **Trong cùng notebook:**
  - probe J25 với id thời gian hằng số, để biết thông tin thời gian đến từ RoPE hay từ thứ tự chuỗi;
  - base với câu TF đổi sang Before/During/After trên video thật, để biết phần đổi cách hỏi giải được bao nhiêu.
- **Tiêu chí dừng (khoá trước khi chạy):** nếu nhánh nhãn xáo trộn đạt bằng nhánh thật, hoặc không có cải thiện nào chuyển sang video thật, thì dừng hướng này.

**J24:** `gated_anchor` trên Daily-Omni, cùng quy trình khoá trước như J23, để kiểm tra không gây hại và tính tổng quát.

## 6. Cách tái lập

| Thí nghiệm | Notebook (`repo/kaggle/`) | Phân tích (`remote_tests/`) |
|---|---|---|
| J14, J15 | nb14, nb15 | analyze_j14.py |
| J17, J18 | nb16, nb17 | analyze_j17_j18.py |
| SEM-CHRONO | nb18, nb19 | server_jobs/semantic_manage.py summary |
| J20 | nb20, nb21 | analyze_j20.py s1.jsonl s2.jsonl |
| J21 | nb22, nb23 | analyze_j21.py s1 s2 --j20 j20_s1 j20_s2 |
| J23 | nb24, nb25 | analyze_j23.py s1 s2 (chỉ file chính, không lấy smoke/) |
| J25 | nb26 (j25b) | analyze_j25.py j25_reason.jsonl j25_probe/ |

- Notebook được dựng bằng `python repo/kaggle/build_notebooks.py`.
- Job cho server được dựng bằng `server_jobs/build_server_job.py <job> <out>`, với các chế độ j12–j19 và j25, có `--shard K/N`.
- File jsonl và gz trong `results/` bị `.gitignore` chặn, nên chỉ đẩy lên báo cáo dạng text (`report.txt`, `protocol.md`). Cần dữ liệu thô thì hỏi manhdungcr7.
