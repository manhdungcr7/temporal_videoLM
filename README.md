# temporal_videoLM — Vì sao video LM thất bại ở bài toán thời gian, và cách sửa không cần huấn luyện

Repo này chứa **toàn bộ** nghiên cứu về lỗi *temporal* (thời gian) của video/omni language model trên benchmark **SVHalluc**, gồm:

- mã nguồn và notebook Kaggle cho mọi thí nghiệm;
- kết quả thô của các thí nghiệm (`results/`);
- nguyên nhân gốc đã xác định, các phương pháp đã thử cùng ưu và khuyết điểm;
- việc đang làm dở và hướng đi tiếp để viết paper.

Mục tiêu của repo là để người khác clone về rồi tự tiếp tục: thí nghiệm, phát triển phương pháp và viết paper.

> **Mới nhất (2026-10-16):** tổng hợp phương pháp và kết quả J14–J25 (bản đồ chặn attention, `gated_anchor` đạt phép thử đăng ký trước, probe thời gian, kế hoạch huấn luyện J22) ở [`docs/ket_qua_j14_j25.md`](docs/ket_qua_j14_j25.md).

- Mô hình chính: **Qwen2.5-Omni-3B Thinker**, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`. Có kiểm tra phụ với Qwen2.5-Omni-7B.
- Benchmark chính: **SVHalluc** (CVPR 2026, arXiv 2606.02642), phần temporal: **TA** (368 câu), **TF** (276 câu), **CMTB** (339 câu). Dataset HF `zcs15/SVHalluc`, revision `211b4f14ab66ae9557b69e302d1f090e0e610839`.
- Benchmark phụ: **Daily-Omni** (HF `liarliar/Daily-Omni`), video tự nhiên.

---

## Mục lục

0. [Tóm tắt một trang](#0-tóm-tắt-một-trang)
- **0A. [Trạng thái hiện tại, phương pháp đang xây dựng và việc làm tiếp (cập nhật 13/10/2026 tối)](#0a-trạng-thái-hiện-tại-phương-pháp-đang-xây-dựng-và-việc-làm-tiếp)** ← **đọc phần này trước**
1. [Bài toán và thiết lập](#1-bài-toán-và-thiết-lập)
2. [Cấu trúc repo](#2-cấu-trúc-repo)
3. [Cách chạy lại](#3-cách-chạy-lại)
4. [Nhật ký thí nghiệm và kết quả (theo giai đoạn)](#4-nhật-ký-thí-nghiệm-và-kết-quả-theo-giai-đoạn)
5. [Nguyên nhân gốc đã xác định](#5-nguyên-nhân-gốc-đã-xác-định)
6. [Các phương pháp đã thử: ưu, khuyết, khả năng CVPR, hướng tiếp](#6-các-phương-pháp-đã-thử)
7. [Kết luận và nguyên tắc thiết kế](#7-kết-luận-và-nguyên-tắc-thiết-kế)
8. [Đang làm dở và lộ trình tới paper](#8-đang-làm-dở-và-lộ-trình-tới-paper)
9. [Cạm bẫy kỹ thuật đã gặp](#9-cạm-bẫy-kỹ-thuật-đã-gặp)
10. [Tài liệu liên quan](#10-tài-liệu-liên-quan)

---

## 0. Tóm tắt một trang

**Bài toán:** mô hình nghe một người nói và xem video. Ví dụ câu hỏi: "Khi người đó nói, hành động được nhắc tới có đang diễn ra không (TA)?", "…xảy ra trước, trong hay sau lời nói (TF)?", "…hành động nào đang được nhìn thấy (CMTB)?". Mọi mô hình mã nguồn mở chỉ đạt quanh mức ngẫu nhiên. Qwen2.5-Omni: TA 50,3, TF 31,5, CMTB 53,1. Các phương pháp giải mã tốt nhất đã công bố chỉ đạt 53,3 / 35,5 / 53,4.

**Nguyên nhân gốc (đã chứng minh):**
1. **Tri giác không phải nút thắt.** Hỏi riêng từng nửa clip thì mô hình nhìn đúng hành động 89–91% (J1, n=218). Nó nghe đúng nội dung lời nói 96%.
2. **Mô hình biết lời nói ở đâu.** Khoảng 300 attention head định vị được lời nói, IoU 0,88 khi audio sạch và 0,83 khi có nhiễu (J5).
3. **Nó không dùng thời gian để chọn bằng chứng.** Nếu ta chọn hộ đúng đoạn thời gian (định tuyến attention), CMTB tăng từ khoảng 52 lên 92–100.
4. **Nhầm nguồn: coi điều NGHE là điều THẤY.** Nếu giữ audio thì dù chọn đúng thời điểm, mô hình vẫn chọn hành động *được nói tới* (CMTB chỉ 59). Lỗi này xảy ra cả khi suy luận bằng văn bản.
5. **TF bị thì của lời nói điều khiển.** Đây là kết quả nhân quả: thay lời nói bằng TTS thì tương lai thì P(Future) = 0,80. Ngay cả khi có bằng chứng hoàn hảo dạng chữ, mô hình vẫn trả lời "Present" nếu không suy luận từng bước.
6. **Báo thời gian bằng chữ hoặc bằng số giây là vô dụng.** Mô hình chỉ dùng được quan hệ có tên: trước / trong / sau.
7. **Đã loại trừ:** thứ tự nhân quả trong mask (postdiction, J4); khoảng im lặng tuyệt đối là lối tắt (J5); "attention dồn về đầu video" (không lặp lại).

**Phương pháp tốt nhất hiện có (tự động, không huấn luyện, không ASR, can thiệp attention):**
- Mô hình **tự tìm mốc thời gian** qua attention của câu hỏi lên token audio, khử độ nổi bật bằng PMI so với một câu hỏi trung tính.
- Ở lớp 13–27, attention video của câu hỏi được dồn về đoạn đó và **tắt audio**.
- Kết quả CMTB: **52,9 → 94,1** (pilot), **73,1 → 92,3 = oracle** (held-out) (J8).
- Bước tìm mốc đi theo câu hỏi với **cả âm thanh không phải lời nói** (chuông, bíp, huýt sáo: đúng 86–91%; attention thô chỉ 7%) (J9).

**Còn mở** *(tình hình mới nhất và việc làm tiếp: xem mục 0A)*:
- TA chưa cải thiện trong cách làm một lượt.
- TF cần suy luận có chủ ý. Pipeline đầy đủ J11 đang được đánh giá.
- Cần làm cho phương pháp chạy **trong một lượt** (thử đầu tiên: áp bias từ lớp 21 thì quá muộn, phải từ lớp 13).
- Cần chứng minh **tổng quát** trên mô hình và benchmark khác.

---

## 0A. Trạng thái hiện tại, phương pháp đang xây dựng và việc làm tiếp

*(cập nhật tối 13/10/2026. Mục 0 ở trên là tóm tắt của giai đoạn trước; mục này là tình hình mới nhất. Chi tiết số liệu ở mục 4.15–4.22.)*

### 0A.1 Mục tiêu của paper và các ràng buộc (nguyên văn yêu cầu của nhóm)

- **Bài toán:** phương pháp giúp mô hình xác định đúng **thời gian diễn ra các sự kiện** trong video và **hiểu thật sự thứ tự trước–sau**,
  tốt cho **nhiều ngữ cảnh**. Nếu đổi cách hỏi hoặc đổi sang bối cảnh khác mà làm sai thì chắc chắn không được nhận. SVHalluc
  chỉ là *một* benchmark để chứng minh bằng thực nghiệm.
- **Phương pháp phải:** hoàn toàn mới; đơn giản (kiểu LayerNorm / residual); dùng cho mọi video LM; **tự động hoàn toàn** (không
  định tuyến theo tên task, không chọn tay theo mẫu); ưu tiên can thiệp **trong mô hình** (attention / Q-former) hơn là ASR;
  âm thanh có thể không phải lời nói (tiếng chó sủa…), và **cảnh** cũng tính như âm thanh.
- **Quy trình do nhóm đặt ra:** nghiên cứu trước, chứng minh hiệu quả **trước khi huấn luyện**; huấn luyện chỉ ở mức nhẹ (tối đa khoảng
  một phiên Kaggle T4×2). Thử trên vài mẫu SVHalluc trước; **chỉ khi gần 100%** mới chạy SVHalluc đầy đủ, rồi benchmark và mô hình khác.
  **Không chạy full benchmark** (kể cả baseline chính thức J4) khi chưa có phương pháp. `nb12` (J10 held-out) và seed `1234`
  giữ lại cho lần xác nhận cuối.

### 0A.2 Những phát hiện mới nhất (đã kiểm chứng)

| # | Phát hiện | Số liệu | Mục |
|---|---|---|---|
| 1 | Thời điểm của **mốc nổi bật** (lời nói) đã nằm sẵn trong mô hình; thời điểm của **một hành động cụ thể** thì không | Có **31–141 trên 576 head** chỉ đúng chunk có tiếng nói (kiểm chéo: 0,91–1,00 so với ngẫu nhiên 0,52); **0 head** chỉ đúng nửa video có sự kiện hình ảnh (0,44–0,56 so với 0,50) | 4.20 |
| 2 | Mù thứ tự **không riêng SVHalluc** | Daily-Omni Event Sequence: đảo ngược video chỉ đổi 35% đáp án, độ chính xác 50 → 41,7 | 4.18 |
| 3 | Mô-đun A (định tuyến attention) **gây hại ngoài SVHalluc** | Daily-Omni, n=220: 66,4 → 54,5 | 4.17 |
| 4 | Câu hỏi TA/TF **không nhắc gì tới hành động** ("the narrated event"), nên tìm sự kiện là bài toán **hai bước** (câu hỏi → nội dung lời nói → chunk hình ảnh khớp). Quan hệ một bước câu hỏi → prompt hình không thể đúng | Giải thích kết quả mục 1 | 4.20 |
| 5 | Prompt ghi kiểu "…content of chunk t is" chỉ mang **định dạng**; prompt kiểu "…in one word:" mang **nội dung thật** | Token dự đoán kế tiếp: `: " . that` ở mọi chunk, so với `Ingredients / Cooking / Stir / Wr(ap)…` | 4.20 |
| 6 | Thêm khối prompt dài làm mô hình **dồn về một đáp án**: TA → "Yes", TF → "Present". Phần "tăng" độ chính xác chỉ là hiệu ứng của mẫu | Xem bảng theo (task, đáp án đúng) ở 4.16 | 4.16 |
| 7 | Đặt prompt ngay sau chunk của nó gây thiên lệch về chunk cuối (RoPE) | Tương quan relevance–t: −0,15 (vị trí chung) so với +0,82 (đặt theo thời gian) | 4.15 |
| 8 | Chỉ huấn luyện m embedding dò (mô hình đóng băng) học được một phần nội dung chunk, chưa đủ | KL 0,78 → 0,48; khớp token 65 → 73% | 4.19 |

### 0A.3 Phương pháp đang xây dựng: **chunk prompts + attention tăng cường theo độ liên quan** ("CQ")

**Đặc tả (do bạn trong nhóm đề xuất):**
1. Mỗi chunk 2 s (một chunk TMRoPE) có **hai prompt dò**: `The essential visual content of chunk t in one word:` (chỉ nhìn **hình** của chunk t) và
   `The essential audio content of chunk t in one word is:` (chỉ nhìn **tiếng** của chunk t); t = 1…n. Gọi là C_{t,v} và C_{t,a}.
   Các prompt **không thấy nhau và không thấy câu hỏi**; chỉ phần câu hỏi/đáp án mới thấy được các prompt.
2. **Vị trí:** mọi prompt cùng position (như thể mỗi prompt đứng ngay trước câu hỏi), tức bắt đầu từ p0 = ngay sau media; câu hỏi nằm ở p0 + độ dài prompt dài nhất.
   *(Đã thử thêm biến thể "prompt đặt ngay sau chunk của nó" (mục 4.15); nhóm chọn lại vị trí chung.)*
3. **Ba điều kiện so sánh** (cùng một lượt chấm xoay thứ tự đáp án):
   - `cq`: có prompt, attention bình thường;
   - `cq_boost_p`: tăng attention của token câu hỏi/đáp án vào các prompt liên quan;
   - `cq_boost_pr`: như trên **và** tăng cả vào **token thô** (video hoặc audio) của chunk và modality tương ứng.
4. **Độ liên quan:** mỗi layer, mỗi head, mỗi token câu hỏi/đáp án lấy **attention của nó tới token cuối của mỗi prompt** (chữ "is" hoặc dấu `:`).
   Bias cộng vào logit attention: `λ · log(r / max r)` (λ = 2). Prompt liên quan nhất giữ nguyên trọng số; prompt khác bị đẩy xuống. Tính **riêng cho từng head**.

**Cài đặt:** `repo/kaggle/jobs/j12_chunk_prompts.py` (hàm attention tuỳ biến `cq_attention` được thay vào `ALL_ATTENTION_FUNCTIONS['sdpa']` trong lúc chạy; mask + position ids tự dựng như `j8_acar.Lay`).
Chạy trên server bằng `server_jobs/build_server_job.py` (mục 3.3). Kết quả: `results/j12*.jsonl`.

**Kết quả hiện tại (chưa đạt):**

| Lần chạy | n | base | cq | cq_boost_p | cq_boost_pr |
|---|---|---|---|---|---|
| J12 (prompt "…is", vị trí chung) | 11 | 36,4 | 54,5 | 63,6 | 63,6 |
| J12d (prompt "…in one word:", vị trí chung) | 28 | 35,7 | 42,9 | 39,3 | 42,9 |
| J12b (chỉ TF, prompt đặt ngay sau chunk) | 10 | 30,0 | 30,0 | 30,0 | 40,0 |

**Vì sao con số trên không đáng tin (kết luận quan trọng nhất):** tách J12d theo (task, đáp án đúng) cho thấy "tăng" là do mô hình **dồn về một đáp án**:

| Task | Đáp án đúng | n | base | cq | cq_boost_p | cq_boost_pr |
|---|---|---|---|---|---|---|
| TA | A (Yes) | 4 | 1/4 | **4/4** | **4/4** | **4/4** |
| TA | B (No) | 4 | 3/4 | **0/4** | **0/4** | **0/4** |
| TF | A (Past) | 4 | 1/4 | 0/4 | 0/4 | 0/4 |
| TF | B (Present) | 4 | 1/4 | 3/4 | 3/4 | **4/4** |
| TF | C (Future) | 2 | 1/2 | 0/2 | 0/2 | 0/2 |
| CMTB | A | 3 | 1/3 | 2/3 | 2/3 | 2/3 |
| CMTB | B | 3 | 1/3 | 0/3 | 0/3 | 0/3 |
| CMTB | C | 4 | 1/4 | 3/4 | 2/4 | 2/4 |

Với CQ, mô hình chọn **A (Yes) ở 8/8 câu TA** và **B (Present) ở 9–10/10 câu TF**. Điểm đã cộng qua mọi cách xoay phương án, nên đây không phải thiên lệch vị trí chữ cái.
Cùng hiện tượng ở J12 cũ (TA đáp `AAA`, TF `BBB`) và J12b (TF `BBBCBBBBBB`).

**Giả thuyết về nguyên nhân (CHƯA tách):**
- (a) Khối prompt là câu **thì hiện tại** ("…is…") đứng ngay trước câu hỏi, mà ta đã biết thì của câu điều khiển TF (nhân quả, mục 4.8) → thử prompt thì quá khứ.
- (b) Văn bản dài chen giữa video và câu hỏi khiến mô hình trả lời từ văn bản → thử prompt có mặt nhưng giữ nguyên chữ mà **che hết media**.
- (c) Câu hỏi đọc được prompt → thử che cho **câu hỏi không nhìn thấy prompt**; nếu kết quả về lại base thì lệch đến từ việc đọc prompt.

**Relevance chưa chỉ đúng chunk sự kiện (J13):** xem 4.20. Prompt "one word" đã có nội dung thật nhưng câu hỏi vẫn không tìm được prompt hình đúng (0/576 head). Lý do có khả năng cao: **hai bước** (phát hiện 4 ở bảng trên).

### 0A.4 Các ứng viên khác (cùng đang được cân nhắc)

| Ứng viên | Tình trạng | Ghi chú |
|---|---|---|
| **Mô-đun A** (định tuyến attention theo mốc, 1 lượt) | CMTB 47 → 79 (P2), 53 → 94 (J8); gây hại ở Daily-Omni (P4) | Cần công tắc tự động; chỉ giải "lúc đó thấy gì" |
| **Soi cô lập từng vùng + đọc theo quy tắc** (P5) | 44 → 77 (n=70) | Cơ chế đúng; bước đọc ra nặn theo đề, không tổng quát |
| **Token dò có huấn luyện nhẹ** (T1, đề xuất của nhóm) | Giai đoạn 1 xong (mục 4.19) | Giai đoạn 2 (nhãn thứ tự tự sinh) chưa làm; xem 8.4 |
| **Pipeline kiểu con người J11 / STMR** | dev2: base 34,1 → CoT 73,2 (TF 81,2) | Nhiều bước, chậm; là baseline mạnh |
| SLA (ghi "đã nhìn lúc nào" vào residual) | **Đã bỏ** | Tiền đề sai (mốc đã nằm trong residual, đích thì không tìm được) |

### 0A.5 Việc nên làm tiếp (theo thứ tự ưu tiên)

1. **Tách nguyên nhân "dồn về một đáp án"** (3 điều kiện ở 0A.3): prompt thì quá khứ; prompt che media; câu hỏi bị che không đọc prompt.
   Báo kết quả **theo (task, đáp án đúng)**. Dùng ≥ 30 câu cân bằng đáp án, seed 777 hoặc 4242 (không dùng 2468 đã dùng để thiết kế).
2. **Đo relevance theo từng token câu hỏi** thay vì trung bình. Các câu SVHalluc là template cố định, nên chấm riêng từng bộ ba (từ, layer, head). Sửa `j13_relevance_heads.py` để ghi từng token.
3. **Phép đo hai bước, không huấn luyện:** dùng các head tìm tiếng nói (J13) để lấy chunk có tiếng nói; sinh vài từ từ prompt audio; so với prompt visual
   của từng chunk (embedding hoặc attention); kiểm tra chunk khớp nhất có rơi vào nửa video có sự kiện nhiều hơn 50% không. Đo offline trên 35 câu của J13d.
4. Nếu 2–3 không cho tín hiệu: **huấn luyện nhẹ** prompt (T1 giai đoạn 2, 8.4).
5. Khi đã có phương pháp: kiểm tra **diễn đạt lại câu hỏi**, rồi **seed held-out `1234` một lần**, rồi benchmark khác (Daily-Omni, TempCompass, NExT-QA, TVBench, Charades-STA) và mô hình khác (Qwen2.5-Omni-7B…).
6. (Tuỳ chọn) chạy hết phần còn dang dở của T2 (39 câu) và P4 (20 câu) nếu cần số cuối.

### 0A.6 Quy tắc làm việc (để khỏi làm lại cái sai)

- **Luôn báo theo (task, đáp án đúng)** hoặc độ chính xác cân bằng. Mô hình dồn về một đáp án rất dễ, và một tỉ lệ tổng gộp che chuyện đó.
- **Đừng kết luận từ dưới ~30 câu.** Các bảng ở mục 0A.3 chỉ có 10–28 câu: khác nhau 1–2 câu là nhiễu.
- Chọn head, siêu tham số, ngưỡng trên **một seed**, đo trên **seed khác**. Seed `1234` là held-out, chỉ dùng một lần.
- Phương pháp **không được dựa vào tên task** hay vào dạng câu hỏi của benchmark; phải làm được cho câu hỏi mở và cho mọi video LM.
- **Bảo mật:** không commit `remote_config*.json` (URL + token server), token Kaggle hay HuggingFace. `.gitignore` đã chặn.

### 0A.7 Cách chạy lại các thí nghiệm gần nhất

```bash
# 1. Build job cho server (từ thư mục gốc của repo). Chia 2 nửa cho 2 server:
python server_jobs/build_server_job.py j12 /tmp/j12_s0.py --shard 0/2 --prompts oneword --out-name j12d_oneword.jsonl
python server_jobs/build_server_job.py j12 /tmp/j12_s1.py --shard 1/2 --prompts oneword --out-name j12d_oneword.jsonl
python server_jobs/build_server_job.py j13 /tmp/j13.py --decode --out-name j13d_oneword.jsonl     # relevance theo head + token dự đoán

# 2. Gửi lên server (xem 3.3 để dựng server). Sau mỗi (re)start server phải chạy fix_ffmpeg trước:
REMOTE_CFG=remote_config5.json python remote_client.py run remote_tests/fix_ffmpeg.py
REMOTE_CFG=remote_config5.json python remote_client.py run /tmp/j12_s0.py --name j12d0 --nowait
REMOTE_CFG=remote_config5.json python remote_client.py status <job_id>
REMOTE_CFG=remote_config5.json python remote_client.py get j12d_oneword.jsonl results/j12d_oneword_s0.jsonl

# 3. Phân tích relevance theo head (file lớn đã nén .gz):
gunzip -k results/j13d_oneword.jsonl.gz
python remote_tests/analyze_j13.py results/j13d_oneword.jsonl
```
Một vài lưu ý: mỗi câu mất khoảng 1–2 phút (clip ≤ 42 s; clip dài hơn bị bỏ, nên 16 mẫu chỉ còn khoảng 11); phiên Kaggle hết sau ~12 h, hãy tải kết quả về thường xuyên.
Ba server (4, 5, 6) đang **rảnh**; các kết quả dang dở của T2 và P4 đã lưu ở `results/`.

---

## 1. Bài toán và thiết lập

### 1.1 SVHalluc temporal
- Clip lấy từ YouCook2. Lời nói nằm ở một nửa clip; **nửa kia là im lặng tuyệt đối** (−120 dB). Đây là một đặc điểm nhân tạo của benchmark, cần lưu ý.
- **TA:** "When the speech is heard, is the narrated event happening at the same time in the video? (A) Yes (B) No"
- **TF:** "Relative to the moment when the speech is heard, when does the narrated event occur in the video? (A) Past (B) Present (C) Future"
- **CMTB:** "When the speech is heard, what visual action is occurring in the video?", 3 lựa chọn là các bước nấu ăn.
- Độ dài clip: trung vị 27 s (p10 15 s, p90 46 s), 17% clip dài hơn 42 s. Đoạn lời nói: trung vị 12 s (7–18 s), chiếm khoảng một nửa clip.
- Chỉ riêng luật vị trí lời nói đã cho TF khoảng 67%. Với CMTB, cho video đen + audio gốc được 65,9%. Benchmark có lối tắt.

### 1.2 Mô hình và cách token hoá
- Qwen2.5-Omni dùng **TMRoPE**: 25 id thời gian mỗi giây. Video và audio được chia thành **chunk 2 s**, mỗi chunk xếp `[video ≈288 token, audio ≈50 token]` (khoảng 144 token hình + 25 token tiếng mỗi giây), dưới mask nhân quả.
- Trên Kaggle (T4×2): chạy FP16, `device_map='balanced'`, `max_memory` 10 GiB/GPU. Clip **> 42 s bị bỏ qua** (OOM). Các job dùng attention eager giới hạn ở 34 s.
- Trên máy local (RTX 3050 4 GB): chạy NF4, chỉ được clip ngắn (khoảng ≤ 14–24 s).

### 1.3 Cách chấm (protocol của repo)
- **Xoay thứ tự lựa chọn** (mỗi cách xoay vòng một lần), lấy log-prob của chữ cái ở token đầu tiên, rồi **cộng theo nội dung lựa chọn**. Cách này triệt tiêu thiên lệch chữ cái.
- Protocol này **khác** với cách của bài báo (sinh chữ rồi parse), nên số base có thể lệch so với bảng công bố. Job `j4_official_protocol.py` để tái lập protocol gốc, **chưa chạy**. Chỉ chạy khi đã chốt phương pháp.
- Lấy mẫu cân bằng theo (task, nhãn):
  - `seed 909` = bộ **pilot**, dùng để thiết kế;
  - `seed 1234` = **held-out**, chỉ dùng để xác nhận;
  - `seed 555` = bộ **dev2**;
  - các seed khác dùng cho chẩn đoán.

### 1.4 Mốc so sánh đã công bố (SVHalluc, Bảng 3)

| Mô hình | TA | TF | CMTB |
|---|---|---|---|
| Gemini 2.5 Pro | 85,17 | 53,89 | 69,25 |
| Qwen3-Omni | 51,11 | 30,60 | 61,75 |
| **Qwen2.5-Omni** | **50,27** | **31,52** | **53,10** |
| video-SALMONN 2 | 50,00 | 33,33 | 48,58 |
| RED (3B), công trình duy nhất có số trên SVHalluc | 50,0 | 29,0 | 53,4 |
| Phương pháp giải mã tốt nhất | 53,3 | 35,5 | 53,4 |

---

## 2. Cấu trúc repo

```
temporal_videoLM/
├── README.md                     ← file này
├── docs/
│   ├── research_log.md           ← nhật ký nghiên cứu theo thời gian (rất chi tiết, nguyên văn)
│   ├── 14_root_causes_and_general_method.md   ← bảng 16 nguyên nhân giả thuyết + kết quả giai đoạn đầu
│   ├── 13_human_inspired_av_event_attention.md
│   └── temporal_related_work_2026-10-11.md    ← tổng quan công trình liên quan (temporal, RoPE, omni)
│
├── kaggle_rootcause.py           ← THƯ VIỆN LÕI: tải model, cắt clip, chấm điểm xoay, mask, hear, whisper, patching…
├── local_qwen/speech_gate.py     ← tạo attention mask 4-D từ TMRoPE (media_layout, gated_mask, binding_mask…)
├── research_topics_learning/single_sample_prompt_lab/lab_*.py
│                                 ← môi trường Kaggle + đọc video đúng chuẩn Qwen (prepare_video, qwen_inputs)
├── build_svhalluc_single_sample_prompt_lab.py  ← cell cài đặt môi trường dùng chung cho mọi notebook
│
├── repo/kaggle/                  ← ĐỢT THÍ NGHIỆM CHÍNH (notebook Kaggle J1–J11)
│   ├── build_notebooks.py        ← sinh toàn bộ nb*.ipynb từ jobs/
│   ├── README.md                 ← thứ tự chạy, thời gian
│   ├── jobs/
│   │   ├── nb_runner.py          ← tải model 1 lần, chạy job_utils + 1 job trong namespace của kaggle_rootcause
│   │   ├── job_utils.py          ← lấy mẫu cân bằng, chấm điểm (≤4 lựa chọn), tạo clip biến thể, vòng lặp có resume + hạn giờ
│   │   ├── j1_event_perception.py … j11_full.py   ← từng thí nghiệm (mô tả trong docstring đầu file)
│   │   ├── j4_official_protocol.py ← tái lập protocol của bài báo (CHƯA chạy, để dành)
│   │   ├── inpass.py             ← mô-đun A: tự định tuyến trong 1 lượt
│   │   ├── p1_two_pointer.py, p2_inpass_eval.py, p4_dailyomni_onepass.py, p5_isolate_verify.py, p6_focused_search.py
│   │   │                         ← loạt P (chỉ chạy trên server, chưa có notebook): hai con trỏ, định tuyến 1 lượt, Daily-Omni, soi cô lập, tìm kiếm tập trung
│   │   ├── t1_probe_distill.py   ← T1: chưng cất cho m embedding dò (chỉ học embedding, mô hình đóng băng)
│   │   ├── t2_order_blindness.py ← T2: mù thứ tự trên Daily-Omni (đảo ngược / đổi chỗ hai nửa video)
│   │   ├── j12_chunk_prompts.py  ← J12: chunk prompts + attention tăng cường theo độ liên quan (phương pháp đang xây dựng)
│   │   └── j13_relevance_heads.py ← J13: đo relevance theo từng head / layer, kèm token dự đoán tại cuối mỗi prompt
│   └── nb*.ipynb                 ← notebook đã build sẵn
│
├── server_jobs/build_server_job.py ← ghép job + helper thành MỘT file để chạy trên server từ xa; có --shard để chia việc cho nhiều server
│
├── kaggle_gpu_server.py, build_kaggle_gpu_server_notebook.py, kaggle_svhalluc_gpu_server.ipynb
│                                 ← server GPU trên Kaggle (HTTP + ngrok) để chạy job từ xa; .ipynb đã build sẵn, hướng dẫn ở mục 3.3A
├── remote_client.py              ← client: ping / info / run FILE / status / get / upload / restart
├── remote_config.example.json    ← mẫu cấu hình (file thật remote_config*.json KHÔNG commit)
├── remote_tests/                 ← các job đã chạy trên server (tense, timing, anchor_text, daily_anchor, STMR, debug…)
│
├── kaggle_anchor_experiment.py, kaggle_temporal_study.py, kaggle_temporal_lab.py, kaggle_temporal_stage2.py,
│   kaggle_binding_lab.py + *_report.py + build_kaggle_*_notebook.py
│                                 ← các giai đoạn ĐẦU (anchor, study, lab, stage2, binding, rootcause)
├── analyze_tense_intervention.py
│
├── local_qwen/*.py               ← các probe chạy local (NF4): temporal_anchor_probe, patching_debug, recency_check…
│
└── results/                      ← kết quả thô (jsonl/json; npz nhỏ). Bản đồ chi tiết ở mục 3.6
```

---

## 3. Cách chạy lại

### 3.1 Yêu cầu
- **Kaggle:** GPU **T4 ×2**, **Internet = On**. Notebook tự tạo venv và cài `transformers==4.57.6`, `accelerate==1.12.0`, `qwen-omni-utils==0.0.8`, `decord`, `av`, `librosa`…
- **Local:** Python 3.12, torch 2.6 (cu124), transformers 4.57.6, bitsandbytes (NF4), ffmpeg. Biến môi trường mà `kaggle_rootcause.py` đọc:
  - `LAB_NF4=1`: tải model dạng 4-bit;
  - `LAB_CACHE=<thư mục HF cache>`;
  - `LAB_LOCAL_DATA=<thư mục chứa samples.json + temporal/...>`: không tải dataset từ HF;
  - `LAB_MAX_DUR=22`: bỏ clip dài hơn;
  - `LAB_NO_WHISPER=1`;
  - `FFMPEG_BIN`.
- Các script trong `local_qwen/` có **đường dẫn Windows tuyệt đối** của máy gốc (`F:\A_star_paper_NLP_ACL\...`). Hãy sửa lại cho máy của bạn.

### 3.2 Chạy một thí nghiệm trên Kaggle (cách chính)
1. Build notebook: `cd repo/kaggle && python build_notebooks.py`. Lệnh này sinh `nb1…nb13*.ipynb`, mỗi notebook nhúng sẵn mã nguồn.
2. Tải notebook lên Kaggle, bật GPU T4×2 và Internet, rồi **Save Version → Save & Run All**.
3. Bên trong notebook:
   - **cell 0** kiểm tra Internet;
   - **cell 1** cài môi trường (khoảng 10 phút);
   - **cell 2** ghi mã nguồn;
   - **cell 3** chạy thử 4 mẫu: dừng ngay nếu có lỗi thật, còn clip quá dài hoặc "no room" không tính là lỗi;
   - **cell 4** chạy đầy đủ, tự dừng trước hạn 12 giờ và giữ kết quả đã chạy;
   - **cell 6** nén kết quả thành `<tên>.zip`.
4. Tải kết quả về. **Chỉ tải file zip**, nếu không sẽ kéo về cả venv nặng vài GB:
   `kaggle kernels output <user>/<slug> -p <dir> --file-pattern "<tên>.zip"`

| Notebook | Job | Câu hỏi | Thời gian |
|---|---|---|---|
| nb1 | j1_event_perception | Mô hình không *thấy*, hay thấy mà không *gắn* được với thời điểm? | 4–5 h |
| nb2 | j2_position_causal | Dùng thời gian hay chỉ dùng vị trí / khoảng im lặng? (đổi chỗ nửa clip, đảo TMRoPE, thêm nhiễu) | ~4 h (**chưa chạy**) |
| nb3 | j3_prior_lens | Prior "No" / "không bao giờ Past" sinh ra ở lớp nào? (logit lens) | 3–4 h (**chưa chạy**) |
| nb4 | j4_postdiction | Mask nhân quả có chặn việc "hiểu lại" cảnh đã thấy không? | ~6 h |
| nb5 | j5_internal_localization | Mô hình có tự biết lời nói ở đâu không (attention theo head, có nhiễu)? | ~4 h |
| nb6/nb7 | j7_chunk_queries | Câu tóm tắt cô lập cho từng đoạn 2 s (pilot / held-out) | 3–6 h |
| nb8/nb9 | j8_acar | Tự tìm mốc bằng attention + phân bổ lại attention (pilot / held-out) | 3–6 h |
| nb10 | j9_nonspeech_anchor | Mốc với âm thanh không phải lời nói | 1–1,5 h |
| nb11/nb12 | j10_auto | J8 + mô hình tự quyết kênh (nhìn / nghe / cả hai) | 3–6 h |
| nb13 | j11_full | Pipeline đầy đủ lấy cảm hứng từ cách con người hiểu video | 3,5–4 h |

### 3.3 Chạy trên server GPU từ xa (nhanh để lặp)
1. **Dựng server trên Kaggle** (lấy ngrok authtoken → thêm Secret → Save & Run All → lấy URL và TOKEN): xem hướng dẫn từng bước ở **mục 3.3A** ngay dưới. Notebook đã build sẵn: `kaggle_svhalluc_gpu_server.ipynb`.
2. Ghi URL và TOKEN vào `remote_config.json` (xem file mẫu `remote_config.example.json`). Có nhiều server thì dùng `remote_config2.json`… và chọn bằng `REMOTE_CFG=remote_config2.json`.
3. Các lệnh:
   ```bash
   python remote_client.py ping | info | restart
   python remote_client.py run remote_tests/fix_ffmpeg.py          # luôn chạy sau mỗi lần (re)start server
   python remote_client.py run my_job.py --nowait                  # job = đoạn Python chạy trong namespace của kaggle_rootcause
   python remote_client.py status <job_id>
   python remote_client.py get <file trong OUT> <đích local>
   python remote_client.py upload <local> <rel>                    # -> OUT/uploads/<rel>
   ```
4. Muốn chạy một job trong `repo/kaggle/jobs` trên server, ghép file theo thứ tự: `time_left = lambda: 10**9` + **khai báo tường minh `PER_LABEL`, `SEED`** + `job_utils.py` + (nếu cần) phần helper của `j8_acar.py` + file job. Xem mục 9 để biết vì sao.
   **Đã có sẵn công cụ ghép:** `python server_jobs/build_server_job.py <j12|j13|plain> OUT.py [...]` (docstring đầu file giải thích tham số: `--shard K/N`, `--prompts oneword|is`, `--per-label`, `--seed`, `--tasks`, `--out-name`; với job không cần helper j8: `plain --job-file <file>`). Ví dụ đầy đủ ở mục 0A.7.
5. **Chia việc cho nhiều server:** build mỗi nửa với `--shard 0/2` và `--shard 1/2`, gửi lên hai server, rồi ghép hai file kết quả theo `idx`. Job có resume theo `idx`, nên chạy lại cùng file không làm lại câu đã xong.
6. **Dừng một job trên server:** dùng `python remote_client.py restart` (kết thúc luôn job đang chạy). Hãy tải kết quả dang dở về trước. Sau restart phải chạy lại `remote_tests/fix_ffmpeg.py`.

### 3.3A Hướng dẫn từng bước: dựng server GPU trên Kaggle và kết nối từ máy của bạn

Server là một notebook Kaggle giữ sẵn mô hình trên GPU và mở cổng ra ngoài qua **ngrok**; máy của bạn gửi mã Python lên bằng `remote_client.py`. Notebook đã build sẵn ở **`kaggle_svhalluc_gpu_server.ipynb`** (có thể build lại bằng `python build_kaggle_gpu_server_notebook.py`).

**Bước 1. Lấy ngrok authtoken**
1. Đăng ký tài khoản miễn phí tại https://dashboard.ngrok.com/signup (có thể đăng nhập bằng Google hoặc GitHub).
2. Sau khi đăng nhập, vào https://dashboard.ngrok.com/get-started/your-authtoken (menu bên trái: **Getting Started → Your Authtoken**).
3. Bấm **Copy** để lấy chuỗi authtoken. Đây là **Authtoken**, không phải *API key*. Giữ kín như mật khẩu.
4. Muốn chạy nhiều server cùng lúc thì mỗi server nên dùng một tài khoản ngrok riêng (gói miễn phí thường chỉ cho một phiên ngrok mỗi tài khoản; kiểm tra lại giới hạn hiện hành trên trang ngrok).

**Bước 2. Tạo notebook trên Kaggle** (nên dùng **tài khoản Kaggle phụ**, xem lưu ý chính sách ở cuối)
1. Tài khoản Kaggle cần đã xác minh số điện thoại (nếu không sẽ không bật được GPU và Internet).
2. **Create → New Notebook → File → Import Notebook** (hoặc *Upload*), chọn `kaggle_svhalluc_gpu_server.ipynb`.
3. **Thêm Secret:** trong notebook, menu **Add-ons → Secrets → Add Secret**
   - *Label*: `NGROK_AUTHTOKEN` (đúng chính tả, viết hoa);
   - *Value*: dán authtoken vừa copy;
   - bật công tắc **Attach** (đính kèm) cho notebook này. Nếu quên bước này, notebook sẽ báo `Không đọc được Kaggle Secret NGROK_AUTHTOKEN`.
4. **Cấu hình phiên** (panel *Session options / Settings* bên phải):
   - *Accelerator*: **GPU T4 ×2**;
   - *Internet*: **On**;
   - *Persistence*: không cần.

**Bước 3. Chạy Save & Run All**
1. Bấm **Save Version** (góc trên bên phải) → chọn **Save & Run All (Commit)** → **Save**. Có thể đóng tab; notebook chạy nền tới 12 giờ.
2. Theo dõi tiến độ: mở **Versions** (hoặc **View Active Events**) → chọn bản đang chạy → xem **Logs**. Giao diện Kaggle thay đổi theo thời gian, hãy tìm mục có tên tương tự.
3. Các cell chạy theo thứ tự: kiểm tra Internet (OK `pypi.org`, `huggingface.co`, `ngrok.com`) → cài môi trường (khoảng 10 phút lần đầu) → ghi file → khởi động server và nạp mô hình (2–4 phút) → mở ngrok.
4. Khi sẵn sàng, log in ra khung sau:
   ```
   ======================================================================
     URL   : NgrokTunnel: "https://xxxx-xxxx.ngrok-free.dev" -> "http://localhost:8765"
     TOKEN : <chuỗi ngẫu nhiên dài>
   ======================================================================
   ```
   **URL** là địa chỉ `https://….ngrok-free.dev` nằm trong dấu ngoặc kép; **TOKEN** là chuỗi ngẫu nhiên bên dưới. Cả hai **thay đổi mỗi lần chạy**.

**Bước 4. Kết nối từ máy của bạn**
1. Ở thư mục gốc của repo, tạo file `remote_config.json` (đã bị `.gitignore` chặn, không bao giờ commit):
   ```json
   {"url": "https://xxxx-xxxx.ngrok-free.dev", "token": "<TOKEN trong log>"}
   ```
   Nhiều server: `remote_config2.json`, `remote_config3.json`… rồi chọn bằng `REMOTE_CFG=remote_config2.json python remote_client.py ...`.
2. Kiểm tra: `python remote_client.py ping` phải in `{"ok": true}`; `python remote_client.py info` cho biết GPU và mô hình.
3. Chạy `python remote_client.py run remote_tests/fix_ffmpeg.py` **một lần sau mỗi lần (re)start server**, rồi gửi job như mục 3.3 / 0A.7.

**Xử lý lỗi thường gặp**
| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `Không kết nối được … Bật Internet` ở cell 0 | Session options → Internet = **On** |
| `Không đọc được Kaggle Secret NGROK_AUTHTOKEN` | Chưa thêm Secret, sai tên, hoặc chưa bật **Attach** cho notebook này |
| ngrok báo authtoken không hợp lệ / đang có phiên khác | Copy sai chuỗi (dùng *Authtoken*, không phải API key), hoặc tài khoản ngrok đó đang chạy một server khác; dùng tài khoản ngrok khác |
| `server không lên được` | Xem log in ra (đuôi `server.log`); thường là hết bộ nhớ GPU hoặc thiếu GPU thứ hai |
| `{"error": "bad token"}` hoặc HTTP 401 | TOKEN trong `remote_config.json` không khớp lần chạy hiện tại; lấy TOKEN mới từ log |
| Không kết nối được tới URL | Phiên đã hết giờ (tối đa 12 giờ) hoặc Kaggle đã ngắt vì rảnh lâu; chạy lại Bước 3 và cập nhật URL + TOKEN |
| Job treo / cần giải phóng GPU | `python remote_client.py restart` (notebook tự khởi động lại server, sau đó nạp lại mô hình), rồi chạy lại `fix_ffmpeg.py` |

**Bảo mật và chính sách**
- **Ai có cả URL lẫn TOKEN đều chạy được mã tuỳ ý trong notebook của bạn.** Không đăng lên GitHub, không gửi vào kênh công khai, không chia sẻ log. Đặt `ALLOW_EXEC = False` ở cell cuối nếu chỉ muốn thử kết nối.
- Server chỉ nghe `127.0.0.1`, chỉ ngrok chuyển tới; token được so sánh theo thời gian hằng số; tải file chỉ trong thư mục kết quả.
- Chưa xác minh được việc mở tunnel từ notebook Kaggle có nằm trong điều khoản sử dụng hay không, và Kaggle có thể tự ngắt notebook rảnh lâu. Notebook **cố ý không có vòng lặp chống ngắt**. Dùng tài khoản phụ, và tải kết quả về thường xuyên.
- Quota GPU của Kaggle theo tuần; mỗi phiên Save & Run All dùng tối đa 12 giờ.

### 3.4 Chạy thử / debug trên máy yếu (1 mẫu)
- Dùng `repo/kaggle/jobs/nb_runner.py` với các biến `LAB_*` ở trên, ví dụ:
  ```bash
  LAB_NF4=1 LAB_LOCAL_DATA=... LAB_MAX_DUR=14 python nb_runner.py <ROOT chứa các .py> <OUT> <ROOT>/j11_full.py '{"LIMIT":7,"HOURS":1,"J8_MAX_DUR":14}'
  ```
- Các clip ngắn (≤ 14 s) dùng tốt để debug: CMTB `1713, 1772, 1474, 1527, 1595`; TA `1472, 1485, 1512, 1525`; TF `1513`.
- Các script debug một mẫu nằm ở `remote_tests/debug_*.py`.

### 3.5 Viết thí nghiệm mới
Một job chỉ cần định nghĩa `item(i, r) -> dict` rồi gọi:
```python
items = sample_items(('TA','TF','CMTB'), PER_LABEL, seed=SEED)
R = run_loop(items, 'ten_file.jsonl', item, show=('base', 'cond_moi'))
```
Các hàm sẵn có:
- `job_utils.py`: `prep_clip`, `score(media, stem, texts, mode, fwd=...)` (fwd tuỳ biến forward, ví dụ để thêm mask), `yes_minus_no`, `forward_record`, `token_groups`, `halves`, `concat_clips`, `with_audio`, `noisy_audio`, `acc`.
- Phần helper của `j8_acar.py` (cách import xem đầu `j10_auto.py`): `Lay`, `forward(inp, lay, special_mask | {layer: mask}, layers, record)`, `profiles` (độ liên quan theo chunk), `pmi`, `bias_mask`, `window_mask`, `top_window`.
- Sau đó thêm một mục vào `NOTEBOOKS` trong `build_notebooks.py` (`extra=[...]` để nhúng file phụ thuộc).

### 3.6 Bản đồ `results/`
| Thư mục / file | Thí nghiệm |
|---|---|
| `anchor_kaggle_results/` | Giai đoạn 1: anchor / crop (CMTB n=55) |
| `study/`, `lab_shard0/` (có `trace/`), `stage2/`, `binding/` | Các giai đoạn đầu (mục 4.2–4.4) |
| `rootcause/s0,s1/` (+ `hidden/*.npz`) | Shard nguyên nhân gốc (n=130) |
| `synthetic_fp16.json`, `synth16/`, `tf_framing.jsonl`, `method_dev.jsonl`, `tense_intervention.jsonl`, `timing_diag.jsonl`, `anchor_text.jsonl`, `scale7b*.jsonl` | Các job server (mục 4.5–4.7) |
| `daily_anchor*.jsonl`, `dailyomni_base_partial.jsonl` | Daily-Omni |
| `stmr_*` | Pipeline văn bản STMR |
| `local_text_tests/` | Trần suy luận bằng văn bản, TF bridge (local) |
| `j1/`, `j4/`, `j5/` (`attn_j5/*.npz`), `j7p/`, `j7h/`, `j8p/`, `j8h/`, `j9/` | Notebook Kaggle J1–J9 |
| `j11_dev2_v1.jsonl`, `j11v2_dev2_partial.jsonl`, `j11v2_dev2.jsonl`, `j10p/`, `j11p/` | J10 / J11 (pilot Kaggle, dev2 trên server) |
| `p1_two_pointer.jsonl`, `p2_inpass.jsonl`, `p5_isolate_verify.jsonl`, `p6_focused_search.jsonl`, `p6v2_focused_search.jsonl` | Loạt P (mục 4.17) |
| `p4_dailyomni_partial.jsonl` | P4: mô-đun A trên Daily-Omni, 220/240 câu (mục 4.17) |
| `t1_probe_distill.json`, `t1_probes.pt` | T1: chưng cất embedding dò (mục 4.19) |
| `t2_order_blindness_partial.jsonl` | T2: mù thứ tự, 81/120 câu (mục 4.18) |
| `j12_chunk_prompts.jsonl`, `j12b_tf_aligned.jsonl`, `j12d_oneword_s0/s1.jsonl` | J12 chunk prompts (mục 4.15–4.16). `s0`/`s1` là hai nửa chạy trên hai server, ghép theo `idx` |
| `j13_relevance_heads.jsonl.gz`, `j13b_relevance_full.jsonl.gz`, `j13d_oneword.jsonl.gz` | J13 relevance theo head (mục 4.20). Giải nén bằng `gunzip -k`. Các khoá: `last` / `mean` = [layer, head, prompt] |

Hidden state lớn của J1/J4 (`hidden_j1`, `hidden_j4`, khoảng 130 MB) không được đưa lên. Chạy lại nb1/nb4 sẽ có lại.

---

## 4. Nhật ký thí nghiệm và kết quả (theo giai đoạn)

Ghi chú chung: n là số câu hợp lệ (clip ≤ 42 s). CI là khoảng tin cậy bootstrap 95%. Chi tiết từng ngày xem `docs/research_log.md`.

### 4.1 Giai đoạn 1: mốc (anchor) và cắt clip, CMTB n=55
| Điều kiện | CMTB |
|---|---|
| base | 52,7 |
| chỉ hình (cả clip) | 41,8 |
| **cắt đúng đoạn lời nói** (`speech_win`) | **96,4** |
| cắt đoạn im lặng (`silent_win`) | 5,5 |

24/26 lỗi của base trùng đúng đáp án của đoạn im lặng. Kết luận: **thông tin có đủ, chỉ là chọn sai thời điểm.** Mô hình tự định vị lời nói kém: IoU trung vị 0,44; hỏi "nửa đầu hay nửa sau" chỉ đúng 57,5% (n=40).

### 4.2 Study / Lab: chặn attention theo thời gian (gate)
- **Gate** (câu hỏi chỉ được đọc video trong cửa sổ lời nói):
  - TA 53,5 → **76,7**, CI [+9,3; +37,2]; recall của Yes 10,5 → 63,2;
  - CMTB 54,8 → 69,4;
  - TF không đổi.
- Hỏi theo từng cửa sổ rồi so (look-and-compare) cho TF 88,1, so với luật vị trí 65,7.
- Khi thêm nhiễu, lợi ích của gate giảm. Một chỗ chưa giải thích: gate trên CMTB có nhiễu ở study giống base, còn ở stage 2 thì +32.

### 4.3 Binding lab (58 câu): **âm tính**
Mask gắn kết cục bộ (audio ↔ video cùng chunk / ±1 chunk, hai chiều) gần như không đổi dự đoán. Giả thuyết "gắn kết bên trong vùng media" bị bác bỏ.

### 4.4 Stage 2 (73 câu, tiêu chí định trước)
- **C1 ĐẠT:** CMTB có nhiễu: base 39,3 / gate 71,4 / gate + chặn audio 89,3 (bằng cắt clip).
- **C3 ĐẠT:** TF có nhiễu, nghe trước rồi nhìn (htl) 83,3 so với luật 63,3.
- **C2 KHÔNG ĐẠT:** TA htl 53,3; chỉ riêng gate mới giúp TA.

### 4.5 Shard nguyên nhân gốc (130 câu)
- **Nghe đúng 96%.** Diễn đạt câu hỏi khác đi không đổi kết quả ở TA/CMTB; ở TF các cách diễn đạt **không bao giờ** thống nhất.
- **TA:** đáp án mặc định khi không có bằng chứng luôn là "No"; 85% lỗi TA trùng với đáp án đó. Chỉ hình (60%) ≈ base (62,9%).
- **CMTB gộp (n=72):**

  | base | gate | chặn audio | **gate + chặn audio** | cắt clip |
  |---|---|---|---|---|
  | 51,4 | 80,6 | 52,8 | **95,8** | 97,2 |

  Phân rã lỗi: 63% chỉ do thời điểm, 20% do cả thời điểm lẫn nguồn, 14% chỉ do audio, 3% do tri giác.
- **Attention dồn về đầu video:** tỉ lệ nửa sau / nửa đầu là 0,44 ở lớp 24–35, và không ưu tiên cửa sổ lời nói.
- **Activation patching:** lỗi đi vào ở **lớp 15–24** (5/10 câu đúng lớp 21).
- **Probe TF** đạt 71% ≈ luật vị trí (68,6%). Kiểm tra lại có kiểm soát vị trí cho thấy probe chủ yếu đọc vị trí lời nói.

### 4.6 Thí nghiệm văn bản và cách diễn đạt TF
- **Câu hỏi văn bản tổng hợp, FP16, n=30 mỗi thì:**
  - với nhãn Past/Present/Future, recall của **Future = 0%** ở mọi thì;
  - với nhãn before/during/after: 93–97%, nhưng chỉ 56,7% khi lời nói ở thì quá khứ (**lối tắt theo thì**);
  - câu Yes/No "cùng lúc": 97–100%.
- **Video thật, n=68:** nhãn Past/Present/Future 29,4 (recall Past = 0) so với before/during/after 42,6 (+13,2, CI [+1,5; +25,0]); luật 69,1.

### 4.7 Định tuyến oracle theo lớp (method_dev, 52 câu)
- CMTB, định tuyến lớp 13–27 (thời gian + nguồn): **100%**. Định tuyến mọi lớp: 94,4%. Chỉ thời gian: 55,6%.
- TA, nghe-rồi-nhìn ở lớp giữa: **92,3** so với base 46,2.
- TF: **81,0** so với luật 71,4.
- **Lưu ý:** cách chọn định tuyến ở đây dựa trên tên task, tức **oracle**, không phải phương pháp.

### 4.8 Can thiệp nhân quả (server)
- **Đổi thì của lời nói bằng TTS** (video giữ nguyên, n=18 TF), hỏi theo nhãn Past/Present/Future:
  - lời nói thì tương lai → P(Future) = 0,80; thì quá khứ → P(Past) = 0,55;
  - chênh P(Future) **+0,74**, CI [+0,56; +0,88]; 14/18 câu đổi đáp án;
  - hỏi theo before/during/after thì hiệu ứng biến mất. TA hầu như không đổi.
- **Thời gian (n=48):** dời lời nói sang nửa kia chỉ đổi đáp án 21% (TA) / 14% (TF); đảo video 11% / 7%; đúng theo đáp án kỳ vọng sau can thiệp 38–58% (mức may rủi). **Mô hình gần như không dùng thứ tự tương đối.**
- **Báo thời gian bằng chữ (n=65):** CMTB 50 → 50; TA 66,7 → 66,7. Định tuyến với cùng cửa sổ: 92,3.
- **Bản 7B (n=69):** TA 57 / TF 28 / CMTB 48; gate + chặn audio và cắt clip **100** (n=23). Lỗi không phải do mô hình nhỏ.

### 4.9 Daily-Omni: video tự nhiên (73 câu có câu nói được trích + hỏi về hình)
Cửa sổ lấy bằng Whisper đánh dấu thời gian từng từ + so khớp gần đúng với câu được trích.

| base | báo bằng chữ | định tuyến thời gian | định tuyến + chặn audio | cắt clip (kiểu TFO) | cắt clip + audio |
|---|---|---|---|---|---|
| 47,9 | 52,1 | 54,8 | 56,2 | 57,5 | 60,3 |

Có lợi khoảng +8 nhưng CI chạm 0, và **không hơn TFO (2609.04242)**.

### 4.10 Trần của bước suy luận (văn bản, bằng chứng hoàn hảo dựng từ nhãn)

| Cách trình bày | TA | TF | CMTB |
|---|---|---|---|
| Không có dòng thời gian | 50 | 33 | 19 |
| Chỉ số giây | 50 | 33 | 36 |
| Đoạn gọi tên tương đối | **100** | 39–46 | 81 |
| Liệt kê gọn "SEEN" theo đoạn | **100** | 43 | **97** |
| Thêm từ đồng nghĩa quan hệ | 50 | 33 | 69–97 |
| Suy luận từng bước (CoT) | 60 | **73** | 80 |
| Neo nghĩa lựa chọn (mô hình tự ánh xạ) | 50 | 40 | 97 |
| Trừ prior (CAD) | 92 | 33 (luôn Future) | 72 |

### 4.11 STMR: pipeline văn bản (pilot 54 câu, bằng chứng thật)

| | TA | TF | CMTB | Chung |
|---|---|---|---|---|
| base | 54 | 24 | 50 | 41 |
| v1 | 92 | 43 | 40 | 54 |
| v2, một bước | **92** | 29 | **90** | 67 |
| v2, CoT (chấm theo chữ cái) | 54 | **86** | 90 | **80** |
| v2, CoT (đọc đáp án từ lời suy luận, bỏ phiếu) | | | 90 | 75,9 |

Phân tích lỗi: khâu nhìn chỉ sai **3/54** câu. Còn lại là lỗi suy luận:
- **(a)** ánh xạ nhãn TF sai;
- **(b)** nhầm nguồn khi suy luận từng bước ("the narrated event … is also happening in the video").

Held-out (v1, n=74): TA 55,6 → 83,3; TF 26,7 → 26,7; CMTB 57,7 → 61,5.

### 4.12 Notebook Kaggle J1–J9

**J1, tri giác so với gắn kết (n=218):**

| | base | Tự nghe + nhìn từng nửa (tắt tiếng) | Dùng Whisper nghe hộ |
|---|---|---|---|
| TA | 46,5 | **90,7** | 95,3 |
| TF | 29,5 | **89,4** | 92,4 |

- Chia theo độ tự tin của bước nhìn: nhóm thấp 70%, nhóm giữa 99%, nhóm cao 100%.
- Probe có kiểm soát vị trí, lớp 21: 74,5 / 83,3 so với mức đa số 65,3 / 54,2. Có mã hoá một phần.

**J4, postdiction (n=137):** cho video đọc mọi audio, kể cả audio đến sau, chỉ đổi 6,6% đáp án; ở các câu "sự kiện trước lời nói" độ chính xác 32,3 → 32,3. Mask hai chiều toàn phần đổi 24,8% đáp án nhưng không tăng. **Bác bỏ.**

**J5, định vị bên trong mô hình (n=77):**
- Khoảng 300 head có r > 0,5 với hoạt động lời nói, khoảng 100 head có r > 0,7, vẫn giữ khi có nhiễu.
- Cửa sổ từ 16 head (chọn trên nửa này, đánh giá trên nửa kia): IoU **0,88 / 0,83**.
- CMTB: gate + chặn audio 96,4; có nhiễu và tự phát hiện lại cửa sổ: **92,9**.
- TA 61 → 72; TF 45 → 29 (gate làm hại TF).

**J7, câu tóm tắt 2 s cô lập (ý của nhóm):**
- pilot: cq 38,9 so với base 40,7;
- held-out: cq 47,7 so với 46,6; không cô lập 39,8;
- "tự chọn đoạn" luôn chọn câu tóm tắt **cuối cùng** (vị trí 1,00); đoạn được chọn có lời nói 32–37%, dưới mức ngẫu nhiên 48–50%.

**J8, ACAR / QTAR:**

| | TA | TF | CMTB |
|---|---|---|---|
| Pilot (n=46): base | 54,5 | 22,2 | 52,9 |
| Pilot: oracle | 54,5 | 27,8 | 88,2 |
| Pilot: **mốc tự động + chặn audio** | 54,5 | 33,3 | **94,1** |
| Pilot: hình + tiếng có PMI, không chặn | 45–64 | 33 | 58,8 |
| Held-out (n=73): base | 50,0 | 29,6 | 73,1 |
| Held-out: oracle | 45,0 | 40,7 | 92,3 |
| Held-out: **mốc tự động + chặn audio** | 45,0 | 40,7 | **92,3** |
| Held-out: mốc PMI + chặn audio | 45,0 | **48,1** | **92,3** |

Phân tích lỗi: **mọi câu sai đều chọn đúng đoạn** (đoạn liên quan nhất nằm trong lời nói ở 97–100% số câu) **nhưng đọc sai**.

**J9, mốc với âm thanh không phải lời nói (n=43):**

| | Hỏi về âm thanh → trúng âm thanh | Hỏi về lời nói → trúng lời nói |
|---|---|---|
| Attention thô | 7% | 95% |
| PMI α=1 | 63% | 98% |
| **PMI α=2** | **86%** | **100%** |
| PMI α=4 | 91% | 100% |

- Theo loại âm thanh: huýt sáo 100, chuông 94, bíp 89, vỗ tay 60.
- Lớp 1–11 không phụ thuộc câu hỏi, nên **không dùng được** để tìm mốc.

### 4.13 Debug trên từng mẫu (GPU yếu)
- **Mốc theo cảnh:** vẽ một hình vuông đỏ ở khoảng thời gian đã biết; độ liên quan theo kênh hình đặt đỉnh đúng ở hình vuông (thô 0,39; PMI 0,53).
- **Bias dời 100% attention video về đúng đoạn, nhưng đáp án không đổi nếu còn audio:** mô hình đọc nội dung lời nói.
- **Đoán kênh đích bằng khối lượng attention: SAI** (video ×0,88 so với audio ×0,97).
- **Mô hình tự đọc câu hỏi và tự chọn kênh: ĐÚNG** với CMTB ("seen" 10/10). TA ra "both" 10/10. TF ra "heard" 10/10, **sai**.
- **QTAR + tắt kênh không cần ở lớp 13–27:** 0,922 so với oracle 0,856. Chỉ tắt ở lớp 21–27 thì không đủ (0,381).
- **Bước "look":**
  - đọc audio ở mọi lớp: sai 3/3;
  - nghe ở lớp 0–12, nhìn khi tắt tiếng ở lớp 13–27: đúng 2/3, nhưng độ chênh chỉ khoảng 0,5;
  - **nói thầm** (mô hình tự diễn đạt điều nghe được rồi dùng câu đó để nhìn): đúng, độ chênh **2–18**;
  - nói thầm + nhìn bằng định tuyến attention: 5/5 câu, ngang cắt clip.
- **Tự định tuyến trong 1 lượt:** độ liên quan tính ở lớp 12–20 tương quan 0,95–0,99 với bản 2 lượt. **Nhưng áp bias từ lớp 21 thì quá muộn** (0,239 so với 0,922). Phải đo sớm hơn và áp từ lớp 13.

### 4.14 J11, pipeline đầy đủ (đang chạy)
- **v1** có 3 lỗi: cửa sổ mốc quá hẹp; ghi "thấy" cho lựa chọn không hề được thấy; câu nói thầm lẫn rác.
- **v2 đã sửa:** cửa sổ = đỉnh PMI, mở rộng theo attention thô ≥ 0,3·max (IoU 0,50 → 0,83 trên 119 câu); chỉ ghi "thấy" khi điểm dương; chỉ giữ câu đầu của câu nói thầm.
- **Giữa chừng trên dev2 (n=24):** base 37,5 / acar 54,2 / một bước 54,2 / CoT 66,7. CMTB: CoT 70 thấp hơn acar 80.
- **Kết quả cuối trên dev2:** base 34,1 → CoT 73,2 (TF 81,2). File `results/j11v2_dev2.jsonl`.

### 4.15 J12: chunk prompts (đặc tả của nhóm), prompt "…is" (n=11) và biến thể đặt vị trí
Prompt: `The essential visual/audio content of chunk t in one short phrase is`. Hai prompt cô lập mỗi chunk 2 s, vị trí chung, boost theo `λ·log(r/max r)` với r = attention tới chữ "is" (mục 0A.3).
- **n=11** (16 mẫu, 4 mẫu > 34 s bị bỏ; seed 2468): base 36,4 / cq 54,5 / cq_boost_p 63,6 / cq_boost_pr 63,6. TA (3): 33 → 67; TF (3): 33 phẳng; CMTB (5): 40 → 60 → 80.
  **Con số này đã bị làm mất giá trị** bởi 4.16: cq cho TA đáp `AAA` khi đáp án là `AAB`, TF đáp `BBB` khi đáp án là `AAB`.
- **Relevance ở layer 18 (trung bình 16 head, token cuối) gần như phẳng:** 0,002–0,005 cho mọi prompt; prompt visual luôn cao hơn audio 1,24–1,39 lần; giữa các chunk cùng modality chỉ chênh 6–27%. Với λ = 2, mức chênh theo thời gian chỉ đổi 0,1–0,5 logit.
  Relevance visual còn **tăng theo t** ở 7/11 câu (tương quan 0,56–0,91).
- **J12b, vị trí chung so với "prompt ngay sau chunk của nó"** (chỉ TF, giới hạn 42 s, n=10): base 30 / boost_p vị trí chung 40 / cq_aligned 30 / boost_p aligned 30 / boost_pr aligned 40. Tương quan relevance–t của prompt visual: **−0,15 (vị trí chung)** so với **+0,82 (aligned)**.
  Giải thích: với RoPE, mỗi token chỉ có một position, nên chỉ chọn được một trong hai: câu hỏi cách đều mọi prompt (vị trí chung), hoặc mỗi prompt cách đều media của nó (aligned). Aligned làm prompt của chunk cuối nằm sát câu hỏi và bị chú ý nhiều hơn.
  Cách có cả hai (chưa làm): giữ vị trí chung và xoay query của prompt t theo `−(p0 − vị trí cuối chunk t)` bên trong hàm attention tuỳ biến; chính xác với token văn bản. Chỉ đáng làm khi có head thật sự chỉ đúng chunk (4.20 cho thấy chưa có).

### 4.16 J12d: prompt "…in one word:" và hiện tượng dồn về một đáp án (n=28)
Prompt: `…visual content of chunk t in one word:` / `…audio content of chunk t in one word is:`. Vị trí chung, token cuối là dấu `:`. Dấu cách sau `:` và dấu xuống dòng giữa prompt được thay bằng dấu cách, vì tokenizer gộp `: ` + `\n` thành một token và làm lệch token cuối.
Chạy trên hai server (mỗi nửa 16 mẫu), seed 2468, giới hạn 42 s, còn 28 câu hợp lệ.

| | TA (8) | TF (10) | CMTB (10) | Tất cả (28) |
|---|---|---|---|---|
| base | 50 | 30 | 30 | 35,7 |
| cq | 50 | 30 | 50 | 42,9 |
| cq_boost_p | 50 | 30 | 40 | 39,3 |
| cq_boost_pr | 50 | 40 | 40 | 42,9 |

Mỗi điều kiện sửa được 7–9 câu nhưng làm hỏng 6–7 câu → **nhiễu**. Bảng theo (task, đáp án đúng) ở mục 0A.3 cho thấy cq **dồn về một đáp án**: TA luôn "Yes", TF luôn "Present" (9–10/10), CMTB đáp án B sai cả 3.
Cùng hiện tượng ở 4.15. Nguyên nhân chưa tách (xem giả thuyết (a)–(c) ở 0A.3).

### 4.17 Loạt P (P1–P6) và P4 (Daily-Omni)
| Thí nghiệm | File | Kết quả |
|---|---|---|
| **P1 hai con trỏ** (nói thầm + `grow_window`, đích lấy từ mục tiêu kiểm chứng) | `p1_two_pointer.py` | Thời điểm của đích đọc từ residual 66–81%, từ attention thô ≈ ngẫu nhiên. TF **93,8** (n=16) khi có mục tiêu kiểm chứng |
| **P2 định tuyến 1 lượt** (đo ở layer 12, áp 13–27) | `p2_inpass_eval.py` | CMTB 47,4 → 78,9; một lượt = hai lượt; TA/TF phẳng |
| **P3 soi cô lập song song** (debug) | `remote_tests/debug_parallel.py` | Một lượt song song khớp soi riêng: tương quan 0,92–1,00 |
| **P5 soi cô lập + đọc theo quy tắc** (70 câu mới) | `p5_isolate_verify.py` | 44,3 → 77,1 (TA 88,2 / TF 79,2 / CMTB 69,0). Bước đọc ra bám theo dạng câu hỏi nên không tổng quát |
| **P6 tìm kiếm tập trung** (tổng quát, mô hình đóng băng, v2) | `p6_focused_search.py` | 36,4 = 36,4 (n=11). Kiểm tra audio cô lập không đáng tin (mô hình nói "heard" ngay cả trong im lặng); với CMTB mô hình tự chọn sai nguồn tham chiếu |
| **P4 mô-đun A trên Daily-Omni** (6 loại, 220/240 câu, dừng sớm để dành máy) | `p4_dailyomni_onepass.py` | base **66,4** → onepass **54,5** → onepass bỏ cổng 57,7 |

P4 theo loại câu hỏi (base → onepass): AV Event Alignment (n=37) 51,4 → 54,1; Comparative (37) 67,6 → 59,5; Context understanding (37) 54,1 → 37,8; Event Sequence (37) 54,1 → 40,5; Inference (36) 86,1 → 66,7; Reasoning (36) 86,1 → 69,4.
**Kết luận:** mô-đun A chỉ có lợi cho câu hỏi "lúc đó thấy/nghe gì"; **gây hại rõ ở mọi loại khác (−8 đến −20)**, vì nó ép attention về một cửa sổ thời gian ngay cả khi câu hỏi cần toàn bộ video. Phương pháp tổng quát phải có công tắc tự động hoặc không ép.

### 4.18 T2: mù thứ tự trên Daily-Omni (81/120 câu, dừng sớm)
Với mỗi câu Event Sequence / AV Event Alignment của Daily-Omni (video 30 s, 4 lựa chọn, xoay 4 thứ tự), so base với (i) **đảo ngược video** (audio giữ nguyên) và (ii) **đổi chỗ hai nửa video**. Nếu mô hình dùng thứ tự, đáp án phải đổi phần lớn.

| Loại | n | base | đảo ngược: đúng / đáp án đổi | đổi chỗ hai nửa: đúng / đáp án đổi |
|---|---|---|---|---|
| Event Sequence | 48 | 50,0 | 41,7 / 35,4% | 47,9 / 22,9% |
| AV Event Alignment | 33 | 51,5 | 48,5 / 30,3% | 54,5 / 24,2% |

Cắt hẳn thứ tự thời gian của video mà 65–77% đáp án **không đổi** → mù thứ tự là lỗi chung, không riêng SVHalluc. Đây là động cơ chính của paper.
Lưu ý: đảo video làm đảo cả hành động; kết quả chỉ nói mô hình ít dùng thứ tự, chưa nói nó dùng gì thay.

### 4.19 T1: chưng cất embedding dò (chỉ huấn luyện m = 8 embedding, mô hình đóng băng)
Ý tưởng của nhóm: m token dò đặt sau media, token câu hỏi/đáp án **chỉ được đọc các token dò (không đọc media)**; học chỉ embedding của dò sao cho phân phối đáp án khớp với khi mô hình nhìn thẳng video ("thầy" là chính mô hình, không cần nhãn).
Dữ liệu: clip 4 s cắt ngẫu nhiên từ video Daily-Omni, 4 câu hỏi chung, đáp án thầy sinh tham lam ≤ 16 token. 160 clip huấn luyện (640 mẫu), 40 clip đánh giá (160 mẫu). 800 bước, lr 5e-3, Adam.

| | khớp token (%) | KL |
|---|---|---|
| Không media, không dò (prior văn bản) | 65,9 | 0,777 |
| Dò chưa huấn luyện | 64,9 | 0,796 |
| Dò sau 200 / 400 / 600 / 800 bước | 73,0 / 60,9 / 73,7 / **73,4** | 0,551 / 0,774 / 0,541 / **0,482** |

- Embedding **có học được** nội dung của clip (KL giảm 38%) nhưng vẫn xa thầy, và đường học không ổn định (bước 400 rơi về mức prior).
- Tài nguyên: khoảng 20 phút cho 800 bước, bộ nhớ 9,9 + 8,0 GB trên T4×2. **Vừa sức "huấn luyện nhẹ"**.
- Giai đoạn 1 dạy *cái gì*, chưa dạy *thứ tự*; thầy cũng mù thứ tự. Giai đoạn 2 cần nhãn tự sinh có thứ tự (ghép clip với thời điểm biết trước).
- File: `results/t1_probe_distill.json`, `results/t1_probes.pt` (8 × 2048 embedding).

### 4.20 J13: relevance theo từng head / layer có chỉ đúng chunk không? (35 câu TA 15 + TF 20)
Cùng bố cục prompt như J12 (vị trí chung, không boost). Ghi attention của token câu hỏi/đáp án tới mỗi prompt ở **mọi layer (36) × mọi head (16) = 576 head**, hai kiểu: `last` (token cuối) và `mean` (trung bình các token câu hỏi/đáp án).
Mục tiêu chấm: **prompt audio → chunk có tiếng nói** (từ VAD) và **prompt visual → nửa video có sự kiện** (lấy từ cách dựng benchmark, chỉ dùng để phân tích). Kiểm chéo: chọn head trên nửa số câu, đo trên nửa còn lại (lặp 20 lần). Phân tích: `remote_tests/analyze_j13.py`.

| Cách đo | Audio → tiếng nói: head đạt ≥ 90% | Audio: đo trên nửa còn lại | Visual → nửa có sự kiện: head ≥ 90% | Visual: đo trên nửa còn lại |
|---|---|---|---|---|
| J13: prompt "…is", relevance ở chữ "is" | 31 (last) / 32 (mean) | 0,91 / 0,99 | **0** | 0,53 / 0,53 |
| J13b: prompt "…is", cộng attention trên **cả prompt** | 141 / 128 | 0,94 / 1,00 | **0** | 0,47 / 0,56 |
| J13d: prompt "one word", relevance ở dấu `:` | 63 / 106 | 1,00 / 0,98 | **0** | 0,44 / 0,53 |
| Ngẫu nhiên | — | **0,52** | — | **0,50** |

- **Audio:** có hàng chục đến hơn trăm head chỉ đúng chunk có tiếng nói gần như mọi lần. Trung bình 16 head hoặc lấy một layer che mất tín hiệu này (trung bình head: 0,51–0,53 ≈ ngẫu nhiên).
  Lưu ý mức chênh về khối lượng của head tốt nhất (chọn bằng kiểm chéo) chỉ +0,4% đến +8% so với ngẫu nhiên; chỉ *prompt mạnh nhất* là đúng ổn định. Chưa loại trừ khả năng đây chỉ là "chunk có tiếng thì nổi bật hơn chunk im lặng", không phụ thuộc câu hỏi.
- **Visual:** **không head nào** ở bất kỳ layer nào, cho cả ba cách đo, chỉ đúng nửa video có sự kiện hơn ngẫu nhiên. Kết quả khớp với J1/P1 (mô hình không tìm được "lúc nào" của một hành động).
- **Token dự đoán kế tiếp tại cuối mỗi prompt** (J13b, J13d):
  - prompt "…is": ở **mọi** chunk, cả hai modality, đều ra `: " . that`; chỉ mang **định dạng**;
  - prompt "…in one word:": prompt visual đoán từ nội dung **của chính chunk đó** (ví dụ video nấu ăn: ` Ingredients / Cooking / Woman`, rồi ` Stir / Heat / Meat`, rồi ` Wr(ap) / Rolling / dum(pling)`), prompt audio thường mở ngoặc kép và đôi khi đoán đúng từ nói ra (` onions`, ` tofu`).
- **Vì sao visual vẫn không lên:** câu hỏi TA/TF là "the narrated event" và hoàn toàn không mô tả hình ảnh, nên không có gì để khớp với từ khoá trong prompt visual. Phải đi hai bước: câu hỏi → nội dung lời nói → chunk hình có nội dung đó (mà thiết kế "prompt không thấy nhau" hiện chặn mất).
- **Chưa đo:** relevance **theo từng token** của câu hỏi/đáp án. "mean" trộn các token có thể nhìn các prompt khác nhau, nên có thể che một token cụ thể chỉ đúng. Với `mean`, tín hiệu audio rõ hơn `last` (106 so với 63 head ở J13d) nên nhiều token cùng nhìn đúng; nhưng với visual cả `last` lẫn `mean` đều 0 head.

### 4.21 Sai sót đã mắc và đã sửa trong đợt này
- Lần chạy J12d đầu bị lẫn cấu hình chỉ-TF của lần trước còn lại trong namespace server; restart server rồi chạy lại sạch (mục 3.3).
- Prompt kết thúc bằng `: ` bị tokenizer gộp với dấu xuống dòng kế tiếp → token cuối là `:\n`, không phải `:`. Đã bỏ dấu cách sau `:` và đổi dấu ngắt giữa prompt thành dấu cách; đã kiểm tra `last_tokens` trong kết quả = `:`.
- Báo cáo ban đầu gộp theo task che hiện tượng dồn đáp án; từ nay báo theo (task, đáp án đúng).

---

## 5. Nguyên nhân gốc đã xác định

| # | Nguyên nhân | Trạng thái | Bằng chứng then chốt |
|---|---|---|---|
| 1 | Tri giác (không nhìn thấy hành động) | **Không phải** | J1: 89–91% |
| 2 | Không nghe được lời nói | **Không phải** | 96% |
| 3 | Không biết lời nói ở đâu | **Không phải** (bên trong mô hình) | J5: IoU 0,88 / 0,83. Nhưng khi được hỏi bằng lời thì trả lời kém (IoU 0,44) |
| 4 | **Không dùng thời gian để chọn bằng chứng** | **Nguyên nhân chính** | Định tuyến 52 → 92–100; báo bằng chữ không giúp; can thiệp thời gian chỉ đổi 7–21% đáp án |
| 5 | **Nhầm nguồn (nghe thành thấy)** | **Nguyên nhân chính** | Giữ audio: 59 so với tắt audio: 94; xảy ra cả trong văn bản |
| 6 | **Lối tắt theo thì của lời nói (TF)** | **Nguyên nhân** | TTS: +0,74 (nhân quả) |
| 7 | **Khung nhãn Past/Present/Future + prior ở các lớp cuối** | **Nguyên nhân** | TF sụp ngay cả với bằng chứng hoàn hảo; CoT 86 |
| 8 | Prior "No" ở TA | Nguyên nhân phụ | 85% lỗi TA = đáp án khi không có bằng chứng |
| 9 | Thời gian dạng chữ / số giây | Mô hình không dùng được | 50 / 33 khi chỉ có số giây |
| 10 | Thứ tự nhân quả (postdiction) | **Bác bỏ** | J4 |
| 11 | Gắn kết cục bộ trong vùng media | **Bác bỏ** | Binding lab |
| 12 | Im lặng tuyệt đối là lối tắt | **Bác bỏ** | J5: nhiễu không đổi base |
| 13 | Attention dồn về đầu video | Yếu / không lặp lại | J4 |
| 14 | Mã hoá trong nhưng không đọc ra | Có một phần | Probe lớp 21 |
| 15 | Do mô hình nhỏ | **Bác bỏ** | 7B mắc y hệt |

---

## 6. Các phương pháp đã thử

| # | Phương pháp | Kết quả | Ưu điểm | Khuyết điểm | CVPR | Hướng tiếp |
|---|---|---|---|---|---|---|
| 1 | Định tuyến oracle (cửa sổ biết trước) | CMTB 96–100 | Cho thấy trần và nguyên nhân nhân quả | Cần biết trước cửa sổ | Không (chỉ là phân tích) | Giữ làm cận trên |
| 2 | STMR (pipeline văn bản) | 41 → 80 (pilot) | Rõ ràng, dễ phân tích lỗi | Giống agent; chậm (khoảng 30 lượt); phụ thuộc văn bản và VAD | Khó | Baseline mạnh |
| 3 | Định tuyến trên Daily-Omni (ASR) | +8 | Có trên dữ liệu tự nhiên | Không hơn TFO; dùng ASR | Không | Thay ASR bằng mốc từ attention |
| 4 | J4 postdiction | 0 | Kiểm định gọn | Không có tác dụng | Đưa vào phân tích | Dừng |
| 5 | J6/J7 câu tóm tắt cô lập | ≈ base | Một lượt, câu chuyện "cô lập cảnh" | Thiên lệch vị trí; câu "Summarize" không mang nội dung | Bản không train hiện tại: không | Đặt câu ngay sau chunk của nó (vị trí theo thời gian); hỏi nhỏ song song kiểu Yes/No; tách nguồn; mẫu PromptEOL; chọn đoạn theo nội dung |
| 6 | **J8: mốc tự động + tắt audio** | **CMTB 92–94 = oracle** | Tự động, trong mô hình, 2 lượt | Chỉ giải CMTB; tắt audio còn cố định | Là hạt nhân | Tự quyết kênh; tổng quát hoá |
| 7 | J9 / QTAR: PMI trên attention | Đúng âm thanh 86–91% | Tổng quát cho mọi loại âm thanh và cảnh | Chỉ chọn thời điểm, không sửa nhầm nguồn | Là thành phần đóng góp | Kiểm tra cảnh ở quy mô lớn |
| 8 | J10: tự quyết kênh | Đang chạy (1 mẫu: 0,92) | Tổng quát cho mọi câu hỏi | TF bị quyết thành "heard" | Có thể | Tách "kênh làm mốc" và "kênh để đọc" |
| 9 | J11: pipeline kiểu con người | Đang chạy | Giải được TF; có câu chuyện nhận thức | Nhiều bước, chậm; một phần dựa vào văn bản | Có thể nếu ablation tốt | Rút gọn: thay bước văn bản bằng can thiệp trong mô hình |
| 10 | Mô-đun A một lượt (P2/P4) | CMTB 47 → 79; **Daily-Omni 66,4 → 54,5** | Tự động, 1 lượt | Gây hại ở mọi loại câu hỏi khác; ép attention vào một cửa sổ | Không (nếu không có công tắc) | Công tắc tự động; chỉ bật khi câu hỏi nhắc tới mốc |
| 11 | Soi cô lập + đọc theo quy tắc (P5) | 44 → 77 (n=70) | Cơ chế đúng, một lượt song song | Bước đọc ra bám dạng câu hỏi | Không (chưa tổng quát) | Thay bước đọc bằng đọc do mô hình tự học |
| 12 | Tìm kiếm tập trung tổng quát, đóng băng (P6) | 36 → 36 | Hoàn toàn tự động | Mô hình 3B không tự quyết đáng tin | Không | Cần huấn luyện nhẹ |
| 13 | **Chunk prompts + boost attention (J12 / J12b / J12d)** | Tăng giả: dồn về Yes / Present | Đúng tinh thần "token dò trong mô hình" | Câu hỏi không khớp được prompt hình (hai bước); khối prompt làm lệch đáp án | Chưa | Mục 0A.5 |
| 14 | **Token dò huấn luyện nhẹ (T1)** | KL 0,78 → 0,48 | Rẻ, không nhãn, mô hình đóng băng | Chưa dạy thứ tự; chỉ embedding thì sức chứa hạn chế | Có thể, nếu giai đoạn 2 thành công | 8.4 |
| 15 | SLA | — | — | Tiền đề sai | Không | Đã bỏ |

**Phản biện sẽ hỏi gì (để chuẩn bị):**
- Khác TFO (Whisper + cắt clip), OmniAgent, Daily-Omni-Agent ở đâu?
- Khác các phương pháp can thiệp attention không cần train (PAI, PASTA, VCD…) ở đâu?
- Có tổng quát sang mô hình và benchmark khác không? Có gây hại không?
- Chi phí suy luận bao nhiêu?

---

## 7. Kết luận và nguyên tắc thiết kế

1. **Chọn thời điểm bằng chính attention của mô hình**, khử độ nổi bật bằng PMI (α=2). Đỉnh PMI cho biết *sự kiện nào*; attention thô cho biết *kéo dài bao lâu*.
2. **Đọc bằng chứng đúng nguồn:** tắt kênh không cần khi đọc. Đây là yếu tố tác động lớn nhất (59 → 94).
3. **Định tuyến phải bắt đầu từ khoảng lớp 13**, nơi bằng chứng đi vào token câu hỏi (patching: lớp 15–24). Bắt đầu từ lớp 21 là quá muộn.
4. **Giữ điều nghe được bằng "nói thầm"** (mô hình tự diễn đạt). Biểu diễn ẩn quá yếu (độ chênh 0,5 so với 2–18).
5. **Biểu diễn thời gian bằng quan hệ có tên** (trước / trong / sau), không bằng số giây.
6. **Câu hỏi quan hệ (TF) cần suy luận có chủ ý**, trên bằng chứng đã tách nguồn (nội dung lời nói khai báo riêng, ghi rõ "chỉ nghe, không phải bằng chứng hình").
7. **Token tóm tắt chèn thêm không mang thông tin khi không huấn luyện**, và bị thiên lệch vị trí RoPE.
8. **Tri giác không phải nút thắt:** đừng tốn công vào việc nâng cấp encoder.
9. **Mô hình biết mốc nổi bật ở đâu nhưng không tìm được một hành động cụ thể** (J13, P1). Chọn head chỉ đúng mốc (hàng chục đến hơn trăm head); không có head nào chỉ đúng sự kiện hình ảnh.
10. **Tìm sự kiện của câu hỏi "the narrated event" là hai bước:** câu hỏi → nội dung lời nói → chunk hình khớp. Một bước (câu hỏi → prompt hình) không thể đúng vì câu hỏi không nhắc gì tới hình ảnh.
11. **Thêm văn bản (prompt dò) vào trước câu hỏi làm mô hình dồn về một đáp án** (TA → Yes, TF → Present). Mọi so sánh phải tách theo (task, đáp án đúng).
12. **Prompt dò phải kết thúc ở chỗ mà mô hình sắp sinh nội dung** (`…in one word:`), không phải ở `…is`, vì token cuối của "…is" chỉ mang định dạng.
13. **Một position chỉ chọn được một trong hai khoảng cách:** câu hỏi → prompt hoặc prompt → media của nó (mục 4.15).

---

## 8. Đang làm dở và lộ trình tới paper

### 8.1 Đang chạy hoặc chờ kết quả (cập nhật tối 13/10/2026)
- **Không có job nào đang chạy.** Ba server GPU (4, 5, 6) đều rảnh.
- `nb12` (J10 held-out): **cố ý chưa chạy.** Chỉ chạy held-out một lần, khi phương pháp đã chốt.
- Dang dở, chưa cần: T2 (81/120 câu), P4 (220/240 câu); cả hai đã lưu ở `results/`.
- Việc làm tiếp: **mục 0A.5** (có thứ tự ưu tiên).

### 8.2 Phương pháp đề xuất cho paper (triển khai thực tế, tổng quát)

**Mô-đun A: tự định tuyến (lõi)**
- Đo độ liên quan theo chunk ở các lớp giữa sớm (cần thử lớp khoảng 8–12, vì 12–20 là muộn). Áp bias + tắt kênh không cần từ lớp 13 **trong cùng một lượt**.
- Prior trung tính cho PMI chỉ phụ thuộc video, **tính một lần cho mỗi video rồi lưu cache**.
- Kênh đích do mô hình tự quyết.
- Áp được cho câu hỏi mở và cho mọi video LM có attention mask + position ids.

**Mô-đun B: bộ nhớ cảnh cô lập (phát triển từ J6/J7)**
- Mỗi cảnh có khe "chỉ nhìn" và "chỉ nghe", đặt **đúng vị trí thời gian của cảnh**, dùng mẫu câu ép tóm tắt vào một token (PromptEOL).
- Khe nhớ không phụ thuộc câu hỏi, nên **tính một lần cho mỗi video**.
- Dùng để đánh chỉ mục / truy xuất theo nội dung cho video dài, và để tách nguồn.
- Biến thể mạnh hơn: **nhiều câu hỏi nhỏ cô lập chạy song song trong một lượt** (cảnh × nội dung, Yes/No đọc tại token cuối của mỗi câu), thay cho khoảng 30 lượt của J11.

**Mô-đun C: suy luận có chủ ý (tuỳ chọn)**
- Chỉ bật khi mô hình nhận ra đây là câu hỏi quan hệ thời gian.

### 8.3 Chứng minh tổng quát (cần có cho CVPR)

| Ngữ cảnh | Benchmark |
|---|---|
| Ảo giác AV theo thời gian | SVHalluc temporal (+ protocol gốc: `j4_official_protocol.py`) |
| Không gây hại | SVHalluc semantic (GSA / FGSA / CMSB) |
| Video AV tự nhiên | Daily-Omni (cả 6 loại; so trực tiếp với TFO) |
| Mô hình chỉ có hình | Qwen2.5-VL-3B trên tập con temporal của MVBench / Video-MME |
| Định vị thời gian zero-shot | Charades-STA: đoạn mô-đun A chọn có trùng nhãn không |
| Mô hình khác | Qwen2.5-Omni-7B, và một họ khác (video-SALMONN 2 / MiniCPM-o) nếu vừa GPU |
| Chi phí | Thời gian và bộ nhớ so với base |

### 8.4 Việc nên làm tiếp theo (theo thứ tự)
*(Bản cũ của mục này đã lỗi thời: mô-đun A 1 lượt đã làm (P2) và gây hại ngoài SVHalluc (P4); mô-đun B là chunk prompts (J12), kết quả ở mục 4.15–4.20.)*
1. **Tách nguyên nhân dồn về một đáp án** và báo mọi kết quả theo (task, đáp án đúng) (0A.3, 0A.5 việc 1).
2. **Đo relevance theo từng token câu hỏi**, và **phép đo hai bước** (câu hỏi → nội dung lời nói → chunk hình khớp) không huấn luyện (0A.5 việc 2–3).
3. Nếu không có tín hiệu: **T1 giai đoạn 2**, huấn luyện nhẹ. Gợi ý thiết kế:
   - mô hình đóng băng; học embedding dò (m = 8 hoặc vài chục, có thể thêm một vài tham số nhỏ ở lớp giữa);
   - dữ liệu **tự sinh, không dùng SVHalluc**: ghép các clip ngắn có thời điểm sự kiện biết trước (cùng cách dựng với `concat_clips`, `halves` trong `job_utils.py`), sinh nhãn "khi nào", "X trước hay Y trước", "lúc nghe Z thì thấy gì", **nhiều cách diễn đạt khác nhau** (để không học thuộc mẫu câu hỏi);
   - mục tiêu: đáp án đúng theo nhãn tự sinh, cộng chưng cất như giai đoạn 1 để không hỏng khả năng chung; ngân sách khoảng một phiên T4×2.
4. **Công tắc tự động cho mọi can thiệp attention** (mô-đun A hiện gây hại ngoài SVHalluc): chỉ can thiệp khi câu hỏi nhắc tới mốc.
5. Đóng gói phương pháp đã chốt, rồi chạy held-out **một lần** (seed `1234`), rồi các benchmark tổng quát (8.3). Kèm kiểm tra diễn đạt lại câu hỏi (mở / nhiều sự kiện / trước–sau theo nhiều cách nói).

---

## 9. Cạm bẫy kỹ thuật đã gặp
- **`PER_LABEL` trên server:** `kaggle_rootcause.py` đặt `PER_LABEL = 1` từ argv, nên job chạy trên server **phải khai báo tường minh** `PER_LABEL`/`SEED`. Nếu không, chỉ chạy 8 câu.
- **`fix_ffmpeg.py`** vá ffmpeg cũ (không có `-fps_mode`) **nhưng cũng xoá** `tf_framing.jsonl` và `tense_intervention.jsonl` trên server. Hãy tải các file đó về trước.
- **4 lựa chọn (Daily-Omni):** mặc định `LETTERS` / `render` chỉ có A–C. Phải dùng `L4` và `prompt_of` trong `job_utils`.
- **Câu tóm tắt cùng độ dài:** đánh số 2 chữ số (`part 01`), vì vị trí song song yêu cầu các câu có cùng số token.
- **Đọc trọng số attention:** cần `eager`. Có thể bật / tắt eager theo từng lớp bằng `self_attn.config._attn_implementation` (xem `inpass.py`).
- **Suy luận từng bước:** phải **đọc đáp án từ lời suy luận** và cắt phần sinh lan man ("Human: …"). Nối "Therefore the answer is (" vào sau đoạn rác cho số sai.
- **Kaggle output:** tải bằng `--file-pattern`, nếu không sẽ kéo về cả venv nặng vài GB.
- **Session server Kaggle hết sau khoảng 12 h:** tải kết quả về thường xuyên.
- **Bash heredoc trên Windows:** dấu `\` trong code Python dễ bị nuốt. Dùng `chr(92)` hoặc sửa file trực tiếp.
- **Tokenizer gộp `: ` với `\n`:** prompt kết thúc bằng dấu cách rồi đến dấu xuống dòng của prompt kế tiếp sẽ thành một token duy nhất, nên "token cuối của prompt" không còn là `:`. Bỏ dấu cách cuối, hoặc ngắt bằng dấu cách.
- **Namespace server giữ biến cũ:** cấu hình của job trước (ví dụ `J12_TASKS`) còn lại và có thể lẫn vào job sau. Sau khi đổi cấu hình, **restart server** rồi chạy lại `fix_ffmpeg.py`.
- **Hàm attention tuỳ biến:** `ALL_ATTENTION_FUNCTIONS['sdpa']` bị thay trong lúc chạy `cq_forward` (rồi khôi phục trong `finally`). Chỉ gắn nhãn `_cq_layer` cho layer LLM (36 layer), vì cùng hàm attention cũng được audio / vision encoder dùng.
- **Out-of-memory khi ghi relevance:** chỉ giữ hàng của câu hỏi/đáp án, và dùng `lm_head` forward pre-hook để chỉ tính logit cho vài vị trí cần.
- **Chuẩn hoá relevance có thể ra NaN** khi prompt không có attention (chia cho 0): thêm epsilon trước khi chuẩn hoá (đã sửa trong `analyze_j13.py`).
- **Báo cáo tổng gộp che hiện tượng dồn đáp án:** luôn tách theo (task, đáp án đúng) (mục 0A.6).

## 10. Tài liệu liên quan
- `docs/temporal_related_work_2026-10-11.md`: tổng quan công trình liên quan: RED (2610.02976), DTR (2604.12582), SyncRA (2609.34363), AVTrace (2609.19991), From Senses to Decisions (2606.10147), Senses Wide Shut (2605.13737), Compositional Failure (2608.27785), VideoRoPE, PAS, VRoPE, OmniAgent (2512.23646), **TFO (2609.04242)**, MLLMs Know When Before Speaking (2605.21954), Video-XL, Gist tokens, Activation Beacon…
- `docs/DECISION_BRIEF.md`: bản tóm tắt để quyết định phương pháp (cập nhật tối 13/10/2026): sự thật đã kiểm chứng, các hướng đã thử, việc nên quyết định tiếp.
- `docs/METHOD_PROPOSAL.md`: đề xuất SLA (**đã bỏ**, giữ làm lịch sử) và lý do.
- `docs/14_root_causes_and_general_method.md`: bảng 16 giả thuyết nguyên nhân và kết quả giai đoạn đầu.
- `docs/research_log.md`: nhật ký đầy đủ theo thời gian.
- Nền tảng nhận thức được dùng: source monitoring (Johnson 1993), working memory / phonological loop (Baddeley), biased competition / attentional template (Desimone & Duncan), event segmentation (Zacks), postdiction (Eagleman & Sejnowski 2000).

---

**Bảo mật:** không bao giờ commit `remote_config*.json` (URL + token của server), token Kaggle hay token HuggingFace. `.gitignore` đã chặn sẵn.
