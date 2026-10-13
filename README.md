# temporal_videoLM — Vì sao video LM thất bại ở bài toán thời gian, và cách sửa không cần huấn luyện

Repo này chứa **toàn bộ** nghiên cứu về lỗi *temporal* (thời gian) của video/omni language model trên benchmark **SVHalluc**, gồm:

- mã nguồn và notebook Kaggle cho mọi thí nghiệm;
- kết quả thô của các thí nghiệm (`results/`);
- nguyên nhân gốc đã xác định, các phương pháp đã thử cùng ưu và khuyết điểm;
- việc đang làm dở và hướng đi tiếp để viết paper.

Mục tiêu của repo là để người khác clone về rồi tự tiếp tục: thí nghiệm, phát triển phương pháp và viết paper.

- Mô hình chính: **Qwen2.5-Omni-3B Thinker**, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`. Có kiểm tra phụ với Qwen2.5-Omni-7B.
- Benchmark chính: **SVHalluc** (CVPR 2026, arXiv 2606.02642), phần temporal: **TA** (368 câu), **TF** (276 câu), **CMTB** (339 câu). Dataset HF `zcs15/SVHalluc`, revision `211b4f14ab66ae9557b69e302d1f090e0e610839`.
- Benchmark phụ: **Daily-Omni** (HF `liarliar/Daily-Omni`), video tự nhiên.

---

## Mục lục

0. [Tóm tắt một trang](#0-tóm-tắt-một-trang)
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

**Còn mở:**
- TA chưa cải thiện trong cách làm một lượt.
- TF cần suy luận có chủ ý. Pipeline đầy đủ J11 đang được đánh giá.
- Cần làm cho phương pháp chạy **trong một lượt** (thử đầu tiên: áp bias từ lớp 21 thì quá muộn, phải từ lớp 13).
- Cần chứng minh **tổng quát** trên mô hình và benchmark khác.

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
│   │   └── inpass.py             ← mô-đun A: tự định tuyến trong 1 lượt (đang debug)
│   └── nb*.ipynb                 ← notebook đã build sẵn
│
├── kaggle_gpu_server.py, build_kaggle_gpu_server_notebook.py
│                                 ← server GPU trên Kaggle (HTTP + ngrok) để chạy job từ xa
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
1. Build notebook server: `python build_kaggle_gpu_server_notebook.py`. Trên Kaggle, đặt Secret `NGROK_AUTHTOKEN` (bật *Attach*) rồi chạy. Notebook in ra `URL` và `TOKEN`. **Dùng tài khoản Kaggle phụ.**
2. Ghi vào `remote_config.json` (xem file mẫu). Có nhiều server thì dùng `remote_config2.json`… và chọn bằng `REMOTE_CFG=remote_config2.json`.
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
| `j11_dev2_v1.jsonl`, `j11v2_dev2_partial.jsonl` | J11 trên server (dev2) |

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

---

## 8. Đang làm dở và lộ trình tới paper

### 8.1 Đang chạy hoặc chờ kết quả (tại thời điểm viết README)
- `nb11` (J10 pilot) và `nb13` (J11 **v1**, còn lỗi) trên Kaggle.
- J11 **v2** trên bộ dev2 (server).
- `nb12` (J10 held-out): **cố ý chưa chạy.** Chỉ chạy held-out một lần, khi phương pháp đã chốt.

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
1. **Mô-đun A trong 1 lượt:** thử đo ở lớp 8–12 và áp từ lớp 13 (`repo/kaggle/jobs/inpass.py`, `remote_tests/debug_inpass.py`).
2. Đọc kết quả nb11 / nb13 / J11 v2 và phân tích lỗi theo từng bước.
3. **Mô-đun B:** debug "hỏi nhỏ song song cô lập, đặt đúng vị trí thời gian" trên 1–2 mẫu, so với nhìn từng đoạn riêng lẻ.
4. Đóng gói phương pháp đã chốt, rồi chạy held-out **một lần**, rồi chạy các benchmark tổng quát (8.3).
5. Nếu muốn có phiên bản huấn luyện: prompt tuning cho khe nhớ (mô hình đóng băng), dữ liệu tự sinh (ghép clip có thời điểm sự kiện biết trước), **không dùng dữ liệu SVHalluc**.

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

## 10. Tài liệu liên quan
- `docs/temporal_related_work_2026-10-11.md`: tổng quan công trình liên quan: RED (2610.02976), DTR (2604.12582), SyncRA (2609.34363), AVTrace (2609.19991), From Senses to Decisions (2606.10147), Senses Wide Shut (2605.13737), Compositional Failure (2608.27785), VideoRoPE, PAS, VRoPE, OmniAgent (2512.23646), **TFO (2609.04242)**, MLLMs Know When Before Speaking (2605.21954), Video-XL, Gist tokens, Activation Beacon…
- `docs/14_root_causes_and_general_method.md`: bảng 16 giả thuyết nguyên nhân và kết quả giai đoạn đầu.
- `docs/research_log.md`: nhật ký đầy đủ theo thời gian.
- Nền tảng nhận thức được dùng: source monitoring (Johnson 1993), working memory / phonological loop (Baddeley), biased competition / attentional template (Desimone & Duncan), event segmentation (Zacks), postdiction (Eagleman & Sejnowski 2000).

---

**Bảo mật:** không bao giờ commit `remote_config*.json` (URL + token của server), token Kaggle hay token HuggingFace. `.gitignore` đã chặn sẵn.
