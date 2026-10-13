# Nguyên nhân gốc của lỗi temporal trên SVHalluc và hướng phương pháp tổng quát

Ngày 12/10/2026. Model: Qwen2.5-Omni-3B. Tài liệu tổng hợp mọi phép đo đã có (local + Kaggle) và tài liệu đã đọc. Mỗi nguyên nhân ghi rõ **bằng chứng** và **mức chắc chắn**. Mục đích: liệt kê đầy đủ, không bỏ sót, trước khi thiết kế phương pháp.

---

## 1. Con người làm gì khi xem một video có lời nói

| Năng lực | Nội dung | Nguồn |
|---|---|---|
| **Theo dõi nguồn gốc** (source monitoring) | Ký ức không có nhãn nguồn sẵn; con người *suy ra* thông tin đến từ mắt thấy, tai nghe hay người khác kể. Nhầm "nghe kể" thành "tự thấy" là lỗi kinh điển, và **thông tin nghe được dễ bị gán sai nguồn hơn** | [Johnson, Hashtroudi & Lindsay 1993](https://pmc.ncbi.nlm.nih.gov/articles/PMC5312673); nghiên cứu reality monitoring theo modality |
| **Định hướng chú ý theo thời gian** | Một tín hiệu (âm thanh, lời nói) cho biết *khi nào* cần nhìn; thị giác kiểm chứng *cái gì* | Crossmodal cueing (Posner paradigm) |
| **Suy luận nguyên nhân chung** | Trước khi ghép nghe và nhìn, não ước lượng chúng có đến từ cùng một sự kiện không | [Causal inference in multisensory perception](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943) |
| **Trọng số theo độ tin cậy** | Không chia đều; nguồn đáng tin hơn cho tác vụ đó được dùng nhiều hơn | [Alais & Burr 2004](https://pubmed.ncbi.nlm.nih.gov/14761661/) |
| **Phân đoạn sự kiện** | Dòng trải nghiệm được cắt thành sự kiện, lưu kèm thời điểm | [Baldassano et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558154/) |

Câu "lát nữa chúng ta thái khoai" khi tay đang cho cà chua vào nồi: con người giữ được ba điều riêng biệt (thấy gì, nghe gì, câu nói nhắc tới cái gì và nó xảy ra khi nào). Để trả lời "lúc nghe câu đó, hình đang làm gì?", họ hỏi **"cái này tôi thấy hay chỉ nghe nói?"** trước khi trả lời.

---

## 2. Bảng nguyên nhân: đầy đủ theo từng khâu xử lý

Ký hiệu mức chắc chắn: **Đã xác nhận** (có phép can thiệp, nhiều mẫu) · **Có dấu hiệu** (một phép đo, mẫu nhỏ) · **Bị bác bỏ / không ủng hộ** · **Chưa đo**.

| # | Khâu | Nguyên nhân giả định | Bằng chứng | Mức chắc chắn |
|---|---|---|---|---|
| 1 | Tiền xử lý | Lấy mẫu 2 fps, ~160×28×28 px/khung làm mất thao tác ngắn | Cắt đúng đoạn đạt 94–96% (CMTB): đầu vào đủ thông tin cho phần lớn câu. 2/55 lỗi còn lại là thao tác ngắn/gần giống | **Có dấu hiệu, nguyên nhân phụ** (~5% lỗi) |
| 2 | Tiền xử lý | Decord đọc sai thứ tự khung (timestamp nhảy lùi) | Codex: 10/12 clip lỗi khi không chuẩn hoá CFR; đã sửa bằng chuẩn hoá 30 fps | **Đã xác nhận, đã khắc phục** trong mọi thí nghiệm |
| 3 | Nhận diện hình | Không phân biệt hành động gần giống (khoá túi / bỏ thịt vào túi) | 2 mẫu lỗi của crop; trace bacon của Codex | **Có dấu hiệu, nguyên nhân phụ** |
| 4 | Nhận diện lời nói | Không hiểu câu nói nhắc hành động gì | `hear` trả đúng "Dicing potato" ở mẫu thử; SVHalluc: transcript không giúp TA/TF | **Chưa đo ở quy mô** (đo được từ dữ liệu giai đoạn 2) |
| 5 | **Định vị lời nói** | Không biết lời nói nằm ở đâu trên dòng thời gian | Timestamp: độ trùng trung vị 0,44; câu hỏi nửa đầu/nửa sau 57,5% (gần ngẫu nhiên 50%), n=40 | **Đã xác nhận cho model 3B** (paper SVHalluc đo Qwen3-Omni 0,88, model khác) |
| 6 | Ghép nghe–nhìn cùng thời điểm | Thứ tự [hình, âm] + mặt nạ nhân quả làm hình không đọc được âm của cùng khoảnh khắc; âm đọc hình của mọi thời điểm | Binding lab, 58 câu: các mặt nạ ghép cục bộ **gần như không đổi câu trả lời nào** | **Bị bác bỏ** như nguyên nhân chính |
| 7 | Mã hoá vị trí | Mốc thời gian tuyệt đối quá thưa / thiên lệch vị trí | Nén chỉ số thời gian: không cải thiện ổn định (TF dồn về Present). Phép đảo thứ tự: không thấy hiệu ứng rõ (n≈10–15) | **Không được ủng hộ** về hành vi |
| 8 | Chú ý khi đọc | Chú ý dồn về đầu video, không theo thời điểm được neo | 6/6 trace: chú ý vào nửa đầu nhiều hơn nửa sau (tỉ lệ 0,61–0,93); không có head nào ổn định neo vào khoảng lời nói (24 clip) | **Có dấu hiệu** (mô tả, chưa nhân quả) |
| 9 | **Chọn thời điểm khi đọc bằng chứng** | Token câu hỏi đọc mọi thời điểm thay vì thời điểm được neo | Chặn khung ngoài khoảng: TA +23,3 [+9,3; +37,2] n=43; CMTB +14,5 [+4,8; +24,2] n=62. Chặn ở token câu hỏi ≈ chặn toàn bộ; chặn chỉ lớp sau yếu hơn | **Đã xác nhận** (trên clip gốc) |
| 10 | **Nhầm nguồn: "nghe nói" thành "thấy"** | Nội dung lời nói quyết định thay bằng chứng hình | 2×2 (pilot n=19): chặn khung 63%, chặn audio 32%, cả hai 95% = crop. Trên clip thay tiếng nền (đếm sơ bộ n=17): gốc 47% → cả hai 88%, trong khi chỉ chặn khung chỉ còn +1,5 điểm ở study trước | **Đã xác nhận ở CMTB** (chờ báo cáo cuối giai đoạn 2) |
| 11 | Tính quan hệ thời gian | Có bằng chứng đúng nhưng không suy ra trước/trong/sau | TF gốc 16–28% (dưới ngẫu nhiên); `htl` 84–88% khi tách bằng chứng theo khoảng. Chưa kiểm tra trực tiếp model có suy luận đúng từ dữ kiện chữ không | **Chưa đo trực tiếp** |
| 12 | Hiểu câu hỏi | Câu TA bị hiểu thành mệnh đề chung ("không nhất thiết xảy ra cùng lúc") | Codex: câu trả lời kiểu phát biểu chung; thêm "in this particular clip" → luôn Yes | **Có dấu hiệu** (vài mẫu) |
| 13 | **Thiên lệch quyết định ở lớp sau** | Model "chốt" đáp án ưa thích, bám yếu vào bằng chứng | TA: recall Yes 10,5%, No 87,5% (n=43); TF dưới mức ngẫu nhiên; logit lens: đáp án lật ở lớp 32–34 (6 trace). Tài liệu: *prior dominance* ở lớp ~25 ([2608.27785](https://arxiv.org/abs/2608.27785)) | **Đã xác nhận như thiên lệch**; cơ chế lớp chưa đo nhân quả |
| 14 | Khoảng cách biểu diễn–hành động | Hidden state biết đáp án đúng nhưng đầu ra không dùng | Tài liệu: [Senses Wide Shut](https://arxiv.org/abs/2605.13737) cho model omni. Ta chưa probe | **Chưa đo** |
| 15 | Mất cân bằng số token | ~144 token hình / ~25 token âm mỗi giây | `balance`: thay đổi dựa trên 1–2 câu, ở TF lệch về Present | **Không được ủng hộ** (mẫu nhỏ) |
| 16 | Dữ liệu đánh giá | Nửa clip im lặng tuyệt đối; chỉ vị trí lời nói đã cho TF 66,7% | Đo trên toàn bộ 397 clip | **Đã xác nhận** — là nhiễu đánh giá, phải kiểm soát |

### Những gì chưa đo, cần đo để bảng "đầy đủ"

1. **#4** độ chính xác nghe lời nói (so `hear` với transcript Whisper chỉ để chấm).
2. **#11** suy luận quan hệ từ dữ kiện chữ ("lời nói ở giây 3–5, hành động ở giây 8–10 → trước/sau?").
3. **#12** độ nhạy với cách diễn đạt câu hỏi, ở quy mô.
4. **#14** probe tuyến tính trên hidden state: "hành động có xảy ra trong khoảng lời nói không" có giải mã được khi đầu ra sai không.
5. **Phân rã lỗi theo nguyên nhân**: với mỗi câu baseline sai, phép sửa nào cứu được (chỉ chặn khung / chỉ chặn audio / cả hai / không gì / cắt clip) → tỉ lệ lỗi do từng nguyên nhân. Dữ liệu 2×2 giai đoạn 2 đã cho phép làm cho CMTB.

---

## 3. Bức tranh nhân quả hiện tại

```
lời nói nhắc sự kiện X (ở thời điểm khác)
        │
        ▼
[5] model không biết lời nói nằm ở đâu ──► [9] token câu hỏi đọc mọi thời điểm (chú ý dồn về đầu video [8])
                                                  │
        [10] nội dung lời nói (X) được dùng như bằng chứng hình ◄── nhầm nguồn
                                                  │
                                                  ▼
                       [13] lớp sau chốt đáp án theo thiên lệch (TA → No, TF dưới ngẫu nhiên)
```

Nhận diện hình (#1, #3) và ghép nghe–nhìn bên trong (#6) **không phải nút thắt chính**. Lỗi chủ yếu nằm ở **khâu đọc bằng chứng của câu hỏi**: chọn sai thời điểm và nhầm nguồn. Điều này khớp với bản đồ đường đi thông tin của [2606.10147](https://arxiv.org/html/2606.10147): modality → token câu hỏi (lớp đầu–giữa) → token cuối.

---

## 4. Phương pháp đề xuất: Đọc bằng chứng có theo dõi nguồn gốc và thời điểm

Tên tạm: **Source- and Time-Monitored Reading (STMR)**. Không huấn luyện, một model, can thiệp ở **token câu hỏi** (đúng chỗ lỗi xảy ra).

| Bước | Con người | Trong model |
|---|---|---|
| A. Đọc câu hỏi | "Câu này hỏi điều tôi **thấy**, điều tôi **nghe**, hay quan hệ giữa chúng? Có neo vào lúc nào không?" | Chính model phân loại câu hỏi (một lượt chữ ngắn): nguồn bằng chứng cần {thấy / nghe / cả hai}; mốc thời gian {theo âm thanh / theo mô tả chữ / không có} |
| B. Định hướng thời gian | Tín hiệu cho biết khi nào cần nhìn | Nếu neo theo âm thanh: khoảng hoạt động của audio (hiện dùng năng lượng; mục tiêu là đặc trưng audio của chính model). Nếu neo theo chữ: điểm liên quan theo từng khoảng (Yes/No theo đoạn). Không có mốc: không giới hạn |
| C. Đọc có kiểm soát nguồn | "Muốn nói là *thấy* thì phải có bằng chứng từ mắt" | Ở token câu hỏi: giới hạn chú ý vào token hình của khoảng được neo; nếu nguồn cần là "thấy" thì chặn nội dung audio khỏi quyết định |
| D. Quan hệ | So thời điểm của các sự kiện đã gắn nguồn | Với câu hỏi quan hệ: nghe nội dung → tìm trong từng khoảng → so sánh (`htl`) |
| E. Mặc định an toàn | Câu hỏi không cần chọn lọc thì xem bình thường | Bộ định tuyến nói "không mốc, cả hai nguồn" → **không can thiệp**, giữ nguyên hiệu năng thông thường |

**Vì sao có thể tổng quát:** nhầm nguồn "nghe → thấy" không chỉ ở temporal. FGSA/CMSB (semantic) của SVHalluc cũng là lỗi đó (vật/hành động được nhắc nhưng không có trong hình), cũng như AVHBench (audio-driven video hallucination). Nếu bước C (chặn audio khi câu hỏi hỏi về thứ *nhìn thấy*) giúp cả semantic lẫn temporal mà không làm hại câu hỏi cần audio, đó là bằng chứng cho tính tổng quát. **Phép thử này chưa làm**: binding lab không có điều kiện chặn audio trên semantic.

**Ranh giới với nghiên cứu đã có:**
- SFFL: chọn modality bằng chứng trước khi suy luận, nhưng **huấn luyện** (GRPO).
- OmniAgent: nghe để định vị rồi nhìn, nhưng bằng **agent và công cụ**.
- DTR/PAS: cân bằng chú ý theo khung **không điều kiện**, model chỉ hình.
- RED/MAD/AVCD/ASD: sửa ở **decoding/logits**, không ở khâu đọc.
- Senses Wide Shut: chỉnh logits từ probe, nhắm câu hỏi có tiền đề sai.
- Điểm có thể mới: **(i)** chỉ ra lỗi ở khâu đọc là *nhầm nguồn + sai thời điểm* bằng phép can thiệp 2×2 có kiểm soát nhiễu; **(ii)** sửa đúng khâu đó trong một lượt, có bộ định tuyến để không làm hại tác vụ khác; **(iii)** liên hệ với lý thuyết theo dõi nguồn gốc. Cần tìm thêm trước khi claim (đặc biệt AVHBench methods, OmniAgent toàn văn).

---

## 5. Kế hoạch kiểm chứng (theo thứ tự, mỗi bước có cửa quyết định)

1. **Báo cáo cuối giai đoạn 2** (đang chạy): C1 (CMTB không lối tắt), C2 (TA), C3 (TF).
2. **Đo nốt nguyên nhân chưa đo** (#4, #11, #12, #14) + phân rã lỗi theo nguyên nhân.
3. **Kiểm tra bước C trên semantic** (FGSA/CMSB/GSA): chặn audio ở token câu hỏi khi câu hỏi hỏi "có thấy không". Nếu giúp FGSA/CMSB mà không hại GSA (cần audio) → bằng chứng tổng quát đầu tiên.
4. **Bộ định tuyến tự động** (bước A) và đo độ chính xác của nó.
5. **Không làm hại**: chạy STMR (có bộ định tuyến) trên benchmark thông thường (Daily-Omni/WorldSense cho nghe–nhìn, MVBench cho video). Tiêu chí: không thấp hơn model gốc.
6. **Model khác**: Qwen2.5-Omni-7B, một model omni khác họ.

Bước nào không đạt thì dừng và điều chỉnh, không mở rộng tiếp.

---

## 6. Cập nhật 12/10: kết quả hai shard root-cause (130 câu hợp lệ, FP16) và giai đoạn 2

### 6.1 Những nguyên nhân đã đóng

| # | Nguyên nhân | Kết quả | Kết luận |
|---|---|---|---|
| 4 | Không nghe đúng nội dung lời nói | Câu model tự nghe khớp ít nhất một từ nội dung với transcript Whisper ở **96,2%** câu (n=130) | **Bác bỏ** |
| 12 | Hiểu sai câu hỏi | TA và CMTB: ba cách diễn đạt cho cùng câu trả lời ở 74–75% câu; độ chính xác gần như không đổi. TF: **0%** câu có ba cách diễn đạt trùng nhau | **Bác bỏ cho TA/CMTB**; TF bất ổn hoàn toàn (xem 11) |
| 13 | Đáp án mặc định (prior) | TA: khi không có nội dung (video đen, câm), model **luôn** trả No. 85% lỗi TA trùng đáp án mặc định; model chỉ đoán Yes ở 26% câu (61% câu gold Yes bị sai) | **Xác nhận: nguyên nhân chính của TA** |
| — | Model có dùng video ở TA không | Chỉ hình: 60%; gốc: 62,9%; 92% lỗi TA trùng đáp án của bản video đen | Ở TA, video gần như **không được dùng** |
| 9/10 | Đọc sai thời điểm / nhầm nguồn (CMTB, gộp 72 câu) | gốc 51,4 · chỉ chặn thời điểm 80,6 · chỉ chặn audio 52,8 · **cả hai 95,8** · cắt clip 97,2. Phân rã 35 lỗi: **63% chỉ do thời điểm**, 20% cần cả hai, 14% chỉ do audio, 3% không sửa được (nhận diện hình) | **Xác nhận và đã định lượng** |
| 8 | Chú ý dồn về đầu video | 24 câu, chú ý của token câu hỏi vào nửa sau / nửa đầu (theo token): 0,70 / 0,72 / **0,44** ở lớp 0–11 / 12–23 / 24–35; trong / ngoài khoảng lời nói ≈ 0,9–1,0 (không ưu tiên khoảng lời nói) | **Xác nhận ở quy mô**; chú ý không bám thời điểm lời nói |
| — | Lỗi vào ở lớp nào (patching) | 10 câu mà gốc sai, sửa đúng: ở 7/10 câu, cả "sửa" và "làm hỏng" chuyển trạng thái trong **lớp 15–24**, và ở 5 câu chuyển đúng tại **lớp 21** | **Cửa sổ đọc bằng chứng ≈ lớp 15–24** |
| 14 | Biết bên trong mà không nói ra | TF: probe tuyến tính trên trạng thái token câu hỏi đạt **71%** (lớp 6, 24, 27), trong khi đầu ra chỉ 27,5% (đa số 35%). Luật vị trí lời nói đạt 68,6% trên cùng câu, nên probe có thể chỉ đang đọc **vị trí lời nói**. TA/CMTB: probe không hơn ngẫu nhiên (mẫu nhỏ) | **Có khoảng cách biểu diễn–hành động ở TF**; nội dung thông tin cần kiểm tra thêm |
| 11a | Không hiểu khung Past/Present/Future | Chữ (FP16): Future bị đoán đúng 0% ở mọi thì; video thật: đổi sang Before/During/After +13,2 [+1,5; +25,0] | **Xác nhận** |
| 11b | Thì của lời nói làm lệch thứ tự | Chữ (FP16): Before/During/After 87–97% nhưng còn **56,7%** khi lời nói ở thì quá khứ | **Xác nhận trên chữ**; can thiệp trên video đang chạy |

### 6.2 Phát hiện phụ về benchmark

- CMTB: chỉ với **video đen + audio gốc**, model đạt **65,9%**, cao hơn baseline có hình (59,1%). Lời nói của nhiều câu CMTB mô tả đúng hành động đang diễn ra, nên một phần CMTB giải được không cần nhìn. Đây là lối tắt thứ hai của benchmark (bên cạnh vị trí lời nói ở TF).

### 6.3 Bức tranh nguyên nhân theo task

- **TA:** model mặc định "No" (prior) và gần như không dùng video. Chặn thời điểm giúp được.
- **TF:** (i) khung nhãn Past/Present/Future không được hiểu; (ii) thì của lời nói lệch phán đoán; (iii) model có thông tin liên quan trong biểu diễn (probe 71%) nhưng đầu ra không dùng; đầu ra bất ổn theo cách diễn đạt.
- **CMTB:** chủ yếu đọc sai thời điểm (63% lỗi), phần còn lại do nhầm nguồn; nhận diện hình hiếm khi là vấn đề (3%).
- **Chung:** model không định vị được lời nói (độ trùng 0,44); chú ý dồn về đầu video và không bám khoảng lời nói; bằng chứng được đọc ở lớp 15–24.

**Còn chờ:** can thiệp thì trên video thật (TTS); dời lời nói / phát ngược video / đảo đồng hồ TMRoPE (model có dùng thông tin thời gian không); cho biết mốc lời nói bằng chữ (tách nguyên nhân #5).
