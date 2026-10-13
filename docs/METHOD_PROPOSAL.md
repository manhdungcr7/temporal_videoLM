> **CẬP NHẬT 13/10/2026 (tối): SLA ĐÃ BỊ BỎ.** Tiền đề của SLA sai: thời điểm của *mốc* (lời nói, âm thanh nổi bật) đã nằm
> sẵn trong residual (đọc được 96–100% ở mọi lớp), còn thời điểm của *đích* (một hành động) thì mô hình không tìm được
> qua attention (≈ ngẫu nhiên, P1). Ghi thêm "đã nhìn lúc nào" vào attention không sửa được chỗ này. Tài liệu bên dưới giữ
> lại làm lịch sử. Hướng đang làm: **chunk prompts + attention tăng cường theo độ liên quan** (README mục 0A và 4.15–4.20),
> và **token dò có huấn luyện nhẹ** (T1, README 4.19). Các phát hiện làm đổi hướng: README 0A.2.

# (Lịch sử) Đề xuất phương pháp: **Self-Locating Attention (SLA)** — attention đọc *cái gì* và báo lại *đã nhìn lúc nào, từ đâu*

> **QUYẾT ĐỊNH CUỐI (cập nhật sau khi kiểm tra tính mới, xem mục 0).** Tên cũ "Episodic Attention" bị bỏ vì trùng tên với
> Episodic Transformer (ICCV 2021). Phần còn lại của tài liệu (mục 1–6) vẫn đúng về động cơ và kế hoạch; mục 0 là định
> nghĩa chốt.

## 0. Phương pháp chốt

**Một dòng:** mỗi head, ngoài nội dung đã đọc, ghi vào residual **mình đã nhìn vào lúc nào và vào nguồn nào**:

    o_i  =  Σ_j a_ij v_j                         (attention thường: CÁI GÌ)
          + W_h · Σ_j a_ij [ τ(t_j) ⊕ e(m_j) ]    (SLA: LÚC NÀO + TỪ ĐÂU của chính chỗ đã nhìn)

- τ(t): mã thời gian trên **đồng hồ chung của hình và tiếng** (TMRoPE). Vài thành phần sin/cos tần số thấp, để hiệu hai
  thời điểm (trước/sau, cách bao xa) là phép tuyến tính.
- e(m): one-hot nguồn (hình / tiếng / chữ).
- W_h **khởi tạo bằng 0**, nên lúc khởi tạo mô hình y hệt bản gốc. Chỉ áp ở các lớp nơi bằng chứng đi vào câu hỏi
  (khoảng 13–24). Vài trăm nghìn tham số.
- Huấn luyện nhẹ trên dữ liệu **tự sinh** (không phải SVHalluc). Tín hiệu huấn luyện buộc các head *nhìn đúng chỗ* (kể cả
  với hành động tinh vi) **và** *báo lại chỗ đã nhìn*.

**Vì sao tối giản mà đủ:** câu hỏi quan hệ (trước/sau/cùng lúc) trở thành **so sánh hai trọng tâm thời gian** do hai head
báo lại. Câu hỏi "thấy hay nghe" trở thành đọc **khối lượng theo nguồn**. Xáo trộn khung hình làm đổi trọng tâm, nên mô
hình hết "mù thứ tự". Đây là các phép tuyến tính đơn giản trên thông tin vốn đã nằm trong attention (mục 2: đọc được
98–100% với sự kiện nổi bật).

**Tính mới (đã kiểm tra, mục 7):**
- Về toán học, Σ a·[τ⊕e] tương đương cộng mã nguồn gốc vào value. Ý "vị trí trong value" đã có cho văn bản (Shaw 2018 —
  tương đối giữa token; RoVE 2606.11275 — xoay value).
- **Khác:** (i) là **nguồn gốc của bằng chứng trên đồng hồ chung đa nguồn** (thời gian tuyệt đối dùng chung cho hình và
  tiếng + mã nguồn), không phải vị trí tương đối giữa token văn bản; (ii) đặt ra để **chữa mù thời gian và nhầm nguồn** với
  chẩn đoán cơ chế đi kèm; (iii) đọc ra **tường minh** qua phép chiếu khởi tạo bằng 0, chỉ ở lớp giữa.
- Không tìm thấy công trình nào làm điều này cho video LM. Các công trình gần: CoPE (vị trí theo ngữ cảnh, văn bản),
  OMCAT (time token), TimeLens (timestamp dạng chữ), MultAttnAttrib / Look Twice / Witness Evidence (dùng attention để quy
  nguồn hoặc làm nổi bật bằng chứng, không đưa trở lại biểu diễn).
- **Bắt buộc có ablation:** SLA so với "RoVE cho video", so với thời gian tuyệt đối dạng chữ, và bỏ mã nguồn.

**Chứng minh trước khi huấn luyện (không huấn luyện):**
- G0: nguồn gốc nằm trong trọng số attention nhưng không đọc tuyến tính được từ residual.
- G1: khi có nguồn gốc đúng của mốc và của đích (đích lấy bằng đọc cô lập từng vùng, tức "soi từng chỗ"), câu hỏi thời gian
  giải được gần như trực tiếp: đã thấy TF 93,8%; cần xác nhận trên mẫu mới cho cả 3 task.

Bản đầy đủ (huấn luyện SLA) chỉ làm sau khi G0 và G1 đạt.

---


> Tài liệu chốt hướng nghiên cứu, viết sau khi tổng hợp mọi thí nghiệm trong repo này và khảo sát tài liệu liên quan
> (tháng 10/2026). Mọi con số có nguồn ở `README.md` và `results/`.

---

## 1. Tóm tắt một đoạn

Video LM hiện nay **thấy đúng** (khoảng 90% khi hỏi riêng từng khoảnh khắc) và **biết bên trong lúc nào sự kiện xảy ra**
(attention định vị được 98–100%). Nhưng chúng **trả lời các câu hỏi về thời gian ở mức may rủi**. Ví dụ: xáo trộn khung hình
mà đáp án không đổi; nhầm điều *nghe* thành điều *thấy*.

Chúng tôi chỉ ra nguyên nhân cơ chế chung:

> **Attention đọc được *cái gì*, nhưng làm mất *lúc nào* và *từ nguồn nào*.**

Thời gian (RoPE / TMRoPE) chỉ được dùng để *quyết định nhìn vào đâu* (điểm q·k). Thứ được ghi lại vào biểu diễn của câu
hỏi là tổng có trọng số của **value**, mà value chỉ chứa nội dung. Sau khi đọc xong, mô hình biết "có cảnh đổ sốt", nhưng
**không còn biết cảnh đó ở thời điểm nào và đến từ hình hay từ tiếng**. Con người thì nhớ sự kiện theo bộ ba
*cái gì – lúc nào – ở đâu*: đó chính là định nghĩa **trí nhớ sự kiện** (episodic memory, Tulving) và **giám sát nguồn**
(source monitoring, Johnson).

**Episodic Attention** là một thay đổi nhỏ trong attention: mỗi lần đọc trả về nội dung **kèm nguồn gốc**, tức thời điểm
và nguồn của thứ đã đọc, được biểu diễn **tương đối với mốc mà chính câu hỏi chọn**.

---

## 2. Thống kê những gì đã biết (từ thí nghiệm của chúng tôi)

| Hiện tượng | Số liệu | Nguồn |
|---|---|---|
| Tri giác từng khoảnh khắc **đúng** | TA 90,7%, TF 89,4% khi tự nghe rồi nhìn từng nửa (base 46,5 / 29,5) | J1, n=218 |
| Nghe đúng nội dung lời nói | 96% | rootcause |
| Bên trong mô hình **biết lúc nào** | Attention của câu hỏi lên audio chọn đúng đoạn lời nói 97–100%; IoU 0,83–0,88, vẫn giữ khi có nhiễu | J5, J8, P1 |
| Biết lúc nào ngay cả với âm thanh không phải lời nói / với cảnh | 86–91% (khi có PMI); hình vuông đỏ trúng (1 mẫu) | J9, debug |
| **Không dùng được thời gian** | Dời sự kiện chỉ đổi 7–21% đáp án; báo thời gian bằng chữ: 50 → 50; chỉ cho số giây: 50 / 33 | timing, anchor_text, ceiling |
| **Nhầm nguồn** (nghe thành thấy) | Chọn đúng thời điểm nhưng giữ audio: CMTB 59; tắt audio: 92–94. Xảy ra cả khi suy luận bằng chữ | J8, STMR |
| Thông tin thời gian **có nhưng không ra đầu ra** | Probe lớp 21: 74,5 / 83,3 so với mức đa số 65,3 / 54,2; đầu ra ở mức may rủi | J1 |
| Bằng chứng đi vào token câu hỏi ở **lớp giữa** | Patching: lớp 15–24. Định tuyến có tác dụng nếu bắt đầu từ lớp 13; từ lớp 21 thì quá muộn | rootcause, debug |
| Prior ngôn ngữ đè bằng chứng | TF bị thì của lời nói điều khiển (TTS: +0,74); TA luôn "No" | tense, rootcause |
| Đã **loại trừ** | Thứ tự nhân quả (J4); gắn kết cục bộ (binding lab); im lặng là lối tắt (J5); mô hình nhỏ (7B mắc y hệt) | J4, binding, J5, 7B |
| Khi có đủ **"lúc nào" của cả hai sự kiện** thì quan hệ thời gian gần như hiển nhiên | TF 93,8% chỉ bằng so sánh hai con trỏ | J11 v2 + P1 |
| Con trỏ đích **không** đọc được bằng lượng attention | ≈ mức may rủi | P1 |
| Đọc cô lập theo thời gian và nguồn chạy song song được trong **1 lượt** | Khớp với đọc riêng từng vùng: tương quan 0,92–1,00 | P3 |

**Bức tranh chung trong tài liệu khớp với các kết quả trên.** Xáo trộn khung hình không ảnh hưởng (2505.14321); TVBench,
Vinoground, TimeBlind và AVTrace đều quanh mức may rủi. Thông tin thời gian đạt đỉnh ở lớp giữa rồi phai dần (TAI,
2610.01595). Đường chính để hiểu video là "câu hỏi truy xuất video" (CVPR 2026); video được đưa vào câu hỏi ở lớp giữa
(Map the Flow, ICLR 2026).

---

## 3. Giả thuyết trung tâm và vì sao nó giải thích mọi thứ

**Giả thuyết:** trong video LM, *nguồn gốc* của bằng chứng (thời điểm, nguồn) chỉ tồn tại trong **trọng số attention** và
không bao giờ được ghi vào **dòng residual**.

| Hiện tượng | Giải thích bằng giả thuyết |
|---|---|
| Biết lúc nào (attention đúng) nhưng trả lời sai | Thông tin nằm ở trọng số, không nằm ở biểu diễn mà đầu ra đọc |
| Xáo trộn khung hình không đổi đáp án | Tổng có trọng số của nội dung không phụ thuộc thứ tự |
| Nghe thành thấy | Value không mang nhãn nguồn |
| Tín hiệu thời gian "phai" ở các lớp sau (TAI) | Không được ghi tường minh nên không được duy trì |
| Câu hỏi quan hệ (trước/sau) thất bại | Cần so sánh *hai* thời điểm, mà mô hình không giữ được thời điểm nào |
| Báo thời gian bằng chữ không giúp | Thông tin thời gian tồn tại; cái thiếu là **gắn** nó với bằng chứng đã đọc |

**Các dự đoán có thể bác bỏ** (cần kiểm tra trước khi tin):
- P1. Từ trọng số attention đọc được thời điểm / nguồn của bằng chứng gần như hoàn hảo. Từ residual của câu hỏi sau khi đọc
  thì kém hơn hẳn, ở **mọi** benchmark thời gian và **mọi** mô hình.
- P2. Đưa nguồn gốc vào (dù thô) làm tăng mạnh các câu hỏi thời gian và **không** ảnh hưởng câu hỏi tĩnh.
- P3. Mức tăng lớn nhất ở câu hỏi quan hệ (trước/sau, thứ tự), vì cần hai nguồn gốc.

---

## 4. Phương pháp: Episodic Attention

### 4.1 Nguyên lý (một câu)
> Mỗi phép đọc của attention trả về **(cái gì, lúc nào, từ đâu)**. *Lúc nào* được biểu diễn **tương đối với mốc mà chính
> câu hỏi chọn**, không bằng số tuyệt đối.

### 4.2 Bản kiến trúc (cho video LM đời mới; thay đổi nhỏ, đặt trong attention)
Với các token media j có thời điểm t_j và nguồn m_j (hình hoặc tiếng):

1. **Mã nguồn gốc trong value:** v_j' = v_j + E_src(m_j) + E_time(t_j − t̂_q).
   - t̂_q là **mốc của câu hỏi**: thời điểm trung bình theo attention của chính câu hỏi ở một lớp giữa (đã chứng minh đọc
     được chính xác, mục 2).
   - E_time là mã quan hệ trơn: trước / trùng / sau, và khoảng cách.
   - Vì vậy đầu ra attention tự động là "nội dung + *nó xảy ra trước/trong/sau mốc* + *thấy hay nghe*".
2. **Khởi tạo bằng 0** cho các phép chiếu E_src, E_time, nên lúc khởi tạo mô hình y hệt bản gốc. Không gây hại theo cấu
   trúc, giống cách residual/LayerNorm được thêm vào.
3. Chỉ áp ở các lớp nơi bằng chứng đi vào câu hỏi (khoảng lớp 13–24). Chi phí: vài phép cộng mỗi lớp, vài nghìn tham số.
4. **Huấn luyện nhẹ:** LoRA + các bảng mã nhỏ, trên dữ liệu **tự sinh, không gán nhãn tay**: ghép clip có thời điểm biết
   trước, chèn âm thanh hoặc sự kiện hình ở thời điểm biết trước, câu hỏi tổng quát về thứ tự / trước–sau / cùng lúc /
   nguồn. **Không dùng dữ liệu SVHalluc.**

**Khác công trình gần nhất ở đâu:**
- *RoPE / VideoRoPE / HoPE / PAS:* chỉ thay đổi *chỗ nhìn* (q·k). Episodic Attention thay đổi *thứ được mang về* (value).
- *RoVE (2606.11275), Shaw 2018:* đưa vị trí *tương đối giữa token* vào value, cho văn bản. Ở đây là **nguồn gốc của bằng
  chứng đa nguồn** (thời điểm tương đối với mốc do câu hỏi chọn + nguồn), kèm chẩn đoán vì sao video LM cần nó.
- *TAI (2610.01595):* bơm lại một vector "chênh lệch thời gian" chung. Ở đây gắn nguồn gốc **vào từng mẩu bằng chứng**.
- *Timestamps dạng chữ / NumPro:* thời gian là ký hiệu bên cạnh; mô hình nhỏ không thao tác được.
- *TG-heads, FV-Action, TFO:* đọc nhiều lượt hoặc cắt clip; không sửa cơ chế của mô hình.

### 4.3 Bản không huấn luyện (để **chứng minh trước** trên mô hình đóng băng)
Mô hình đóng băng chưa biết đọc mã nguồn gốc trong value. Vì vậy ta **tạo nguồn gốc bằng chính cơ chế của nó** và đưa
lại cho nó ở dạng nó hiểu được:

1. **Mốc (lúc nào):** attention của câu hỏi ở lớp 12, khử nổi bật bằng PMI so với câu hỏi trung tính. Prior trung tính
   được lưu cache cho mỗi video.
2. **Đọc có nguồn gốc:** một lượt duy nhất, gồm các phép đọc cô lập theo **(vùng thời gian quanh mốc × nguồn)** đặt đúng vị
   trí thời gian. Kết quả của mỗi phép đọc được gắn sẵn nhãn *lúc nào* (trước/trong/sau mốc) và *từ đâu* (thấy/nghe).
3. **Trả lời từ bằng chứng có nguồn gốc:** mô hình nhận các mẩu bằng chứng dưới dạng **quan hệ có tên** ("thấy X *trước* mốc";
   "nghe Y *trong* mốc") và tự suy luận. Câu hỏi quan hệ thì suy luận có chủ ý.

Bản này **không phải phương pháp cuối cùng**. Nó là phép kiểm tra rằng *khi nguồn gốc được trả lại cho mô hình, các câu
hỏi thời gian được giải*. Nếu bản này đạt gần trần trên SVHalluc, giả thuyết đủ mạnh để huấn luyện bản kiến trúc.

### 4.4 Vì sao tổng quát (không bị nặn theo SVHalluc)
- Không có khái niệm "lời nói", "Past/Present/Future" hay "hành động được nhắc tới" trong cơ chế. Mốc là **bất kỳ thứ gì
  câu hỏi chú ý tới** (âm thanh, cảnh, vật thể). Nguồn là hình/tiếng; với mô hình chỉ có hình thì chỉ có thời gian.
- Áp được cho mọi câu hỏi cần thời gian: thứ tự hành động, trước/sau, đếm, tốc độ/hướng, định vị, đồng bộ AV, và câu hỏi
  mở.
- Với câu hỏi tĩnh, phần nguồn gốc không thay đổi gì (khởi tạo bằng 0 / mốc trải đều), nên không gây hại.

---

## 5. Kế hoạch kiểm chứng (có cổng dừng)

| Cổng | Việc | Tiêu chí qua |
|---|---|---|
| G0 | Kiểm tra P1 (nguồn gốc nằm ở trọng số, không nằm ở residual) trên SVHalluc | Chênh lệch rõ |
| G1 | Bản không huấn luyện trên **vài mẫu** SVHalluc (cả TA/TF/CMTB, các mẫu chưa dùng để thiết kế) | ≈ 100% |
| G2 | **Toàn bộ** SVHalluc temporal (protocol chính thức + protocol xoay) | Gần trần, vượt SOTA rõ |
| G3 | Benchmark khác (TVBench, TempCompass, Vinoground, Daily-Omni, AVHBench) và mô hình khác (Qwen2.5-VL, Qwen2.5-Omni-7B…) | Tăng nhất quán, không hại câu hỏi tĩnh |
| G4 | Bản kiến trúc (huấn luyện nhẹ trên dữ liệu tự sinh) | Bằng hoặc hơn bản không huấn luyện, chi phí ≈ base |

## 6. Rủi ro (nói trước)
- Bản không huấn luyện cần nhiều bước (tốn lượt forward). Nó chỉ là bằng chứng, không phải sản phẩm cuối.
- Ở bước "trả lời từ bằng chứng có nguồn gốc", mô hình 3B suy luận quan hệ còn yếu (TF cần suy luận có chủ ý). Đó chính là
  lý do bản kiến trúc đưa nguồn gốc vào **biểu diễn** thay vì qua lời.
- 100% trên SVHalluc không thể bảo đảm (nhãn TF dựa trên "gần/xa", có clip mơ hồ). Mục tiêu nghiêm túc: gần trần và vượt
  SOTA rõ rệt.
- Bản kiến trúc cần A100 trong một khoảng ngắn.
