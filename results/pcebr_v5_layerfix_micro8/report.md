# OB-TVOR v5 — sửa VO theo từng tầng, micro8, 2026-10-16

**Native 5/8; v5 gốc 5/8; bản sửa chính F2 5/8.**

Đúng tám ID micro cũ, bốn mỗi shard, rotation0. Không dùng kết quả micro để chọn scale/layers. Giữ scale .045×gate, sáu tầng giữa14/16/18/20/22/24, salience H18/H27, patch selection, grid clock và hàm bivector gốc. Không huấn luyện thêm, không chạy QA40/reserve400 trong lượt này. Hai server chạy các mẫu khác nhau, không phải replicas.

Proposal không nhất quán: phần lập luận yêu cầu T_VO của chính mỗi tầng, nhưng lời nhắn cuối ghi T_VO tầng18. Lượt này kiểm chứng cách hiểu từng tầng. Không coi việc sửa code là bằng chứng chẩn đoán lệch không gian là nguyên nhân duy nhất.

## Các nhánh đã cố định trước khi chạy

- F1: dùng banks cũ từ output block18; thay VO18 bằng VO của mỗi tầng đích, tạo một vector riêng cho mỗi tầng; giữ hook cuối block.
- **F2, bản sửa chính:** lấy media states ở input của chính mỗi tầng đích, trước input_layernorm; giữ cùng patch indices và phép residual pooling gốc; dùng LN/V/O của tầng đó; tạo vector riêng và giữ hook cuối block.
- F3: dùng nguyên vectors F2, nhưng tangent update residual sau attention, trước MLP. Thực hiện bằng hook attention output và delta tương ứng; đây là diễn giải vị trí can thiệp khác, được báo cáo riêng.
- F2 reversed/content-only/random: kiểm tra kênh thời gian, nội dung và một đối chứng vector ngẫu nhiên với cùng strength/layers/rows. Random chỉ một fixed seed idx+101, không phải mean ba seeds như control Stage-0 cũ.
- F2 late/reversed: tương tự với sáu tầng28–33; chấm sensitivity, không tự coi sensitivity là correctness.
- Zero-scale và v5-original replay: kiểm tra native và cấu hình cũ khớp trước khi so bản sửa.

VO ở đây vẫn là phép biến đổi vector tổng hợp qua LN/V/O và RepeatGQA, không tái tạo attention weights theo query/head. Pooling và LN không giao hoán nói chung; không gọi nó là exact native-attention output. Các banks lấy từ native prepass, không được tính lại sau từng steering update. M1/M2 chưa có kết nối QA trong proposal, nên không tự thêm pipeline learned mới.

## MC: mọi cấu hình

| Cấu hình | Đúng | Sai | Cứu so native | Làm sai thêm | Mean TV/native |
|---|---:|---:|---|---|---:|
| native | 5 | 3 | [] | [] | 0.000000 |
| v5_original | 5 | 3 | [] | [] | 0.020530 |
| F1_layer_vo18banks_block | 5 | 3 | [] | [] | 0.029575 |
| F2_layer_inputbanks_block | 5 | 3 | [] | [] | 0.048734 |
| F2_reversed | 5 | 3 | [] | [] | 0.018807 |
| F2_content_only | 5 | 3 | [] | [] | 0.038576 |
| F2_random | 5 | 3 | [] | [] | 0.021053 |
| F2_late | 5 | 3 | [] | [] | 0.009136 |
| F2_late_reversed | 5 | 3 | [] | [] | 0.010155 |
| F3_layer_inputbanks_attention_residual | 5 | 3 | [] | [] | 0.053195 |
| zero_scale_replay | 5 | 3 | [] | [] | 0.000000 |

## Từng mẫu

| ID | Task/gold | Native | V5 gốc | F1 | F2 chính | F3 attention | F2 đảo | F2 content | F2 random |
|---|---|---|---|---|---|---|---|---|---|
| 1449 | TA/A | A | A | A | A | A | A | A | A |
| 2150 | TF/C | A | A | A | B | B | A | B | A |
| 1929 | TF/C | C | C | C | C | C | C | C | C |
| 2054 | TA/B | A | A | A | A | A | A | A | A |
| 1956 | CMTB/A | A | A | A | A | A | A | A | A |
| 1513 | TF/A | B | B | B | B | B | B | B | B |
| 2264 | CMTB/B | B | B | B | B | B | B | B | B |
| 1821 | TF/C | C | C | C | C | C | C | C | C |

## Đúng/sai theo nhóm

| Task/gold | N | Native đúng/sai | V5 gốc đúng/sai | F2 đúng/sai |
|---|---:|---|---|---|
| TA/A | 1 | 1/0 | 1/0 | 1/0 |
| TA/B | 1 | 0/1 | 0/1 | 0/1 |
| TF/A | 1 | 0/1 | 0/1 | 0/1 |
| TF/B | 0 | Chưa đo | Chưa đo | Chưa đo |
| TF/C | 3 | 2/1 | 2/1 | 2/1 |
| CMTB/A | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/B | 1 | 1/0 | 1/0 | 1/0 |
| CMTB/C | 0 | Chưa đo | Chưa đo | Chưa đo |

TF: A=Past,B=Present,C=Future. TA A=Yes,B=No. CMTB letters là nội dung lựa chọn của từng câu. Các cell N=0 không được suy rộng thành accuracy0%.

## Kết luận từ lượt sửa

Cả ba bản sửa F1/F2/F3 đều5/8, không cứu hoặc làm hỏng câu đúng theo MC. F2/F3 đổi ID2150 từ A/Past sang B/Present, nhưng gold là C/Future: sai→sai. F2 content-only cũng đổi đúng câu này sang B; chưa thấy lợi ích về đáp án từ kênh order. Các nhánh reverse/random/late cũng5/8, không được diễn giải là bằng chứng temporal reasoning được khôi phục.

F2 tác động lên xác suất mạnh hơn v5 gốc: mean TV/native=0.048734 so 0.020530. Mid/late swap-TV ratio≈4.195. Tác động lớn hơn hoặc sensitivity cao hơn chưa đem lại câu trả lời đúng hơn.

Lượt này sửa và kiểm chứng một sai khác triển khai cụ thể theo cách hiểu VO từng tầng. Nó chưa xác nhận lệch không gian là nguyên nhân chính gây lỗi, hoặc rằng chỉ sửa mapping/vị trí hook là đủ. Không dùng kết quả này để bác bỏ mọi phương pháp latent temporal reasoning; cũng chưa có cơ sở tăng quy mô ablation chỉ vì parity qua.

## Câu trả lời không có choices

32 tokens, greedy, cùng quy tắc bỏ options và vị trí token của v5 gốc. Chỉ native và F2 được sinh mới; văn bản v5 gốc được dẫn từ archive và ghi rõ. Giữ vectors từ prepass như đường sinh gốc.

Regex alignment: {"native": {"aligned": 6, "generic": 0, "timestamp_beyond_duration": 0, "total": 8}, "F2_layer_inputbanks_block": {"aligned": 6, "generic": 0, "timestamp_beyond_duration": 0, "total": 8}}. Đây là heuristic có các phản ví dụ phủ định/sai event đã được ghi ở Stage-0 v5; không phải semantic QA ground truth độc lập.

Đọc trực tiếp văn bản cho thấy ít nhất hai giới hạn mà số regex không phản ánh: ID1821 đổi mốc so sánh từ thời điểm speech sang thời điểm boiling liquid xuất hiện, nhưng vẫn được chấm aligned vì từ after. ID1513 vẫn nói after dù gold Past, đồng thời kết thúc ở “speech is at182.” trên clip13.074s. Số182 không có đơn vị và nằm cuối đoạn32 tokens bị cắt, nên không tự gán nhãn timestamp182giây; regex timestamp không bắt được chuỗi này. Các nhận xét này đối chiếu nội dung câu trả lời với câu hỏi/gold, chưa là annotation AV độc lập.

**ID1449 — TA/A**

Native: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can see the chickpeas being added to

F2: Yes, the narrated event of adding cooked chickpeas is happening at the same time in the video. You can see the chickpeas being added to

V5 gốc (archive): Yes, the narrated event of adding chickpeas to the pot is happening at the same time in the video. You can see the chickpeas being

**ID2150 — TF/C**

Native: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

F2: The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

V5 gốc (archive): The narrated event of "Pour in boiling water" happens right after the speech. So, if the speech starts at a certain time, the boiling water is

**ID1929 — TF/C**

Native: The narrated event of getting the spice blend together happens right after the onions are being stirred in the video. So, it's likely that the narration starts a

F2: The narrated event of getting the spice blend together happens right after the onions are being stirred in the pot. So, it's likely around the 10

V5 gốc (archive): The narrated event happens right after the onions are stirred in the pot. So, it's likely that the stirring of the onions is part of the process described

**ID2054 — TA/B**

Native: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

F2: Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

V5 gốc (archive): Yes, the narrated event is happening at the same time in the video. The person is dipping the chicken in the egg wash and then into the flour mix

**ID1956 — CMTB/A**

Native: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

F2: Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. It's a

V5 gốc (archive): Well, when the speech is heard, it's like the person is adding flour to the pot. You can see the flour being poured in. And then,

**ID1513 — TF/A**

Native: The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, it's right

F2: The event of pouring the sauce mixture over the grated potatoes and mixing it happens right after the speech. So, if the speech is at 182.

V5 gốc (archive): The narrated event of pouring the sauce mixture over the grated potatoes and mixing until everything is well - combined happens right after the speech. So, if the speech

**ID2264 — CMTB/B**

Native: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisking action in the video. So

F2: Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the whisk moving around in the bowl. It

V5 gốc (archive): Well, when the speech is heard, it seems like the person is whisking the ingredients together. You can see the hand moving the whisk around in the bowl

**ID1821 — TF/C**

Native: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

F2: The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the boiling liquid is shown in the video. So, it

V5 gốc (archive): The narrated event of putting a piece of lemon zest on the bottom of each bowl happens right after the moment when the speech is heard. So, it's

## Kiểm toán

Native logits replay max error=0.0; v5-original probability replay max error=0; zero-scale logits khớp native8/8. Native freeform khớp archive8/8.
Mỗi trong12 tầng/mẫu được tính lại VO từ bank input đã lưu và xây lại vector đúng từng tensor trên server. Max parity error=1.71363354e-07. Trọng số LN/V/O của12 tầng có SHA giống nhau giữa hai runtime.
Mean swap TV mid=0.035838, late=0.008544. Đây là sensitivity trước đảo coordinate trong vector, không phải physical AV time reversal hoặc QA accuracy trên video đảo.
Hai raw archives và từng entry được kiểm tra SHA local, không extract. Raw banks/vectors/selected indices và logits/probabilities được lưu; không archive toàn bộ activation tensor mọi tokens. Không còn forward/pre hooks; CUDA khỏe. Source gốc được giữ nguyên.
Lượt đầu lỗi device ở attention-residual branch do Accelerate chuyển hidden input qua GPU sau PyTorch pre-hook. Bản devicefix chuyển residual sang cùng device/dtype attention output. Chưa có sample nào hoàn tất chấm ở lượt lỗi; không đổi mẫu hoặc công thức. Source trước sửa và job logs được lưu tại device_error_attempt. Archive job xếp sau lượt lỗi cũng báo thiếu results; log được giữ.

Kết quả là development micro đã được xem nhiều lần. Chỉ kết luận về các cấu hình và tám mẫu này; chưa đủ bằng chứng generalization, bảo toàn ngữ nghĩa/cú pháp hoặc khả năng được accept CVPR.
