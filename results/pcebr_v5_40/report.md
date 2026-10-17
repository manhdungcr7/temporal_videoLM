# OB-TVOR v5 — đánh giá đủ frozen research40, 2026-10-16

**Native: 18 đúng/22 sai (45.0%). V5: 19 đúng/21 sai (47.5%). Sửa được 1 câu, làm sai thêm 0 câu; đổi prediction ở 1/40 mẫu.**

Đúng bộ 40 mẫu đã cố định: mỗi nhóm task/gold có 5 mẫu, cả video và decoded audio ≤45 giây. Hai server lần này chạy hai shards 20 mẫu; không phải hai replicas của cùng 40 mẫu. Có đủ 110 cyclic option rotations, cộng log-softmax answer-letter scores sau ánh xạ về nội dung lựa chọn gốc, đúng protocol các bảng 40 mẫu trước.

Đường inference giữ nguyên v5: grid interpolation, query-conditioned top 15% visual / top 25% audio patch pooling, VO block 18, vector content/order với tan(theta)=.85, gate không sàn, tangent update .045×gate ở blocks14/16/18/20/22/24. Hàm run_routed được lấy trực tiếp từ AST source gốc, không viết lại công thức. Tính vector riêng cho từng rotation, không giả định invariant khi đổi choices.

Không có huấn luyện, tuning thresholds, thay samples hoặc chọn phương pháp theo gold. Các model M1 và classifier M2 không được dùng trong hàm M3 gốc, nên kết quả QA này không phải áp dụng classifier B8 để tự chọn đáp án. Chưa chạy generation tự do trên cả 40 mẫu hoặc reserve400. Đây vẫn là bộ development đã được xem qua nhiều vòng.

## Đúng/sai theo task và đáp án chuẩn

| Task/gold | N | Native đúng | Native sai | V5 đúng | V5 sai |
|---|---:|---:|---:|---:|---:|
| TA/A | 5 | 5 | 0 | 5 | 0 |
| TA/B | 5 | 3 | 2 | 3 | 2 |
| TF/A | 5 | 0 | 5 | 0 | 5 |
| TF/B | 5 | 1 | 4 | 1 | 4 |
| TF/C | 5 | 4 | 1 | 4 | 1 |
| CMTB/A | 5 | 3 | 2 | 3 | 2 |
| CMTB/B | 5 | 2 | 3 | 2 | 3 |
| CMTB/C | 5 | 0 | 5 | 1 | 4 |
| **Tổng** | **40** | **18** | **22** | **19** | **21** |

TA: A=Yes,B=No. TF: A=Past,B=Present,C=Future. CMTB A/B/C là các hành động khác nhau theo từng câu hỏi, không có một nghĩa cố định cho mỗi chữ.

| Task | Native đúng/tổng | V5 đúng/tổng |
|---|---|---|
| TA | 8/10 | 8/10 |
| TF | 5/15 | 5/15 |
| CMTB | 5/15 | 6/15 |

TF prediction counts: `{"native": {"B": 5, "C": 9, "A": 1}, "v5": {"B": 5, "C": 9, "A": 1}}`. Không đánh giá bias Present chỉ từ accuracy tổng; xem phân bố predictions và các nhóm gold ở trên.

## Những câu đổi đáp án

| ID | Task/gold | Native | V5 | Kết quả |
|---|---|---|---|---|
| 1898 | CMTB/C | A | C | Cứu |

ID 1898: native top-2 score gap=8.94069672e-08; v5 gap=1.01562504. Scores đầy đủ ở summary.json và per_sample.json.
Native gần như hòa điểm hai lựa chọn; kết quả đổi argmax ở đây cần được diễn giải như phá hòa, chưa đủ để chứng minh khôi phục suy luận thời gian.

## Rotation 0 riêng, không gộp với score chính

| Cấu hình | Đúng | Sai | Tổng |
|---|---:|---:|---:|
| native | 23 | 17 | 40 |
| v5 | 18 | 22 | 40 |

Rotation0 riêng: cứu 1 câu, làm sai thêm 6 câu. IDs được cứu=[2252]; IDs làm sai thêm=[2166, 2223, 1660, 1964, 2007, 2336].

Score rotation 0 được báo cáo để nối với 8 micro Stage-0. Score chính là tổng các rotations, do đó một sample có thể đổi prediction khi gộp rotations. Không trộn hai kiểu đánh giá khi so gain.

## Tất cả mẫu

| ID | Task/gold | Native | V5 | Native đúng? | V5 đúng? |
|---|---|---|---|---|
| 1427 | TF/A | B | B | 0 | 0 |
| 1449 | TA/A | A | A | 1 | 1 |
| 1457 | TA/A | A | A | 1 | 1 |
| 1513 | TF/A | B | B | 0 | 0 |
| 1528 | TA/A | A | A | 1 | 1 |
| 1584 | TF/B | C | C | 0 | 0 |
| 1660 | TF/B | C | C | 0 | 0 |
| 1704 | TF/A | B | B | 0 | 0 |
| 1716 | CMTB/B | B | B | 1 | 1 |
| 1733 | TF/B | C | C | 0 | 0 |
| 1758 | CMTB/C | A | A | 0 | 0 |
| 1766 | TA/B | A | A | 0 | 0 |
| 1772 | CMTB/B | C | C | 0 | 0 |
| 1821 | TF/C | C | C | 1 | 1 |
| 1830 | CMTB/A | B | B | 0 | 0 |
| 1867 | TA/A | A | A | 1 | 1 |
| 1876 | TF/A | C | C | 0 | 0 |
| 1898 | CMTB/C | A | C | 0 | 1 |
| 1929 | TF/C | C | C | 1 | 1 |
| 1952 | CMTB/A | A | A | 1 | 1 |
| 1956 | CMTB/A | A | A | 1 | 1 |
| 1964 | TF/C | C | C | 1 | 1 |
| 1980 | CMTB/B | C | C | 0 | 0 |
| 1990 | CMTB/A | C | C | 0 | 0 |
| 1997 | TA/A | A | A | 1 | 1 |
| 2007 | TF/B | B | B | 1 | 1 |
| 2054 | TA/B | A | A | 0 | 0 |
| 2073 | TA/B | B | B | 1 | 1 |
| 2109 | TA/B | B | B | 1 | 1 |
| 2150 | TF/C | B | B | 0 | 0 |
| 2166 | TF/A | C | C | 0 | 0 |
| 2216 | TA/B | B | B | 1 | 1 |
| 2223 | TF/C | C | C | 1 | 1 |
| 2242 | CMTB/C | A | A | 0 | 0 |
| 2252 | CMTB/C | A | A | 0 | 0 |
| 2264 | CMTB/B | B | B | 1 | 1 |
| 2281 | TF/B | A | A | 0 | 0 |
| 2336 | CMTB/C | B | B | 0 | 0 |
| 2355 | CMTB/B | C | C | 0 | 0 |
| 2366 | CMTB/A | A | A | 1 | 1 |

## Kiểm toán

- Manifest SHA: `0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
- Đủ 40 mẫu duy nhất, 110 rotations; mỗi cell đủ 5. Native replay max error=0.0.
- So với native lịch sử R3: 0 predictions đổi; max score drift=0.185913824. Chi tiết `[]`.
- Cả 8 micro rotation0 v5 predictions khớp Stage-0 v5. Max probability difference≈4.19088488e-08.
- Parity checks không đạt: 0/110; zero-gate rotations: 0/110; mean gate=0.946413. Các checks được ghi log, không loại mẫu theo kết quả kiểm tra.
- Mean TV giữa native/v5 mỗi rotation=0.032517, max=0.165674; 8/110 rotations đổi argmax riêng. Gate min/max=0.899726/0.967631.
- Raw logits/logprobs và aggregation được tính lại trên server và local. Hai archives và từng entry SHA đã kiểm tra; không extract. Source gốc, runner hash, raw banks/vectors và từng rotation được lưu.
- Hooks đã được tháo; CUDA khỏe ở cả hai server. Không đọc reserve400 hoặc thay nhãn benchmark.

## Nhận xét

Gain đo được là +1/40 câu (+2.5 điểm phần trăm). Đây là kết quả trực tiếp trên QA40; các accuracy 67,7/59,9% ở M2 Stage-0 là bài phân loại proxy khác, không thể thay thế score QA này.

Chỉ một câu CMTB được cứu, đúng mẫu 1898 có native gần như hòa A/C; TA và TF không có câu nào đổi prediction sau aggregation. TF/Past vẫn 0/5. Kết quả này chưa xác nhận giảm temporal hallucination có tính tổng quát. Ở thứ tự lựa chọn gốc, v5 giảm 5 câu đúng; lợi ích của cấu hình hiện tại chưa ổn định theo thứ tự lựa chọn. Không dùng gain của cyclic aggregation để bỏ qua tổn thất này.

Đánh giá tiếp cần tập trung vào các câu được cứu/làm hỏng và quan hệ event/time/source thật. Không thể suy bảo toàn cú pháp, temporal correctness hoặc CVPR acceptance chỉ từ parity, cosine hoặc sensitivity. Không chỉnh strength/gate trên bảng 40 này rồi gọi kết quả mới là held-out confirmation.

Các tệp chính: summary.json, per_sample.csv, per_sample.json, completion_audit.json, protocol_s*.json, raw_artifacts_s*.tar.gz.
