# Vòng giới hạn CLC vào token trả lời cuối, 2026-10-17

**Không nhánh last_ nào hơn native 23/40.** Nhánh last_shared22_native đạt 22/40 và last_layerlocal_native đạt 21/40; cả hai không cứu câu nào. Giảm phạm vi can thiệp chưa là giải pháp cho QA temporal.

Đủ40 cùng inputs/directions của vòng layer-local; giữ hệ số .042×gamma, năm tầng, không CAOR/CT ở decode cuối. Chỉ đổi phạm vi từ toàn bộ question/options/assistant sang token assistant cuối. Không train, không chọn hệ số theo gold. Cùng38 origins, development đã tái sử dụng.

| Nhánh | Đúng/40 | Ngoài micro8 đúng/32 | Cứu/hại native | McNemar p chưa chỉnh |
|---|---:|---:|---:|---:|
| native | 23 | 18 | 0/0 | 1.0000 |
| fulltail_shared22_mixed_replay | 19 | 13 | 4/8 | 0.3877 |
| last_shared22_mixed_replay | 21 | 16 | 0/2 | 0.5000 |
| last_shared22_native | 22 | 17 | 0/1 | 1.0000 |
| last_layerlocal_native | 21 | 16 | 0/2 | 0.5000 |
| last_layerlocal_ev | 21 | 16 | 0/2 | 0.5000 |
| last_layerlocal_negated | 21 | 16 | 0/2 | 0.5000 |
| last_random_101 | 22 | 17 | 0/1 | 1.0000 |
| last_random_202 | 22 | 17 | 0/1 | 1.0000 |
| last_random_303 | 20 | 15 | 0/3 | 0.2500 |

Các nhánh last_ dùng cùng directions đã lưu trước vòng này. Negated và random có cùng rows/layers/strength. Phạm vi token ít hơn cũng giảm tổng can thiệp, vì vậy không được quy mọi khác biệt cho việc sửa representation của option.

| Task/gold | native | fulltail_shared22_mixed_replay | last_shared22_mixed_replay | last_shared22_native | last_layerlocal_native | last_layerlocal_ev | last_layerlocal_negated | last_random_101 | last_random_202 | last_random_303 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMTB/A | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| CMTB/B | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| CMTB/C | 3 | 1 | 3 | 3 | 3 | 3 | 2 | 3 | 2 | 2 |
| TA/A | 5 | 4 | 4 | 4 | 5 | 5 | 5 | 5 | 5 | 5 |
| TA/B | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| TF/A | 2 | 3 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 |
| TF/B | 2 | 1 | 1 | 2 | 1 | 0 | 1 | 1 | 2 | 0 |
| TF/C | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

Kiểm toán: hai route sources trùng SHA; archives/input hashes/parent direction hashes khớp; native và CLC toàn-tail replay sai số xác suất tối đa 0.0; backbone versions không đổi. 17/40 mẫu giữ cache full-pass fallback của vòng trước.

Wrapper đầu tiên dùng gamma trước làm tròn thay vì gamma sáu chữ số trong profile của lượt gốc. Parity đã phát hiện sai khác; không nhận prediction từ lượt lỗi. Đã sửa wrapper dùng đúng profile gamma và chạy lại, giữ ngưỡng replay1e-6; log initial_replay_error.json nằm trong raw archives. Đây là lỗi wrapper đã sửa, không phải thay đổi phương pháp để nâng điểm.

Chưa có independent test, option rotations mới, interval ground truth, hoặc free-form semantic scoring. Không chọn nhánh tốt nhất trong bảng làm bằng chứng xác nhận.

| ID | Gold | native | fulltail_shared22_mixed_replay | last_shared22_mixed_replay | last_shared22_native | last_layerlocal_native | last_layerlocal_ev | last_layerlocal_negated | last_random_101 | last_random_202 | last_random_303 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1427 | A | B | A | B | B | B | B | B | B | B | B |
| 1449 | A | A | A | A | A | A | A | A | A | A | A |
| 1457 | A | A | A | B | B | A | A | A | A | A | A |
| 1513 | A | B | A | B | B | B | B | B | B | B | B |
| 1528 | A | A | B | A | A | A | A | A | A | A | A |
| 1584 | B | C | B | C | C | C | C | C | C | C | C |
| 1660 | B | B | C | B | B | A | A | A | A | B | A |
| 1704 | A | A | A | A | A | A | A | A | A | A | A |
| 1716 | B | B | B | B | B | B | B | B | B | B | B |
| 1733 | B | C | C | C | C | C | C | C | C | C | C |
| 1758 | C | A | A | A | A | A | A | A | A | A | A |
| 1766 | B | A | A | A | A | A | A | A | A | A | A |
| 1772 | B | C | C | C | C | C | C | C | C | C | C |
| 1821 | C | C | C | C | C | C | C | C | C | C | C |
| 1830 | A | B | B | B | B | B | B | B | B | B | B |
| 1867 | A | A | A | A | A | A | A | A | A | A | A |
| 1876 | A | C | C | C | C | C | C | C | C | C | C |
| 1898 | C | C | A | C | C | C | C | C | C | C | C |
| 1929 | C | C | C | C | C | C | C | C | C | C | C |
| 1952 | A | B | A | B | B | B | B | B | B | B | B |
| 1956 | A | A | A | A | A | A | A | A | A | A | A |
| 1964 | C | C | A | C | C | C | C | C | C | C | C |
| 1980 | B | C | C | C | C | C | C | C | C | C | C |
| 1990 | A | C | C | C | C | C | C | C | C | C | C |
| 1997 | A | A | A | A | A | A | A | A | A | A | A |
| 2007 | B | B | A | A | B | B | A | B | B | B | A |
| 2054 | B | A | A | A | A | A | A | A | A | A | A |
| 2073 | B | B | B | B | B | B | B | B | B | B | B |
| 2109 | B | B | B | B | B | B | B | B | B | B | B |
| 2150 | C | A | B | A | A | A | A | A | A | A | A |
| 2166 | A | A | C | A | A | C | A | A | A | A | A |
| 2216 | B | B | B | B | B | B | B | B | B | B | B |
| 2223 | C | C | A | C | C | C | C | C | C | C | C |
| 2242 | C | C | C | C | C | C | C | C | C | C | C |
| 2252 | C | A | A | A | A | A | A | A | A | A | A |
| 2264 | B | B | B | B | B | B | B | B | B | B | B |
| 2281 | B | A | A | A | A | A | A | A | A | A | A |
| 2336 | C | C | B | C | C | C | C | B | C | B | B |
| 2355 | B | C | C | C | C | C | C | C | C | C | C |
| 2366 | A | A | A | A | A | A | A | A | A | A | A |
