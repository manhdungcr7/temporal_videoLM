# Vòng layer-local CLC, 2026-10-17

**Nhánh chính layerlocal_native đạt 18/40 so native 23/40; cứu 0, hại 5.** Đổi sang contrast theo từng tầng chưa cải thiện QA ở protocol này. shared22_native đạt 23/40, cứu 4/hại 4; bỏ bivector giảm tác hại so công thức cũ nhưng chưa hơn baseline.

Đã chạy đủ40 câu cùng frozen manifest, rotation0, chia hai shard khác mẫu. Cùng hệ số .042×gamma và năm tầng14/16/18/20/22; không train, không tìm hệ số theo gold. Mọi nhánh cuối bỏ CAOR/CT, nhưng vẫn dùng CAOR để xây contrast. Đây là dữ liệu development đã dùng nhiều vòng.

| Cấu hình | Đúng/40 | Ngoài micro8 đúng/32 | Cứu/hại so native | McNemar p chưa chỉnh đa kiểm định |
|---|---:|---:|---:|---:|
| native | 23 | 18 | 0/0 | 1.0000 |
| CAOR_only | 20 | 15 | 0/3 | 0.2500 |
| shared22_mixed_replay | 19 | 13 | 4/8 | 0.3877 |
| shared22_native | 23 | 17 | 4/4 | 1.0000 |
| layerlocal_native | 18 | 13 | 0/5 | 0.0625 |
| layerlocal_ev | 19 | 14 | 1/5 | 0.2188 |
| layerlocal_negated | 20 | 15 | 2/5 | 0.4531 |
| random_101 | 21 | 16 | 1/3 | 0.6250 |
| random_202 | 20 | 15 | 1/4 | 0.3750 |
| random_303 | 19 | 14 | 0/4 | 0.1250 |

shared22_native và layerlocal_native dùng cùng .45 evidence + .35 order đã normalize; chỉ khác tầng lấy hướng. shared22_mixed_replay tái lập công thức cũ có .20 bivector. layerlocal_ev là ablation bỏ order; negated và random là đối chứng. Không chọn nhánh thắng sau khi xem số liệu làm claim xác nhận.

| Task/gold | N | native | CAOR_only | shared22_mixed_replay | shared22_native | layerlocal_native | layerlocal_ev | layerlocal_negated | random_101 | random_202 | random_303 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CMTB/A | 5 | 2 | 2 | 3 | 3 | 2 | 2 | 2 | 3 | 2 | 2 |
| CMTB/B | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| CMTB/C | 5 | 3 | 3 | 1 | 2 | 1 | 1 | 3 | 3 | 2 | 2 |
| TA/A | 5 | 5 | 4 | 4 | 5 | 5 | 4 | 4 | 4 | 4 | 4 |
| TA/B | 5 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 4 | 3 |
| TF/A | 5 | 2 | 1 | 3 | 3 | 1 | 3 | 3 | 2 | 2 | 1 |
| TF/B | 5 | 2 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 0 | 1 |
| TF/C | 5 | 4 | 4 | 2 | 3 | 3 | 3 | 3 | 3 | 4 | 4 |

Cosine trung bình giữa hướng local và hướng22: {"14": 0.013716647610999643, "16": 0.06369002617429942, "18": 0.21771618500351905, "20": 0.497961338609457, "22": 0.9999999836087227}. Khác hướng không tự chứng minh khác ý nghĩa temporal.

Kiểm toán: 38 origins; SHA archive/source/từng artifact khớp; source effective hai shard trùng; backbone versions không đổi; native/CAOR/CLC cũ replay sai số xác suất tối đa 0.0. No-op tối đa 0.0. Có 17 mẫu giữ full-pass fallback theo ngưỡng .02 cũ.

Các khoảng bootstrap trong summary.json lấy mẫu theo recording origin (5000 draws, seed20261017), chỉ diễn tả bất định nội bộ tập40; không khắc phục việc tái sử dụng development set hay đa kiểm định.

| ID | Gold | native | CAOR_only | shared22_mixed_replay | shared22_native | layerlocal_native | layerlocal_ev | layerlocal_negated | random_101 | random_202 | random_303 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1427 | A | B | B | A | A | B | A | A | B | B | B |
| 1449 | A | A | A | A | A | A | A | A | A | A | A |
| 1457 | A | A | A | A | A | A | B | A | A | A | A |
| 1513 | A | B | B | A | A | B | B | B | B | B | B |
| 1528 | A | A | B | B | A | A | A | A | A | A | B |
| 1584 | B | C | C | B | B | C | C | C | C | A | A |
| 1660 | B | B | C | C | B | B | B | A | A | A | B |
| 1704 | A | A | A | A | A | A | A | A | A | A | A |
| 1716 | B | B | B | B | B | B | B | B | B | B | B |
| 1733 | B | C | C | C | C | C | C | C | C | C | C |
| 1758 | C | A | A | A | A | A | A | A | A | A | A |
| 1766 | B | A | A | A | A | A | A | A | A | B | A |
| 1772 | B | C | C | C | C | C | C | C | C | C | C |
| 1821 | C | C | C | C | C | C | C | C | C | C | C |
| 1830 | A | B | B | B | B | B | B | B | B | B | B |
| 1867 | A | A | A | A | A | A | A | B | B | A | A |
| 1876 | A | C | C | C | C | C | C | C | C | C | C |
| 1898 | C | C | C | A | A | A | A | C | C | C | C |
| 1929 | C | C | C | C | C | C | C | C | C | C | C |
| 1952 | A | B | B | A | A | B | B | B | A | B | B |
| 1956 | A | A | A | A | A | A | A | A | A | A | A |
| 1964 | C | C | C | A | A | A | A | C | C | C | C |
| 1980 | B | C | C | C | C | C | C | C | C | C | C |
| 1990 | A | C | C | C | C | C | C | C | C | C | C |
| 1997 | A | A | A | A | A | A | A | A | A | B | A |
| 2007 | B | B | B | A | A | A | A | B | B | A | A |
| 2054 | B | A | A | A | A | A | A | A | A | A | A |
| 2073 | B | B | B | B | B | B | B | B | B | B | B |
| 2109 | B | B | B | B | B | B | B | A | B | B | B |
| 2150 | C | A | A | B | B | B | B | A | B | A | A |
| 2166 | A | A | B | C | C | C | A | A | A | A | B |
| 2216 | B | B | B | B | B | B | B | B | B | B | B |
| 2223 | C | C | C | A | C | C | C | B | B | C | C |
| 2242 | C | C | C | C | C | C | C | C | C | C | C |
| 2252 | C | A | A | A | A | A | A | C | A | A | A |
| 2264 | B | B | B | B | B | B | B | B | B | B | B |
| 2281 | B | A | A | A | A | A | A | A | A | A | A |
| 2336 | C | C | C | B | C | B | B | B | C | B | B |
| 2355 | B | C | C | C | C | C | C | C | C | C | C |
| 2366 | A | A | A | A | A | A | A | A | A | A | A |

Nguồn phương pháp: [Tuned Lens](https://arxiv.org/html/2303.08112v4) nêu representation drift và bias của raw logit lens trên các LM được khảo sát; không chứng minh đây là nguyên nhân lỗi Qwen. [CAA](https://aclanthology.org/2024.acl-long.828/) đã dùng residual contrasts để steering, nên phép đổi này chưa có claim novelty.

Phạm vi: chưa chạy option rotations mới, chưa chấm free-form độc lập, chưa có event/interval gold cho40, chưa chứng minh grounding hoặc triển khai tổng quát.
