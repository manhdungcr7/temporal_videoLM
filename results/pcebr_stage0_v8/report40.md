# V8 CLC-only — mở rộng frozen40, 2026-10-17

Đã lưu 40/40 mẫu. Cùng manifest frozen40, rotation 0, mỗi task/gold có 5 mẫu, mọi clip ≤45 giây. Hệ số CLC .042 giữ nguyên từ micro8; gamma theo pipeline đã sửa. Không chọn lại mẫu hoặc huấn luyện.

CLC-only được chọn sau exploratory ablation micro8 (6/8 so native5/8). Vì 40 có chứa 8 mẫu đó, báo cáo riêng 32 mẫu ngoài micro. Cả 40 là development set đã dùng trong các vòng trước, không phải test mới. Đây là decode trực tiếp: chưa chạy đầy đủ rollout bridge trên 40.

| Cấu hình | Toàn bộ đúng/sai | Ngoài micro8 đúng/sai |
|---|---|---|
| native | 23/17 | 18/14 |
| CAOR_only | 20/20 | 15/17 |
| CLC_only | 19/21 | 13/19 |
| full_direct | 16/24 | 13/19 |

| Task/gold | N | Native đúng/sai | CAOR đúng/sai | CLC-only đúng/sai | Full direct đúng/sai |
|---|---:|---|---|---|---|
| TA/A | 5 | 5/0 | 4/1 | 4/1 | 1/4 |
| TA/B | 5 | 3/2 | 3/2 | 3/2 | 4/1 |
| TF/A | 5 | 2/3 | 1/4 | 3/2 | 3/2 |
| TF/B | 5 | 2/3 | 1/4 | 1/4 | 2/3 |
| TF/C | 5 | 4/1 | 4/1 | 2/3 | 1/4 |
| CMTB/A | 5 | 2/3 | 2/3 | 3/2 | 2/3 |
| CMTB/B | 5 | 2/3 | 2/3 | 2/3 | 2/3 |
| CMTB/C | 5 | 3/2 | 3/2 | 1/4 | 1/4 |

CLC-only cứu IDs: [1427, 1513, 1584, 1952]; làm hỏng IDs: [1528, 1660, 1898, 1964, 2007, 2166, 2223, 2336].

| ID | Task/gold | Native | CLC | CAOR | Full direct | Gate |
|---|---|---|---|---|---|---:|
| 1427 | TF/A | B | A | B | A | 0.9944 |
| 1449 | TA/A | A | A | A | B | 0.9924 |
| 1457 | TA/A | A | A | A | B | 0.9913 |
| 1513 | TF/A | B | A | B | B | 0.9928 |
| 1528 | TA/A | A | B | B | B | 0.9913 |
| 1584 | TF/B | C | B | C | B | 0.9941 |
| 1660 | TF/B | B | C | C | B | 0.9958 |
| 1704 | TF/A | A | A | A | A | 0.9941 |
| 1716 | CMTB/B | B | B | B | B | 0.9944 |
| 1733 | TF/B | C | C | C | C | 0.9902 |
| 1758 | CMTB/C | A | A | A | A | 0.995 |
| 1766 | TA/B | A | A | A | B | 0.9968 |
| 1772 | CMTB/B | C | C | C | C | 0.9926 |
| 1821 | TF/C | C | C | C | C | 0.9928 |
| 1830 | CMTB/A | B | B | B | B | 0.9951 |
| 1867 | TA/A | A | A | A | A | 0.9959 |
| 1876 | TF/A | C | C | C | C | 0.9951 |
| 1898 | CMTB/C | C | A | C | A | 0.9927 |
| 1929 | TF/C | C | C | C | A | 0.9933 |
| 1952 | CMTB/A | B | A | B | B | 0.9948 |
| 1956 | CMTB/A | A | A | A | A | 0.9928 |
| 1964 | TF/C | C | A | C | A | 0.9933 |
| 1980 | CMTB/B | C | C | C | C | 0.9911 |
| 1990 | CMTB/A | C | C | C | C | 0.996 |
| 1997 | TA/A | A | A | A | B | 0.9945 |
| 2007 | TF/B | B | A | B | A | 0.992 |
| 2054 | TA/B | A | A | A | A | 0.992 |
| 2073 | TA/B | B | B | B | B | 0.9916 |
| 2109 | TA/B | B | B | B | B | 0.9946 |
| 2150 | TF/C | A | B | A | B | 0.9909 |
| 2166 | TF/A | A | C | B | A | 0.9951 |
| 2216 | TA/B | B | B | B | B | 0.9929 |
| 2223 | TF/C | C | A | C | B | 0.9927 |
| 2242 | CMTB/C | C | C | C | C | 0.9954 |
| 2252 | CMTB/C | A | A | A | A | 0.9932 |
| 2264 | CMTB/B | B | B | B | B | 0.9886 |
| 2281 | TF/B | A | A | A | A | 0.9924 |
| 2336 | CMTB/C | C | B | C | B | 0.9941 |
| 2355 | CMTB/B | C | C | C | C | 0.9966 |
| 2366 | CMTB/A | A | A | A | A | 0.9911 |

Micro8 overlap replay CLC max probability error: 0.0.

Các kết quả này kiểm chứng một ablation của V8, không chứng minh cơ chế joint binding hoặc novelty. Cần kiểm tra lựa chọn luân phiên, interval/event ground truth, câu hỏi/contexts độc lập và absence hard negatives trước khi đánh giá tổng quát hóa.

## Kiểm toán

Đủ40 ID, 38 recording origins, clip dài nhất 43.1s. Cả40 prompt tail tokenizer roundtrip khớp; backbone không đổi trên cả hai shard. Native replay max error=0.0; kernel-noop max error=0.0. 17/40 mẫu dùng full-pass fallback thay vì nới ngưỡng cache. Hai raw result archives được kiểm tra SHA local.

Wrapper mở rộng ban đầu thiếu gc trong namespace sau một mẫu/shard; đã sửa và resume các ID còn lại, giữ nguyên mẫu/công thức. Logs lỗi gốc được giữ. Native23/40 là baseline cùng lượt rotation0, không phải bảng cyclic-rotation cũ.
