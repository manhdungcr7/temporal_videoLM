# ChronoDeSmear Round 5 trên SVHalluc semantic — kết quả (đối chiếu protocol viết trước)

Chạy trên Kaggle (2 notebook, mỗi notebook 30 mẫu), 2026-10-15. 60/60 mẫu, không lỗi, không thay mẫu. Manifest sha `52c3f399e7…`, runtime sha `fcee077d81…` giống nhau ở hai nửa.
Bộ: GSA / FGSA / CSMB, mỗi task 2 ô (A = Yes, B = No), 10 mẫu mỗi ô, seed 31337, clip <= 40 s, mọi phép xoay đáp án. Protocol: `protocol.md`. Số liệu: `table.md`, `summary.json`, `per_sample.csv`.

## Bảng (mỗi ô là đúng/sai; cộng lại bằng 10)
|Method|GSA/A|GSA/B|FGSA/A|FGSA/B|CSMB/A|CSMB/B|Đúng/60|Độ chính xác cân bằng GSA/FGSA/CSMB (trung bình)|Dự đoán A–B mỗi task|Cứu/làm sai so native (p, Holm)|
|---|---|---|---|---|---|---|---|---|---|---|
|C0 native|8/2|7/3|10/0|4/6|9/1|2/8|40|75 / 70 / 55 (**66,7**)|11-9 ; 16-4 ; 17-3|—|
|C1 causal de-smear|6/4|1/9|9/1|3/7|9/1|1/9|29|35 / 60 / 50 (48,3)|15-5 ; 16-4 ; 18-2|4/15 (p=0,019; Holm 0,134)|
|C2 stratified dynamic|7/3|2/8|6/4|3/7|9/1|5/5|32|45 / 45 / 70 (53,3)|15-5 ; 13-7 ; 14-6|9/17 (p=0,169; Holm 0,592)|
|C3 balanced modal|9/1|0/10|6/4|4/6|8/2|7/3|34|45 / 50 / 75 (56,7)|19-1 ; 12-8 ; 11-9|11/17 (p=0,345; Holm 0,690)|
|C4 stem + media|10/0|1/9|5/5|1/9|9/1|1/9|27|55 / 30 / 50 (45,0)|19-1 ; 14-6 ; 18-2|4/17 (p=0,007; Holm 0,065)|
|C5 ChronoDeSmear đầy đủ|10/0|2/8|7/3|3/7|10/0|3/7|35|60 / 50 / 65 (58,3)|18-2 ; 14-6 ; 17-3|7/12 (p=0,359; Holm 0,690)|
|C6 xáo trộn (ensemble 3 seed)|9/1|1/9|10/0|1/9|9/1|1/9|31|50 / 55 / 50 (51,7)|18-2 ; 19-1 ; 18-2|2/11 (p=0,022; Holm 0,135)|

## Đối chiếu từng điều kiện của protocol
| Ứng viên | (1) Cân bằng >= native + 5 | (2) Holm p < 0,05 | (3) Hơn đối chứng C6 >= 5 | (4) Task giảm tối đa | (5) Dồn <= 75% |
|---|---|---|---|---|---|
|C1|không (−18,3)|không|không (−3,3)|40 điểm|không (90%)|
|C2|không (−13,3)|không|không (+1,7)|30|có (75%)|
|C3|không (−10,0)|không|có (+5,0; ghép cặp p=0,65)|30|không (95%)|
|C4|không (−21,7)|không|không (−6,7)|40|không (95%)|
|C5|không (−8,3)|không|có (+6,7; ghép cặp p=0,39)|20|không (90%)|

**Không cấu hình nào thoả điều kiện (1), (2), (4).** Kết quả là **âm tính**: mọi ứng viên **thấp hơn native**, và số câu bị làm sai luôn nhiều hơn số câu được cứu (4–17 so với 12–17 ở các cấu hình tốt nhất). C1 và C4 tệ hơn native ở mức p chưa hiệu chỉnh < 0,05 (Holm 0,134 và 0,065: chưa đủ để kết luận "có hại").

## Diễn giải
- **Native đã khá tốt** ở các task này (66,7 cân bằng), nhưng nghiêng về "Yes" ở FGSA (80% dự đoán A) và CSMB (85%): ô B (No) đúng 4/10 và 2/10. Đây là điểm yếu thật của native, và là chỗ lẽ ra phương pháp có thể cải thiện; nó không cải thiện được.
- **Các ứng viên cũng không cân bằng hơn:** C3, C4, C5 còn dồn mạnh hơn native (tới 95% một chữ). Cùng kiểu "đổi chỗ sai" với kết quả ở temporal.
- **Nhìn theo đối chứng ngẫu nhiên:** C5 hơn C6 6,7 điểm cân bằng nhưng ghép cặp chỉ 8 câu so với 4 (p = 0,39), và C6 cũng thấp hơn native 15 điểm. Mọi cấu hình, kể cả hướng ngẫu nhiên có cùng norm, đều **làm hỏng** native. Phần "phương pháp nào thoát khỏi nhiễu" không có bằng chứng.
- **Mức can thiệp lớn:** C2 đổi đáp án native ở 26/60 câu, C5 ở 19/60, C6 ở 13/60. Thêm vector vào nhiều layer đổi quyết định rất nhiều nhưng không đổi theo hướng đúng.

## Kết luận
Theo đúng tiêu chí đã nêu trước, **ChronoDeSmear không "tốt" trên semantic**, nên **không có cơ sở để viết paper** từ phương pháp này. Cùng chiều với kết quả temporal (C2 21/40 so với native 18/40, p = 0,63). Không tinh chỉnh trên bộ này.

Giới hạn: 60 mẫu, một seed chọn mẫu; kiểm định ghép cặp công suất thấp (với p < 0,05 cần chênh khoảng 9 so với 2). "Âm tính" nghĩa là không có bằng chứng về lợi ích và có xu hướng gây hại, không phải chứng minh phương pháp vô ích với mọi cấu hình hay mọi bộ dữ liệu.
