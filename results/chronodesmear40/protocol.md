# ChronoDeSmear Round5 — protocol trước thực nghiệm, 2026-10-15

Đề xuất được gọi là của Yoshua Bengio theo quy ước cuộc trao đổi; tên persona
trong file gốc không xác nhận tác giả ngoài đời hoặc bảo chứng của hội nghị.
`proposal_original.txt` là mô tả đầu, `proposal_original.py` là toàn bộ mã trong
attachment tiếp theo, được lưu nguyên văn trước khi chạy.

## Dữ liệu và mô hình

Qwen/Qwen2.5-Omni-3B, 36 decoder blocks, FP16, 2xT4 mỗi server. Đúng manifest
`research40_manifest.json`, SHA định danh bộ mẫu
`0349065d7d1d14726a90bd26891ab8279ff2ff31b32d168fdadb188e4b99a459`.
5 mỗi TA/A,B; TF/A,B,C; CMTB/A,B,C. Raw video/decoded audio<=45s. Tất cả110
cyclic answer rotations. Không sampling lại, bỏ lỗi, thay mẫu hoặc chọn method
theo gold. Đây là development đã dùng nhiều lần, không phải held-out.

Server numbering theo kết nối mới: server1=engagingly-unfried-corrie;
server2=excluding-pep-varying. Không ghi token vào artifact nghiên cứu.
Hai phiên mới không có decoder cache, nên encode lại đúng prompt native cũ:
render(stem,choices,perm)+'\nAnswer with only the letter of the best answer.'.
Không dùng cut/reencode full clip. Giữ budget max_pixels lịch sử. Native
decoder replay kiểm tra mỗi rotation (ngưỡng.12 đã dùng ở các lượt cũ); ghi
thực tế sai số và so baseline cũ từng ID, không ép baseline phải bằng18/40.

## Phương pháp và tham số giữ nguyên mã đề xuất

C0 native. C1 de-smear media tại block index8/10/12/14/16, lambda V=.26,A=.18,
lag1.8–12s, same-source mean subtraction + norm matching. C1 tự nó cần1 pass.
Selector lấy hidden22 native, normalized tokens, innovation global/local=.55/.45,
local previous window dt>.5 và<=3s, 5 bins theo min/max visual timestamps,
bottom45% mỗi bin. Bins không phải theo [0,duration] như lời mô tả.

Branches static+A, V-only, hard-sync, anti-sync đều không de-smear. Chỉ nhánh
C3 phụ thuộc energy-derived speech window (không phải VAD đã xác nhận speech).

Delta thực tế trong code, được giữ nguyên:

```
D_raw = .5*(H_ds+H_base) - H_staticA + .35*(H_Vall-H_staticA)
D = center_options(D_raw, factor=.55)
D_bal = .5*D + .5*center_options(.5*(H_Vall+H_sync)-H_anti)
```

C2 inject D không de-smear final. C3 inject D_bal + de-smear final. C4 inject
D_raw chỉ span trước lựa chọn + de-smear final, recurrence gain khác C2.
C5 inject D + de-smear final + thought từ native hidden22, scale.18, block0–2.
Thought mix V/A=.55/.45, top48,temp.65. C6 ba permutations seed42/1337/2027,
mỗi seed forward riêng, trung bình log-prob; cùng C2 (không de-smear/thought).
Lưu score từng seed, không chỉ ensemble. Gain C2/C3/C5/C6=.24/.30/.30/.24 tại
10/12/14/16; C4=.28/.35/.35/.28. Không tuning sau khi xem gold.

## Sửa kỹ thuật công khai

- Không chạy sampler<=50s, cut, unlink hoặc catch-and-skip trong mã gốc.
- Decoder-only replay từ encoded input, lm_head chỉ token cuối, use_cache=False.
- Mask theo device/layer, boolean causal triangle thay quadratic int64 indices.
- Echo means dùng prefix sums/searchsorted thay dense NxN@NxD, giữ predicate.
  Static local means cũng vector hóa; kiểm tra đối chiếu bản gốc bằng synthetic tests.
- Biên stem/options được xác định bằng exact embedding match, thay ước lượng
  độ dài token không tính assistant suffix. Span C4 là sau media đến option marker;
  có thể gồm delimiter trước lexical stem, ghi biên chính xác trong từng rotation.
- Mọi rotation lưu atomically; resume chỉ cùng manifest/runtime SHA; lỗi được
  ghi và dừng, không xóa output. Hook cleanup bằng finally.

## Những claim chưa được chứng minh / ablation chưa trực giao

C2 tên stratified nhưng dùng H_ds và H_Vall; không chỉ H_base-H_staticA. C3
đồng thời đổi delta/de-smear, C4 đổi span/gain. Không gọi cả matrix là factorial
trực giao. C6 là control cho C2, không phải full C5.

Phép cộng/trừ hidden nonlinear không triệt sạch background/audio. Subtract mean
theo token options không bảo đảm không đổi prior nhãn. Low innovation chưa là
background oracle, high innovation chưa là event oracle. De-smear không chỉ
xóa echo: có thể xóa temporal context hữu ích; cần control direction/lag.

Hàm `compute_av_resonance_and_shift_diagnostics` thực tế chỉ cosine ba vùng
visual với audio prototype toàn clip; không có audio/video shift, timestamp
annotation hoặc kiểm chứng đúng event. Chỉ báo representation similarity.
Không gọi nó là event grounding accuracy hoặc causal evidence. C5 static
counterfactual vẫn ưu tiên visual, thought55/45 cũng không invariant khi đổi nguồn.

## Đánh giá

Mỗi task/gold: đúng/sai, confusion TF, rescue/regression so C0 cùng lượt;
paired exact McNemar và hiệu chỉnh nội bộ. C6 report từng seed và ensemble.
Đối chiếu baseline cũ theo ID. Không ghép oracle theo task, không coi gain trên
development là chứng cứ tổng quát. Mục tiêu nội bộ30/40 và>=3/5 từng ô không
phải tiêu chuẩn nhận CVPR. Đo runtime nguyên suite riêng với runtime từng pass.

Tiếp nối sẽ đăng ký riêng trước chạy, ưu tiên factorial sạch và kiểm chứng
causal-time hypothesis; không hướng agent và không luật theo dạng câu hỏi.
