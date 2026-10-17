# Gửi Yoshua Bengio — v6 locked: còn hai điểm cần chốt trước khi chạy

Yoshua, bản cập nhật đã sửa nhiều điểm của review trước: CAOR dùng query-row de-sinking và output differential; M2 có common physical partition; C6 dùng một inner-product scorer; C8b giữ global branch, C8c là control riêng; loss innovation=1; có relation filter, norm matching cấp token, diagnostics sau balancing và ghi rõ hai-pass inference.

SHA cậu công bố cũng khớp: source đính kèm khi chuẩn hóa CRLF→LF có SHA256 `a03c8c28ba97a373a2a0824f1555496610da183b7b810e576f234274ceab6fab`. Bản byte gốc Windows có SHA `779e8281087d92a3c4d6085c645fc11431cebe73e1f47cd4f889d63fab686bc2`; đây là khác line endings, không phải nghi vấn hai phiên bản thuật toán khác nhau.

**Tôi chưa chạy source hoặc gửi job GPU. Theo chỉ dẫn người dùng, cần hỏi cậu hai điểm code/spec còn khác trước khi thực nghiệm.**

## 1. Nhánh “exact legacy v5” chưa tái hiện salience/gate của v5

Tại [dòng2412](stage0_pcebr_v6_locked_original.py:2412), code gọi nhánh là `Exact Legacy v5 Bivector State`, lấy post-block states như v5 nhưng vẫn gọi **hàm extract_query_conditioned_salient_slots của v6**. Hàm này đã thay đổi nhiều thao tác trước khi tính banks và gate:

| Thành phần | V5 source đã chạy | Hàm hiện được dùng trong nhánh legacy mới |
|---|---|---|
| Khử hướng query theo r0 | Không có | Có, hệ số.65 |
| Media H18 trước pooling | Raw H18, trừ.85×mean | Layer-normalized H18, trừ.90×mean |
| Hợp salience H18/H27 | .55/.45 | .50/.50 và trừ delimiter baseline |
| Gate/temperature | Ngưỡng.04, scale14, denominator.08 | Ngưỡng.045, scale16, thêm prominence filter |

Các thay đổi nằm ở [hàm extraction dòng2004](stage0_pcebr_v6_locked_original.py:2004). `use_legacy_v5_phi=True` chỉ đổi phần Phi/rel_scalar ở bước sau; không khôi phục banks, logits salience hay gate cũ. Vì vậy nhánh hiện tại là tangent-v5 trên evidence mới, chưa phải exact replay OB-TVOR v5 và không được mặc định cùng baseline5/8 hoặc cùng logits.

**Nhờ cậu chọn rõ:**

- Muốn exact baseline v5: dùng nguyên extraction + bivector + tangent của source v5 đã archive tại `results/pcebr_stage0_v5/stage0_pcebr_v5_original.py`, và đối chiếu logits/probabilities với archive micro8.
- Muốn giữ control hiện tại: đổi tên thành biến thể dùng v6 salience, bỏ claim exact legacy và báo cáo v5 archive riêng với đúng provenance.

Không cần đổi phương pháp CAOR chính để giải quyết điểm này, nhưng cần chốt baseline nào thực sự được so sánh.

## 2. Delta attention được tính FP32 hay mixed precision theo V?

Proposal chốt “vi phân cùng độ chính xác FP32”, nhưng [dòng2639–2642](stage0_pcebr_v6_locked_original.py:2639) là:

```python
p_nat_f = torch.softmax(logits_nat, dim=-1)
p_cf_f = torch.softmax(logits_cf, dim=-1)
dp = (p_cf_f - p_nat_f).to(vg.dtype)
delta_attn = torch.matmul(dp, vg).transpose(1, 2).float()
```

Logits/softmax là FP32, nhưng khi V của backbone là FP16 như runtime T4 đã dùng, **dp bị lượng tử hóa về FP16 trước phép nhân**, và đầu ra matmul FP16 mới được chuyển về FP32. `.float()` phía sau không khôi phục các chênh lệch xác suất đã mất khi cast. Đây không phải cùng phép tính với `(p_cf_f-p_nat_f) @ vg.float()` trong FP32. Ngay cả khi phần cứng dùng accumulation precision cao hơn, đầu vào dp vẫn đã bị cast.

Đây có thể là lựa chọn tốc độ/bộ nhớ hợp lý, nhưng chưa khớp câu mô tả FP32. **Cậu muốn giữ mixed precision và sửa spec, hay thực sự tính dp@V trong FP32 rồi cast tại output?** Nhờ chốt trước khi chạy để không tự thay đổi cường độ số học của can thiệp. No-op dp=0 có thể qua ở cả hai lựa chọn, nên zero-perturbation parity không tự xác nhận hai cách tương đương khi dp khác0.

## Góp ý thêm, không phải yêu cầu tự đổi thuật toán

- Định nghĩa W trong code và proposal hiện khớp ở phép midpoint quadrature trên elementary segments. Với mixture-uniform interval distributions, CDF difference tuyến tính từng đoạn, nhưng positive/negative parts có thể có điểm đổi dấu nằm giữa segment. Midpoint quadrature không bảo đảm exact W_plus/W_minus nếu chưa chia thêm ở các nghiệm đó. Có thể giữ feature hiện tại và gọi đúng là xấp xỉ; không suy exact Wasserstein tổng quát chỉ từ test dịch support10/12. Đây là giới hạn toán học cần ghi đúng, không phải tôi tự thay công thức trong lượt chạy.
- `zero_noop` dùng gamma của sample. Nếu gamma0, `run_caor_routed` đi vào nhánh bypass ở dòng2563; test không chứng minh kernel thực sự đã được gọi. Nên ghi invocation counts hoặc dùng gate1 riêng cho control zero-perturbation để kiểm tra được cả sample không có evidence. Đây là yêu cầu về diễn giải control, chưa có kết quả chạy cho thấy sample nào gặp trường hợp đó.
- `reverse_full_profile` đảo các slot-level terms nhưng giữ `token_innov` tại token cũ (dòng2190–2194). Theo đặc tả “đảo thứ tự K slots” có thể giữ cách này, nhưng cần gọi rõ là slot-profile reversal; không diễn giải thành physical AV reversal hoặc đảo toàn bộ bias từng token.

**Nhờ cậu xác nhận/sửa hai điểm chính trên. Tôi dừng ở preflight, chưa chạy M0/M1/M2/M3 và chưa có kết quả v6.**
