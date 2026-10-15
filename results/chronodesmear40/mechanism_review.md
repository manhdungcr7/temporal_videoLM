# Kiểm tra giả thuyết ChronoDeSmear trước khi xem toàn bộ kết quả

2026-10-15. Những nhận xét dưới đây đến từ công thức/code, không dựa vào chọn
nhãn đúng/sai để đổi phương pháp. Lượt tái lập giữ nguyên các phép tính đề xuất.

**Hệ số .55 bị triệt tiêu bởi normalization của recurrence.** Với một row delta
d khác0, proposal center xong nhân.55, rồi tiêm
beta*||h||/||.55*d||*(.55*d) = beta*||h||/||d||*d.
Vì vậy .55 không giảm độ lớn perturbation, trừ trường hợp epsilon clamp ở norm
rất nhỏ. Mean-centering vẫn đổi hướng d, nhưng attenuation không còn tác dụng.

**Zero mean trước injection không giữ zero mean sau injection.** Mỗi token có
scale khác nhau ||h_i||/||d_i||; tổng delta chuẩn hóa có thể khác0. Kể cả mean0
thì nonlinear downstream không bảo đảm prior nhãn bất biến. Token mean còn
khác với mean theo choice vì các lựa chọn có độ dài khác nhau.

**Mã C2 khác biểu thức nền tĩnh đơn giản.** D=.5*(H_ds+H_base)-H_staticA+
.35*(H_Vall-H_staticA). Nó không chỉ H_base-H_staticA và đã phụ thuộc de-smear
ở lượt trích delta. Gain C4 khác gain C2; C6 chỉ đối chứng C2. Do đó không gọi
matrix hiện tại là orthogonal factorial.

**Không có bảo đảm exact background cancellation.** Không thể khai triển f
phi tuyến thành V_in+V_out+A với hệ số1 ở mọi nhánh. Các nhánh còn thay đổi
attention normalization, nguồn context và mọi hidden state sau đó. Delta có
thể chứa thay đổi readout hoặc distribution shift, không chỉ event evidence.

**C3 không kiểm tra riêng việc cộng lại V_out.** Thay H_base bằng H_Vall trong
nhánh .5*(...+H_sync)-H_anti rồi blend với delta khác không tương đương cộng
.5 V_out dưới model đã được chứng minh. Nó còn đổi de-smear tại pass cuối.

Ngay cả trong toy model tuyến tính mà chính đề xuất dùng, đặt Base=Vin+Vout+A,
Vall=Vin+Vout, Sync=Vin, Anti=Vout+A thì:

```
old_C2 = .5*(Base+Sync)-Anti = Vin-.5*Vout-.5*A
new_C3_subbranch = .5*(Vall+Sync)-Anti = Vin-.5*Vout-A
new_C3_subbranch-old_C2 = -.5*A
```

Thay Base bằng Vall không bù Vout trong toy model; còn trừ thêm audio. Blend
với delta stratified có thể đổi kết quả, nhưng không chứng minh đúng giải thích
"bù +.5Vout". Cũng với H_ds=H_base trong toy model, raw strat delta thực tế
trở thành 1.35*Vdyn-.35*A, không phải chỉ+Vdyn hoặc audio cancellation sạch.
Đây là kiểm tra tính nhất quán của giải thích trong giả định đơn giản, không
khẳng định transformer thật tuyến tính hoặc xác định hệ số ngữ nghĩa của nó.

**Static selection không có oracle sự kiện.** 45% low-innovation mỗi bin chỉ
giữ số token phân tầng, không đảm bảo mỗi timestamp/event được giữ. Camera
motion có thể high-innovation, hành động chậm có thể low-innovation. Không
suy ra zero temporal bias hoặc đúng background từ5bins.

**Resonance không phải grounding validation.** Hàm có tên shift diagnostics
không thực hiện shift. Nó so ba visual prototypes với audio prototype toàn
clip bằng cosine, không annotations independent. Các con số chỉ là độ tương
hợp representation, không xác nhận event/time/source đúng.

**Causal history có thể hữu ích.** [Causality Matters](https://ojs.aaai.org/index.php/AAAI/article/view/37856)
phân tích temporal cues qua inter-frame attention rồi query trong các VideoLM.
Suy luận của chúng tôi: high-pass bỏ context có thể làm hại; cần đo chứ không
đồng nhất mọi accumulation với hallucination. Paper không chứng minh cơ chế
riêng trên Qwen3B/jointAV của chúng ta.

[Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1) mô tả
time-interleaving2s, visual trước/audio sau trong chunk và TMRoPE. Causal token
order không đồng nghĩa chính xác physical time order giữa mọi cặp token AV.
Native Past0/5/Future4/5 không đủ xác định nguyên nhân là causal echo.

**Option centering chưa phải cơ chế free-form.** C2/C3/C5 cần vùng choices
để center (C4 chỉ chọn span stem). Không router theo TA/TF không đồng nghĩa
độc lập định dạng câu hỏi. Khi người dùng không đưa choices, thuật toán phải
định nghĩa lại phép center hoặc bỏ thành phần; đó là biến thể chưa được kiểm
chứng. Harness mọi suite chấm ABC không tự làm mọi cơ chế benchmark-specific:
C1, N2 pure delta, layer ensemble và last-token MLP/attention gating có định
nghĩa không cần choices. Nhưng option mean-centering là thành phần thuật toán,
không chỉ scoring wrapper, nên không claim nguyên bản này đã phủ QA tự do.
