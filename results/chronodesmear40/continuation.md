# Trạng thái tiếp nối — 2026-10-15

Ba suite đã được hiện thực, protocol được ghi trước chạy, jobs xếp tuần tự trên
mỗi server. Không cần tải lại/encode lại40 hoặc submit lại toàn bộ khi chat bị
ngắt. Xem JSON job state và raw records thực tế, không dựa snapshot văn bản này.

1. `chronodesmear40`: 7 cấu hình Yoshua Bengio + lưu riêng3 shuffled seeds.
2. `chronodesmear_followup40`: pure delta/factor ladder, random static control,
   source-symmetric delta, bounded same-layer feedback, fixed layer ensemble,
   native layer lens tại10 blocks.
3. `late_path40`:6 can thiệp attention/MLP muộn + native reference.

Kết nối trong remote_config.json / remote_config2.json được ignore; không ghi
token vào báo cáo. Server numbering theo URL mới trong protocol.md.

Job IDs:

|Suite|Server1|Server2|
|---|---|---|
|Cache đã xong|674ae646ec|5512812fd8|
|Round5|bbdbd08878|ccc5ead4e5|
|Factor/depth tiếp nối|c6dc612762|5850ba5726|
|Late path|db65c8367f|acefbd6e61|

Local Python dùng `.venv/Scripts/python.exe`; WindowsApps interpreter cần quyền
chạy ngoài sandbox của phiên này. Không dùng python alias trống. Watcher local
`server_jobs/chronodesmear_watch.py` đang checkpoint 3 suite; nếu dừng có thể
khởi động lại watcher mà không submit lại jobs. Nó tự tải tar raw, kiểm tra55
rotations/shard và recompute aggregate, sau đó lưu `archive_audit_s*.json`.

Report generator:

```
.venv/Scripts/python.exe server_jobs/report_chronodesmear_research.py chronodesmear40
.venv/Scripts/python.exe server_jobs/report_chronodesmear_research.py chronodesmear_followup40
.venv/Scripts/python.exe server_jobs/report_chronodesmear_research.py late_path40
```

Chỉ gọi report suite khi đủ40; report chưa có không đồng nghĩa job không chạy.
Nếu job error, đọc job_s*.json/errors_remote, sửa nguyên nhân, lưu phiên lỗi và
chỉ resume phần thiếu; đừng đổi sample/gain theo nhãn hoặc ghi đè kết quả có
runtime SHA khác. Native baseline phải đối chiếu cùng phiên. Các artifacts
copy-reference không tính là method mới hoặc inference tái lập độc lập.

CPU scientific tests:3 passed, interval mean vs dense equation, norm/causal
identity, selector vs original, mask causal/source isolation. Smoke trên mỗi
server qua1mẫu trước Round5; followup/late wrapper cũng smoke1 rồi resume full.

`mechanism_review.md` có các vấn đề trước khi xem kết quả đủ: .55 bị norm-rescale
triệt magnitude; C3 không bù V_out ngay trong toy algebra; resonance không có
shift test; matrix không orthogonal. `related_work.md` lưu nguồn chính chủ mới đọc.

Chưa gọi bất kỳ candidate nào thành công/CVPR-ready trước khi đủ số liệu và
kiểm chứng tổng quát. Cuối mỗi suite cần in bảng đúng/sai, đánh giá riêng mỗi
ý tưởng và fit với yêu cầu, rồi lưu report local.
