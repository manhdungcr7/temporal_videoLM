Native media geometry40 — 2026-10-15

Hoàn tất40 recordings, một native forward/recording, rotation0. Không đổi inputs/model/answers. Các vector cùng recording pooled theo2s, riêng nguồn. Đây là diagnostic, không phải một method QA mới.

|Block|Source|Within-recording cosine|Between-recording cosine|LOO-centered within|LOO-centered between|
|---|---|---|---|---|---|
|0|video|0.9399|0.8548|0.6157|-0.0169|
|0|audio|0.6167|0.6078|0.0120|0.0067|
|8|video|0.9420|0.8745|0.5701|-0.0152|
|8|audio|0.7359|0.7242|0.0435|0.0073|
|16|video|0.9153|0.8464|0.4898|-0.0101|
|16|audio|0.7839|0.7700|0.0918|0.0139|
|22|video|0.9231|0.8770|0.4217|-0.0093|
|22|audio|0.8171|0.8024|0.1138|0.0112|
|30|video|0.9422|0.9128|0.3834|-0.0037|
|30|audio|0.8454|0.8284|0.1273|0.0085|
|35|video|0.8896|0.7701|0.5377|-0.0110|
|35|audio|0.7787|0.7251|0.2333|-0.0039|

Raw cosine giữa clips cao sẽ cho thấy shared representation/anisotropy là giải thích khả dĩ; cosine nội clip cao không tự đo được phần trăm static background. Bộ40 có38 nguồn gốc; between cosine loại cặp cùng nguồn, centering leave-one-origin-out với trọng số bằng nhau mỗi nguồn. Chỉ phân tích, không tạo đáp án hoặc fit theo gold.

Không có event/time annotations độc lập, nên không dùng cosine để tuyên bố grounded đúng sự kiện, xác nhận audio/video liên kết đúng thời gian, hoặc chứng minh một thành phần background đã bị loại sạch. Native decoder media states có thể chứa cross-source context.

Native replay max=0.00000000. Features40 file npz, metadata40 file json; source/runtime SHA được lưu; protocol trước chạy ở protocol.md.
