$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $repoRoot 'results'
$outputStem = Join-Path $resultRoot 'yoshua_bengio_all40_feedback_20261015'
$cellNames = @('TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C')
$suiteNames = @('argon_latent40','argon_ccld40','latent_reopen40','symmetric_latent40','query_latent40','latent_slot_opt40','latent_slot_read15','presentation_resample40')
$manifestPath = Join-Path $resultRoot 'research40_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$goldById = @{}
foreach ($sample in $manifest.selected) {
    if ($goldById.ContainsKey([string]$sample.idx)) { throw 'Duplicate manifest ID' }
    if ($sample.duration -gt 45 -or $sample.decoded_audio_duration -gt 45) { throw 'Duration exceeds 45s' }
    $goldById[[string]$sample.idx] = $sample
}
if ($goldById.Count -ne 40) { throw 'Manifest must have exactly 40 IDs' }
$allRows = [System.Collections.Generic.List[object]]::new()
$allCounts = [System.Collections.Generic.List[object]]::new()
$sourceFiles = [System.Collections.Generic.List[string]]::new()
$sourceFiles.Add($manifestPath)
$tables = @{}
$baselineById = @{}

foreach ($suite in $suiteNames) {
    $folder = Join-Path $resultRoot $suite
    $summaryPath = Join-Path $folder 'summary.json'
    $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $sourceFiles.Add($summaryPath)
    if ($suite -eq 'argon_latent40') { $summary = $summary.methods }
    $rows = [System.Collections.Generic.List[object]]::new()
    if ($suite -eq 'latent_reopen40') {
        foreach ($shard in @('results_s1.jsonl','results_s2.jsonl')) {
            $rawPath = Join-Path $folder $shard
            $sourceFiles.Add($rawPath)
            foreach ($line in Get-Content -LiteralPath $rawPath -Encoding UTF8) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $sample = $line | ConvertFrom-Json
                if ($sample.status -ne 'ok') { throw 'Unsuccessful D/G record' }
                foreach ($prediction in $sample.predictions.PSObject.Properties) {
                    $rows.Add([pscustomobject]@{suite=$suite;idx=[string]$sample.idx;task=$sample.task;gold=$sample.gold;method=$prediction.Name;prediction=[string]$prediction.Value;correct=[int]($prediction.Value -eq $sample.gold)})
                }
            }
        }
    } else {
        $csvPath = Join-Path $folder 'per_sample.csv'
        $sourceFiles.Add($csvPath)
        foreach ($sample in Import-Csv -LiteralPath $csvPath -Encoding UTF8) {
            if ($suite -eq 'argon_latent40') {
                foreach ($method in $summary.PSObject.Properties.Name) {
                    $prediction = $sample.$method
                    $rows.Add([pscustomobject]@{suite=$suite;idx=[string]$sample.idx;task=$sample.task;gold=$sample.gold;method=$method;prediction=$prediction;correct=[int]($prediction -eq $sample.gold)})
                }
            } else {
                if ([int]$sample.correct -ne [int]($sample.prediction -eq $sample.gold)) { throw 'Stored correctness disagrees with prediction' }
                $rows.Add([pscustomobject]@{suite=$suite;idx=[string]$sample.idx;task=$sample.task;gold=$sample.gold;method=$sample.method;prediction=$sample.prediction;correct=[int]$sample.correct})
            }
        }
    }
    foreach ($row in $rows) {
        $expected = $goldById[$row.idx]
        if ($null -eq $expected -or $row.task -ne $expected.task -or $row.gold -ne $expected.gold) { throw 'Manifest identity/gold mismatch' }
        if ($row.prediction -notin @('A','B','C')) { throw 'Invalid prediction' }
        if ($row.task -eq 'TA' -and $row.prediction -eq 'C') { throw 'TA prediction has nonexistent choice' }
        if ($row.method -eq 'R3_M0_base') { $baselineById[$row.idx] = $row.prediction }
        if ($row.method -in @('R4_C0_baseline','S0_native','Q0_native','L0_native','E0_native','P0_original_native_reference')) {
            if ($baselineById[$row.idx] -ne $row.prediction) { throw 'Native baseline differs across suites' }
        }
        $allRows.Add($row)
    }
    $table = [System.Collections.Generic.List[string]]::new()
    $table.Add('|Method|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tong dung/sai|TF du doan A/B/C|')
    $table.Add('|---|---|---|---|---|---|---|---|---|---|---|')
    foreach ($method in $summary.PSObject.Properties.Name) {
        if ($suite -eq 'latent_reopen40' -and $method -notmatch '^[DG]') { continue }
        $methodRows = @($rows | Where-Object method -EQ $method)
        if ($methodRows.Count -ne 40 -or @($methodRows.idx | Sort-Object -Unique).Count -ne 40) { throw ('Incomplete/duplicate method: '+$method) }
        $entry = [ordered]@{suite=$suite;method=$method}
        $parts = [System.Collections.Generic.List[string]]::new()
        $parts.Add($method)
        $correctTotal = 0
        foreach ($cellName in $cellNames) {
            $cellRows = @($methodRows | Where-Object { ($_.task+'/'+$_.gold) -eq $cellName })
            if ($cellRows.Count -ne 5) { throw 'Cell size differs from 5' }
            $correct = [int](($cellRows | Measure-Object -Property correct -Sum).Sum)
            $correctTotal += $correct
            $stored = $summary.$method.cells.$cellName
            if ($suite -eq 'latent_reopen40') { $storedCorrect = [int]$stored[0]; $storedN = [int]$stored[1] }
            else { $storedCorrect = [int]$stored.correct; $storedN = [int]$stored.correct+[int]$stored.wrong }
            if ($correct -ne $storedCorrect -or $storedN -ne 5) { throw ('Summary mismatch: '+$method+' '+$cellName) }
            $entry[$cellName+'_correct'] = $correct
            $entry[$cellName+'_wrong'] = 5-$correct
            $parts.Add(('{0}/{1}' -f $correct,(5-$correct)))
        }
        $storedTotal = if ($suite -eq 'latent_reopen40') { [int]$summary.$method.total } else { [int]$summary.$method.correct }
        if ($storedTotal -ne $correctTotal) { throw 'Summary total mismatch' }
        $entry.correct = $correctTotal
        $entry.wrong = 40-$correctTotal
        $entry.accuracy_percent = 2.5*$correctTotal
        $parts.Add(('{0}/{1}' -f $correctTotal,(40-$correctTotal)))
        $tfCounts = @('A','B','C' | ForEach-Object { $choice=$_; @($methodRows | Where-Object { $_.task -eq 'TF' -and $_.prediction -eq $choice }).Count })
        $entry.tf_pred_A = $tfCounts[0]; $entry.tf_pred_B = $tfCounts[1]; $entry.tf_pred_C = $tfCounts[2]
        $parts.Add(($tfCounts -join '/'))
        $table.Add('|'+($parts -join '|')+'|')
        $allCounts.Add([pscustomobject]$entry)
    }
    $tables[$suite] = (($table -join "`n").Replace('Tong dung/sai','Tổng đúng/sai').Replace('TF du doan A/B/C','TF dự đoán A/B/C'))
}

$report = @'
Tổng hợp thực nghiệm và phản hồi cho Yoshua Bengio — 2026-10-15

Tên Yoshua Bengio dưới đây là cách gọi bên đề xuất theo yêu cầu trong cuộc trao đổi. Báo cáo không xác nhận người nghiên cứu ngoài đời là tác giả hoặc đã bảo chứng các ý tưởng này. Tên thư mục/method gốc được giữ để đối chiếu artifact.

Kết quả cao nhất là D1/D2: 27 đúng, 13 sai; nhưng cả hai dự đoán Present cho toàn bộ 15 câu TF. Trong các đề xuất CCLD, C2/C7 cao nhất: 26 đúng, 14 sai. Chưa có phương pháp đồng thời tốt ở mọi ô và đáp ứng đầy đủ yêu cầu triển khai nhiều ngữ cảnh.

Quy ước: mọi ô là ĐÚNG/SAI, cộng lại bằng 5; cột tổng cộng lại bằng 40. TA:A=Yes,B=No. TF:A=Past,B=Present,C=Future. CMTB:A/B/C là nội dung lựa chọn riêng của từng câu. Cột cuối là SỐ LẦN DỰ ĐOÁN, không phải số câu đúng.

Sử dụng đúng 40 ID, 5 mẫu/ô, video và decoded audio <=45s; mọi cyclic option rotations, tổng 110 rotations/suite. Nhãn dùng cho phân tầng và đánh giá, không router chọn phương pháp theo nhãn. Đây là development đã tái sử dụng, không phải held-out. Không trộn các báo cáo cũ 28/29/32 mẫu <=50s vào những bảng này.

Sửa lỗi trình bày ở câu trả lời trước: native TA/B đúng 3, sai 2; không phải 0 đúng/5 sai. Native tổng 18 đúng, 22 sai. Báo cáo và CSV gốc đã ghi đúng; bản tổng hợp này đếm lại từ dự đoán.

**1. Đề xuất ban đầu của Yoshua Bengio: M0–M9**

{{argon_latent40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|M0 Native|Đối chứng 18/40; TA8/10, TF5/15, CMTB5/15. Là mốc cùng phiên, không thay bằng baseline lịch sử 19/40.|
|M1 Text-only PMI|16/40, giảm 2 câu. Prior correction không gắn riêng một task, nhưng cấu hình hiện tại không đem lại lợi ích; prior văn bản không mặc nhiên là bias thuần. Giữ làm đối chứng, không coi là cải tiến.|
|M2 Gate+clock+foreground+PMI|26/40, CMTB13/15 nhưng TF5/15 và Present15/15. Chưa phù hợp: hard gate/cửa sổ làm hẹp evidence và clock dùng từ nhãn. Tín hiệu đáng giữ là cải thiện binding, không phải đã sửa order.|
|M3 Recurrent L22 về tầng sớm|21/40, CMTB13/15 nhưng TA5/10 và TF3/15. Feedback là giả thuyết dùng rộng được, pipeline hiện còn clock/mask/PMI và thiếu mapping giữa tầng. Chưa đạt.|
|M4 Continuous thought kèm clock/mask/PMI|20/40, TA/B1/5 và CMTB/C0/5. Thought liên tục có thể dùng tổng quát, nhưng kết quả còn lẫn nhiều thành phần; chưa chứng minh alignment AV hoặc temporal grounding.|
|M5 Latent handoff|22/40, TA4/10, TF5/15, Present14/15. Chưa phù hợp với nhiều hướng reasoning vì giảm quyền đọc audio theo tầng cố định; không mặc định mọi câu hỏi đều nên chuyển từ audio sang visual.|
|M6 Unified blend|20/40, thấp hơn M2 sáu câu. Heuristic blend theo skew prior chưa là độ tin cậy evidence. Chưa đạt; không nên tiếp tục ghép thành phần chỉ vì từng nhánh cứu một số câu.|
|M7 Không inject latent trực tiếp|19/40; là control clock/mask/foreground/PMI, không phải bằng chứng latent mới. Còn thành phần phụ thuộc cửa sổ/clock và chưa đạt hiệu quả.|
|M8 Recurrent injection control|24/40, CMTB13/15, TF3/15. Không phải recurrence thuần: delta vẫn hình thành từ hard/anti có clock. Chưa đáp ứng yêu cầu; cần tách nguồn delta khỏi gate/clock trước khi quy gain cho recurrence.|
|M9 Thought injection control|19/40, TF8/15 và phân bố TF6/4/5. Đáng chú ý vì hồi phục Past3/5, nhưng CMTB4/15 và còn thought theo cửa sổ. Đây là giả thuyết nên tách ra kiểm chứng, không phải phương pháp tổng quát đã thành công.|

Lưu ý triển khai: thành phần mang tên Value-Clock ở lượt này cộng vào residual layer output, không trực tiếp sửa v_proj. Không dùng tên đó để suy ra đã hiện thực value-space clock injection đúng nghĩa. Shared vocabulary coordinates cũng không bảo đảm zero modality gap.

**2. Đề xuất CCLD tiếp nối của Yoshua Bengio: C0–C7**

{{argon_ccld40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|C1 M9 control|19/40, tái lập từng dự đoán M9. Control hợp lệ; còn thought theo cửa sổ. Chưa là cải tiến hoặc bằng chứng TMRoPE tự hiểu mọi quan hệ thời gian.|
|C2 Bỏ clock/PMI khỏi M8|26/40, CMTB15/15 nhưng TA/A1/5, TF/Future0/5; TF đoán11/4/0. Chưa phù hợp: vẫn hard-sync theo cửa sổ. Cũng bỏ foreground/desmear nên không phải ablation chỉ bỏ clock. Giữ làm đối chứng quan trọng cho đánh đổi giữa binding và order.|
|C3 Visual recurrence|21/40, TF Present15/15. Không speech-window/từ đáp án/task router: phù hợp hơn về thiết kế. Nhưng chỉ khuếch đại visual delta, feedback khác tầng, chưa chứng minh nhiều hướng reasoning; hiệu quả chưa đạt.|
|C4 Shuffled delta|14/40. Đây là negative control, không phải candidate triển khai. C3 hơn 7 câu cho thấy hướng vector có tác dụng trong đối chứng này; một seed chưa đủ chứng minh cơ chế ngữ nghĩa/nhân quả.|
|C5 Global M9+visual recurrence|21/40, TF Present15/15, không hơn C3. Không dùng speech-window trong biến thể global này; phù hợp hơn về cấu trúc nhưng chưa đạt hiệu quả và vẫn bất đối xứng visual. Không kết luận M9 bổ sung được temporal order.|
|C6 CCLD unmasked|25/40, Future0/5; TF đoán5/10/0. Pass cuối unmasked nhưng 45% delta lấy từ hard-sync theo cửa sổ: toàn thuật toán chưa window-free. Chưa đáp ứng yêu cầu.|
|C7 C6+late audio damping|26/40, Future0/5. Chỉ sửa một câu TA/B idx2054 so C6, không đổi câu TF. Vẫn window và giảm audio cố định từ block16: chưa phù hợp mọi câu cần audio; chưa chứng minh late handoff hữu ích tổng quát.|

Trong CCLD, H_Q gồm stem, choices và assistant format. Vì thế thay đổi lựa chọn/format có thể góp vào gain. Phép trừ hidden states của transformer phi tuyến không được coi là chứng minh đã khử sạch prior hoặc giữ nguyên mọi evidence.

C2 so native cứu17 câu, làm sai9; C7 cứu16, làm sai8. Exact McNemar p lần lượt 0.1686 và 0.1516; Holm p của 7 so sánh C1–C7 đều 1.0. N nhỏ nên chưa chứng minh gain ổn định, cũng không kết luận phương pháp hoàn toàn vô ích. Hiệu chỉnh lượt này không xóa selection bias của cả lịch sử thử nghiệm.

**3. Các phép thử của tôi: bỏ clock, mở lại query và local encoding**

{{latent_reopen40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|D1 Bỏ clock khỏi M2|27/40 nhưng Present15/15. Bác giả thuyết clock là điều kiện cần cho collapse trong cấu hình này; không xác định riêng tác dụng clock ở M8. Vẫn gate/window nên chưa đạt yêu cầu tổng quát.|
|D2 Mở native attention từ block24|27/40 nhưng Present15/15. Mở muộn không sửa collapse; còn pipeline M2 nên chưa phù hợp.|
|D3 Query đọc global|20/40, TF0/4/11. Chuyển thiên lệch sang Future; thay cả quyền đọc audio và thời gian nên không tách được hai tác dụng. Chưa đạt.|
|G1 Source-local encoding, global read|19/40. Không speech-window/từ đáp án/router; query giữ fullAV. Phù hợp hơn về cấu trúc, nhưng ngưỡng local3s là giả định cần kiểm tra và hiệu quả chưa đạt.|
|G2 G1+mở lại từ24|19/40, cùng dự đoán G1 trên lượt này. Không có bằng chứng lợi ích bổ sung.|
|G3 Source-local reflection|21/40, Past0/5 và TF dự đoán1/2/12. Phản hồi cùng tầng không đủ; thiên Future còn mạnh. Phù hợp hơn về thiết kế nhưng chưa đạt hiệu quả.|

**4. Phản hồi đối xứng cùng tầng: S0–S6**

{{symmetric_latent40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|S1 Visual delta cùng tầng|19/40, TA/A0/5. Không window nhưng vẫn ưu tiên visual; chỉ là đối chứng cho lỗi truyền khác tầng. Chưa đạt.|
|S2 Joint AV interaction|22/40, TF dự đoán5/0/10; Present0/5, TA/A1/5. Có cấu trúc phù hợp nhất với mục tiêu jointAV trong nhóm này: không gate theo speech, không ưu tiên chiều audio/visual cố định. Nhưng kết quả chưa đạt; chỉ cứu9/làm sai5 so native, p=0.424.|
|S3 Chiếu loại phần đơn modality|18/40. Không chứng minh phần còn lại chính là suy luận jointAV; phép chiếu có thể bỏ thông tin hữu ích. Phù hợp ý định, không cải thiện.|
|S4 S3 trên hai tầng|19/40. Thêm tầng chưa đem lại cải thiện rõ. Chưa có căn cứ tăng số tầng tiếp theo cùng hướng.|
|S5 Joint feedback chỉ vào stem|16/40. Giảm can thiệp trực tiếp lên choices/format, hợp mục tiêu tổng quát hơn; kết quả giảm. Không chứng minh gain cũ chỉ do choices, vì can thiệp còn khác về phạm vi token.|
|S6 Visual delta kiểu C3 chỉ vào stem|16/40. Đối chứng cho lựa chọn vị trí can thiệp; chưa cải thiện, vẫn visual-asymmetric.|

S v1 có lỗi replay vượt ngưỡng, được giữ riêng implementation_v1 và không trộn vào bảng chính. S v2 giữ ngưỡng, dùng full fallback; không thay mẫu hay công thức để bỏ qua lỗi.

**5. Thought điều kiện theo câu hỏi: Q0–Q5**

{{query_latent40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|Q1 Global prototype|17/40. Không window/router và chỉ phản hồi vào stem, phù hợp hơn về cách áp dụng; không đem lại lợi ích.|
|Q2 Question prototype|18/40, TF7/15 nhưng TA6/10; đổi nơi sai chứ không tăng tổng. Không giả định hướng modality cố định, nhưng chưa đạt hiệu quả.|
|Q3 Evidence contrast|17/40. Không hơn Q2 hoặc control Q5; chưa chứng minh phần chênh là evidence hữu ích.|
|Q4 Contrast+joint delta|17/40. Không có bằng chứng hai thành phần bổ sung lẫn nhau.|
|Q5 Shuffled thought|17/40. Negative control, không phải candidate triển khai; kết quả ngang Q3/Q4 làm yếu lập luận rằng hướng thought đang cung cấp grounding có ích.|

**6. Tối ưu latent memory theo video: L0–L3**

{{latent_slot_opt40}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|L1 Memory chưa tối ưu|19/40. Control cho hiệu ứng thêm memory; không chứng minh latent optimization.|
|L2 Memory được tối ưu|18/40, giữ nguyên argmax native trên40/40. Không task router/speech-window/handoff, phù hợp hơn về thiết kế. Tuy nhiên loss tái dựng teacher nội bộ chưa sửa temporal QA; chưa đạt.|
|L3 Target thời gian bị xáo|17/40. Negative control; L2 chỉ hơn một câu và không hơn native/L1, chưa chứng minh temporal fitting đem lại gain downstream.|

L2 tăng video self-match113/527 lên458/527 và audio257/514 lên440/514. Các số này đo khớp index nội bộ cùng model, không phải event grounding có annotation độc lập. Mean TV phân phối đáp án=0.0463: xác suất có đổi dù argmax không đổi. Không suy ra model bỏ qua memory tại mọi tầng hoặc mọi latent optimization đều thất bại.

**7. Giữ nguyên slots, đổi tầng đọc: E0–E5**

{{latent_slot_read15}}

|Ý tưởng|Đánh giá theo yêu cầu và góp ý|
|---|---|
|E1 Memory chưa tối ưu, read15|16/40, thấp hơn read23 ba câu. Mở sớm gây hại trong cấu hình này.|
|E2 Memory tối ưu, read15|17/40, thấp hơn read23 một câu; so E4 cứu1/làm sai2. Phù hợp về cách áp dụng nhưng không sửa lỗi chỉ bằng thời điểm đọc.|
|E3 Target xáo, read15|17/40, ngang E2. Chưa có lợi ích downstream rõ của đúng thứ tự target.|
|E4 Optimized read23|18/40, điểm tham chiếu L2 được sao chép, không phải candidate mới được fit/chạy lại toàn bộ.|
|E5 Unoptimized read23|19/40, điểm tham chiếu L1 được sao chép, không phải một cải tiến độc lập mới.|

Read23 parity kiểm tra lại bằng0.0. Slots giữ nguyên từng giá trị; so E2/E4 tách riêng thời điểm đọc, không lẫn việc fit lại.

**8. Đối chứng clock đầu vào: P0–P2**

{{presentation_resample40}}

P1 lấy frame theo index đạt19/40; P2 theo presentation timestamp đạt18/40. Đây là input fidelity control, không phải đóng góp latent reasoning hay định vị sự kiện bằng oracle. Cả hai giữ audio/budget; P1 so native còn thay decoder và số frame khả dụng, P2 so P1 mới tách quy tắc chọn frame. Chưa sửa temporal hallucination. P0 là điểm native tham chiếu được sao chép.

P đủ40 aggregate records và110 rotation audit entries. Do tunnel server1 mất ở cuối lượt, raw per-rotation scores của19 mẫu server1 chưa archive được; đúng ID thiếu1898 được hoàn tất trên server2, không thay mẫu. Không gọi lượt P là đã lưu đủ raw110 scores. Các suite M/C/DG/S/Q/L/E có artifacts riêng và không bị thiếu raw theo sự cố P.

**Góp ý gửi Yoshua Bengio**

1. Giữ hai quan sát thực nghiệm: synchrony-style intervention giúp CMTB mạnh; thought-only M9 hồi phục một phần TF/Past. Không suy ra đã sửa joint temporal reasoning hoặc đã chứng minh cơ chế nội tại.
2. Không dùng điểm26–27/40 làm tiêu chí chọn duy nhất. D1/D2 TF chỉ5/15 bằng native dù tổng tăng9 câu; C2 tăng CMTB lên15/15 nhưng Future0/5, TA/A1/5. Báo đủ8 ô, phân phối dự đoán, rescue/regression và tính bất định.
3. Bỏ clock dùng từ nhãn và cửa sổ lời nói làm mặc định cho toàn bài toán. Bảo toàn full timeline, provenance audio/video; không cố định trình tự tìm audio rồi khớp visual hoặc giảm audio ở tầng muộn.
4. Đừng đề xuất lại đúng same-layer delta, orthogonal interaction, stem-only thought hay self-reconstruction slots chỉ bằng cách đổi gain/layer: các nhánh này đã được chạy và chưa đạt. Nếu quay lại phải có giả thuyết mới và bằng chứng xác định nút thắt.
5. Ablation cần trực giao: cùng prompt/thought/scale/mask, chỉ đổi một thành phần; đối chứng nhiều random direction/seed, cùng norm và compute. Unmasked pass cuối không làm nguồn delta tự hết phụ thuộc window.
6. Kiểm chứng quan hệ thời gian bằng biến đổi audio/video giữ nguyên nội dung: dịch tương đối, đổi thứ tự, nhiều sự kiện, kiểm tra câu cần audio và câu cần visual. Với biến đổi nhãn không xác định được chắc chắn, phải gán nhãn/kiểm chứng lại; không tự tạo gold sai.
7. Hướng mới có căn cứ hơn là latent reasoner/adapter được giám sát bằng event–source–time trên train pool tách video/nguồn khỏi40 dev, kết hợp temporal hard negatives có kiểm chứng. Kiểm tra grounding và đáp án cùng nhau. Đây là đề xuất chưa chạy, không có điểm kết quả.
8. Thiết kế không có task router mới chỉ thỏa một phần yêu cầu. Để đánh giá triển khai cần held-out/free-form, audio-led/visual-led, reverse lookup, nhiều sự kiện, order/count/duration/causal; đo latency/VRAM từng method, không dùng thời gian cả suite làm runtime triển khai.

Về mục tiêu CVPR: chủ đề phù hợp hướng nghiên cứu, nhưng hiện chưa có phương pháp được chứng minh đủ mạnh/tổng quát hoặc có novelty đã xác lập. Mốc30/40 và >=3/5 mỗi ô chỉ là tiêu chí phát triển nội bộ, không phải tiêu chí nhận bài. Hiện không phương pháp nào đạt cả hai. Các primitive đã được đối chiếu tài liệu ở báo cáo trước; bản này không thực hiện một literature review mới hay xác nhận quyền ưu tiên.

Artifacts gốc: report.md, summary.json, per_sample.csv/JSONL, protocol.md trong các thư mục ở từng bảng. Bản tổng hợp được tạo từ các dự đoán đã lưu, không chạy inference mới hoặc thay kết quả cũ. CSV kèm theo chứa đủ task/gold/prediction để liệt kê chính xác ID đúng/sai mỗi method. Summary JSON nguồn được đối chiếu với số đếm tái tính cho mọi ô.
'@

foreach ($suite in $suiteNames) { $report = $report.Replace(('{{'+$suite+'}}'),$tables[$suite]) }
$utf8Bom = [System.Text.UTF8Encoding]::new($true)
[System.IO.File]::WriteAllText(($outputStem+'.md'),$report,$utf8Bom)
$allRows | Export-Csv -LiteralPath ($outputStem+'_predictions.csv') -NoTypeInformation -Encoding UTF8
$allCounts | Export-Csv -LiteralPath ($outputStem+'_counts.csv') -NoTypeInformation -Encoding UTF8
$hashes = @($sourceFiles | Sort-Object -Unique | ForEach-Object { $hash=Get-FileHash -LiteralPath $_ -Algorithm SHA256; [pscustomobject]@{path=$_.Substring($repoRoot.Length+1);sha256=$hash.Hash.ToLowerInvariant()} })
[ordered]@{date='2026-10-15';kind='local_recount_not_new_inference';manifest_ids=$goldById.Count;method_entries=$allCounts.Count;prediction_rows=$allRows.Count;all_cells_match_archived_summary=$true;baseline_predictions_match_across_suites=$true;source_hashes=$hashes} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($outputStem+'_audit.json') -Encoding UTF8
Write-Output ('Validated '+$allCounts.Count+' suite/method entries; '+$allRows.Count+' predictions; 5 samples per cell; all archived summary counts match.')
Write-Output ('Report: '+$outputStem+'.md')
