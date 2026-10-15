"""One reviewable local index for Argon's request and the subsequent research."""
import json
from pathlib import Path
import argon_ccld_manage as common

ROOT=common.ROOT
PHASES=['latent_reopen40','argon_ccld40','symmetric_latent40','query_latent40',
        'latent_slot_opt40','latent_slot_read15','presentation_resample40']
DESCRIPTIONS={
    'latent_reopen40':'D/G: bỏ clock, mở lại query và đọc evidence cục bộ',
    'argon_ccld40':'C0–C7: đề xuất CCLD của Argon',
    'symmetric_latent40':'S0–S6: đúng tầng, norm giới hạn và tương tác đối xứng',
    'query_latent40':'Q0–Q5: thought theo câu hỏi và đối chứng scaffold',
    'latent_slot_opt40':'L0–L3: thật sự tối ưu input latent, backbone đóng băng',
    'latent_slot_read15':'E0–E5: giữ nguyên slots, đọc sớm hơn',
    'presentation_resample40':'P0–P2: kiểm tra đồng hồ frame đầu vào'}

def main():
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    allrows={};missing=[]
    for phase in PHASES:
        folder=ROOT/'results'/phase
        rows=[json.loads(x) for s in (1,2) if (folder/f'results_s{s}.jsonl').exists()
              for x in (folder/f'results_s{s}.jsonl').read_text().splitlines() if x.strip()]
        if len(rows)!=40:missing.append((phase,len(rows)));continue
        assert {r['idx'] for r in rows}=={r['idx'] for r in manifest['selected']}
        assert all(r['manifest_sha256']==manifest['manifest_sha256'] for r in rows)
        allrows[phase]=rows
    assert not missing,missing
    lines=['# Đánh giá Argon CCLD và các lượt tiếp nối trên development40','',
        '2026-10-15. Toàn bộ các suite dưới đây hoàn tất đúng40/40 ID, mỗi ô task/ground-truth có5 mẫu; raw video và decoded audio đều<=45s. Mọi suite dùng tất cả110 cyclic option rotations. Không thay mẫu khi lỗi.',
        '',f'Manifest SHA256: `{manifest["manifest_sha256"]}`.',
        '', '**Kết quả chính: C2 và C7 của Argon đạt26 đúng/14 sai, ngang tổng M2 cũ. C3/C5 vẫn dự đoán Present15/15. Chưa tìm được phương pháp giải quyết tốt đồng thời các ô temporal và đủ bằng chứng triển khai đa ngữ cảnh.**',
        '', '40 mẫu này đã được dùng nhiều lần để chọn và phân tích phương pháp. Chúng là development, không phải held-out. Không chọn method theo task/nhãn hoặc ghép đáp án bằng oracle. Độ đa dạng câu hỏi thực tế chưa được xác nhận.',
        '', '## Đề xuất Argon: kết quả đúng/sai từng ô','',
        'TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future; CMTB:A/B/C là nội dung lựa chọn riêng từng câu. Trong các ô, `đúng/sai` cộng lại bằng5.',
        '',(ROOT/'results/argon_ccld40/table.md').read_text().strip(),
        '', '![Recall từng ô và phân phối nhãn TF](argon_ccld40/figures/ccld_development40.png)',
        '', '## Đánh giá, nhận xét và góp ý với Argon','',
        'C2 đạt CMTB15/15 nhưng TA/Yes1/5 và TF/Future0/5; TF dự đoán Past11,Present4,Future0. Đây là thay đổi đánh đổi, không phải giải quyết joint temporal reasoning.',
        'C3/C5 không dùng speech-window trong đường suy luận tương ứng, nhưng vẫn collapse Present15/15. Vì vậy phép trừ visual-blind không bảo đảm giữ đúng temporal evidence hay loại bỏ thiên lệch. C3 hơn shuffled C4 thêm7 câu; một hướng có ý nghĩa hơn nhiễu chưa đủ chứng minh cơ chế nhân quả hoặc độ chính xác cao.',
        'C6/C7 vẫn lấy45% delta từ hard_sync theo cửa sổ; unmasked pass2 không làm toàn bộ thuật toán window-free. C7 còn giảm audio từ block16 bằng hằng số, chưa phù hợp mọi hướng reasoning hoặc câu hỏi cần nghe chi tiết. C6/C7 đều TF/Future0/5. C7 chỉ khác C6 trên một dự đoán TA/B idx2054 (sai thành đúng); không đổi câu TF nào.',
        'Phân biệt giả thuyết với chứng minh: hiệu hidden state của một mạng phi tuyến không triệt sạch prior và không bảo toàn100% tín hiệu. H_Q trong triển khai Argon gồm stem, choices và format; tác dụng lên readout lựa chọn có thể lẫn với grounding. C2 cũng bỏ foreground/desmear; C1 thought-window/.22 khác C5 global/.20, nên các đối chứng chưa hoàn toàn trực giao.',
        'Lượt D1 giữ gate/foreground/PMI nhưng bỏ clock đạt27/40 và vẫn Present15/15. Nó chứng minh lexical clock không cần thiết cho collapse M2 trong cấu hình này; không xác định riêng tác dụng clock ở delta M8. M9 tái lập19/40, TF8/15, chưa chứng minh TMRoPE tự suy luận đúng mọi quan hệ thời gian.',
        'So native cùng lượt, C2 cứu17/làm sai9 và C7 cứu16/làm sai8. Exact paired p lần lượt.1686/.1516; toàn bộ7 phép so sánh C1–C7 có Holm-adjusted p=1.0000. Không coi đây là chứng cứ gain ổn định; hiệu chỉnh nội bộ này cũng chưa xử lý toàn bộ quá trình chọn nhiều lượt.',
        '', '## Đáp ứng lưu ý về tổng quát và CVPR','',
        'C3/C5 đáp ứng hình thức tốt hơn: fullAV ở pass cuối, không router theo task, không VAD window hoặc từ đáp án. Nhưng cơ chế ưu tiên phần đóng góp visual và kết quả chưa chứng minh hiệu quả cho audio-led, visual-led, reverse lookup, nhiều sự kiện, count/duration, causal hoặc free-form QA.',
        'C1/C2/C6/C7 còn dựa cửa sổ. C7 có chính sách giảm một nguồn cố định. Không nên trình bày các cấu hình này như phương pháp đáp ứng đầy đủ yêu cầu nhiều ngữ cảnh của người dùng.',
        'Chưa có cấu hình nào được xác nhận đủ mới, đủ mạnh hoặc sẽ được nhận CVPR. Các primitive counterfactual, activation steering, weighted vocabulary embedding và latent optimization đã có tiền lệ. Đóng góp cần được xác định cụ thể và kiểm chứng trên held-out, đa benchmark và can thiệp thời gian có kiểm soát.',
        '', '## Các hướng riêng đã hiện thực và chạy sau Argon','',
        '|Suite|Cấu hình nổi bật|Kết quả|Điểm thất bại chính|',
        '|---|---|---:|---|',
        '|D/G|D1 bỏ clock; D2 mở lại từ24|27/40 mỗi cấu hình|TF vẫn Present15/15|',
        '|S|S2 tương tác cùng tầng|22/40|TF/Present0/5; S1/S3/S4/S5/S6 lần lượt19/18/19/16/16|',
        '|Q|Q2 prototype theo stem|18/40|Ngang native; Q1/Q3/Q4/Q5 đều17|',
        '|L|L2 input slots được tối ưu|18/40|L1 chưa fit19, L3 đảo target17; feature self-match không thành QA gain|',
        '|E|E2 cùng slots, đọc từ15|17/40|E1 chưa fit16, E3 đảo target17; E4 đọc từ23=18|']
    p=json.loads((ROOT/'results/presentation_resample40/summary.json').read_text())
    lines.append(f'|P|P2 presentation-time|{p["P2_pyav_presentation_time"]["correct"]}/40|P1 frame-index={p["P1_pyav_frame_index"]["correct"]}/40; input correctness control|')
    lines += ['', 'Tối ưu latent đã chạy32 Adam steps/clip/variant, source-only teachers ở block22, fixed2s slots, cosine+within-source contrastive loss, backbone đóng băng. Tất cả parameters được giữ nguyên khi so E2/E4. Video slot self-match từ113/527 lên458/527; audio từ257/514 lên440/514, nhưng L2 vẫn18/40 và giữ nguyên từng argmax trên40 câu. Mean TV của phân phối lựa chọn=.0463; xác suất có đổi, không suy ra mọi layer bỏ qua memory. Đó là bằng chứng mục tiêu tái dựng nội bộ chưa đủ cho downstream temporal QA ở cấu hình này, không phải bằng chứng mọi latent optimization đều thất bại.',
        'E2 so với E4 chỉ cứu1 câu, làm sai2; parity dựng lại read23 bằng0. Đổi riêng giai đoạn đọc không sửa được thất bại của các slot đã fit. Cần mục tiêu grounding/decision có supervision độc lập thay vì chỉ lặp lại feature của chính model.',
        '', '## Research đã dùng và ranh giới với hướng tương lai','',
        '[LatentOmni](https://arxiv.org/html/2605.22012v1) kết hợp latentAV với feature supervision và đồng bộ vị trí. [Unsilencing Visual Latents](https://arxiv.org/html/2605.02735v1) đã có inference-time latent optimization trong backbone đóng băng; không claim primitive đó mới.',
        '[RED](https://arxiv.org/html/2610.02976v1) nghiên cứu đóng góp audio/video/interaction ở decoding; [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/) là tiền lệ steering hidden states. Các nguồn này không chứng minh công thức CCLD hoạt động trên40 mẫu của chúng ta.',
        '[AHA](https://arxiv.org/html/2512.24052v2) dùng preference hard negatives cho temporal/audio hallucination trên Qwen2.5-Omni-7B. Đó là hướng supervision liên quan, nhưng audio-only và thiết kế dữ liệu của paper không tự giải quyết jointAV tổng quát ở model3B. Không chạy checkpoint7B thay model mục tiêu hoặc tuyên bố đã tái lập DPO.',
        '[PyAV time documentation](https://pyav.org/docs/stable/api/time.html) giải thích pts*time_base. Đối chứng P giữ audio và budget nguyên, tách PyAV index/PyAV PTS. Nó kiểm tra clock đầu vào, không làm oracle xác định sự kiện.',
        'Hướng tiếp theo có căn cứ hơn: huấn luyện riêng adapter/latent reasoner trên train pool tách video/source khỏi40 mẫu, với segment grounding có kiểm chứng và temporal counterfactual hard negatives giữ nguyên nội dung. Bao phủ cả hai hướng modality, nhiều sự kiện và quan hệ count/duration/order; không đặt luật suy luận theo câu chữ của SVHalluc. Phải đánh giá đúng sự kiện và khoảng thời gian cùng câu trả lời. Đây là đề xuất chưa được chạy; không đưa điểm dự kiến hoặc gọi là phương pháp thành công.',
        '[Thiết kế chi tiết và các điều kiện chống supervision sai](latent_research_next_grounded_training.md).',
        '', '## Tái lập và artifacts','',
        'Baseline Argon và M9 giống lịch sử trên từng ID (0/40 thay đổi). Native replay CCLD max0.0; shuffled row-norm error5.72e-6. Suite S v1 gặp prefix replay.03125>.03; giữ log riêng implementation_v1, v2 dùng full fallback và không nới ngưỡng. Các sửa kỹ thuật không thay mẫu hoặc scale.',
        'C/S/Q/L/E đều có JSONL từng sample, mọi110 rotation JSON và audit tính lại điểm. L có40 checkpoint input slots lưu local trong latent_slot_opt40/fits, tổng25,741,416bytes, SHA từng file ở fit_archive_audit.json. Chỉ input slots được fit; không sửa backbone.',
        'Server1 mất tunnel ở cuối đối chứng P, khi19/20 sample đã lưu local. Đúng ID còn thiếu1898 được chuyển sang server2, cùng code toán học và C0 tham chiếu đã archive; không thay mẫu. P có đủ40 aggregate records/110 rotation audit entries, nhưng raw per-rotation files của19 mẫu server1 chưa tải được trước khi tunnel offline. Không gọi các aggregate records là đã archive đủ raw110 rotation scores; xem recovery_protocol.json và raw archive coverage riêng. Điều này không ảnh hưởng raw110 của Argon hoặc các suite C/S/Q/L/E đã tải xong.',
        'Code trong server_jobs; mỗi folder có protocol trước chạy, report, summary, per_sample.csv, job statuses và rotations. proposal_original.txt/.py là bản Argon nguyên gốc; không thực thi phần sampling/xóa folder của code đó. Thay bằng đúng manifest40<=45s và native raw input theo yêu cầu người dùng. Những persona/tên tác giả trong đề xuất không phải xác nhận thật của nhà nghiên cứu hoặc hội nghị.',
        '', '## Bảng đầy đủ từng suite','']
    for phase in PHASES:
        folder=ROOT/'results'/phase
        lines += [f'### {DESCRIPTIONS[phase]}','',f'[Báo cáo]({phase}/report.md) · [Protocol]({phase}/protocol.md) · [Số liệu]({phase}/summary.json)','',
            (folder/'table.md').read_text().strip() if (folder/'table.md').exists() else 'Xem report.md cho bảng đúng/sai.','']
    path=ROOT/'results/argon_and_latent_research_20261015.md'
    path.write_text(common.markdown_lines(lines),encoding='utf-8')
    print('SAVED',str(path))

if __name__=='__main__':main()
