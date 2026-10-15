"""Complete Round5/continuation review; only completed, audited local results."""
import csv,json
from pathlib import Path
from report_chronodesmear_research import ROOT,SUITES,build

def main():
    for suite,baseline in SUITES.items(): build(suite,baseline)
    geometry=ROOT/'results/media_geometry40'
    assert (geometry/'report.md').exists()
    summaries={suite:json.loads((ROOT/'results'/suite/'summary.json').read_text()) for suite in SUITES}
    for suite in SUITES:
        assert all((ROOT/'results'/suite/f'archive_audit_s{s}.json').exists() for s in (1,2))
    candidates={m:v for suite,summary in summaries.items() for m,v in summary.items()
        if m not in ('R5_C0_baseline','N0_native','N1_desmear_reference','N6_random_stratified','P0_native_reference','P6_last_mlp_random')
        and not m.startswith('R5_C6')}
    best=max(v['correct'] for v in candidates.values())
    best_names=[m for m,v in candidates.items() if v['correct']==best]
    passed=[m for m,v in candidates.items() if v['correct']>=30 and min(c['correct'] for c in v['cells'].values())>=3]
    lines=['Yoshua Bengio — ChronoDeSmear Round5, cải tiến và kiểm chứng cơ chế','',
        '2026-10-15. Hoàn tất ba suite QA trên đúng40/40 mẫu và diagnostic geometry40. Mọi bảng QA là đúng/sai. Bộ có5 mẫu/ô, duration<=45s,110 rotations/suite,38 nguồn gốc; development tái sử dụng.',
        '',f"Điểm cao nhất trong các candidate/diagnostic không phải random control: **{best}/40**, {', '.join(best_names)}. Candidate đạt mốc nội bộ30/40 và>=3/5 mọi ô: {passed or 'không có'}. Mốc này không phải tiêu chuẩn nhận CVPR.",
        '', 'Baseline native18/40. Round5 C2 tốt nhất21/40; full C5 chỉ18/40. Không có căn cứ tuyên bố đã giải quyết jointAV temporal hallucination hoặc đủ novelty/generalization cho CVPR.',
        '', '**1. Đề xuất của Yoshua Bengio: kết quả đầy đủ**','',
        (ROOT/'results/chronodesmear40/table.md').read_text(encoding='utf-8'),
        '', '[Report từng ý tưởng và paired tests](chronodesmear40/report.md) · [Protocol](chronodesmear40/protocol.md) · [Phản biện cơ chế](chronodesmear40/mechanism_review.md) · [Literature đã đọc](chronodesmear40/related_work.md)',
        '', 'C1 có cấu trúc dùng rộng nhưng kết quả16/40. C2 tăng CMTB5->10/15, giảm TA8->7/10 và TF5->4/15. C3 còn speech-window và không phải phép bù V_out độc lập; TF2/15. C4 stem+media16/40. C5 full giữ TA8/10 nhưng đổi TF5->2/15 lấy CMTB5->8/15. C6 random ensemble cũng có Past5/5 nhưng tổng18/40; một ô cao không chứng minh grounding.',
        '', 'C2/C3/C5 còn cần options để mean-center; không task router không đồng nghĩa đã có thuật toán QA tự do. C5 source contrast vẫn chủ yếu visual. Không coi zero window là đủ để đáp ứng nhiều ngữ cảnh.',
        '', '**2. Các cải tiến và factor controls đã chạy tiếp**','',
        (ROOT/'results/chronodesmear_followup40/table.md').read_text(encoding='utf-8'),
        '', '[Report/so ghép từng yếu tố và layer lens](chronodesmear_followup40/report.md) · [Protocol trước chạy](chronodesmear_followup40/protocol.md)',
        '', 'N2 pure delta; N3 thêm center; N4 thêm de-smear; N5 thêm thought. N6 random static cùng budget. N7 phản thực tế static theo cả hai nguồn. N8 cùng-layer22/norm cap/lexical stem, là candidate riêng vì đổi nhiều yếu tố. N9 ensemble các tầng cố định. N1 chỉ copy-reference C1, không inference độc lập.',
        '', '**3. Kiểm chứng attention/MLP muộn**','',
        (ROOT/'results/late_path40/table.md').read_text(encoding='utf-8'),
        '', '[Report và parity audit](late_path40/report.md) · [Protocol](late_path40/protocol.md)',
        '', 'P1 giảm MLP last row, P2 giảm attention, P3 tăng MLP, P4 giảm MLP ở stem, P5 giảm cả hai last row, P6 random update control. Blocks28–35, scale cố định. Không dùng task/label router hoặc xóa media; P0 copy-reference native.',
        '', '**4. Những gì các phép thử cơ chế cho phép kết luận**','']
    follow_analysis=json.loads((ROOT/'results/chronodesmear_followup40/analysis.json').read_text())
    lines.extend(['|Fixed block|Đúng/40|Đúng ở tầng này nhưng sai cuối|Sai ở tầng này nhưng đúng cuối|','|---|---|---|---|'])
    for li,entry in follow_analysis['layer_lens'].items():
        lines.append(f"|{li}|{entry['correct']}|{len(entry['late_lost'])}|{len(entry['late_gained'])}|")
    lines.extend(['', 'Logit lens chưa được calibrated. Không chọn layer theo gold; một ca đúng ở giữa/sai cuối không chứng minh model đã grounded đúng sự kiện rồi bị xóa. Late-path là can thiệp thực nhưng vẫn cần source/time-grounded evaluation để xác nhận nguyên nhân.',
        '', '[Geometry raw/centered cosine trong và khác nguồn video](media_geometry40/report.md). Đây là measurement, không phải QA method hoặc event-time oracle.',
        '', 'Block22 video pooled2s: within cosine0.9231, different-origin between0.8770; leave-one-origin-out centered tương ứng0.4217/-0.0093. Similarity cao có thành phần chung giữa các clip; không suy tỷ lệ background90% hoặc quy causal echo từ cosine. Không coi centering này là một method QA đã thử.',
        '', 'Block22 lens16/40 vs cuối18/40:8 câu đúng giữa/sai cuối và10 câu sai giữa/đúng cuối. Đó là hai chiều thay đổi. Không có bằng chứng cho khẳng định chung tầng cuối chỉ xóa lời giải đúng. Late-path28–35 không đổi bất kỳ TF prediction; P3+1 chỉ sửa CMTB1898, baseline sample đó exact tie. Kết quả không loại mọi cơ chế khác/tầng khác hoặc chứng minh model không có temporal information.',
        '', '**5. Góp ý gửi Yoshua Bengio**','',
        '1. Sửa giải thích trước khi tăng thành phần: normalization recurrence triệt hệ số.55 về magnitude; center tokens không bảo đảm label prior bất biến. Trong toy algebra, thay Base bằng Vall không bù V_out mà trừ thêm audio. Các phép trừ hidden nonlinear không có bảo đảm background cancellation sạch.',
        '2. Không gọi matrix hiện tại orthogonal: C2 đã dùng H_ds/V-only; C3 đổi delta+de-smear; C4 đổi span+gain; C6 đối chứng C2, không full C5. Những chênh lệch đó đều đã ghi protocol, không âm thầm đổi bản đề xuất.',
        '3. Không gọi resonance Đầu/Giữa/Cuối là shift/grounding verification. Hàm gốc không dịch input và không có event interval labels độc lập. Không suy phần trăm background từ cosine.',
        '4. Ưu tiên giữ đúng sự kiện, nguồn và interval trong toàn timeline, không cố định audio-first/visual-first. Bản dùng option-slice cần định nghĩa và thử bản free-form riêng. Không tối ưu bằng luật câu hỏi/nhãn SVHalluc.',
        '5. Các nhánh mới ở lượt này cũng phải đánh giá theo đủ8ô, không chỉ gain tổng hoặc hết Present collapse. Không lặp đúng các nhánh này bằng cách đổi vài gain rồi gọi mới; cần giả thuyết sửa nút thắt được đo.',
        '6. Để theo mục tiêu CVPR, cần novelty cụ thể ngoài temporal contrast/activation steering đã có tiền lệ, event/source/time validation độc lập, held-out tách source và nhiều ngữ cảnh. Agent workflow không được dùng trong các suite này; scalar gating/ensemble cũng chỉ là diagnostics, không tự tạo novelty.',
        '', '**Tái lập**','',
        'Ba suite đều có40 aggregate records,55 raw rotations/server, tổng330 rotation JSON đã tải và recompute score. Geometry có40 feature archives riêng. Correct/wrong, per-sample predictions, controls và errors nằm trong từng folder. Tên Yoshua Bengio là quy ước gọi bên đề xuất trong cuộc trao đổi, không xác nhận tác giả ngoài đời.',
        '', 'Các kết quả này chưa đạt mục tiêu nghiên cứu. Không có claim acceptance, không đổi gold/mẫu, không đưa best-layer/oracle choice thành phương pháp triển khai.'])
    target=ROOT/'results/yoshua_chronodesmear_round5_review_20261015.md'
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (target.parent/'yoshua_chronodesmear_round5_all_predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['suite','idx','task','gold','method','prediction','correct'])
        for suite in SUITES:
            with (ROOT/'results'/suite/'per_sample.csv').open(encoding='utf-8') as source:
                for row in csv.DictReader(source): writer.writerow([suite]+[row[k] for k in ('idx','task','gold','method','prediction','correct')])
    print(json.dumps(dict(report=str(target),best_correct=best,best_methods=best_names,criterion_passed=passed)))

if __name__=='__main__': main()
