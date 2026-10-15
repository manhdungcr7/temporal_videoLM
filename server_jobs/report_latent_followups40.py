"""Write complete local reports for the two independent latent follow-ups."""
import argparse,json,statistics
from pathlib import Path
import argon_ccld_manage as report

ROOT=report.ROOT
NOTES={
    'symmetric_latent40':dict(baseline='S0_native',title='Phản hồi latent đúng tầng và tương tác hai nguồn',
        interpretations=[
            'S1 là đối chứng visual-asymmetric ở đúng block22. S2 dùng tương tác đại số hai nguồn; S3 chiếu bỏ span hai chênh lệch unimodal; S4 dùng hai block18/22. S5 giới hạn S3 vào stem. S6 giữ delta/cross-layer rescaling của Argon C3 nhưng giới hạn vào stem.',
            'S1–S5 không khuếch đại delta nhỏ; S6 tái lập rescaling của Argon để tách tác dụng của vị trí token. Hiệu hidden-state và phép chiếu không được coi là phép phân rã nhân quả đã được xác nhận.',
            'Đối chứng này kiểm tra ba giả thuyết khác nhau: truyền sai tầng, nhiễu do tăng norm và can thiệp trực tiếp vào choices/format. Hiệu quả phải đánh giá trên cả8 ô, không chỉ tổng hoặc việc hết collapse.',
            'V1 dừng ở mẫu thứ3 do prefix replay.03125 vượt ngưỡng.03. Hai record v1 được loại khỏi bảng chính và lưu ở implementation_v1. V2 giữ ngưỡng, dùng full execution khi native prefix check không qua; không đổi công thức/scale hoặc mẫu.',
            'S1/S6 vẫn ưu tiên visual; S2–S5 đối xứng nguồn ở phép xây delta. Không cấu hình nào có clock từ nhãn, speech window, task router hoặc audio handoff. Phạm vi câu hỏi rộng vẫn cần kiểm chứng ngoài40 development.']),
    'query_latent40':dict(baseline='Q0_native',title='Thought có điều kiện câu hỏi và đối chứng scaffold',
        interpretations=[
            'Q1 dùng global prototype, Q2 dùng prototype của stem câu hỏi đã đọcAV, Q3 lấy chênh prototype với trường hợp không có nội dung AV, Q4 thêm tương tác cùng tầng, Q5 xáo chiều thought củaQ3 với cùng norm.',
            'Mọi thought đi vào stem trước choices, không vào trực tiếp đáp án/format. Không có cửa sổ lời nói hoặc thứ tự modality cố định. Đây là thay đổi ở nguồn thought và chỗ phản hồi, không phải phát minh mới weighted vocabulary embeddings.',
            'Q3 cần hơn cảQ2 và shuffled Q5 trước khi gán lợi ích cho phần chênh theo bằng chứng. Chuẩn hóa một chênh lệch nhỏ có thể khuếch đại nhiễu; prototype_difference_norm được lưu để kiểm tra.',
            'Cache chỉ là tối ưu tính toán: native prefix check<=.03, full fallback khi không qua, và kiểm tra mọi điều kiện ở rotation cache-eligible đầu của mỗi worker. Baseline dùng full native pass.',
            'Không có mã theo dạng câu hỏi hoặc nhãn, nhưng điều đó chưa chứng minh khả năng triển khai đa ngữ cảnh. Cần held-out, free-form, reverse reasoning, nhiều sự kiện và kiểm tra thực sự theo can thiệp thời gian.']),
    'latent_slot_opt40':dict(baseline='L0_native',title='Tối ưu latent slots theo nguồn và thời gian',
        interpretations=[
            'L1 thêm memory chưa tối ưu; L2 tối ưu input slots với source-only teachers; L3 cùng số bước và constraints nhưng đảo target theo thời gian trong từng nguồn. Backbone đóng băng và nhãn benchmark không vào loss.',
            'Tái dựng đúng target nội bộ không đồng nghĩa định vị đúng sự kiện thật. Teacher là hidden states của chính model và dùng clock native, không phải annotation sự kiện độc lập.',
            'L2 cần hơn L1 để chứng minh lợi ích optimization, và hơn L3 để chứng minh sự sắp xếp theo thời gian quan trọng. Nếu feature reconstruction tăng nhưng QA không tăng, chưa sửa được khoảng cách biểu diễn và quyết định.',
            'Memory chỉ được đọc từ block23, sau layer22 được fit. Lượt read15 giữ nguyên các slot để kiểm tra riêng nút thắt thời điểm đọc.',
            'Không speech-window, clock từ đáp án, loại câu hỏi hay thứ tự modality cố định. Đây là một thích nghi của latent optimization đã có tiền lệ, chưa phải bằng chứng novelty đủ mạnh cho CVPR.']),
    'presentation_resample40':dict(baseline='P0_original_native_reference',title='Đối chứng lấy frame theo chỉ số và theo presentation timestamp',
        interpretations=[
            'P0 là điểm native gốc được sao chép, không thay đầu vào. P1 dùng PyAV và chỉ số frame cách đều. P2 dùng cùng decoded pictures nhưng chọn hình đang hiển thị tại mỗi thời điểm grid native. Cả hai giữ nguyên audio, FPS, số frame và geometry của original utility.',
            'P1 so với P0 thay decoder và số frame gốc khả dụng; không gọi là decoder-only ablation. P2 so với P1 mới kiểm tra riêng quy tắc chọn chỉ số/thời gian trên cùng decoded pictures.',
            'PTS nhân time_base, trừ container origin. Các bước PTS không đơn điệu được stable sort cùng index ảnh, không đoán từ average FPS hoặc API Decord. Black trước frame đầu và hold sau frame cuối được ghi lại.',
            'Đây là kiểm tra đầu vào có căn cứ từ pixel audit, không phải novel temporal reasoning, event oracle hoặc bảo đảm cải thiện. Dù có tăng QA cũng chưa đủ cho đóng góp CVPR; nếu không tăng, chỉ bác lợi ích QA của quy tắc này trên development40.']),
    'latent_slot_read15':dict(baseline='E0_native',title='Đối chứng thời điểm đọc memory: block15 so với23',
        interpretations=[
            'E1/E2/E3 đọc cùng memory chưa fit/đã fit/fit target bị đảo từ block15. E4/E5 là các điểm read23 được sao chép từ lượt gốc, không huấn luyện hoặc chạy lại candidate.',
            'Các latent được giữ nguyên từng số float. Chỉ đổi tầng bắt đầu cho query đọc memory. Một rotation mỗi worker dựng lại read23 để kiểm tra parity trước khi so sánh.',
            'Mở quyền đọc sớm có thể giúp sử dụng evidence, cũng có thể làm nhiễu readout. Các slot chỉ có loss ở block22, nên không mặc định mọi feature ở tầng sớm đã được align tốt.',
            'Phương pháp không chọn theo task/nhãn và giữ toàn bộ AV. Tuy vậy kết quả40 development không chứng minh tổng quát, triển khai free-form hoặc chấp nhận CVPR.'])}

def main(phase):
    cfg=NOTES[phase];folder=ROOT/'results'/phase;report.FOLDER=folder
    summary=report.summary(cfg['baseline'])
    rows=[json.loads(x) for s in (1,2) for x in (folder/f'results_s{s}.jsonl').read_text().splitlines() if x.strip()]
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    assert len(rows)==40 and {r['idx'] for r in rows}=={r['idx'] for r in manifest['selected']}
    assert all(r['manifest_sha256']==manifest['manifest_sha256'] for r in rows)
    audits=[a for r in rows for a in r['audit']];assert len(audits)==110
    best=max((m for m in summary if m!=cfg['baseline']),key=lambda m:summary[m]['correct'])
    eligible=[m for m,v in summary.items() if v['correct']>=30 and all(c['correct']>=3 for c in v['cells'].values())]
    lines=[f'# {cfg["title"]}:40/40','',
        'Ngày2026-10-15. Cùng40 video cố định<=45s, năm mỗi task/nhãn, tất cả110 cyclic rotations. Đây là development đã dùng nhiều lần; không dùng nhãn trong suy luận.',
        '', '**Bảng ghi đúng/sai.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động của mỗi câu.',
        '',(folder/'table.md').read_text().strip(),'',
        f'Cấu hình có số đúng cao nhất trong lượt này: **{best}: {summary[best]["correct"]}/40**.',
        ('Đạt ngưỡng development30/40 và ít nhất3/5 mỗi ô: '+', '.join(eligible)+'.') if eligible else
        '**Không cấu hình nào đạt30/40 và ít nhất3/5 trong mọi ô.**',
        '', '## So ghép cặp với native cùng lượt','',
        '|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p, chưa hiệu chỉnh|',
        '|---|---:|---:|---:|---:|']
    for m,v in summary.items():
        lines.append(f"|{m}|{v['rescued']}|{v['damaged']}|{v['rescued']-v['damaged']:+d}|{v['paired_p']:.4f}|")
    lines+=['','## Nhận xét và phạm vi','']+cfg['interpretations']
    lines+=['','Không thể suy ra phương pháp đủ mới hoặc đủ mạnh cho CVPR chỉ từ kết quả này. Các primitive counterfactual/activation steering/soft embedding đã có tiền lệ; cần đối chứng mạnh và kiểm chứng tổng quát.',
        '', '## Kiểm tra triển khai','',
        f'Manifest `{manifest["manifest_sha256"]}`.',
        f'Thời gian trung bình của toàn suite và mọi rotations: {statistics.mean(r["seconds"] for r in rows):.1f}s/mẫu; không phải latency của một method.',
        'Raw sample scores: results_s1/s2.jsonl. Summary/table/per_sample.csv và protocol.md được giữ riêng từng họ. Mọi lỗi kỹ thuật và bản triển khai cũ được giữ, không quietly skip hoặc thay sample.']
    if 'replay_error' in audits[0]:lines.append(f'Native replay max={max(a["replay_error"] for a in audits):.8f}.')
    if 'prefix_used' in audits[0]:
        lines.append(f'Native prefix fallback ở{sum(not a["prefix_used"] for a in audits)}/110 rotations; condition fallback ở parity checks đầu worker={sum(len(a.get("full_intervention_fallback",[])) for a in audits)}.')
    if phase=='latent_slot_opt40':
        lines += ['',f'Zero-memory identity max={max(a["zero_memory_error"] for a in audits):.8f}.',
            '', '## Tái dựng slot theo teacher nội bộ','',
            '|Memory|Video cosine|Video đúng slot|Audio cosine|Audio đúng slot|',
            '|---|---:|---:|---:|---:|']
        for m in ('L1_unoptimized_memory','L2_optimized_memory','L3_time_shuffled_target'):
            values=[]
            for source in ('video','audio'):
                d=[r['reconstruction'][m][source] for r in rows];n=sum(x['n'] for x in d)
                cosine=sum(x['cosine']*x['n'] for x in d)/n
                correct=sum(x['retrieval_correct'] for x in d)
                values.extend([f'{cosine:.4f}',f'{correct}/{n}'])
            lines.append('|'+m+'|'+'|'.join(values)+'|')
        lines.append('Đúng slot nghĩa là nearest target của cùng nguồn trùng index thời gian nội bộ. Có thể giải mã clock hoặc self-match; không coi số này là độ chính xác event grounding thật.')
    if phase=='latent_slot_read15':
        errors=[v for a in audits for v in a['read23_parity'].values()]
        lines += ['',f'Read23 reconstruction parity max={max(errors):.8f}. Không refit parameters.',
            '',f'E2 so với E4: cứu{sum(r["predictions"]["E2_optimized_read15"]==r["gold"] and r["predictions"]["E4_optimized_read23_reference"]!=r["gold"] for r in rows)} câu và làm sai{sum(r["predictions"]["E2_optimized_read15"]!=r["gold"] and r["predictions"]["E4_optimized_read23_reference"]==r["gold"] for r in rows)} câu.']
    if phase=='presentation_resample40':
        clocks=[r['clock'] for r in rows]
        lines += ['',f'Tổng native frame slots={sum(c["native_frames"] for c in clocks)}, cùng chỉ số chọn giữa P1/P2={sum(c["same_selected_indices"] for c in clocks)}.',
            f'Black trước ảnh đầu={sum(c["black_slots"] for c in clocks)} slots; hold sau PTS cuối={sum(c["targets_after_last_pts"] for c in clocks)} slots; clips có bước PTS không đơn điệu={sum(c["nonmonotone_steps"]>0 for c in clocks)}/40.']
    sensitivity=folder/'decision_sensitivity_audit.json'
    if sensitivity.exists():
        decision=json.loads(sensitivity.read_text())
        lines += ['', '## Độ nhạy quyết định','',
            'Từ scores có sẵn, không chạy lại inference. TV đo trên phân phối lựa chọn: trung bình log-prob qua rotations rồi chuẩn hóa. Đây không phải sensor attention hoặc độ chính xác grounding sự kiện.',
            '',f'Đối chiếu với `{decision["baseline"]}`.','',
            '|Method|Đổi dự đoán/40|Mean TV|Median TV|',
            '|---|---:|---:|---:|']
        for m,d in decision['comparisons'].items():
            lines.append(f'|{m}|{d["changed_predictions"]}|{d["mean_total_variation"]:.4f}|{d["median_total_variation"]:.4f}|')
        if phase=='latent_slot_opt40':
            lines.append('L2 giữ nguyên mọi40 argmax native dù xác suất có thay đổi. Điều này không chứng minh model bỏ qua memory ở mọi layer; chỉ chứng minh slot fit này chưa sửa quyết định cuối.')
    (folder/'report.md').write_text(report.markdown_lines(lines),encoding='utf-8')
    print((folder/'table.md').read_text());print('Saved',str(folder/'report.md'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=list(NOTES));args=parser.parse_args();main(args.phase)
