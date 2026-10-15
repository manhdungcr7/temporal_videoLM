"""Produce the Vietnamese review only after all frozen40 records are present."""
import json,statistics
from pathlib import Path
import argon_ccld_manage as manager

ROOT=manager.ROOT;FOLDER=manager.FOLDER

def load(folder):
    return {r['idx']:r for s in (1,2) for r in
            [json.loads(x) for x in (folder/f'results_s{s}.jsonl').read_text().splitlines() if x.strip()]}

def paired(rows,first,second):
    a=sum(r['predictions'][first]==r['gold'] and r['predictions'][second]!=r['gold'] for r in rows.values())
    b=sum(r['predictions'][first]!=r['gold'] and r['predictions'][second]==r['gold'] for r in rows.values())
    return a,b

def main():
    report=manager.summary();rows=load(FOLDER)
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    assert set(rows)=={r['idx'] for r in manifest['selected']},'Report requires exactly the frozen40 IDs'
    assert all(r['manifest_sha256']==manifest['manifest_sha256'] for r in rows.values())
    assert all(sum(r['task']+'/'+r['gold']==c for r in rows.values())==5 for c in manager.CELLS)
    assert sum(len(r['audit']) for r in rows.values())==110
    historical=load(ROOT/'results/argon_latent40')
    audits=[a for r in rows.values() for a in r['audit']]
    replay=max(a['replay_error'] for a in audits)
    shuffle=max(a['shuffle_norm_error'] for a in audits)
    oldbase_changes=[idx for idx,r in rows.items() if r['predictions']['R4_C0_baseline']!=historical[idx]['predictions']['R3_M0_base']]
    m9_changes=[idx for idx,r in rows.items() if r['predictions']['R4_C1_m9_thought_ctrl']!=historical[idx]['predictions']['R3_M9_thought_only']]
    candidates=[m for m in report if m!='R4_C0_baseline']
    ordered=sorted(candidates,key=lambda m:report[m]['paired_p']);holm={};previous=0.
    for rank,m in enumerate(ordered):
        previous=max(previous,min(1.,report[m]['paired_p']*(len(ordered)-rank)));holm[m]=previous
    eligible=[m for m,v in report.items() if v['correct']>=30 and all(c['correct']>=3 for c in v['cells'].values())
              and v['correct']>report['R4_C0_baseline']['correct']]
    best=max(candidates,key=lambda m:report[m]['correct'])
    c3,c4=paired(rows,'R4_C3_causal_vis_rec','R4_C4_shuffled_rec_ctrl')
    c7_changes=[dict(idx=idx,task=r['task'],gold=r['gold'],before=r['predictions']['R4_C6_ccld_unmasked'],
        after=r['predictions']['R4_C7_ccld_late_handoff']) for idx,r in rows.items()
        if r['predictions']['R4_C6_ccld_unmasked']!=r['predictions']['R4_C7_ccld_late_handoff']]
    notes={
        'R4_C1_m9_thought_ctrl':'Thought M9 dùng cửa sổ âm thanh; đây là đối chứng tái lập, không phải nhánh không phụ thuộc cửa sổ.',
        'R4_C2_m8_noclk_nopmi':'Bỏ clock/PMI nhưng vẫn dùng sync/anti và cũng bỏ foreground/desmear cũ. Không phải thí nghiệm chỉ thay clock.',
        'R4_C3_causal_vis_rec':'Không dùng cửa sổ hoặc từ chỉ đáp án. Tuy nhiên chỉ khuếch đại phần chênh do visual và truyền vector từ block22 sang các block sớm.',
        'R4_C4_shuffled_rec_ctrl':'Đối chứng giữ norm theo từng query row, xáo chiều với một seed cố định; không phải nhiều hướng ngẫu nhiên độc lập.',
        'R4_C5_m9_plus_causal_rec':'Global thought và visual delta không cần cửa sổ; đáp ứng ràng buộc hình thức tốt hơn, nhưng vẫn chưa kiểm chứng câu hỏi thiên về audio hoặc nhiều bước khác.',
        'R4_C6_ccld_unmasked':'Pass2 không mask, nhưng delta vẫn lấy45% từ hard_sync theo cửa sổ. Vì vậy toàn phương pháp còn phụ thuộc cửa sổ.',
        'R4_C7_ccld_late_handoff':'Còn hard_sync và giảm audio cố định từ block16. Chính sách này chưa có căn cứ áp dụng cho mọi câu hỏi cần audio.'}
    lines=['# Đánh giá Argon Round4 CCLD trên đúng40 mẫu',
        '', 'Ngày2026-10-15. Đã chạy đủ8 cấu hình trên hai server Kaggle, mỗi server20 mẫu.',
        'Mô hình Qwen2.5-Omni-3B FP16. Mỗi ô có5 video đã lọc<=45s trước khi lấy mẫu; không thay mẫu hoặc bỏ lỗi.',
        'Mỗi câu dùng mọi cyclic answer rotations, tổng110 rotations. Đây là development đã dùng nhiều lần.',
        '', '**Trong bảng: đúng/sai, không phải đúng/tổng.** TA:A=Yes,B=No; TF:A=Past,B=Present,C=Future. CMTB:A/B/C là lựa chọn hành động riêng của từng câu.',
        '',(FOLDER/'table.md').read_text().strip(),
        '',f"Cấu hình có số đúng cao nhất trong lượt này: **{best} = {report[best]['correct']}/40 ({100*report[best]['correct']/40:.1f}%)**.",
        ('Có cấu hình đạt tiêu chí development30/40 và ít nhất3/5 ở mọi ô: '+', '.join(eligible)+'.') if eligible else
        '**Không cấu hình nào đạt tiêu chí development30/40 và ít nhất3/5 ở mọi ô.**',
        '', '## So ghép cặp với baseline cùng lượt', '',
        '|Method|Cứu baseline sai|Làm sai baseline đúng|Δ đúng|Exact paired p|Holm p (7 phép so)|',
        '|---|---:|---:|---:|---:|---:|']
    for m in candidates:
        v=report[m];lines.append(f"|{m}|{v['rescued']}|{v['damaged']}|{v['rescued']-v['damaged']:+d}|{v['paired_p']:.4f}|{holm[m]:.4f}|")
    lines+=['','Các p-value hai phía dùng exact McNemar. Holm chỉ hiệu chỉnh7 so sánh nội bộ lượt này; không xóa selection bias từ toàn bộ lịch sử thử nghiệm.',
        '', '## Nhận xét từng cấu hình', '']
    for m in candidates:
        v=report[m];tf=v['tf_pred_ABC'];n_tf=sum(tf)
        collapse=f" Dự đoánTF A/B/C={tf}; "+('collapse vào một nhãn.' if max(tf)==n_tf else 'không collapse hoàn toàn vào một nhãn; điều này chưa chứng minh suy luận đúng.')
        lines.append(f"- **{m}:** {v['correct']} đúng, {v['wrong']} sai. {notes[m]}{collapse}")
    lines+=['','## Kiểm chứng các khẳng định của đề xuất', '',
        f'C3 hơn shuffled trên{c3} câu và thua trên{c4} câu. Chênh đúng={c3-c4:+d}; cần xem đối chứng này trước khi gán lợi ích cho hướng ngữ nghĩa của delta.',
        f'C7 khác C6 trên{len(c7_changes)}/40 dự đoán: {c7_changes}. Không có thay đổi dự đoán nào ở TF; lợi ích của late audio damping trong lượt này chỉ nằm ở một câu TA/B, chưa chứng minh khả năng tổng quát.',
        'Không có chứng minh phép trừ hidden state triệt tiêu sạch prior/thì/nhầm nguồn. Attention và MLP phi tuyến, nên hiệu trạng thái cũng có thể làm mất tín hiệu hữu ích hoặc đưa can thiệp ra khỏi phân phối quen thuộc.',
        'M9 cải thiện TF không chứng minh TMRoPE tự so được trước/trong/sau. Muốn kiểm chứng cần đổi thời gian tương đối khi giữ nội dung, kiểm tra mọi nhãn và loại bỏ lối tắt vị trí lời nói.',
        'Các cấu hình không hoàn toàn trực giao: C2 khác M8 cũ ở foreground/desmear; C1 dùng thought theo cửa sổ với scale.22, C5 dùng global thought với scale.20; C6/C7 vừa đổi delta vừa giữ nhánh sync.',
        'Trong code này, H_Q được lấy từ mọi token sau media: có cả stem, lựa chọn và assistant-format tokens. Vì vậy can thiệp có thể đổi đường đọc lựa chọn/format, chưa được tách khỏi grounding. Hướng riêng S5/S6 kiểm tra giới hạn can thiệp vào đúng stem câu hỏi.',
        '', '## Phù hợp với yêu cầu tổng quát và mục tiêu CVPR', '',
        'C3/C5 phù hợp hơn với yêu cầu không router theo task, không chèn từ chỉ đáp án và không chọn cửa sổ bằng VAD. Chúng giữ nguyên fullAV cho pass cuối. Tuy nhiên, không phụ thuộc câu chữ trong code không đồng nghĩa đã tổng quát về hiệu quả.',
        'C1/C2/C6/C7 còn phụ thuộc cửa sổ; C7 còn mặc định giảm audio. Chưa thể coi đây là phương pháp triển khai cho mọi kiểu câu hỏi hay mọi hướng reasoning. Cần audio-only/visual-only, reverse lookup, nhiều sự kiện, nhiều khoảng lời nói, count/duration/causal và free-form QA ngoài SVHalluc.',
        'Chưa có đủ bằng chứng để gọi là đóng góp mới đủ mạnh cho CVPR. Counterfactual contrasts và activation steering đã có tiền lệ; cần novelty cụ thể, ablation sạch, cải thiện ổn định, held-out và kiểm chứng bằng can thiệp thời gian.',
        '[RED](https://arxiv.org/abs/2610.02976) phân rã hỗ trợ audio/video/interaction ở decoding; [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/) đã can thiệp hidden states; [LatentOmni](https://arxiv.org/abs/2605.22012) dùng huấn luyện cho latentAV. Các cơ chế không giống hoàn toàn CCLD, nhưng chặn các claim novelty quá rộng.',
        '', '## Kiểm tra triển khai và khả năng tái lập', '',
        f'Manifest `{manifest["manifest_sha256"]}`. Native/decoder replay max abs error={replay:.8f}; shuffled row-norm max error={shuffle:.8f}.',
        f'Baseline khác lịch sử trên{len(oldbase_changes)}/40 câu: {oldbase_changes}. M9 khác lịch sử trên{len(m9_changes)}/40 câu: {m9_changes}. Mọi so ghép cặp chính dùng C0 của cùng lượt.',
        f'Thời gian trung bình cho toàn suite8 cấu hình và tất cả rotations: {statistics.mean(r["seconds"] for r in rows.values()):.1f}s/mẫu; đây không phải latency của một cấu hình triển khai.',
        'Các thay đổi cần thiết so code gửi được ghi trong protocol.md: giữ đúng40 và<=45s, không reencode full-cut, giữ prompt baseline, decoder replay, mask theo device, bool triangle, không xóa output và không silently skip.',
        '', '## Góp ý tiếp theo', '',
        'Giữ C3/C5 như giả thuyết cần kiểm chứng, không mặc định trộn thêm nhiều thành phần. Tách cùng một thought/scale để đo delta; thử same-layer và không khuếch đại delta nhỏ; bổ sung đối chứng phản hồi audio đối xứng và several random directions.',
        'Hướng riêng đã chuẩn bị: S1–S6 trong results/symmetric_latent40, được xếp sau Argon trên mỗi server. Hướng này kiểm tra phản hồi đúng tầng, phần tương tác hai nguồn, giới hạn norm và stem-only controls; không dùng cửa sổ, clock từ vựng hoặc policy giảm cố định một modality. S1/S6 là đối chứng visual-asymmetric, không phải ứng viên đối xứng.',
        'Audit đồng hồ đầu vào được lưu riêng trong results/presentation_clock_audit. TimestampAPI Decord có thể không khớp pixel; chưa dùng nó làm oracle để thay đổi đầu vào CCLD.',
        '', '[Tổng hợp các lượt tiếp nối đã chạy](../argon_and_latent_research_20261015.md): same-layer symmetric feedback, query-conditioned thoughts, fitted source/time latent memory và đối chứng thời điểm đọc.',
        '', 'Artifacts: proposal_original.txt/.py, protocol.md, results_s1/s2.jsonl, summary.json, per_sample.csv, table.md và job_s1/s2.json.']
    (FOLDER/'report.md').write_text(manager.markdown_lines(lines),encoding='utf-8')
    (FOLDER/'audit_summary.json').write_text(json.dumps(dict(n=40,rotations=110,replay_error=replay,
        shuffle_norm_error=shuffle,baseline_changed_ids=oldbase_changes,m9_changed_ids=m9_changes,
        holm_p=holm,c7_vs_c6_changes=c7_changes,eligible=eligible),indent=2))
    print('Saved results/argon_ccld40/report.md')

if __name__=='__main__':main()
