"""Exact per-cell and paired counts for Argon's fixed40 experiment."""
import csv,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/argon_latent40'
MANIFEST=json.loads((ROOT/'results/research40_manifest.json').read_text())
CELLS=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def pred(score):return chr(65+max(range(len(score)),key=lambda i:score[i]))

def summarize():
    rows=[r for server in (1,2) for r in load(FOLDER/f'results_s{server}.jsonl')]
    selected={r['idx']:r for r in MANIFEST['selected']}
    assert len(rows)==len({r['idx'] for r in rows})==40,'Require all40 exactly once'
    assert {r['idx'] for r in rows}==set(selected),'Wrong sample set'
    methods=list(rows[0]['scores'])
    for row in rows:
        assert row['manifest_sha256']==MANIFEST['manifest_sha256']
        assert (row['task'],row['gold'])==(selected[row['idx']]['task'],selected[row['idx']]['gold'])
        assert set(row['scores'])==set(methods)
        assert all(math.isfinite(x) for score in row['scores'].values() for x in score)
    baseline={r['idx']:r for server in (1,2) for r in load(ROOT/f'results/research40_baseline_s{server}.jsonl')}
    summary={}
    for method in methods:
        counts={c:dict(correct=0,wrong=0,total=0) for c in CELLS};rescue=[];regression=[];hist_change=[]
        for row in rows:
            correct=pred(row['scores'][method])==row['gold'];cell=row['task']+'/'+row['gold']
            counts[cell]['correct']+=int(correct);counts[cell]['wrong']+=int(not correct);counts[cell]['total']+=1
            base_ok=pred(row['scores'][methods[0]])==row['gold']
            if correct and not base_ok:rescue.append(row['idx'])
            if base_ok and not correct:regression.append(row['idx'])
            old=pred(baseline[row['idx']]['scores']['base']);new=pred(row['scores'][methods[0]])
            if old!=new:hist_change.append(dict(idx=row['idx'],old=old,new=new))
        n=sum(c['correct'] for c in counts.values());discord=len(rescue)+len(regression)
        p=min(1.,2*sum(math.comb(discord,x) for x in range(min(len(rescue),len(regression))+1))/2**discord) if discord else 1.
        summary[method]=dict(correct=n,wrong=40-n,total=40,accuracy=100*n/40,
            cells=counts,tasks={t:dict(correct=sum(c['correct'] for cell,c in counts.items() if cell.startswith(t+'/')),
                total=sum(c['total'] for cell,c in counts.items() if cell.startswith(t+'/'))) for t in ('TA','TF','CMTB')},
            rescues=rescue,regressions=regression,paired_exact_p=p,
            minimum_cell_correct=min(c['correct'] for c in counts.values()))
    data=dict(manifest_sha256=MANIFEST['manifest_sha256'],n=40,methods=summary,
        historical_baseline_changes=hist_change,
        historical_baseline_correct=sum(pred(baseline[r['idx']]['scores']['base'])==r['gold'] for r in rows),
        prediction_counts={m:{t:{label:sum(pred(r['scores'][m])==label for r in rows if r['task']==t)
            for label in ('AB' if t=='TA' else 'ABC')} for t in ('TA','TF','CMTB')} for m in methods},
        max_replay_logit_error=max(a['replay_error'] for r in rows for a in r['audit']),
        max_tokens=max(a['tokens'] for r in rows for a in r['audit']),
        max_dense_mask_MB=max(a['mask_MB'] for r in rows for a in r['audit']),
        max_seconds=max(r['seconds'] for r in rows),total_sample_seconds=sum(r['seconds'] for r in rows))
    (FOLDER/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (FOLDER/'per_sample.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['idx','task','gold',*methods])
        for row in sorted(rows,key=lambda r:r['idx']):writer.writerow([row['idx'],row['task'],row['gold'],*[pred(row['scores'][m]) for m in methods]])
    lines=['# Argon LatentChrono-CoT: kết quả thực nghiệm40 mẫu','',
        'Đã hoàn tất40/40, 5 mẫu mỗi task/nhãn; cùng raw AV và mọi cyclic answer rotations.',
        'Bảng dưới ghi **đúng/sai**. A/B/C là nhãn ground-truth, không phải tên điều kiện.',
        '', '|Method|'+ '|'.join(CELLS)+'|Tổng đúng/sai|Accuracy|',
        '|---|'+ '|'.join(['---']*(len(CELLS)+2))+'|']
    for m,z in summary.items():
        lines.append('|'+m+'|'+ '|'.join(f"{z['cells'][c]['correct']}/{z['cells'][c]['wrong']}" for c in CELLS)+
            f"|{z['correct']}/{z['wrong']}|{z['accuracy']:.1f}%|")
    lines+=['','## So sánh paired với M0','', '|Method|Cứu câu sai|Làm hỏng câu đúng|Δ số đúng|Exact paired p (chưa điều chỉnh multiple tests)|',
        '|---|---|---|---|---|']
    for m,z in summary.items():lines.append(f"|{m}|{len(z['rescues'])}|{len(z['regressions'])}|{z['correct']-summary[methods[0]]['correct']:+d}|{z['paired_exact_p']:.4f}|")
    lines+=['','## Kiểm tra kỹ thuật','',
        f"Manifest `{data['manifest_sha256']}`. Max sequence={data['max_tokens']}; dense fp16 mask={data['max_dense_mask_MB']}MiB/device; max native/decoder replay error={data['max_replay_logit_error']:.5f}.",
        f"Baseline lịch sử đúng {data['historical_baseline_correct']}/40; khác dự đoán {len(hist_change)}/40 mẫu so với baseline rerun phiên này.",
        'Dữ liệu raw: results_s1.jsonl, results_s2.jsonl; dự đoán từng câu: per_sample.csv; số đếm đầy đủ: summary.json.',
        'Chi tiết đề xuất và thay đổi triển khai: protocol.md; nguyên văn: proposal_original.txt và proposal_original.py.',
        '', '## Phân tích và lời khuyên', '',
        'M2 `true_gate_audio` đứng đầu với26/40, hơn M0 là8 câu. Nó cứu14 câu M0 sai và làm sai6 câu M0 đúng; paired exact McNemar p=0.115, chưa điều chỉnh cho nhiều cấu hình. Trên N=40 đây là tín hiệu đáng kiểm tra tiếp, chưa phải bằng chứng thống kê thuyết phục. M8 đạt24/40; so ghép cặp, M2 cứu3 câu M8 sai và làm sai1 câu M8 đúng (p=0.625). M2 hơn M8 hai câu, nhưng chưa tách được tác dụng gate/clock/foreground/PMI khỏi latent loop.',
        '', 'M2 cải thiện CMTB lên13/15 từ5/15 của M0, nhất là CMTB/C từ0/5 lên4/5; TF/B tăng từ1/5 lên5/5. Đổi lại TA/A giảm từ5/5 xuống3/5 và M2 vẫn0/5 ở TF/A lẫn TF/C. Có vẻ cấu hình đổi prior/decision boundary theo hướng lợi cho vài cell temporal binding, trong khi làm sụp hai nhãn thời gian. Các latent method M3–M6 chỉ đạt50–55%; unified M6 đạt50%, không hơn rõ thành phần đơn. Text-only PMI làm baseline giảm18 xuống16.',
        '', 'M2 đoán Present cho15/15 câu TF: TF/B5/5 đến từ collapse, không chứng minh hiểu đồng thời. M8 tên “recurrent only” là đối chứng gần đúng: delta vẫn lấy từ hard/anti và clock. M2 và M7 cũng dùng salience/clock từ phép chiếu vocabulary chung, dù không inject continuous thought hay delta trực tiếp. M4 đạt20/40 nhưng gồm cả clock/mask/PMI; M9 thought-only đạt19/40 và TF8/15 so với5/15 của M0. Chưa có factorial ablation đủ sạch để quy đóng góp cho top-48 projection, clock, mask, hay recurrence.',
        '', 'Mỗi cell chỉ có5 mẫu. Không method nào đạt development criterion30/40 hoặc ít nhất3/5 ở mọi cell. M2 được chọn sau10 cấu hình trên chính bộ development này nên có selection bias; đây không phải held-out test. Baseline rerun đạt18/40 và khác dự đoán baseline lịch sử1 mẫu. So sánh nội bộ vẫn dùng cùng raw AV, cyclic rotations và replay khớp chính xác.',
        '', 'Lời khuyên Argon: giữ latent feedback như giả thuyết, nhưng chưa gọi là đã tái hiện Coconut/recurrent-depth hoặc chứng minh “zero modality gap”. Coconut hồi tiếp last hidden state làm next input embedding; M4 ở đây qua top-k logit-lens, tính kỳ vọng token embeddings rồi cộng residual vào block0–2, là cơ chế khác ([Coconut](https://arxiv.org/abs/2412.06769)). Recurrent Depth dùng recurrent block được huấn luyện, khác nối block22 về block10 tùy ý ([Recurrent Depth](https://arxiv.org/abs/2502.05171)). Unsilencing Visual Latents tối ưu latent với contrastive alignment và confidence progression; top-k expectation chưa triển khai objective đó ([Unsilencing Visual Latents](https://arxiv.org/abs/2605.02735)).',
        '', 'Bước tiếp theo: preregister ablation trên held-out mới, giữ cố định mask/gate/clock, so M0, M2, M8, M4 và thiết kế factorial tách từng thành phần. Đo per-sample, dùng McNemar và hiệu chỉnh nhiều phép thử; tập trung xem TA/A và TF/A,C có tiếp tục sụp không. Chỉ mở rộng nếu lợi thế M2 lặp lại ngoài development40.',
        '', 'Job hoàn tất40/40, không OOM; native/decoder replay sai số logit tuyệt đối tối đa0.0. Thời gian trung bình khoảng70 giây/mẫu với10 điều kiện và cyclic rotations.', '']
    (FOLDER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({m:{k:v for k,v in z.items() if k in ('correct','wrong','accuracy')} for m,z in summary.items()},indent=2))
    return data

if __name__=='__main__':summarize()
