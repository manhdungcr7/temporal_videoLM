"""Build Vietnamese reports from completed archived scores; no model inference."""
import csv,json,math
from pathlib import Path
import numpy as np
from chronodesmear_manage import ROOT,CELLS,summary

SUITES={'chronodesmear40':'R5_C0_baseline','chronodesmear_followup40':'N0_native','late_path40':'P0_native_reference'}
NOTES={
'R5_C0_baseline':('Đối chứng','Native cùng phiên; so từng ID với baseline cũ.'),
'R5_C1_causal_desmear_1p':('Thiết kế không task/window','Trừ history có thể mất context hữu ích; điểm không xác nhận tự nó đã xóa causal echo.'),
'R5_C2_stratified_dyn_cd':('Chưa đầy đủ: không window nhưng cần choices','Delta thực tế dùng H_ds và V-only; không chỉ base-static. Static selector không phải event oracle; option-centering chưa có định nghĩa free-form.'),
'R5_C3_balanced_modal_cd':('Chưa phù hợp đầy đủ','Còn speech-window; nhiều yếu tố đổi cùng lúc. Không diễn giải gain là đã bù riêng V_out.'),
'R5_C4_stem_plus_media':('Thiết kế không task/window','Không chạm choices trực tiếp; gain khác C2 nên chưa ablation riêng phạm vi token.'),
'R5_C5_chronodesmear_full':('Chưa đầy đủ: fullAV/no window nhưng cần choices','Static delta vẫn ưu tiên visual; option-centering phụ thuộc MC format. Không suy ra event grounding hoặc novelty từ tên full.'),
'R5_C6_multiseed_shuffled':('Đối chứng cho C2','Ba forward/seed và ensemble log-prob; không là shuffled full C5. Phải xem từng seed và compute.'),
'N0_native':('Đối chứng','Native rerun, kiểm tra parity với Round5.'),
'N1_desmear_reference':('Tham chiếu sao chép C1','Không tính là inference mới độc lập.'),
'N2_pure_delta':('Không task/window','Pure H_AV-H_staticA; giữ recurrence gain cố định để bỏ confound H_ds/V-only trong delta.'),
'N3_pure_delta_centered':('Không task/window; cần choices','So N2 tách riêng mean-centering; hệ số.55 bị norm-rescaling triệt tiêu về magnitude. Chưa có bản free-form.'),
'N4_centered_plus_desmear':('Không task/window; cần choices','So N3 thêm de-smear final, giữ delta/gain cố định. Chưa có bản free-form.'),
'N5_centered_desmear_thought':('Không task/window; cần choices','So N4 thêm thought; weighted vocabulary projection không mặc nhiên semantic alignment. Chưa có bản free-form.'),
'N6_random_stratified':('Negative selection control','Cùng số token mỗi bin nhưN3; một seed2027, không phải event-negative có nhãn.'),
'N7_source_symmetric':('JointAV tốt hơn về cấu trúc, còn cần choices','Đối xứng phép xây delta cho cả hai nguồn, giữ full timeline; center vẫn phụ thuộc options. Không claim causal decomposition.'),
'N8_symmetric_same_layer_stem':('Phù hợp hơn về jointAV','Same-layer22, norm cap15%, lexical stem. Đổi nhiều yếu tố soN7 nên là candidate riêng.'),
'N9_fixed_depth_ensemble':('Diagnostic/readout baseline','Layer22/26/30/34/35 cố định, không chọn bằng gold. Không claim novelty ensemble.'),
'P0_native_reference':('Đối chứng sao chép','Native Round5 cùng inputs/model, không inference baseline mới toàn bộ.'),
'P1_last_mlp_half':('Diagnostic không task/window','MLP x.5 tại last row block28–35; gain không chứng minh event grounding.'),
'P2_last_attn_half':('Diagnostic không task/window','Attention output x.5 tại cùng row/blocks; không attention-mass oracle.'),
'P3_last_mlp_amplify':('Diagnostic hướng đối nghịch','MLP x1.5; kiểm tra chiều tác dụng, không chọn scale theo nhãn.'),
'P4_stem_mlp_half':('Diagnostic phạm vi token','MLP x.5 chỉ lexical stem; không trực tiếp can thiệp choices/format.'),
'P5_last_both_half':('Diagnostic cả hai nhánh','Cùng giữ fullAV/causality, scale attention vàMLP. Không ablation một yếu tố soP1/P2.'),
'P6_last_mlp_random':('Negative control có giới hạn','Một fixed random direction/layer, norm khớp local current MLP update; không trajectory matching hoàn hảo.'),
}

def load_rows(folder):
    return [json.loads(line) for server in (1,2) for line in (folder/f'results_s{server}.jsonl').read_text().splitlines() if line.strip()]

def pair(rows,first,second):
    rescued=[r['idx'] for r in rows if r['predictions'][first]==r['gold'] and r['predictions'][second]!=r['gold']]
    damaged=[r['idx'] for r in rows if r['predictions'][first]!=r['gold'] and r['predictions'][second]==r['gold']]
    return dict(first=first,second=second,rescued=rescued,damaged=damaged,delta=len(rescued)-len(damaged))

def origin_signflip_p(rows,method,baseline,origins):
    differences={}
    for row in rows:
        origin=origins[row['idx']]
        difference=int(row['predictions'][method]==row['gold'])-int(row['predictions'][baseline]==row['gold'])
        differences[origin]=differences.get(origin,0)+difference
    observed=abs(sum(differences.values()));distribution={0:1}
    for value in differences.values():
        if value==0: continue
        new={}
        for total,count in distribution.items():
            for signed in (-value,value): new[total+signed]=new.get(total+signed,0)+count
        distribution=new
    return sum(count for total,count in distribution.items() if abs(total)>=observed)/sum(distribution.values())

def build(suite,baseline):
    folder=ROOT/'results'/suite
    state=summary(folder,baseline)
    if state['completed']!=40: raise RuntimeError((suite,'incomplete',state['completed']))
    counts=json.loads((folder/'summary.json').read_text());rows=load_rows(folder)
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    origins={s['idx']:s['video_path'].replace('\\','/').split('/')[-2] for s in manifest['selected']}
    cluster_p={m:origin_signflip_p(rows,m,baseline,origins) for m in counts if m in rows[0]['predictions']}
    assert all(entry['n']==40 and all(c['n']==5 for c in entry['cells'].values()) for entry in counts.values())
    old={int(r['idx']):r['R3_M0_base'] for r in csv.DictReader((ROOT/'results/argon_latent40/per_sample.csv').open())}
    baseline_diff=[dict(idx=r['idx'],before=old[r['idx']],after=r['predictions'][baseline]) for r in rows if old[r['idx']]!=r['predictions'][baseline]]
    names={'chronodesmear40':'Yoshua Bengio: ChronoDeSmear Round5','chronodesmear_followup40':'Tiếp nối: factor ladder và source-symmetric feedback','late_path40':'Đối chứng attention/MLP cuối'}
    best=max(v['correct'] for v in counts.values())
    best_names=[m for m,v in counts.items() if v['correct']==best]
    passed=[m for m,v in counts.items() if v['correct']>=30 and min(c['correct'] for c in v['cells'].values())>=3]
    lines=[names[suite], '', 'Ngày2026-10-15. Hoàn tất40/40 ID; 5 mỗi task/gold, duration<=45s, mọi110 cyclic rotations.',
        '', 'Đây là development đã tái sử dụng, không phải held-out. Mọi ô là **đúng/sai**; TA:A=Yes/B=No, TF:A=Past/B=Present/C=Future; CMTB là lựa chọn riêng mỗi câu.',
        '',f"Điểm cao nhất: **{best}/40**, cấu hình {', '.join(best_names)}. Cấu hình đạt mục tiêu nội bộ30/40 và>=3/5 mọi ô: {passed or 'không có'}. Mục tiêu này không phải luật nhận CVPR.",
        '',(folder/'table.md').read_text().replace('Correct/wrong','Đúng/sai').replace('TF predictions','TF dự đoán'),
        '', '**Nhận xét từng ý tưởng**', '', '|Method|Phù hợp về thiết kế|Đánh giá từ kết quả và cơ chế|','|---|---|---|']
    for m,v in counts.items():
        fit,note=NOTES.get(m,('Đối chứng random seed','Xem riêng seed, không chỉ ensemble.'))
        weak=[c for c,x in v['cells'].items() if x['correct']<3]
        lines.append(f"|{m}|{fit}|{v['correct']} đúng/{v['wrong']} sai; ô<3/5: {', '.join(weak) or 'không có'}. {note}|")
    lines.extend(['','**So ghép cặp với native cùng lượt**','','|Method|Cứu câu native sai|Làm sai câu native đúng|Exact paired p|Holm p trong suite|','|---|---|---|---|---|'])
    for m,v in counts.items():
        if m==baseline: continue
        lines.append(f"|{m}|{v['rescued']}|{v['damaged']}|{v['paired_p']:.4f}|{v.get('holm_p',1):.4f}|")
    lines.extend(['','Paired exact McNemar hai phía; Holm chỉ hiệu chỉnh nội bộ suite, không xử lý mọi lần chọn phương pháp trong lịch sử. N nhỏ và tái dùng dev giới hạn suy luận gain ổn định.',
        '',f'Baseline khác lịch sử trên{len(baseline_diff)}/40 mẫu: {baseline_diff}.',
        '', '**Giới hạn và artifacts**','',
        'Không có task router, gold-dependent selection hay ghép oracle. Không suy ra CVPR acceptance/novelty hoặc broad deployment từ điểm40 này. Các bước mask/steering/soft thoughts và scalar gating cần đối chiếu prior art; kết quả local không tự xác lập novelty.',
        '', 'Protocol trước chạy: [protocol.md](protocol.md). Raw sample scores: results_s1/s2.jsonl. Chi tiết từng câu: per_sample.csv. Mọi per-rotation scores và archive audit được lưu riêng; kiểm tra trạng thái audit trước khi gọi raw archive hoàn tất.'])
    archive_audits=[]
    for server in (1,2):
        p=folder/f'archive_audit_s{server}.json'
        if p.exists(): archive_audits.append(json.loads(p.read_text()))
    lines.extend(['',f'Raw archives đã audit: {len(archive_audits)}/2 server; tổng rotations={sum(a["rotations"] for a in archive_audits)}.'])
    if archive_audits:
        ties={}
        for audit in archive_audits:
            for m,ids in audit['tie_ids'].items(): ties.setdefault(m,[]).extend(ids)
        lines.extend(['',f'Các argmax có margin<1e-6 (tie chọn A theo np.argmax, không sửa hậu kiểm): {ties}.'])
    extra=dict(baseline_differences=baseline_diff,source_origins=len(set(origins.values())),source_cluster_signflip_p=cluster_p)
    changes={}
    def normalized(values,rotations):
        x=np.array(values,dtype=float)/rotations;x=x-x.max();p=np.exp(x)
        return p/p.sum()
    for method in rows[0]['scores']:
        tv=[]
        for row in rows:
            num_rotations=len(row['audit'])
            p=normalized(row['scores'][method],num_rotations)
            p0=normalized(row['scores'][baseline],num_rotations)
            tv.append(float(.5*np.abs(p-p0).sum()))
        changes[method]=dict(changed_predictions=sum(r['predictions'][method]!=r['predictions'][baseline] for r in rows),
            mean_tv=float(np.mean(tv)),median_tv=float(np.median(tv)))
    extra['decision_changes']=changes
    lines.extend(['','**Độ nhạy quyết định**','','|Method|Đổi đáp án/40|Mean TV|','|---|---|---|'])
    for method,record in changes.items(): lines.append(f"|{method}|{record['changed_predictions']}|{record['mean_tv']:.4f}|")
    lines.extend(['','TV dùng mean log-prob qua rotations rồi normalize trên choices. Không đổi argmax không đồng nghĩa mọi hidden state/path bị bỏ qua; đây không phải event-grounding metric.'])
    lines.extend(['',f'Bộ40 có{len(set(origins.values()))} nguồn gốc. McNemar theo clip chỉ thăm dò khi có cùng nguồn lặp. Paired sign-flip theo source origin (chưa hiệu chỉnh multiple tests): {cluster_p}. Cả hai loại kiểm định đều chưa xử lý selection bias do tái dùng development.'])
    if suite=='chronodesmear40':
        lines.extend(['','**Góp ý gửi Yoshua Bengio**','',
            'Giữ fullAV/no task routing là tiến bộ về thiết kế. Tuy nhiên C2 không phải delta nền tĩnh thuần, C3 vẫn dùng cửa sổ và không kiểm chứng riêng việc bù V_out; C6 là control C2. Không gọi matrix này orthogonal factorial.',
            '', 'Center options không bảo đảm prior bất biến: hệ số.55 bị per-row norm rescaling triệt tiêu về magnitude. Trong toy linearization của đề xuất, thay H_base bằng H_Vall không bù V_out mà còn trừ thêm audio. Xem [mechanism_review.md](mechanism_review.md).',
            '', 'Cosine resonance ba vùng không phải shift test hoặc event grounding. Không có annotation independently verified thì không claim đã xác nhận temporal link. Causal accumulation có thể mang context hữu ích, không mặc định toàn bộ là echo gây lỗi.',
            '', 'Option-centering trong C2/C3/C5 cần choices, không chỉ là scoring wrapper. Chưa có định nghĩa hoặc đánh giá bản không choices, vì thế chưa đáp ứng đầy đủ yêu cầu QA tự do/mọi ngữ cảnh. Không task router là điều kiện khác với độc lập định dạng câu hỏi.',
            '', '[Causality Matters](https://ojs.aaai.org/index.php/AAAI/article/view/37856) phân tích temporal pathway qua inter-frame và query trên các VideoLM; [Qwen2.5-Omni Technical Report](https://arxiv.org/html/2503.20215v1) mô tả2s interleaving/TMRoPE. Đây là căn cứ đặt giả thuyết/controls, không là bằng chứng hiệu quả của ChronoDeSmear.'])
        extra['C5_vs_C2']=pair(rows,'R5_C5_chronodesmear_full','R5_C2_stratified_dyn_cd')
    if suite=='chronodesmear_followup40':
        pairs=[('N3_pure_delta_centered','N2_pure_delta'),('N4_centered_plus_desmear','N3_pure_delta_centered'),
            ('N5_centered_desmear_thought','N4_centered_plus_desmear'),('N3_pure_delta_centered','N6_random_stratified'),
            ('N7_source_symmetric','N3_pure_delta_centered'),('N8_symmetric_same_layer_stem','N7_source_symmetric')]
        extra['factor_pairs']=[pair(rows,*p) for p in pairs]
        lines.extend(['','**Các thay đổi từng yếu tố**','','|So sánh A với B|A cứu/B sai|A sai/B đúng|Chênh số đúng|','|---|---|---|---|'])
        for p in extra['factor_pairs']: lines.append(f"|{p['first']} vs {p['second']}|{len(p['rescued'])}|{len(p['damaged'])}|{p['delta']:+d}|")
        lens={}
        lines.extend(['','**Logit lens native theo tầng cố định**','','|Block index|TA/A|TA/B|TF/A|TF/B|TF/C|CMTB/A|CMTB/B|CMTB/C|Tổng đúng/sai|Đúng tầng này/sai cuối|Sai tầng này/đúng cuối|','|---|---|---|---|---|---|---|---|---|---|---|---|'])
        for li in ('4','8','12','16','18','22','26','30','34','35'):
            cells={c:sum(r['layer_predictions'][li]==r['gold'] for r in rows if r['task']+'/'+r['gold']==c) for c in CELLS}
            correct=sum(cells.values())
            late_lost=[r['idx'] for r in rows if r['layer_predictions'][li]==r['gold'] and r['predictions'][baseline]!=r['gold']]
            late_gained=[r['idx'] for r in rows if r['layer_predictions'][li]!=r['gold'] and r['predictions'][baseline]==r['gold']]
            lens[li]=dict(correct=correct,cells=cells,late_lost=late_lost,late_gained=late_gained)
            lines.append('|'+ '|'.join([li]+[f'{cells[c]}/{5-cells[c]}' for c in CELLS]+[f'{correct}/{40-correct}',str(len(late_lost)),str(len(late_gained))])+'|')
        extra['layer_lens']=lens
        lines.extend(['','Logit lens chưa calibrated và có thể đọc prior chữ cái; không dùng gold để chọn layer cho N9. Correct-at-middle/wrong-final là quan sát readout, chưa đủ chứng minh causal late overwrite.',
            '', 'N8 đổi cả layer, norm cap và token span so N7 nên không gán chênh lệch riêng cho một yếu tố. Low-innovation/static và mean norm ratio chỉ là proxy representation, không event annotation.'])
    if suite=='late_path40':
        extra['path_pairs']=[pair(rows,'P1_last_mlp_half','P3_last_mlp_amplify'),pair(rows,'P1_last_mlp_half','P6_last_mlp_random')]
        temporal_changes={m:[r['idx'] for r in rows if r['task']=='TF' and r['predictions'][m]!=r['predictions'][baseline]] for m in rows[0]['scores']}
        extra['temporal_prediction_changes']=temporal_changes
        lines.extend(['',f'TF prediction changes so native: {temporal_changes}. P3 chỉ sửa idx1898 CMTB/C từ A sang C; native sample này có exact score tie qua rotations, nên không xem +1 là gain grounding ổn định. P1/P2/P5/P6 làm sai idx2109 TA/B; P4 giữ nguyên mọi argmax.'])
        lines.extend(['','Suy luận về attention/MLP muộn phải đọc cùng layer lens và control random. Tăng accuracy hoặc đổi TF distribution chưa chứng minh restored grounding; null result chưa chứng minh model không liên kết thời gian. Không đề xuất scalar gating đơn lẻ như novelty CVPR.'])
        audits=[a for row in rows for a in row['audit']]
        parity=[c['full_parity'] for a in audits for c in a['condition_audit'].values() if c.get('full_parity') is not None]
        extra['cache_audit']=dict(rotations=len(audits),max_full_condition_parity=max(parity) if parity else None,
            max_native_prefix_error=max(a['native_prefix_error'] for a in audits),
            max_native_reference_error=max(a['native_spotcheck'] for a in audits),
            full_condition_calls=sum(not c['prefix_used'] for a in audits for c in a['condition_audit'].values()))
        lines.extend(['',f'Cache audit: {extra["cache_audit"]}. Fallback full giữ ngưỡng.03; không nới để ép chạy cached. Thời gian condition có thể gồm parity reference, không dùng làm latency deployment đầy đủ.'])
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (folder/'analysis.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
    return dict(suite=suite,best_correct=best,best_methods=best_names,n=40,archive_servers=len(archive_audits))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('suite',choices=list(SUITES)+['all']);args=parser.parse_args()
    for suite,baseline in SUITES.items():
        if args.suite in ('all',suite): print(json.dumps(build(suite,baseline)))
