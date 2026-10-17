"""Local aggregation, archive verification and descriptive Stage-0 v6 report."""
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/pcebr_stage0_v6'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,value):(D/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
main=[read(f'main_results_s{s}.json') for s in (1,2)]
audits=[read(f'archive_audit_s{s}.json') for s in (1,2)]
for key in ('M0','M1_native','M1_pooled','M2'):assert main[0][key]==main[1][key],key
source=(D/'stage0_pcebr_v6_final_original.py').read_text(encoding='utf-8')
source_sha=hashlib.sha256(source.encode()).hexdigest()
assert source_sha=='6108077a93f19e67f2a58b9d334b448d853ca117a1bb30c317dc8c6f49f11515'
archives=[]
for s in (1,2):
    inp=read(f'input_audit_s{s}.json')
    assert inp['source_lf_sha256']==source_sha
    for name,digest in inp['caches'].items():assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==digest
    index=read(f'archive_index_s{s}.json');path=D/f'raw_artifacts_s{s}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected={x['name']:x for x in audits[s-1]['files']}
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)
        for m in members:
            assert m.isfile() and not m.issym() and '..' not in Path(m.name).parts
            item=expected[m.name];b=tar.extractfile(m).read()
            assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
    completion=read(f'completion_s{s}.json')
    assert completion['parameter_versions_unchanged'] and completion['hooks_before']==completion['hooks_after'] and completion['cuda_health']==[2.,2.]
    archives.append(dict(**index,verified=True))

manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
rows=[];parities=[]
for s in (1,2):
    sample_rows=main[s-1]['M3']['samples'];micro=read(f'm3_audit_s{s}.json')
    assert [r['idx'] for r in sample_rows]==[r['idx'] for r in manifest[f'server{s}'][:4]]
    assert len(sample_rows)==len(micro)==4
    for r,a in zip(sample_rows,micro):
        assert r['kernel_noop_layer_invocations']==6
        assert a['tail_roundtrip'] and a['hook_option_disjoint']
        assert not any(x in a['freeform_input_text'] for x in ('(A)','(B)','(C)','Answer with only the letter'))
        rows.append(dict(server=s,**r));parities.append(a['legacy_same_runtime_archived_source_parity'])
pred_keys=('mc_pred_native','mc_pred_legacy_v5_vo_tangent','mc_pred_routed_fwd','mc_pred_no_desink','mc_pred_order_sign_rev','mc_pred_slot_profile_rev','mc_pred_rel_only','mc_pred_order_only')
totals={k:sum(r[k]==r['gold_letter'] for r in rows) for k in pred_keys}
cells={}
for task,letters in (('TA','AB'),('TF','ABC'),('CMTB','ABC')):
    for gold in letters:
        rr=[r for r in rows if r['task']==task and r['gold_letter']==gold]
        cells[task+'/'+gold]=dict(n=len(rr),correct={k:sum(r[k]==gold for r in rr) for k in pred_keys})
avg=lambda k:statistics.mean(r[k] for r in rows)
summary=dict(total=8,totals=totals,cells=cells,
    rescued=[r['idx'] for r in rows if r['mc_pred_native']!=r['gold_letter'] and r['mc_pred_routed_fwd']==r['gold_letter']],
    harmed=[r['idx'] for r in rows if r['mc_pred_native']==r['gold_letter'] and r['mc_pred_routed_fwd']!=r['gold_letter']],
    changed=[r['idx'] for r in rows if r['mc_pred_native']!=r['mc_pred_routed_fwd']],
    means={k:avg(k) for k in ('confidence_gate_gamma_true_query','confidence_gate_gamma_wrong_query','confidence_gate_gamma_semantic_wrong_query','tv_late_kslot_swap','tv_mid_kslot_swap','tv_mid_slot_profile_swap','tv_mid_routed_vs_nat','tv_mid_irrel_slot_vs_nat','tv_mid_rand3seed_vs_nat')},
    legacy_parities=parities,replicas_identical_M0_M1_M2=True)
control_rows=[r for s in (1,2) for r in read(f'control_accuracy_s{s}.json')]
assert len(control_rows)==8 and all(r['full_replay_probability_error']==0 for r in control_rows)
summary['original_control_correct']={k:sum(r['conditions'][k]['correct'] for r in control_rows) for k in control_rows[0]['conditions']}
moments=[r for s in (1,2) for r in read(f'wasserstein_signed_mean_identity_s{s}.json')]
summary['signed_w_equals_mean_lag_max_abs_error']=max(r['absolute_error'] for r in moments)
summary['heuristic_freeform_aligned']={branch:sum(r[f'freeform_{branch}_audit']['heuristic_clause_aligned'] for r in rows) for branch in ('native','routed')}
hist=[]
for s in (1,2):
    old=read_old=json.loads((ROOT/'results/pcebr_stage0_v5'/f'm3_audit_s{s}.json').read_text(encoding='utf-8'))
    current=read(f'm3_audit_s{s}.json')
    for a,b in zip(old,current):
        assert a['result']['idx']==b['result']['idx']
        hist.append(dict(idx=b['result']['idx'],native_probability_max_abs_drift=max(abs(x-y) for x,y in zip(a['probabilities']['p_nat'],b['probabilities']['p_nat'])),
            v5_probability_max_abs_drift=max(abs(x-y) for x,y in zip(a['probabilities']['p_mid_fwd'],b['probabilities']['p_legacy_v5'])),
            native_prediction_equal=a['result']['mc_pred_native']==b['result']['mc_pred_native']))
save('historical_comparison.json',hist)
save('summary.json',summary);save('m3_per_sample.json',rows)
flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
with (D/'m3_per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
save('completion_audit.json',dict(archives=archives,source_lf_sha256=source_sha,summary=summary))

lines=['# CAOR-WEMD / PC-EBR Stage-0 v6 — kết quả và kiểm toán, 2026-10-16','',
f"**Micro8: native {totals['mc_pred_native']}/8, legacy v5 {totals['mc_pred_legacy_v5_vo_tangent']}/8, CAOR full {totals['mc_pred_routed_fwd']}/8. Cứu {len(summary['rescued'])}, làm hỏng {len(summary['harmed'])}.**",'',
'M0/M1/M2 chạy lại cùng dữ liệu trên hai server T4; đây là hai replicas, không thêm video độc lập. M1 có 252 pseudo pairs: 202 train/16 origins, 50 held-out/4 origins đã xem ở các vòng trước. Native cache là post-media last-token readout block index27; pooled là media-token mean cùng block. Nhãn mô tả/event presence chưa được annotation độc lập.','',
'Source gốc giữ nguyên; wrapper chỉ lưu observations và kiểm tra bổ sung. Xem [protocol.md](protocol.md) và [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md). M3 chỉ chạy 8 mẫu rotation0 đúng manifest; chưa chạy đủ QA40 hoặc reserve400.','',
'## M0','',f"Kiểm tra gốc qua cả hai server. Audit độc lập ví dụ CDF giao nhau: `{json.dumps(read('extra_unit_audit_s1.json'))}`.",'',
'Phi có rank số học6 trên bộ toy đã định; local non-media mass invariance và reversal checks là unit checks, không phải bằng chứng temporal QA đúng. Zero-noop ép gate1 và gọi kernel đủ6 layer mỗi mẫu; kiểm tra parity logit được báo riêng.','',
'## M1 — số đúng qua ba seeds','',
'Global/Exact/Multipos có mẫu số50. Global chấm mô tả top1 bằng chuỗi normalized content, không yêu cầu đúng recording/interval; Exact yêu cầu đúng index trong cùng recording/modality; Multipos chấp nhận normalized description trùng trong group. Nonnull có mẫu số31 theo substring filter, chưa phải nhãn event tồn tại. Số sai = mẫu số − số đúng. Các seed không tạo thêm held-out origins.','']
for key in ('M1_native','M1_pooled'):
    lines += [f'### {key}','','| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global / Exact / Multipos / Nonnull % |','|---|---|---|---|---|---|']
    for name,v in main[0][key]['summary'].items():
        means=' / '.join(str(v[k]) for k in ('mean_global_pct','mean_within_exact_pct','mean_within_multipos_pct','mean_within_nonnull_pct'))
        lines.append(f"| {name} | {v['global_hits_3seeds']} | {v['exact_hits_3seeds']} | {v['multipos_hits_3seeds']} | {v['nonnull_hits_3seeds']} | {means} |")
    lines.append('')
lines += ['## M2 — probes và đối chứng','',
'Các episodes là biến đổi từ cùng4 held-out origins. Reversal/Roll tác động interval coordinates/banks; tên “Physical” trong source không có nghĩa đã dịch waveform hoặc đảo frame. Acc3 dùng argmax riêng3 relation logits trên subset không foil. Paired chỉ tính cặp còn đủ thuận/đảo sau balancing.','']
pi=0
for suite,probes in main[0]['M2'].items():
    if suite=='foil_proxy_and_hop_audit':continue
    lines += [f'### {suite}','','| Probe | Đúng/sai/tổng4 | Acc4 % | Acc3 % | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |','|---|---|---:|---:|---|---|---|---|---|']
    for name,v in probes.items():
        a=audits[0]['probes'][pi];pi+=1
        assert round(100*a['correct4']/a['total'],1)==v['overall_4way_pct']
        cc=[f"{a['by_class'][str(c)]['correct']}/{a['by_class'][str(c)]['total']}" for c in range(4)]
        paired=f"{a['paired_correct']}/{a['paired_total']}" if a['paired_total'] else '—'
        lines.append(f"| {name} | {a['correct4']}/{a['total']-a['correct4']}/{a['total']} | {v['overall_4way_pct']} | {v['overall_3way_no_foil_pct']} | {' | '.join(cc)} | {paired} |")
    lines.append('')
lines += ['Foil selection và hop diagnostics trước/sau balancing lưu đầy đủ trong main_results_s1/s2.json. Foils vẫn dựa vào caption/embedding disjointness, không phải annotation event vắng mặt. “Oracle hop1 anchor” thay vector context cho bridge, vẫn giữ lg1/p1 trong time features; chưa phải oracle sửa toàn bộ hop1 localization hoặc một upper bound toán học.','',
'## M3 — micro8','',
'| ID | Task/gold | Native | V5 | Full | No-desink | Order-sign đảo | Slot-profile đảo | Rel-only | Order-only |','|---|---|---|---|---|---|---|---|---|---|']
for r in rows:lines.append(f"| {r['idx']} | {r['task']}/{r['gold_letter']} | "+' | '.join(r[k] for k in pred_keys)+' |')
lines += ['', '| Task/gold | N | Native đúng/sai | V5 đúng/sai | Full đúng/sai |','|---|---:|---|---|---|']
for name,c in cells.items():
    n=c['n'];vals=[f"{c['correct'][k]}/{n-c['correct'][k]}" if n else 'Chưa đo' for k in pred_keys[:3]]
    lines.append(f"| {name} | {n} | {' | '.join(vals)} |")
lines += ['',f"Cứu: {summary['rescued']}; làm hỏng: {summary['harmed']}; thay đổi prediction: {summary['changed']}.",'',
    '| ID | Gate true / permuted / semantic wrong | TV full/native | TV random/native | TV mid sign swap / slot profile swap | Kernel no-op error |','|---|---|---:|---|---:|']
for r in rows:
    lines.append(f"| {r['idx']} | {r['confidence_gate_gamma_true_query']} / {r['confidence_gate_gamma_wrong_query']} / {r['confidence_gate_gamma_semantic_wrong_query']} | {r['tv_mid_routed_vs_nat']} | {r['tv_mid_rand3seed_vs_nat']} | {r['tv_mid_kslot_swap']} / {r['tv_mid_slot_profile_swap']} | {r['kernel_noop_zero_perturb_max_err']} |")
lines += ['',f"Trung bình: `{json.dumps(summary['means'])}`.",'',
'Legacy parity được kiểm tra bằng original v5 source trên cùng inputs/runtime; không mặc định trùng bit với lần chạy lịch sử khác runtime. Kết quả từng mẫu lưu trong m3_audit_s1/s2.json.','',
'Accuracy của các control có sẵn, chạy lại cùng source/hệ số/seeds để ghi prediction: '+json.dumps(summary['original_control_correct'])+'. Full replay probability error=0 ở8/8. Các control khớp L2 norm của token-bias profile theo modality; random/irrelevant tắt row de-sinking và audio damping, nên không coi là toàn bộ attention differential cùng norm.','',
f"Audit signed transport: max |(gated W+−W−)−gated normalized mean lag| = {summary['signed_w_equals_mean_lag_max_abs_error']:.3g}. Xem [math_note_signed_transport.md](math_note_signed_transport.md); điều này áp dụng riêng term W trong M3, không suy toàn bộ M2/M3 chỉ có mean-time.",'',
'### Sinh tự do32 tokens','',
'Điểm heuristic_clause_aligned chỉ là rule-based audit, không phải semantic accuracy được annotation. Đọc trực tiếp văn bản, quan hệ được nói tới và reference event. '+json.dumps(summary['heuristic_freeform_aligned'])+'. Ví dụ2150 và1821 nói rõ event sau speech nhưng bị flag generic/false; không suy chất lượng QA từ số heuristic này.','']
for r in rows:
    lines += [f"**{r['idx']} — {r['task']}/{r['gold_letter']} ({r['gold_answer_text']})**",'',f"Native: {r['freeform_32tok_native']}",'',f"CAOR: {r['freeform_32tok_routed']}",'',f"Native audit: `{json.dumps(r['freeform_native_audit'],ensure_ascii=False)}`",'',f"CAOR audit: `{json.dumps(r['freeform_routed_audit'],ensure_ascii=False)}`",'']
lines += ['## Artifacts','',
f"Hai archive được kiểm tra SHA256 và từng entry local, không extract. Mỗi server có {audits[0]['raw_files']} raw PT files; {audits[0]['retrieval_queries']} query rows được recompute; {len(audits[0]['probes'])} linear probes được tính lại từ logits. Backbone parameter version counters giữ nguyên, hook count trước/sau bằng nhau, CUDA khỏe. Không suy novelty/CVPR acceptance từ Stage-0.",'']
lines += ['Lỗi JSON serialization ở observation wrapper được giữ tại audit_wrapper_error; source thuật toán không đổi. M0/M1 đã hoàn thành được dùng tiếp; M2 chạy lại sau khi sửa ghi metadata, M3 chạy sau đó. Audit thêm về signed-Wasserstein/mean-lag và teacher/oracle feature differences lưu riêng; không dùng chúng để thay phương pháp mainrun.','']
(D/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
