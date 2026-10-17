"""Reaggregate and document v5 original runs and explicitly exploratory controls."""
import ast
import csv
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/pcebr_stage0_v5'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,obj):(D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
main=[read(f'main_results_s{s}.json') for s in (1,2)]
raw=[read(f'archive_audit_s{s}.json') for s in (1,2)]
controls=[read(f'controls_s{s}.json') for s in (1,2)]
keys=('Algebraic_Bivector_and_ZeroFloor_Unit_Checks','M1_native_orthogonal_factorial','M1_pooled_orthogonal_factorial','M2_incremental_ablation_and_learned_hop2')
for key in keys:assert main[0][key]==main[1][key],key
source=(D/'stage0_pcebr_v5_original.py').read_bytes()
assert source==(ROOT/'server_jobs/stage0_pcebr_v5_bivector_vo_routing.py').read_bytes()
ast.parse(source.decode('utf-8'))
source_tree=ast.parse(source.decode('utf-8'));metric_ns={'re':re}
metric_nodes=[]
for node in source_tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('STOPWORDS','GENERIC_EVASION_PATTERNS') for t in node.targets):metric_nodes.append(node)
    if isinstance(node,ast.FunctionDef) and node.name in ('content_tokens','audit_freeform_generation_quality'):metric_nodes.append(node)
exec(compile(ast.Module(body=metric_nodes,type_ignores=[]),'<original_freeform_metric>','exec'),metric_ns)
metric_counterexamples=[]
for text,task,gold in [('It is not happening at the same time in the video.','TA','Yes'),('The video shows flour being put away; milk and egg are not being added.','CMTB','add milk and one egg whisk together'),('After the speech, the camera changes angle.','TF','Future')]:
    metric_counterexamples.append(dict(text=text,task=task,gold=gold,audit=metric_ns['audit_freeform_generation_quality'](text,30.,task,gold)))
save('freeform_metric_counterexamples.json',metric_counterexamples)
archive_checks=[]
for s in (1,2):
    inp=read(f'input_audit_s{s}.json');assert inp['source_sha256']==hashlib.sha256(source).hexdigest()
    for name,digest in inp['caches'].items():assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==digest
    index=read(f'archive_index_s{s}.json');path=D/f'raw_artifacts_s{s}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected={f['name']:f for f in raw[s-1]['files']}
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)
        for member in members:
            assert member.isfile() and not member.issym() and '..' not in Path(member.name).parts and member.name.startswith('artifacts/')
            value=tar.extractfile(member).read();meta=expected[Path(member.name).name]
            assert len(value)==meta['bytes'] and hashlib.sha256(value).hexdigest()==meta['sha256']
    assert raw[s-1]['retrieval_metrics_recomputed_exact'] and raw[s-1]['retrieval_queries']==1800
    assert len(raw[s-1]['probes'])==27
    assert controls[s-1]['intervention_hook_count']==0 and controls[s-1]['cuda_ok']
    archive_checks.append(dict(**index,verified=True))

manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
rows=[];extra=[]
for s in (1,2):
    part=main[s-1]['M3_bivector_vo_tangent_routing']['samples'];aud=read(f'm3_audit_s{s}.json')
    assert [r['idx'] for r in part]==[r['idx'] for r in manifest[f'server{s}'][:4]]
    assert len(part)==len(aud)==4
    for r,a,sample in zip(part,aud,manifest[f'server{s}'][:4]):
        assert r['gold_letter']==sample['gold'] and r['native_replay_max_err']==0
        assert a['tail_roundtrip'] and not set(a['hook_rows'])&set(a['option_rows'])
        assert all(marker not in a['freeform_input_text'] for marker in ('(A)','(B)','(C)','Answer with only the letter'))
        rows.append(dict(shard=s,**r))
    for a in controls[s-1]['routing_ablations']:
        assert a['conditions']['zero_scale_parity']['native_logit_max_diff']==0
        extra.append(a)
avg=lambda k:statistics.mean(r[k] for r in rows)
cells={}
for task,letters in [('TA','AB'),('TF','ABC'),('CMTB','ABC')]:
    for letter in letters:
        rr=[r for r in rows if r['task']==task and r['gold_letter']==letter]
        cells[f'{task}/{letter}']=dict(total=len(rr),native_correct=sum(r['mc_native_correct'] for r in rr),routed_correct=sum(r['mc_routed_correct'] for r in rr))
extra_summary={name:sum(r['conditions'][name]['correct'] for r in extra) for name in extra[0]['conditions']}
presence_ablations=controls[0]['std_plus_presence_no_phi_c5']
assert presence_ablations==controls[1]['std_plus_presence_no_phi_c5']
summary=dict(replicas_identical_M1_M2=True,source_unmodified=True,mc_native_correct=sum(r['mc_native_correct'] for r in rows),mc_routed_correct=sum(r['mc_routed_correct'] for r in rows),micro_total=8,cells=cells,
    mean_gate_true=avg('confidence_gate_gamma_true_query'),mean_gate_wrong=avg('confidence_gate_gamma_wrong_query'),
    mean_swap_tv_late=avg('tv_late_kslot_swap'),mean_swap_tv_mid=avg('tv_mid_kslot_swap'),mean_tv_routed=avg('tv_mid_routed_vs_nat'),mean_tv_random=avg('tv_mid_rand3seed_vs_nat'),
    freeform_regex_aligned_native=sum(r['freeform_native_audit']['semantic_gold_aligned'] for r in rows),freeform_regex_aligned_routed=sum(r['freeform_routed_audit']['semantic_gold_aligned'] for r in rows),
    matched_exploratory_correct=extra_summary,std_plus_presence_no_phi_c5=presence_ablations,M2={})
m2=main[0]['M2_incremental_ablation_and_learned_hop2']
for suite,probes in m2.items():
    if suite=='foil_proxy_audit':continue
    b8=probes['B8_full_bary_phi_presence_c5_zca_connected']
    base=probes['B1b_unmodified_softmax_mean_time_c5_zca']
    best=max(v['overall_4way_pct'] for k,v in probes.items() if k.startswith(('B1_','B1b_','B2_','B3_','B4_')))
    summary['M2'][suite]=dict(full_c5=b8['overall_4way_pct'],matched_mean_time_c5=base['overall_4way_pct'],best_control=best,gap_best_pp=round(b8['overall_4way_pct']-best,1),full3_c5=b8['overall_3way_no_foil_pct'],during_c5=b8['by_state']['During'],n=b8['n_eval_episodes'])
save('summary.json',summary)
save('completion_audit.json',dict(summary=summary,archives=archive_checks,source_sha256=hashlib.sha256(source).hexdigest()))

lines=['# OB-TVOR / PC-EBR Stage-0 v5 — kết quả và kiểm toán, 2026-10-16','',
'**V5 sửa đúng parity và uniform gate. M2 full C5 đạt 67,7/59,9/34,6% trên Natural/Reversal/Roll; bỏ Phi vẫn đạt 67,2/57,3/34,0%. M3 giữ nguyên toàn bộ MC predictions: native và routed đều 5/8. Chưa xác nhận cải thiện temporal QA để chuyển A0–A7.**','',
'Kết quả của cấu hình gốc được báo cáo riêng với các đối chứng exploratory bổ sung. Xem đánh giá và hướng cải tiến ở [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md).','',
'Hai runtime mới, cùng 252 pseudo pairs: 202 train từ 16 origins và 50 held-out từ 4 origins đã xem trong các vòng trước. Ba seeds 20261015/16/17. M1: 36 fits/server × 180 steps; M2: 2 bridge fits × 120 steps, 27 probes × 200 steps. Hai replicas trùng kết quả M1/M2, không tăng số origins độc lập. Chỉ 8 mẫu micro cũ, 4/shard, rotation 0; chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.','',
'Source gốc 1939 dòng và alias có cùng SHA. Wrapper chỉ lưu observations/checkpoints; không đổi công thức, seeds, nhãn hoặc ngân sách mainrun. Native pairs là post-media readout block index 27; pooled là mean media states cùng block. Pseudo descriptions và substring Nonnull filter chưa phải event/time ground truth độc lập.','',
'## M0 — đại số','',f"```json\n{json.dumps(main[0][keys[0]],indent=2)}\n```",'',
'Cosine thuận/đảo xấp xỉ .1611 được quyết định bởi tan(theta)=.85, với hai kênh trực giao even/odd. Đây là kiểm tra representation đổi đúng theo định nghĩa; không đo temporal correctness. Uniform gate bằng 0 sửa đúng sàn .35 của v4.','',
'## M1 — đúng qua ba seeds','',
'Global/Exact/Multipos có mẫu số 50, Nonnull có 31 theo substring filter. Số sai của từng seed = mẫu số trừ số đúng.','']
for key in keys[1:3]:
    lines += [f'### {key}','','| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |','|---|---|---|---|---|---|']
    for name,v in main[0][key]['summary'].items():
        means='/'.join(f'{v[k]:.2f}' for k in ('mean_global_pct','mean_within_exact_pct','mean_within_multipos_pct','mean_within_nonnull_pct'))
        lines.append(f"| {name} | {v['global_hits_3seeds']} | {v['exact_hits_3seeds']} | {v['multipos_hits_3seeds']} | {v['nonnull_hits_3seeds']} | {means} |")
    lines.append('')
lines += ['C2 và C3 cùng within-rec weight .20, giúp tách tác dụng chuẩn hóa. Tuy nhiên C3 không phải exact replay C3 v4: v5 khởi tạo mean_shift từ media đã pre-whiten, v4 dùng media normalize trước whitening. C4/C5 có ZCA riêng mỗi modality nhưng _pre_whiten_media vẫn cộng nhánh (x_in−txt_mean)*txt_inv_std trước normalize; đây là hybrid chưa loại bỏ việc trừ text mean trên media như prose tuyên bố. C8 là shuffle của txt_diag_v4, chưa phải shuffle khớp C5 ZCA/hybrid.','',
'## M2 — mọi lớp và đối chứng','',
'Các episode là nhiều biến đổi/cặp từ cùng 4 held-out origins; không phải video độc lập. 3-way dùng argmax riêng ba relation logits trên nonfoil subset. Paired metric chỉ xét các cặp còn đủ thuận/đảo sau balancing.','']
pi=0
for suite,probes in m2.items():
    if suite=='foil_proxy_audit':continue
    lines += [f'### {suite}','','| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |','|---|---|---:|---:|---|---|---|---|---|']
    for name,v in probes.items():
        a=raw[0]['probes'][pi];pi+=1
        assert round(100*a['correct4']/a['total'],1)==v['overall_4way_pct']
        cc=[f"{a['by_class'][str(c)]['correct']}/{a['by_class'][str(c)]['total']}" for c in range(4)]
        pp=f"{a['paired_correct']}/{a['paired_total']}" if a['paired_total'] else '—'
        lines.append(f"| {name} | {a['correct4']}/{a['total']-a['correct4']}/{a['total']} | {v['overall_4way_pct']} | {v['overall_3way_no_foil_pct']} | {' | '.join(cc)} | {pp} |")
    lines.append('')
lines += ['### Foil proxy audit','',f"```json\n{json.dumps(m2['foil_proxy_audit'],indent=2)}\n```",'',
'Foils được phân hạng strict/lexical/string; tất cả vẫn là proxy từ mô tả, chưa xác nhận event vắng mặt. Thêm raw affinity có thể cải thiện phân biệt recording/query khác nguồn; phải đọc riêng relation-only và coordinate-shuffle trước khi quy gain 4-way cho temporal reasoning. B7 dùng C1; B8 và các controls B2/B3/B4 dùng C5. Baseline khớp B8 là B1b.','',
'### Đối chứng exploratory bỏ riêng Phi, giữ C5 mean-time và presence','',
'Cùng chuẩn hóa train, linear head, seed, lr=.035, weight decay=.04, 200 steps như v5. Không thay mainrun. Raw X/y/logits/checkpoint được lưu trong expl_presence_probe_*.pt.','',
'| Suite | Mean-time+presence, không Phi: đúng/tổng | Acc4% | Acc3% | Full B8 Acc4% |','|---|---|---:|---:|---:|']
for (suite,part),a in zip([(k,v) for k,v in m2.items() if k!='foil_proxy_audit'],presence_ablations):
    lines.append(f"| {suite} | {a['correct']}/{a['total']} | {a['accuracy4']} | {a['accuracy3']} | {part['B8_full_bary_phi_presence_c5_zca_connected']['overall_4way_pct']} |")
lines += ['',
'## M3 — 8 mẫu micro','',
'| ID | Task/gold | Native→routed/reversed | True/wrong gate | Fwd/rev cosine | Swap TV late/mid |','|---|---|---|---|---:|---|']
for r in rows:
    lines.append(f"| {r['idx']} | {r['task']}/{r['gold_letter']} | {r['mc_pred_native']}→{r['mc_pred_routed_fwd']}/{r['mc_pred_routed_rev']} | {r['confidence_gate_gamma_true_query']}/{r['confidence_gate_gamma_wrong_query']} | {r['fwd_vs_rev_vector_cosine']} | {r['tv_late_kslot_swap']}/{r['tv_mid_kslot_swap']} |")
lines += ['',f"Native đúng {summary['mc_native_correct']}/8, routed đúng {summary['mc_routed_correct']}/8. Mean true gate={summary['mean_gate_true']:.6f}; wrong-dimension-permuted-query gate={summary['mean_gate_wrong']:.6f}. Mean TV swap late={summary['mean_swap_tv_late']:.6f}, mid={summary['mean_swap_tv_mid']:.6f}; routed/native={summary['mean_tv_routed']:.6f}, random/native={summary['mean_tv_random']:.6f}.",'',
'| Task/gold | N | Native đúng/sai | Routed đúng/sai |','|---|---:|---|---|']
for name,c in cells.items():
    n=c['total'];nc=c['native_correct'];rc=c['routed_correct']
    lines.append(f'| {name} | {n} | {nc}/{n-nc} | {rc}/{n-rc} |' if n else f'| {name} | 0 | Chưa đo | Chưa đo |')
lines += ['','Đây là micro rotation 0; không so trực tiếp với bảng 40 mẫu có cyclic rotations. Wrong-query trong code là hoán vị các chiều hidden của cùng stem, không phải một câu hỏi có nghĩa nhưng hỏi event khác. VO chỉ được tính ở block 18; cùng vector được inject vào các blocks 14/16/18/20/22/24 hoặc 28–33. Không được mô tả là VO theo đúng từng block.','',
'### Văn bản sinh 32 tokens','',
f"Regex semantic_gold_aligned: native {summary['freeform_regex_aligned_native']}/8, routed {summary['freeform_regex_aligned_routed']}/8. Đây là chấm bằng regex/keyword, chưa phải annotation semantic độc lập; các phản ví dụ ở controls_s*.json.",'']
lines += ['Ba phản ví dụ chấm nhầm được chạy lại trực tiếp từ hàm gốc tại [freeform_metric_counterexamples.json](freeform_metric_counterexamples.json): phủ định TA, phủ định action CMTB và quan hệ TF nói về sự kiện khác đều có thể được regex tính aligned.','']
for r in rows:
    lines += [f"**{r['idx']} — {r['task']}/{r['gold_letter']} ({r['gold_answer_text']})**",'',f"Native: {r['freeform_32tok_native']}",'',f"Routed: {r['freeform_32tok_routed']}",'',f"Native audit: `{json.dumps(r['freeform_native_audit'],ensure_ascii=False)}`",'',f"Routed audit: `{json.dumps(r['freeform_routed_audit'],ensure_ascii=False)}`",'']
lines += ['## Đối chứng exploratory cùng cường độ .045×gate','',
'Các nhánh này được thêm riêng sau thiết kế gốc để tách raw/VO và content/order, không chỉnh mainrun hoặc chọn siêu tham số theo QA. Chỉ thêm MC forwards, không sinh lại text.','',
'| ID | Native | Full VO+order | Raw cùng cường độ | VO content-only | Gold |','|---|---|---|---|---|---|']
for a in extra:
    lines.append(f"| {a['idx']} | {a['main_native']} | {a['main_full']} | {a['conditions']['low_strength_raw_tangent']['prediction']} | {a['conditions']['vo_content_only_tangent']['prediction']} | {a['gold']} |")
lines += ['',f"Tổng đúng: native {summary['mc_native_correct']}/8; full {summary['mc_routed_correct']}/8; raw low-strength {extra_summary['low_strength_raw_tangent']}/8; VO content-only {extra_summary['vo_content_only_tangent']}/8. Zero-scale replay khớp native logits tuyệt đối 8/8.",'',
'Không suy bảo toàn cú pháp/ngữ nghĩa 100% từ việc giữ norm hoặc bước tiếp tuyến. Chỉ thay biên độ giữa v4/v5 cũng có thể giảm generic captioning; đối chứng cùng cường độ ở trên kiểm tra phần nào đóng góp của VO và order.','',
'## Tái lập và artifacts','',
f"Mỗi server có {raw[0]['raw_files']} raw PT files, 1.800 retrieval query rows được tính lại đúng, 27 linear probe outputs được tổng hợp từ logits. Hai archives và từng entry được kiểm tra SHA local, không extract. Checkpoints ZCA lưu cả tensors ngoài state_dict. Native replay, token roundtrip, hook/option disjointness và prompt không choices đều qua 8/8; mọi intervention hooks đã tháo và CUDA khỏe.",'',
'Mainrun, controls, source gốc, proposal và các job logs được lưu riêng. Names Yoshua/Yann là quy ước vai trò; không phải xác nhận tác giả thật. Novelty, tổng quát hóa và CVPR acceptance chưa được xác nhận từ Stage-0 này.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
with (D/'m3_per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps(summary,ensure_ascii=False,indent=2))
