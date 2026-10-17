"""Aggregate checked v7 artifacts without importing torch locally."""
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/pcebr_stage0_v7'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,x):(D/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
main=[read(f'main_results_s{s}.json') for s in (1,2)]
audits=[read(f'archive_audit_s{s}.json') for s in (1,2)]
source_sha=hashlib.sha256((D/'stage0_pcebr_v7_final_original.py').read_text(encoding='utf-8').encode()).hexdigest()
assert source_sha=='d27eb87a7014e941bd7a7a920dbe3ce074338cf616757608cc476712d7c8ee80'
replicas={k:main[0][k]==main[1][k] for k in ('M0','M1_native','M1_pooled','M2')}
assert all(replicas.values()),replicas
verified=[]
for s in (1,2):
    inp=read(f'input_audit_s{s}.json');assert inp['source_lf_sha256']==source_sha
    for n,h in inp['caches'].items():assert hashlib.sha256((ROOT/'results'/n).read_bytes()).hexdigest()==h
    index=read(f'archive_index_s{s}.json');path=D/f'raw_artifacts_s{s}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected={x['name']:x for x in audits[s-1]['files']}
    with tarfile.open(path,'r:gz') as t:
        members=t.getmembers();assert len(members)==len(expected)
        for m in members:
            assert m.isfile() and not m.issym() and '..' not in Path(m.name).parts
            b=t.extractfile(m).read();e=expected[m.name]
            assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
    c=read(f'completion_s{s}.json')
    assert c['parameter_versions_unchanged'] and c['hooks_before']==c['hooks_after'] and c['cuda_health']==[2.,2.]
    verified.append(dict(**index,verified_every_entry=True))
rows=[dict(server=s,**r) for s in (1,2) for r in main[s-1]['M3']['samples']]
assert len(rows)==8
assert all(r['native_replay_max_err']<1e-5 and r['zero_scale_replay_max_err']<1e-5 and r['kernel_noop_zero_perturb_max_err']<1e-5 and r['kernel_noop_layer_invocations']==6 for r in rows)
for s in (1,2):
    for row in read(f'm3_audit_s{s}.json'):
        assert row['legacy_same_runtime_archived_source_parity']['probability_max_abs_error']<1e-6
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
assert [r['idx'] for r in rows]==[r['idx'] for s in (1,2) for r in manifest[f'server{s}'][:4]]
pred_keys=[k for k in rows[0] if k.startswith('mc_pred_')]
counts={k:sum(r[k]==r['gold_letter'] for r in rows) for k in pred_keys}
cells={}
for task,gs in [('TA','AB'),('TF','ABC'),('CMTB','ABC')]:
    for gold in gs:
        group=[r for r in rows if r['task']==task and r['gold_letter']==gold]
        cells[f'{task}/{gold}']=dict(n=len(group),correct={k:sum(r[k]==gold for r in group) for k in pred_keys})
controls=[r for s in (1,2) for r in read(f'control_accuracy_s{s}.json')]
assert len(controls)==8
assert all(r['full_replay_probability_error']<1e-6 and r['caor_replay_probability_error']<1e-6 for r in controls)
control_counts={k:sum(r['conditions'][k]['correct'] for r in controls) for k in controls[0]['conditions']}
couplings=[r for s in (1,2) for r in read(f'coupling_actual_input_audit_s{s}.json')]
mean_keys=[k for k in rows[0] if k.startswith('tv_') or k.startswith('confidence_gate_gamma_')]
summary=dict(n=8,counts=counts,cells=cells,controls=control_counts,replicas=replicas,
    rescued=[r['idx'] for r in rows if r['mc_pred_native']!=r['gold_letter'] and r['mc_pred_routed_fwd']==r['gold_letter']],
    harmed=[r['idx'] for r in rows if r['mc_pred_native']==r['gold_letter'] and r['mc_pred_routed_fwd']!=r['gold_letter']],
    changed=[r['idx'] for r in rows if r['mc_pred_native']!=r['mc_pred_routed_fwd']],
    means={k:statistics.mean(r[k] for r in rows) for k in mean_keys},
    heuristic_freeform={b:sum(r[f'freeform_{b}_audit']['heuristic_clause_aligned'] for r in rows) for b in ('native','routed')},
    coupling_max_errors={k:max(r[k] for r in couplings) for k in ('analytic_pnull_error','pnull_change_with_fixed_inputs_zero_omega','J_aug_row_error','J_aug_col_error','J_cond_row_error','J_cond_col_error','marginal_only_phi_channels_9_10_11_12_max_change')})
save('summary.json',summary);save('m3_per_sample.json',rows)
flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]
with (D/'m3_per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
save('completion_audit.json',dict(source_lf_sha256=source_sha,archives=verified,summary=summary))

lines=['# JEP-CLC / PC-EBR Stage-0 v7 — kết quả và kiểm toán, 2026-10-16','',
    f"**Micro8: native {counts['mc_pred_native']}/8; CAOR-only {counts['mc_pred_caor_attn_only']}/8; v7 full {counts['mc_pred_routed_fwd']}/8. Cứu {len(summary['rescued'])}, làm hỏng {len(summary['harmed'])}. M0 gốc thất bại assertion dòng827 trên cả hai server.**",'',
    '**Audit cơ chế: audio salience/gate bằng0 trên cả8 mẫu, kéo joint_gate và toàn bộ14D Psi về0. Phần joint event-pair mới chưa hoạt động trên micro8; outer routing/CLC vẫn có thể tác động. Khi real mass≈1e−14 bị chia cho denominator clamp1e−12, J_cond có tổng≈.01 thay vì1. Xem [deep_mechanism_audit.json](deep_mechanism_audit.json).**','',
    'M1/M2 dùng cùng252 pseudo pairs (202 train/16 origins;50 held-out/4 origins đã xem) trên hai runtimes2xT4. Đây là replicas, không tăng số recordings độc lập. Native cache là post-media last-token readout block27; pooled là mean media states cùng block. Source gốc giữ nguyên SHA; M3 gọi trained C6 + M2 probe + trained six-scalar bridge. Chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.','',
    'Xem [protocol.md](protocol.md), [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md) và [math_note_augmented_dustbin.md](math_note_augmented_dustbin.md).','',
    '## M0 — lỗi của bản gốc và diagnostic','',
    'Bản gốc dừng trước M1 vì assert `bimod_std > .30`. Feature13 là std_J−std_uniform, không phải raw std. Ví dụ bimodal có raw std=.4 bằng baseline nên feature=0; synchronous≈−.4. W+=W−≈.175 và max feature difference=1, nên vector14D vẫn phân biệt hai ví dụ. Không đổi ngưỡng để tuyên bố pass. Diagnostic riêng quan sát mọi assert và gọi các pha M1–M3 nguyên bản sau đó; logs nguồn lỗi được giữ ở original_m0_failure và local job_run_s*.json.','',
    'Original M0 diagnostic:','',f"```json\n{json.dumps(read('m0_diagnostic_s1.json')['observed_original_return'],ensure_ascii=False,indent=2)}\n```",'',
    'Các kiểm tra toy không xác nhận event matching, null calibration hoặc QA correctness.','',
    '## M1 — từng seed','',
    'Global/Exact/Multipos có mẫu số50; Nonnull31 theo substring filter. Global chấm normalized caption content, không yêu cầu đúng recording/interval. Exact là đúng index trong cùng recording/modality; Multipos chấp nhận mô tả trùng trong group. Số sai = mẫu số−số đúng. Ba seeds không tạo thêm held-out origins.','']
for key in ('M1_native','M1_pooled'):
    lines += [f'### {key}','','| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |','|---|---|---|---|---|---|']
    for name,v in main[0][key]['summary'].items():
        means='/'.join(str(v[k]) for k in ('mean_global_pct','mean_within_exact_pct','mean_within_multipos_pct','mean_within_nonnull_pct'))
        lines.append(f"| {name} | {v['global_hits_3seeds']} | {v['exact_hits_3seeds']} | {v['multipos_hits_3seeds']} | {v['nonnull_hits_3seeds']} | {means} |")
    lines.append('')
lines += ['## M2 — số đếm từ raw logits','',
    'Pool held-out có4 origins; bank sizes8/3/5/9 mỗi modality. Điều kiện bank>=4 loại p-gN4cbmunQ khỏi target contexts, còn3 recording contexts đủ điều kiện; origin bị loại vẫn có thể cung cấp foil queries. Foil là text proxy chưa được annotation absence. Reversal/Roll đổi coordinates/bank assignments, không dịch waveform hoặc đảo frames. Acc3 dùng argmax riêng3 relation logits trên nonfoil subset. Paired chỉ xét các cặp còn đủ thuận/đảo sau balancing.','']
probe_index=0;m2tables={}
for suite,values in main[0]['M2'].items():
    if not suite.startswith('PartB'):continue
    b8=next(k for k in values if k.startswith('B8_full'))
    order=[b8]+[k for k in values if k!=b8]
    byname={}
    for name in order:
        a=audits[0]['probes'][probe_index];b=audits[1]['probes'][probe_index];assert a==b
        assert a['total']==values[name]['n_eval_episodes']
        assert abs(100*a['correct4']/a['total']-values[name]['overall_4way_pct'])<.11
        byname[name]=a;probe_index+=1
    m2tables[suite]=byname
    lines += [f'### {suite}','','| Probe | Đúng/sai/tổng 4-way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng |','|---|---|---:|---:|---|---|---|---|---|']
    for name in values:
        a=byname[name];cls=[f"{a['by_class'][str(c)]['correct']}/{a['by_class'][str(c)]['total']}" for c in range(4)]
        paired=f"{a['paired_correct']}/{a['paired_total']}" if a['paired_total'] else '—'
        lines.append(f"| {name} | {a['correct4']}/{a['total']-a['correct4']}/{a['total']} | {100*a['correct4']/a['total']:.1f} | {100*a['correct3']/a['total3']:.1f} | {' | '.join(cls)} | {paired} |")
    lines.append('')
assert probe_index==len(audits[0]['probes'])
save('m2_per_probe_counts.json',m2tables)
lines += ['### Foil/hop audit','','```json',json.dumps(main[0]['M2']['foil_proxy_and_hop_audit'],ensure_ascii=False,indent=2),'```','',
    '## M3 — cố định8 mẫu rotation0','','| ID | Task/gold | Native | CAOR-only | V7 full | True/wrong semantic gate | TV full/native |','|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['idx']} | {r['task']}/{r['gold_letter']} | {r['mc_pred_native']} | {r['mc_pred_caor_attn_only']} | {r['mc_pred_routed_fwd']} | {r['confidence_gate_gamma_true_query']}/{r['confidence_gate_gamma_semantic_wrong_query']} | {r['tv_mid_routed_vs_nat']} |")
lines += ['','| Task/gold | N | Native đúng/sai | CAOR-only đúng/sai | Full đúng/sai |','|---|---:|---|---|---|']
for cell,c in cells.items():
    nn=c['n'];v=[f"{c['correct'][k]}/{nn-c['correct'][k]}" if nn else 'Chưa đo' for k in ('mc_pred_native','mc_pred_caor_attn_only','mc_pred_routed_fwd')]
    lines.append(f"| {cell} | {nn} | {' | '.join(v)} |")
lines += ['',f'All source MC conditions correct/8: `{json.dumps(counts)}`.',f'Means: `{json.dumps(summary["means"])}`.','',
    'Gamma measures concentration/excess with fixed thresholds, not an independently calibrated event-presence probability. Source 2x2 gate diagnostics also vary excess cap and therefore do not isolate a true factorial design. Mid/late TV sensitivity is not directional accuracy; full mid-vs-reversed includes CLC while late is attention-only.','',
    '### Original-control accuracy and exploratory CLC controls','','| Condition | Đúng/sai/8 |','|---|---|']
for k,v in control_counts.items():lines.append(f'| {k} | {v}/{8-v}/8 |')
lines += ['','Original random/irrelevant controls are attention-only. Exploratory random-CLC seeds101/202/303 keep full attention profiles and CLC strength fixed; sink-matched contrast changes only the foil sink correction before reconstructing the latent direction. Option-free-prepass control recomputes support/foil/reverse L22 states with choices absent. These are separate post-mainrun controls, not mainrun changes or tuned variants.','',
    'The original generated freeform prompt excludes choices but reuses a CLC direction built from the MC prompt with choices. Therefore original freeform is not fully option-independent.','',
    '### Actual-input coupling audit','',f'```json\n{json.dumps(couplings,indent=2)}\n```','',
    'Important: all8 actual-input audits have joint_gate0. The stored Psi vectors must be read with deep_mechanism_audit.json: low real mass≈1e−14 is divided by clamp(min=1e−12), returning J_cond total≈.01 and breaking its normalized/marginal claims. Multiplication by joint_gate0 masks the resulting Phi channels. The outer routing gamma can remain nonzero because it uses max(alpha_a,alpha_v) plus an excess cap, while the joint gate uses their product. Thus a non-additive omega does not establish active pair binding.','',
    'Under converged positive-mass dual-calibrated construction, real-event mass=a² and P_null=1−a²; at fixed marginals/score/gate pair compatibility supplies no extra null evidence. The independent CPU checks validate this with nonzero gate; the8 QA inputs all use the zero-gate regime and therefore do not independently test positive-gate coupling. Wasserstein/Bhattacharyya/mean-lag channels depend only on prescribed marginals. The lag RBF/synchrony/dispersion channels can still use joint interactions. See the derivation and numerical residuals; finite iteration can produce departures.','',
    '## Generated text and heuristic audit','',
    f"Original heuristic counts: native {summary['heuristic_freeform']['native']}/8, routed {summary['heuristic_freeform']['routed']}/8. This rule has been revised after earlier samples were observed; it is not independently annotated semantic accuracy.",'']
for r in rows:
    lines += [f"### {r['idx']} — {r['task']}/{r['gold_letter']}",'',f"Native: {r['freeform_32tok_native']}",'',f"V7: {r['freeform_32tok_routed']}",'',f"Native audit: `{json.dumps(r['freeform_native_audit'],ensure_ascii=False)}`",f"V7 audit: `{json.dumps(r['freeform_routed_audit'],ensure_ascii=False)}`",'']
lines += ['## Provenance','',f'Source LF SHA256 `{source_sha}`; all archive entries verified locally without extraction.','',
    'Native replay and active zero-noop results are in m3_audit*.json; same-runtime archived-v5 replay is checked from raw probabilities. Parameter version counters remained unchanged and hooks/CUDA health checks completed. Fixed preprocessing and sample IDs preserved; historical exact logits need not match fresh-runtime caches.','',
    f"Archive metadata: `{json.dumps(verified)}`."]
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
