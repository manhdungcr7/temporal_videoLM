"""Generate the v3 results/gate report from archived artifacts only."""
import ast
import csv
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/pcebr_stage0_v3'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,x):(D/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
main=[read(f'main_results_s{s}.json') for s in (1,2)]
controls=[read(f'controls_s{s}.json') for s in (1,2)]
raw=[read(f'archive_audit_s{s}.json') for s in (1,2)]
for key in ('M1_native_pairs','M1_pooled_pairs','M2_interval_algebra_and_counterfactuals','Algebraic_Unit_Checks'):assert main[0][key]==main[1][key],key
source=(D/'stage0_pcebr_mechanism_check_v3_original.py').read_text(encoding='utf-8')
fixed=(D/'stage0_pcebr_mechanism_check_v3_fixed.py').read_text(encoding='utf-8')
assert source.replace('zm_all = m_loc.encode_media(media_rec_feats, sources_t)',
    'zm_all = m_loc.encode_media(media_rec_feats, sources_t).detach()')==fixed
ast.parse(fixed)
archives=[]
for s in (1,2):
    path=D/f'raw_artifacts_s{s}.tar.gz';index=read(f'archive_index_s{s}.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected={x['name']:x for x in raw[s-1]['files']}
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==143
        for m in members:
            assert m.isfile() and not m.issym() and m.name.startswith('artifacts/') and '..' not in Path(m.name).parts
            value=tar.extractfile(m).read();name=m.name.split('/')[-1]
            assert hashlib.sha256(value).hexdigest()==expected[name]['sha256']
    assert raw[s-1]['retrieval_metrics_recomputed_exact'] and raw[s-1]['retrieval_queries']==2600
    assert len(raw[s-1]['probes'])==21
    freeze=read(f'freeze_repair_audit_s{s}.json');assert all(x['weights_unchanged'] and x['trainable_parameters']==0 for x in freeze)
    archives.append(dict(server=s,bytes=path.stat().st_size,files=len(members),sha256=index['sha256'],verified=True))

rows=[];audit=[]
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
for s in (1,2):
    part=main[s-1]['M3_mid_vs_late_routing_micro_probe'];r=part['samples'];a=read(f'm3_audit_s{s}.json')
    assert [x['idx'] for x in r]==[x['idx'] for x in manifest[f'server{s}'][:4]]
    assert len(a)==4
    for x in a:
        assert x['exact_tail_token_roundtrip']
        assert not set(x['hook_rows'])&set(x['option_rows'])
        assert all(marker not in x['freeform_input_text'] for marker in ('(A)','(B)','(C)','Answer with only the letter'))
    rows.extend(dict(shard=s,**x) for x in r);audit+=a
mean=lambda key:statistics.mean(x[key] for x in rows)
ratio=mean('tv_mid_kslot_swap')/mean('tv_late_kslot_swap')
gates=dict(unit_checks=True,M1_algebra=True,M1_localization=True,M2={},M3={})
lines=['PC-EBR Stage-0 v3 — kết quả và phản biện, 2026-10-15','',
    '**Cổng 0 v3 không qua các điều kiện đã công bố. Unit checks và group-max invariance qua; hiệu năng M1, các gate M2 và phần sensitivity của M3 không qua. Chưa chạy A0–A7 hoặc reserve400.**','',
    'Hai runtime mới; cùng4 cache pairs,202 train/50 held-out từ16/4 recording origins;3seeds. M1/M2 và unit results giống nhau giữa hai replicas. '
    'Bản gốc lỗi backward-through-graph lần2 ở M2. Repair: đóng băng feature teachers M1 trước train probes; weights được so bitwise với checkpoint và không đổi. '
    'M1 không chạy lại. Bản script standalone được sửa bằng một `.detach()` ở media feature extraction; công thức/loss/labels/seeds không đổi. '
    'M3 đủ8ID (4/shard, rotation0), không thay mẫu, cả hai GPU runtime hoàn tất bình thường.','',
    '**1. Unit checks**','',
    'Basis diag=[2,0,0,0], lowrank=[2,2,2,2]; mean-zero counterexample đổi winner0→2, Max-Zero giữ winner0. '
    'Đây là identity toán học đã được triển khai đúng; không tự chứng minh retrieval hoặc temporal grounding.','',
    '**2. M1 — số đúng theo3seeds và mean metrics**','',
    'Global/Exact/Multipos có50queries; Nonnull có31 theo NULL_PATTERNS mới (v2 là32), nên không so trực tiếp Nonnull% giữa hai phiên bản như cùng denominator. '
    'Held-out4origins đã được xem qua nhiều vòng; replicas/seeds không làm tăng số samples độc lập.','']
for key,label in [('M1_native_pairs','Native post-media readout'),('M1_pooled_pairs','Pooled media')]:
    part=main[0][key];conds=part['crossed_3seed_factorial']
    lines += [f'**{label}**','','| Condition | Global đúng/50 qua3seeds | Exact đúng/50 qua3seeds | Multipos mean% | Nonnull mean% |','|---|---|---|---:|---:|']
    for name,v in conds.items():
        lines.append(f"| {name} | {v['global_hits_3seeds']} | {v['exact_hits_3seeds']} | {v['mean_within_multipos_pct']:.2f} | {v['mean_within_nonnull_pct']:.2f} |")
    for candidate,context in [('C4_ctx_plus_max_zero_hier_diag','C1_ctx_only_diag'),('C7_ctx_plus_max_zero_hier_lowrank','C5_ctx_only_lowrank')]:
        cand=conds[candidate];ctx=conds[context];shuf=conds['C8_shuffled_within_rec_ctrl']
        passed=all(cand[k]>ctx[k] and cand[k]>shuf[k] for k in ('mean_within_exact_pct','mean_within_multipos_pct','mean_within_nonnull_pct'))
        gates['M1_localization'] &= passed
    lines.append('')
all_group=[x for a in raw for x in a['all_seed_group_checks']]
gates['M1_algebra']=all(x['max_error']==0 and x['winning_group_agreement']==1 for x in all_group)
lines += ['Group maxima error=0 và winning-group agreement100% cho C4/C7 ở mọi3seeds, cả hai representations. '
    'Nhưng native Exact C1→C4:32→24%; C5→C7:24.67→22%. Pooled:38→30% và28.67→25.33%. '
    'C7 Nonnull còn thua C8 shuffle (native17.20% so19.33%; pooled23.67% so24.73%). Vì vậy gate performance M1 fail. '
    'Pooled context-only diag C1 tốt nhất đơn giản:16/50global,19/50exact,23/50multipos,14/31nonnull. '
    'Max-Zero bảo toàn winning group, không bảo toàn đúng semantic slot: native C1global22%→C4global16% dù group giữ nguyên.','',
    '**3. M2 — gate so với đầy đủ controls**','',
    'Natural/Reverse/Roll có lần lượt232/576/752held-out episodes, cân bằng4lớp. Đây là synthetic episodes tái dùng4origins, không phải video QA độc lập. '
    '3-way là argmax riêng3relation logits trên subset không foil; khác với chỉ bỏ foil khỏi4-way predictions.','']
probeindex=0
for suite,part in main[0]['M2_interval_algebra_and_counterfactuals'].items():
    if not suite.startswith('PartB'):continue
    lines += [f'**{suite}**','','| Probe | Đúng/tổng4-way | 4-way% | 3-way% | Before% | During% | After% | Foil% |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,v in part.items():
        actual=raw[0]['probes'][probeindex];probeindex+=1
        assert round(100*actual['correct']/actual['n'],1)==v['overall_4way_pct']
        lines.append(f"| {name} | {actual['correct']}/{actual['n']} | {v['overall_4way_pct']:.1f} | {v['overall_3way_no_foil_pct']:.1f} | "+' | '.join(f"{v['by_state'][c]:.1f}" for c in ('Before','During','After','Null_Foil'))+' |')
    full=part['B7_pcebr_pos_debiased_full'];uniform=part['B4_uniform_retrieval_sim0_control']
    controls4=[v['overall_4way_pct'] for n,v in part.items() if n.startswith(('B1_','B2_','B3_','B4_'))]
    controls3=[v['overall_3way_no_foil_pct'] for n,v in part.items() if n.startswith(('B1_','B2_','B3_','B4_'))]
    gap4=full['overall_4way_pct']-max(controls4);gap3=full['overall_3way_no_foil_pct']-max(controls3)
    gates['M2'][suite]=dict(uniform_pass=uniform['overall_4way_pct']<=28 and uniform['overall_3way_no_foil_pct']<=35,
        full_min_gap4=round(gap4,1),full_min_gap3=round(gap3,1),full_during_recall=full['by_state']['During'],
        performance_pass=gap4>=10 and full['by_state']['During']>=35)
    lines += ['',f"Uniform gate pass; full4-way gap so control tốt nhất={gap4:+.1f}pp,3-way gap={gap3:+.1f}pp. During={full['by_state']['During']:.1f}%.",'']
lines += ['Uniform25%/33.3% đáp ứng ngưỡng cụ thể; nhưng features bằng0 là tính chất được mã hóa vào operator. '
    'Uniform có thể luôn đoán foil:foil recall100% cùng0% ở3lớp khác không có nghĩa nhận ra thiếu bằng chứng. '
    'Full natural thua wrong-event31.9%; reversal thua baseline35.4%; roll thua coordinate-shuffle28.2%. '
    'Không bộ nào đạt yêu cầu +10pp. During gate chỉ pass Natural/Reverse, fail Roll22.9%.','',
    '**4. M3 — dữ liệu đầy đủ8mẫu**','',
    '| Shard | ID | Cos(fwd,rev) | TVswap late | TVswap mid | TV M1+M2/native | TV irrelevant/native | TV random/native | Native replay error |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['shard']} | {r['idx']} | {r['fwd_vs_rev_vector_cosine']:.4f} | {r['tv_late_kslot_swap']:.4f} | {r['tv_mid_kslot_swap']:.4f} | {r['tv_mid_m1m2_vs_nat']:.4f} | {r['tv_mid_irrel_slot_vs_nat']:.4f} | {r['tv_mid_rand3seed_vs_nat']:.4f} | {r['native_replay_max_err']} |")
gates['M3']=dict(native_replay_pass=all(x['native_replay_max_err']<=1e-5 for x in rows),
    option_span_pass=all(x['exact_tail_token_roundtrip'] and not set(x['hook_rows'])&set(x['option_rows']) for x in audit),
    freeform_generation_pass=all(x['freeform_gen_ok'] for x in rows),cosine_pass=all(x['fwd_vs_rev_vector_cosine']>-.95 for x in rows),
    mid_late_ratio=ratio,ratio_pass=ratio>=2,M1M2_mean_TV=mean('tv_mid_m1m2_vs_nat'),
    irrelevant_mean_TV=mean('tv_mid_irrel_slot_vs_nat'),random_mean_TV=mean('tv_mid_rand3seed_vs_nat'),
    controls_pass=mean('tv_mid_m1m2_vs_nat')>max(mean('tv_mid_irrel_slot_vs_nat'),mean('tv_mid_rand3seed_vs_nat')))
lines += ['',f"Gộp8mẫu: TVswap mid={mean('tv_mid_kslot_swap'):.6f},late={mean('tv_late_kslot_swap'):.6f},ratio={ratio:.3f}×<2.0. "
    f"TV M1+M2/native={mean('tv_mid_m1m2_vs_nat'):.6f},irrelevant={mean('tv_mid_irrel_slot_vs_nat'):.6f},random={mean('tv_mid_rand3seed_vs_nat'):.6f}. "
    'Không vượt random control. Native replay0 trên8/8; span roundtrip đúng8/8,hook disjoint options8/8,free-form generation nonempty8/8. '
    'Không dùng các pass kỹ thuật này để thay cho gate sensitivity hoặc temporal correctness.','',
    '**Sinh tự hồi quy sau bỏ choices — nguyên văn outputs, không phải accuracy**','',
    '| ID | Generated tokens | Text |','|---|---:|---|']
for r in audit:lines.append(f"| {r['idx']} | {len(r['generated_tokens'])} | {r['freeform_generated_text'].replace('|','/')} |")
lines += ['',
    '**5. Audit còn quan trọng trước đề xuất kế tiếp**','',
    '- M2 không dùng các giá trị `sim_mat` hierarchical; chỉ dùng `.device` rồi tính lại media/text affinities bằng m_loc. Context m_ctx/Max-Zero không nằm trong downstream scoring path. M3 cũng chỉ dùng m_loc, không dùng toàn hierarchical model.',
    '- Tất cả20origins train/held-out không có foil nào đạt text-cos<.55; code dùng fallback chỉ khác chuỗi caption. Đây không phải verified absence bằng annotation.',
    '- Natural During dùng khoảng cách centers≤2.1s, bao gồm adjacent2s windows; Roll tại anchor sát biên có thể gán Before/After cho displacement1slot=2s, không nhất quán định nghĩa Natural. B3 vẫn tách role banks khi src_a==src_b.',
    '- M3 có đường audio-first→visual cố định, và đảo logits/weights visual trong khi evidence value bank V18 giữ nguyên; chưa phải raw AV reversal có bảo toàn event identity. Nó chưa đáp ứng nhiều chiều reasoning như user yêu cầu.',
    '- Positive-part triệt exact uniform, nhưng tiny logit peak1e-4 đã cho normalized Phi=[1,1,1,0,0,0] dù margin≈5.6e-6; M3 cộng14%norm không gate theo uncertainty/null. Không suy calibration hoặc không hallucination từ algebra.',
    '- sec lấy từ temporal RoPE/25 khớp decoder positions, chưa phải verified PTS:idx2150 video position clock đến15.52s dù clip12.167s; idx1513 đến16.04s dù clip13.074s. Cần clock provenance từ encoder/frame metadata trước khi gọi slots là khoảng vật lý.',
    '- Sinh được12tokens không chứng minh grounded answer. Ví dụ2150 sinh “The event occurs at 1:00” trong clip12.167s; cần kiểm tra độc lập, không coi freeform_gen_ok là temporal correctness.','',
    '**6. Artifacts và giới hạn**','',
    '143rawPTfiles/server gồm36M1checkpoints,52retrievalmatrices,2bundles,32episodefiles (8 của attempt lỗi,24 của resume),21probes. '
    '2600retrievalqueryrows/server recompute đúng; SHA archives và mọi entry đã kiểm local, không extract. '
    'Teacher weights freeze không đổi; hooks cuối bằng0 ởcả2runtime. M1/M2 replicas trùng. Không chạy bảng40/A0–A7/reserve400. '
    'Các phần fail chỉ bác bỏ cấu hình và dự đoán cụ thể này, không bác bỏ toàn bộ latent reasoning hoặc mọi tầng muộn.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
save('gates.json',gates)
save('completion_audit.json',dict(archives=archives,only_feature_freeze_repair=True,M1_M2_replica_equal=True,
    micro_ids=[x['idx'] for x in rows],native_replay_max_error=max(x['native_replay_max_err'] for x in rows),
    all_tail_roundtrips=True,all_options_removed_from_freeform_input=True,all_hook_option_spans_disjoint=True,
    M3_ratio=ratio,QA40_run=False,A0_A7_run=False,reserve400_run=False))
with (D/'m3_per_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(report=str(D/'report.md'),gates=gates),ensure_ascii=False,indent=2))
