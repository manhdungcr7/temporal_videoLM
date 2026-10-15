"""Build the v4 handoff report and verify every downloaded archive entry."""
import ast
import collections
import csv
import hashlib
import json
import re
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/pcebr_stage0_v4'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,obj):(D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

main=[read(f'main_results_s{s}.json') for s in (1,2)]
raw=[read(f'archive_audit_s{s}.json') for s in (1,2)]
ctrl=[read(f'controls_s{s}.json') for s in (1,2)]
for key in ('Algebraic_and_TinyNoise_Unit_Checks','M1_native_clean_comparison','M1_pooled_clean_comparison','M2_shrinkage_calibrated_connected_to_C1'):
    assert main[0][key]==main[1][key],key
source=(D/'stage0_pcebr_v4_original.py').read_bytes()
assert source==(ROOT/'server_jobs/stage0_pcebr_v4_clock_and_token_salience.py').read_bytes()
ast.parse(source.decode('utf-8'))
fixed=(D/'stage0_pcebr_v4_goldfix.py').read_text(encoding='utf-8')
assert source.decode('utf-8').replace("gold_letter = sample.get('answer', 'A')", "gold_letter = sample.get('gold', sample.get('answer', 'A'))").replace('\r\n','\n')==fixed
ast.parse(fixed)
archives=[]
for s in (1,2):
    inp=read(f'input_audit_s{s}.json')
    assert hashlib.sha256(source).hexdigest()==inp['source_sha256']
    for name,digest in inp['caches'].items():assert hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()==digest
    index=read(f'archive_index_s{s}.json');path=D/f'raw_artifacts_s{s}.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==index['sha256']
    expected={f['name']:f for f in raw[s-1]['files']}
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==index['raw_files']==len(expected)
        for member in members:
            assert member.isfile() and not member.issym() and member.name.startswith('artifacts/') and '..' not in Path(member.name).parts
            data=tar.extractfile(member).read();name=Path(member.name).name
            assert len(data)==expected[name]['bytes']
            assert hashlib.sha256(data).hexdigest()==expected[name]['sha256']
    assert raw[s-1]['retrieval_queries']==1500
    assert raw[s-1]['retrieval_metrics_recomputed_exact']
    assert len(raw[s-1]['probes'])==21
    assert ctrl[s-1]['intervention_hook_count']==0 and ctrl[s-1]['cuda_ok']
    archives.append(dict(**index,verified=True))

manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
rows=[];audits=[];gold_corrections=[]
for s in (1,2):
    ms=main[s-1]['M3_clock_calibrated_patch_salience_routing']['samples'];ma=read(f'm3_audit_s{s}.json')
    assert [r['idx'] for r in ms]==[r['idx'] for r in manifest[f'server{s}'][:4]]
    assert len(ma)==4
    for row,audit,sample in zip(ms,ma,manifest[f'server{s}'][:4]):
        assert row['native_replay_max_err']==0 and audit['tail_roundtrip']
        assert not set(audit['hook_rows'])&set(audit['option_rows'])
        assert all(x not in audit['freeform_input_text'] for x in ('(A)','(B)','(C)','Answer with only the letter'))
        row=dict(shard=s,**row)
        gold=sample['gold']
        options=re.findall(r'\(([A-D])\)\s*(.*?)(?=\s*\([A-D]\)|$)',sample['question'])
        gold_text=dict(options)[gold].strip()
        original_gold=row['gold_letter'];original_text=row['gold_answer_text']
        row['gold_letter']=gold;row['gold_answer_text']=gold_text
        original_audits={}
        corrected_audits={}
        for kind in ('native','routed'):
            text=row[f'freeform_24tok_{kind}'].strip().lower()
            audit_name=f'freeform_{kind}_audit'
            original_audits[kind]=row[audit_name]
            ca=dict(row[audit_name]);match=None
            if gold_text.lower() in ('yes','no'):
                if text.startswith('yes'):match=gold_text.lower()=='yes'
                elif text.startswith('no'):match=gold_text.lower()=='no'
            ca['yes_no_polarity_matches_gold']=match
            row[audit_name]=ca;corrected_audits[kind]=ca
        gold_corrections.append(dict(idx=sample['idx'],original_script_gold=original_gold,manifest_gold=gold,original_script_gold_text=original_text,manifest_gold_text=gold_text,original_audits=original_audits,corrected_audits=corrected_audits))
        row['native_prediction']='ABCD'[max(range(len(audit['probabilities']['p_nat'])),key=lambda i:audit['probabilities']['p_nat'][i])]
        row['routed_prediction']='ABCD'[max(range(len(audit['probabilities']['p_mid_fwd'])),key=lambda i:audit['probabilities']['p_mid_fwd'][i])]
        rows.append(row);audits.append(audit)
save('corrected_gold_audit.json',gold_corrections)
mean=lambda key:statistics.mean(x[key] for x in rows)
cells={}
for task,letters in [('TA','AB'),('TF','ABC'),('CMTB','ABC')]:
    for letter in letters:
        rr=[r for r in rows if r['task']==task and r['gold_letter']==letter]
        cells[f'{task}/{letter}']=dict(total=len(rr),native_correct=sum(r['native_prediction']==letter for r in rr),routed_correct=sum(r['routed_prediction']==letter for r in rr))
allprobes=raw[0]['probes']
m2=main[0]['M2_shrinkage_calibrated_connected_to_C1']
summary=dict(replicas_identical_M1_M2=True,original_script_unmodified=True,n_micro=8,cells=cells,
    mean_tv_swap_mid=mean('tv_mid_kslot_swap'),mean_tv_swap_late=mean('tv_late_kslot_swap'),
    mean_tv_routed=mean('tv_mid_routed_vs_nat'),mean_tv_random=mean('tv_mid_rand3seed_vs_nat'),
    mean_tv_irrelevant=mean('tv_mid_irrel_slot_vs_nat'),
    near_invariant_vector_max_error=max(a['vec_reverse_max_abs'] for a in audits),incorrect_original_gold_count=sum(x['original_script_gold']!=x['manifest_gold'] for x in gold_corrections),M2={})
for suite,part in m2.items():
    if suite=='verified_foil_audit':continue
    full=part['B7_shrinkage_calibrated_full_c1_connected']
    controls=[v['overall_4way_pct'] for k,v in part.items() if k.startswith(('B1_','B2_','B3_','B4_'))]
    summary['M2'][suite]=dict(full4=full['overall_4way_pct'],best_control4=max(controls),gap_pp=round(full['overall_4way_pct']-max(controls),1),during=full['by_state']['During'],reference_v3_gain10pp_during35_met=full['overall_4way_pct']-max(controls)>=10 and full['by_state']['During']>=35)
save('completion_audit.json',dict(summary=summary,archives=archives,script_sha256=hashlib.sha256(source).hexdigest()))
save('summary.json',summary)

lines=['# CQ-TSPR / PC-EBR Stage-0 v4 — kết quả, 2026-10-15','',
'**V4 chạy xong trên cả hai server với nguyên công thức của đề xuất. C3 cải thiện retrieval; bản M2 đầy đủ vẫn thua đối chứng tốt nhất trên cả ba suites. M3 có lỗi bất biến khi đảo thời gian: hai vector gần như giống nhau, nên phép thử sensitivity không đo được hướng thời gian như dự định. Chưa đủ bằng chứng chuyển A0–A7.**','',
'Hai runtime T4 mới đều khỏe; M1/M2 trùng toàn bộ giữa hai replicas. Có 252 pairs: 202 train/16 origins và 50 held-out/4 origins; ba seeds cố định 20261015/16/17. M1 có 30 fits/server (5 conditions × 3 seeds × 2 representations), 180 steps/fit. M2 có 21 probes/server, 200 steps/probe. Chỉ 8 ID micro, 4/shard, rotation 0, cùng các lượt trước. Chưa chạy QA40 đầy đủ, A0–A7 hoặc reserve400.','',
'Bản gốc 1515 dòng được lưu cùng proposal; alias trong server_jobs có SHA giống bản gốc. Wrapper chỉ chèn quan sát để lưu checkpoints, scores, episodes, probabilities và generated texts; không đổi công thức, nhãn, seeds hoặc siêu tham số. Không cần devicefix/autograd repair ở mainrun v4. Prepare thêm input_ids vào cache và dùng T4 SDPA explicit-KV-repeat như v3.','',
'Native pairs là post-media readout block index 27; pooled là mean media tokens cùng block. Đây là pseudo descriptions từ model, không phải annotation event/time độc lập. 50 queries đến từ 4 origins đã xem nhiều lần; ba seeds và hai replicas không tăng số nguồn held-out độc lập. Nonnull có 31 queries theo substring filter; không gọi đó là 31 event ground truths.','',
'## M0 — ổn định số','',
f"```json\n{json.dumps(main[0]['Algebraic_and_TinyNoise_Unit_Checks'],indent=2)}\n```",'',
'Shrinkage sửa được counterexample nhiễu 1e-4, giữ peak support ≈.2 và relational feature rất nhỏ. Đây là unit check ổn định số, chưa xác nhận confidence calibration hoặc correctness khi logits nhọn nhưng chọn nhầm event.','',
'## M1 — đầy đủ số đúng qua ba seeds','',
'Các cột Global/Exact/Multipos có mẫu số 50; Nonnull có 31. Sai = mẫu số trừ số đúng của từng seed. Đây là retrieval, không phải số câu QA40.','']
for key,title in [('M1_native_clean_comparison','Native readout'),('M1_pooled_clean_comparison','Pooled media')]:
    lines += [f'### {title}','','| Condition | Global đúng/50 | Exact đúng/50 | Multipos đúng/50 | Nonnull đúng/31 | Mean Global/Exact/Multipos/Nonnull % |','|---|---|---|---|---|---|']
    for name,value in main[0][key]['summary'].items():
        metrics='/'.join(f'{value[k]:.2f}' for k in ('mean_global_pct','mean_within_exact_pct','mean_within_multipos_pct','mean_within_nonnull_pct'))
        lines.append(f"| {name} | {value['global_hits_3seeds']} | {value['exact_hits_3seeds']} | {value['multipos_hits_3seeds']} | {value['nonnull_hits_3seeds']} | {metrics} |")
    lines.append('')
lines += ['C3 native: Global 11→15/50, Exact 16→20/50, Multipos 21→26/50, Nonnull 10→12/31 ở cả 3 seeds. Pooled: Global 16→[22,21,21]/50; Exact 19→[21,20,20]/50. Đây là tiến bộ đáng giữ làm baseline. Tuy nhiên C3 đổi đồng thời standardization và within-rec loss weight 0→.20; C2 dùng .25, nên chưa cô lập đóng góp whitening. Code chỉ chuẩn hóa diagonal mean/std, không tính full covariance whitening. Pooled shuffle Exact [18,18,20] gần C1 [19,19,19]; C8 Global seed 3=23 còn hơn C3. Chưa có xác nhận localization/generalization ổn định trên nguồn mới.','',
'## M2 — mọi probe và từng lớp','',
'Natural có 232 held-out episodes (58/lớp), Reversal 576 (144/lớp), Roll 156 (39/lớp). Roll v3 có 752 episodes, nên v3/v4 không cùng evaluation set. Tất cả vẫn phụ thuộc 4 held-out origins. 3-way là argmax riêng ba relation logits trên nonfoil subset; không phải lọc 4-way predictions. Paired metric chỉ trên các cặp còn cả fwd/rev sau balancing/truncation.','']
for suite_idx,(suite,part) in enumerate((k,v) for k,v in m2.items() if k!='verified_foil_audit'):
    lines += [f'### {suite}','','| Probe | Đúng/sai/tổng4way | Acc4% | Acc3% | Before đúng/tổng | During đúng/tổng | After đúng/tổng | Foil đúng/tổng | Paired đúng/tổng (%) |','|---|---|---:|---:|---|---|---|---|---|']
    for j,(name,value) in enumerate(part.items()):
        audit=allprobes[suite_idx*7+j]
        assert round(100*audit['correct4']/audit['total'],1)==value['overall_4way_pct']
        counts=[f"{audit['by_class'][str(c)]['correct']}/{audit['by_class'][str(c)]['total']}" for c in range(4)]
        paired=f"{audit['paired_correct']}/{audit['paired_total']} ({value.get('paired_both_directions_correct_pct','—')})" if audit['paired_total'] else '—'
        lines.append(f"| {name} | {audit['correct4']}/{audit['total']-audit['correct4']}/{audit['total']} | {value['overall_4way_pct']} | {value['overall_3way_no_foil_pct']} | {' | '.join(counts)} | {paired} |")
    lines.append('')
lines += ['Full B7 đạt37.9/41.3/21.2%; đối chứng tốt nhất47.0/43.2/28.8%; gap−9.1/−1.9/−7.6pp. Reversal paired full53.7% còn dưới mean-time59.3%. During full62.1/63.9/15.4%; tăng During ở hai suites đầu không đồng nghĩa thắng tổng thể. Phi-only Natural44.0% còn hơn full37.9%, trong khi mean-time vẫn47.0%. Uniform25.0/33.3% nhưng có thể dự đoán một lớp cố định: During100% ở uniform Roll là constant-class, không phải năng lực temporal. Theo gain10pp/During35 của v3 làm mốc tham chiếu, cả3suite đều không đạt; v4 không được xem là một preregistration gate mới.','',
'Foil train chỉ5/16origins qua filter;11/16fallback. Held-out0/4qua,4/4fallback. Điều này trái claim100%strict/no fallback trong proposal. Kể cả qua cosine/Jaccard cũng chỉ là khác mô tả văn bản, không xác nhận event vắng mặt trong recording. C1sim_mat được dùng thật ở Natural/Reversal; Roll dùng C1encoder cho truy vấn hợp thành rồi khử projection của hop1, chưa phải learned multi-hop reasoning.','',
'## M3 — micro-probe và generation','',
'| Shard/ID | Task/gold | Gate | Cosfwd/rev | SwapTVlate/mid | Routed/nativeTV | IrrelevantTV | RandomTV | MCnative→routed |','|---|---|---:|---:|---|---:|---:|---:|---|']
for row in rows:
    lines.append(f"| {row['shard']}/{row['idx']} | {row['task']}/{row['gold_letter']} | {row['confidence_gate_gamma']} | {row['fwd_vs_rev_vector_cosine']} | {row['tv_late_kslot_swap']}/{row['tv_mid_kslot_swap']} | {row['tv_mid_routed_vs_nat']} | {row['tv_mid_irrel_slot_vs_nat']} | {row['tv_mid_rand3seed_vs_nat']} | {row['native_prediction']}→{row['routed_prediction']} |")
lines += ['',f"MeanTVswaplate={mean('tv_late_kslot_swap'):.6f},mid={mean('tv_mid_kslot_swap'):.6f};meanrouted={mean('tv_mid_routed_vs_nat'):.6f},irrelevant={mean('tv_mid_irrel_slot_vs_nat'):.6f},random={mean('tv_mid_rand3seed_vs_nat'):.6f}. Max|vecfwd−vecrev|={summary['near_invariant_vector_max_error']:.9g}. Khi cảswapTV gần0, tỷ sốmid/late không có ý nghĩa để suy khả năng temporal giữa tầng. Đúng hướng thời gian phải được đo bằng correctness sau biến đổi hợp lệ.",'',
'### Đúng/sai theo task và gold trên8micro,rotation0','',
'| Task/gold | N | Native đúng/sai | Routed đúng/sai |','|---|---:|---|---|']
for cell,value in cells.items():
    n=value['total'];nc=value['native_correct'];rc=value['routed_correct']
    lines.append(f'| {cell} | {n} | {nc}/{n-nc} | {rc}/{n-rc} |')
lines += ['', '**Các cellsN=0 chưa được đánh giá. Bảng8micro không đại diện kết quả40mẫu hoặc score cyclic-option rotations110 của các vòng trước.**','',
'### Văn bản24tokens','']
lines += [f"Tổng MC micro: native đúng {sum(r['native_prediction']==r['gold_letter'] for r in rows)}/8, routed đúng {sum(r['routed_prediction']==r['gold_letter'] for r in rows)}/8. Đây là rotation 0 trên 8 mẫu, không phải kết quả đầy đủ QA40.",'']
lines += ['**Lỗi nhãn trong script gốc:** `sample.get("answer","A")` mặc địnhA vì manifest dùng`gold`. Sai5/8nhãn; bảng này chấm lại từ outputs/probabilities đã lưu bằng manifestgold, không chạy lại model và không sửa mainrun. Raworiginalaudit được giữ nguyên; so sánh nhãn/cực tính sửa ởcorrected_gold_audit.json.','']
for row in rows:
    lines += [f"**{row['idx']} — {row['task']},gold{row['gold_letter']} ({row['gold_answer_text']})**",'',f"Native: {row['freeform_24tok_native']}",'',f"Routed: {row['freeform_24tok_routed']}",'',f"Audit native: `{json.dumps(row['freeform_native_audit'],ensure_ascii=False)}`",'',f"Audit routed: `{json.dumps(row['freeform_routed_audit'],ensure_ascii=False)}`",'']
lines += ['Regex chỉ phát hiện mốc vượt thời lượng, không phát hiện mọi timestamp sai hoặc chứng minh một mốc là clip-relative. Không nhắc timestamp/không trả YesNo không tự là cải thiện. Câu trả lời chung chung, thiếu nội dung hoặc bị cắt ở24tokens cần được tính vào chất lượng. Không dùng8mẫu để tunegate/strength rồi gọi là validation mới.','',
'### Đồng hồ và cấu trúc vector','',
'| ID | Clipduration | Audio25Hzspan | Cachedvmax | Gv | Patches/griduniform | Mappedvmax |','|---|---:|---:|---:|---:|---|---:|']
for row in rows:
    c=row['clock_provenance_audit']
    lines.append(f"| {row['idx']} | {c['clip_duration_sec']} | {c['audio_25hz_duration_sec']} | {c['cached_visual_sec_max']} | {c['n_visual_temporal_grids_Gv']} | {c['patches_per_visual_grid']}/{c['uniform_patches_across_grids']} | {c['calibrated_visual_sec_max']} |")
lines += ['',
'Mappingg→(g+.5)T/G là giả định chia đều theo duration, chưa khôi phục framePTS. Cần lưu sampledframeindices/timestamps,temporalpatchgrouping và padding. Code không assert audio-span khớp duration hoặc countsuniform. Không nên gọi oracle hay kết luận lỗi đồng hồ model đã được chứng minh.','',
'Với phép đảoP trên lưới đối xứng, p′=Pp,V′=PV nên tổngp′V′=tổngpV:ev_bound giữ nguyên. rel_scalar đổi dấu, chron_dir cũng đổi dấu, nên tíchrel_scalar×chron_dir giữ nguyên. Vì thế vector cuối là even/invariant dưới đảo; đây là lỗi thiết kế readout temporal, không phải bằng chứng backbone không biết thứ tự. Dữ liệu thực và toy controls ghi ởcontrols_s*.json.','',
'Code gamma=max(.35,min(1,max(alphaV,alphaA))) có floor.35; uniform vẫninject.077norm/layer (.22×.35), trái mô tả gamma→0 khi không có evidence. Top15% cosine theo câu hỏi là salienceheuristic, chưa được xác nhận là định vị đúng event hoặc phân biệt narration với action. A/V cộng đối xứng chưa tự chứng minh conditional binding hoặc suy luậnA→A,V→V.','',
'## Tái lập và giới hạn','',
f"Mỗi server{raw[0]['raw_files']}rawPT:30projectors,30retrievalmatrices,2bundles,24episodetensors,21probes,4M3banks;1500retrievalqueryrows/server được tính lại đúng. Archive và từngentry SHA được kiểm tra local, khôngextract. Native replay0,tokenroundtrip/hookdisjoint/freeform bỏchoices đủ8/8. Source/cachehash và metadata được lưu; secrets chỉ ởignoredconfigs.",'',
'Việc upload ban đầu bị auto-review từ chối; đã kiểm kê read-only xác nhận runtime mới trống. Sau câu trả lời rõ ràng của người dùng “Đồng ý tải lên cả hai server”, mới tải4cache,manifest vàcode. Không thay dữ liệu hoặc gửi sang đích khác.','',
'Phản biện và đề xuất tiếp: [feedback_to_yoshua_bengio.md](feedback_to_yoshua_bengio.md). Các định danhpaper sai và giới hạn literature: [literature_audit.md](literature_audit.md). TênYoshua/Yann là quy ước vai trò trong trao đổi, không phải xác nhận tác giả thật của các nhà khoa học.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in rows]
with (D/'m3_per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
print(json.dumps(summary,ensure_ascii=False,indent=2))
