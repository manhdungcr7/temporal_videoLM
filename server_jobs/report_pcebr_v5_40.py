"""Verify and report the complete frozen40 evaluation, without fitting or selecting variants."""
import ast
import collections
import csv
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/pcebr_v5_40'
def read(name):return json.loads((D/name).read_text(encoding='utf-8'))
def save(name,data):(D/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
samples={r['idx']:r for r in manifest['selected']}
rows=[];rotations=[];archives=[]
for server in (1,2):
    part=read(f'results_s{server}.json')
    assert len(part)==20 and [r['idx'] for r in part]==[r['idx'] for r in manifest[f'server{server}']]
    for row in part:
        sample=samples[row['idx']]
        assert sample['duration']<=45 and sample['decoded_audio_duration']<=45
        assert row['task']==sample['task'] and row['gold']==sample['gold']
        rows.append(dict(server=server,**row))
    protocol=read(f'protocol_s{server}.json');audit=read(f'archive_audit_s{server}.json');idx=read(f'archive_index_s{server}.json')
    assert protocol['manifest_sha256']==manifest['manifest_sha256']
    source=(ROOT/'results/pcebr_stage0_v5/stage0_pcebr_v5_original.py').read_bytes()
    assert protocol['source_sha256']==hashlib.sha256(source).hexdigest()
    assert protocol['runner_sha256']==hashlib.sha256((ROOT/'server_jobs/pcebr_v5_40.py').read_bytes()).hexdigest()
    assert audit['all_rotations_reaggregated_exact'] and audit['all_raw_logits_logprobs_verified']
    assert audit['hooks']==0 and audit['cuda_ok']
    archive=D/f'raw_artifacts_s{server}.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==idx['sha256']
    expected={f['name']:f for f in audit['files']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)==idx['files']
        for m in members:
            assert m.isfile() and not m.issym() and '..' not in Path(m.name).parts
            data=tar.extractfile(m).read();meta=expected[m.name]
            assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
            if m.name.startswith('rotations/') and m.name.endswith('.json'):
                record=json.loads(data);rotations.append(record)
    archives.append(dict(**idx,sha_verified=True,all_entries_verified=True))
assert len(rows)==40 and len({r['idx'] for r in rows})==40
assert len(rotations)==sum(r['rotation_count'] for r in rows)==110
assert set(samples)=={r['idx'] for r in rows}

# Reaggregate every rotation locally, independently of the server summary.
byrot=collections.defaultdict(list)
for r in rotations:byrot[r['idx']].append(r)
for row in rows:
    rr=sorted(byrot[row['idx']],key=lambda r:r['rotation'])
    assert len(rr)==row['rotation_count']
    for method in ('native','v5'):
        total=[0.]*len(row['scores'][method])
        for r in rr:
            assert sorted(r['perm'])==list(range(len(total)))
            assert r['manifest_sha256']==manifest['manifest_sha256']
            for slot,original in enumerate(r['perm']):total[original]+=r['logprobs'][method][slot]
        assert total==row['scores'][method]
        assert 'ABCD'[max(range(len(total)),key=lambda j:total[j])]==row['predictions'][method]

prior={}
for server in (1,2):
    for line in (ROOT/f'results/argon_latent40/results_s{server}.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r.get('status')=='ok':prior[r['idx']]=r
prior_diffs=[];max_native_score_drift=0.
for row in rows:
    if row['idx'] in prior:
        old=prior[row['idx']]
        if old['predictions']['R3_M0_base']!=row['predictions']['native']:
            prior_diffs.append(dict(idx=row['idx'],old=old['predictions']['R3_M0_base'],new=row['predictions']['native']))
        max_native_score_drift=max(max_native_score_drift,max(abs(a-b) for a,b in zip(old['scores']['R3_M0_base'],row['scores']['native'])))

micro={}
for server in (1,2):
    for a in json.loads((ROOT/f'results/pcebr_stage0_v5/m3_audit_s{server}.json').read_text(encoding='utf-8')):micro[a['result']['idx']]=a
micro_probability_error=0.
for idx,audit in micro.items():
    r=next(r for r in byrot[idx] if r['rotation']==0)
    for method,key in [('native','p_nat'),('v5','p_mid_fwd')]:
        probability=[math.exp(v) for v in r['logprobs'][method]]
        micro_probability_error=max(micro_probability_error,max(abs(a-b) for a,b in zip(probability,audit['probabilities'][key])))
    assert 'ABCD'[max(range(len(r['logprobs']['v5'])),key=lambda j:r['logprobs']['v5'][j])]==audit['result']['mc_pred_routed_fwd']

def tally(part,method,rotation0=False):
    field='rotation0_predictions' if rotation0 else 'predictions'
    correct=sum(r[field][method]==r['gold'] for r in part)
    return dict(correct=correct,wrong=len(part)-correct,total=len(part),accuracy=round(correct*100/len(part),2))
rescued=[r['idx'] for r in rows if r['predictions']['native']!=r['gold'] and r['predictions']['v5']==r['gold']]
harmed=[r['idx'] for r in rows if r['predictions']['native']==r['gold'] and r['predictions']['v5']!=r['gold']]
changed=[r for r in rows if r['predictions']['native']!=r['predictions']['v5']]
summary=dict(total=40,rotations=110,manifest_sha256=manifest['manifest_sha256'],
             native=tally(rows,'native'),v5=tally(rows,'v5'),rotation0={m:tally(rows,m,True) for m in ('native','v5')},
             rescued=rescued,harmed=harmed,changed_ids=[r['idx'] for r in changed],
             historical_native_changed=prior_diffs,historical_native_max_score_difference=max_native_score_drift,
             stage0_micro_probability_max_error=micro_probability_error,
             max_native_replay_error=max(r['native_replay_max_error'] for r in rotations),
             parity_failures=sum(not r['parity_checks']['original_micro_asserts_pass'] for r in rotations),
             zero_gate_rotations=sum(r['parity_checks']['zero_gate'] for r in rotations),
             mean_gate=statistics.mean(r['gate'] for r in rotations),by_cell={},by_task={})
for task,letters in [('TA','AB'),('TF','ABC'),('CMTB','ABC')]:
    part=[r for r in rows if r['task']==task]
    summary['by_task'][task]={method:tally(part,method) for method in ('native','v5')}
    for letter in letters:
        cell=[r for r in part if r['gold']==letter];assert len(cell)==5
        summary['by_cell'][f'{task}/{letter}']={method:tally(cell,method) for method in ('native','v5')}
tf=[r for r in rows if r['task']=='TF']
summary['tf_predictions']={m:dict(collections.Counter(r['predictions'][m] for r in tf)) for m in ('native','v5')}
summary['rotation0_rescued']=[r['idx'] for r in rows if r['rotation0_predictions']['native']!=r['gold'] and r['rotation0_predictions']['v5']==r['gold']]
summary['rotation0_harmed']=[r['idx'] for r in rows if r['rotation0_predictions']['native']==r['gold'] and r['rotation0_predictions']['v5']!=r['gold']]
summary['rotation0_changed']=[dict(idx=r['idx'],task=r['task'],gold=r['gold'],predictions=r['rotation0_predictions']) for r in rows if r['rotation0_predictions']['native']!=r['rotation0_predictions']['v5']]
rotation_tvs=[sum(abs(math.exp(a)-math.exp(b)) for a,b in zip(r['logprobs']['native'],r['logprobs']['v5']))/2 for r in rotations]
summary['probability_audit']=dict(mean_rotation_tv=statistics.mean(rotation_tvs),max_rotation_tv=max(rotation_tvs),
    changed_rotation_argmax=sum(max(range(len(r['perm'])),key=lambda j:r['logprobs']['native'][j])!=max(range(len(r['perm'])),key=lambda j:r['logprobs']['v5'][j]) for r in rotations),
    min_gate=min(r['gate'] for r in rotations),max_gate=max(r['gate'] for r in rotations))
summary['changed_margins']=[]
for r in changed:
    item=dict(idx=r['idx'],task=r['task'],gold=r['gold'],predictions=r['predictions'],scores=r['scores'])
    for method in ('native','v5'):
        ordered=sorted(r['scores'][method],reverse=True)
        item[method+'_top2_gap']=ordered[0]-ordered[1]
    summary['changed_margins'].append(item)
save('summary.json',summary)
save('completion_audit.json',dict(summary=summary,archives=archives,all_rotations_reaggregated_local=True))
save('per_sample.json',rows)
flat=[]
for r in sorted(rows,key=lambda r:r['idx']):
    item={k:r[k] for k in ('idx','task','gold','duration','rotation_count')}
    item.update(native=r['predictions']['native'],v5=r['predictions']['v5'],
                native_correct=r['predictions']['native']==r['gold'],v5_correct=r['predictions']['v5']==r['gold'],
                native_rotation0=r['rotation0_predictions']['native'],v5_rotation0=r['rotation0_predictions']['v5'])
    flat.append(item)
with (D/'per_sample.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

n=summary['native'];v=summary['v5']
lines=['# OB-TVOR v5 — đánh giá đủ frozen research40, 2026-10-16','',
f"**Native: {n['correct']} đúng/{n['wrong']} sai ({n['accuracy']}%). V5: {v['correct']} đúng/{v['wrong']} sai ({v['accuracy']}%). Sửa được {len(rescued)} câu, làm sai thêm {len(harmed)} câu; đổi prediction ở {len(changed)}/40 mẫu.**",'',
'Đúng bộ 40 mẫu đã cố định: mỗi nhóm task/gold có 5 mẫu, cả video và decoded audio ≤45 giây. Hai server lần này chạy hai shards 20 mẫu; không phải hai replicas của cùng 40 mẫu. Có đủ 110 cyclic option rotations, cộng log-softmax answer-letter scores sau ánh xạ về nội dung lựa chọn gốc, đúng protocol các bảng 40 mẫu trước.','',
'Đường inference giữ nguyên v5: grid interpolation, query-conditioned top 15% visual / top 25% audio patch pooling, VO block 18, vector content/order với tan(theta)=.85, gate không sàn, tangent update .045×gate ở blocks14/16/18/20/22/24. Hàm run_routed được lấy trực tiếp từ AST source gốc, không viết lại công thức. Tính vector riêng cho từng rotation, không giả định invariant khi đổi choices.','',
'Không có huấn luyện, tuning thresholds, thay samples hoặc chọn phương pháp theo gold. Các model M1 và classifier M2 không được dùng trong hàm M3 gốc, nên kết quả QA này không phải áp dụng classifier B8 để tự chọn đáp án. Chưa chạy generation tự do trên cả 40 mẫu hoặc reserve400. Đây vẫn là bộ development đã được xem qua nhiều vòng.','',
'## Đúng/sai theo task và đáp án chuẩn','',
'| Task/gold | N | Native đúng | Native sai | V5 đúng | V5 sai |','|---|---:|---:|---:|---:|---:|']
for cell,values in summary['by_cell'].items():
    a=values['native'];b=values['v5']
    lines.append(f"| {cell} | 5 | {a['correct']} | {a['wrong']} | {b['correct']} | {b['wrong']} |")
lines += [f"| **Tổng** | **40** | **{n['correct']}** | **{n['wrong']}** | **{v['correct']}** | **{v['wrong']}** |",'',
'TA: A=Yes,B=No. TF: A=Past,B=Present,C=Future. CMTB A/B/C là các hành động khác nhau theo từng câu hỏi, không có một nghĩa cố định cho mỗi chữ.','',
'| Task | Native đúng/tổng | V5 đúng/tổng |','|---|---|---|']
for task,values in summary['by_task'].items():
    a=values['native'];b=values['v5'];lines.append(f"| {task} | {a['correct']}/{a['total']} | {b['correct']}/{b['total']} |")
lines += ['',f"TF prediction counts: `{json.dumps(summary['tf_predictions'])}`. Không đánh giá bias Present chỉ từ accuracy tổng; xem phân bố predictions và các nhóm gold ở trên.",'',
'## Những câu đổi đáp án','',
'| ID | Task/gold | Native | V5 | Kết quả |','|---|---|---|---|---|']
for r in changed:
    outcome='Cứu' if r['idx'] in rescued else 'Làm hỏng' if r['idx'] in harmed else 'Sai→sai'
    lines.append(f"| {r['idx']} | {r['task']}/{r['gold']} | {r['predictions']['native']} | {r['predictions']['v5']} | {outcome} |")
if not changed:lines.append('| — | — | — | — | Không có câu đổi prediction |')
for item in summary['changed_margins']:
    lines += ['',f"ID {item['idx']}: native top-2 score gap={item['native_top2_gap']:.9g}; v5 gap={item['v5_top2_gap']:.9g}. Scores đầy đủ ở summary.json và per_sample.json."]
    if item['native_top2_gap']<1e-6:
        lines += ['Native gần như hòa điểm hai lựa chọn; kết quả đổi argmax ở đây cần được diễn giải như phá hòa, chưa đủ để chứng minh khôi phục suy luận thời gian.']
lines += ['', '## Rotation 0 riêng, không gộp với score chính','',
'| Cấu hình | Đúng | Sai | Tổng |','|---|---:|---:|---:|']
for m,c in summary['rotation0'].items():lines.append(f"| {m} | {c['correct']} | {c['wrong']} | 40 |")
lines += ['',
f"Rotation0 riêng: cứu {len(summary['rotation0_rescued'])} câu, làm sai thêm {len(summary['rotation0_harmed'])} câu. IDs được cứu={summary['rotation0_rescued']}; IDs làm sai thêm={summary['rotation0_harmed']}.",'',
'Score rotation 0 được báo cáo để nối với 8 micro Stage-0. Score chính là tổng các rotations, do đó một sample có thể đổi prediction khi gộp rotations. Không trộn hai kiểu đánh giá khi so gain.','',
'## Tất cả mẫu','',
'| ID | Task/gold | Native | V5 | Native đúng? | V5 đúng? |','|---|---|---|---|---|']
for r in flat:lines.append(f"| {r['idx']} | {r['task']}/{r['gold']} | {r['native']} | {r['v5']} | {int(r['native_correct'])} | {int(r['v5_correct'])} |")
lines += ['', '## Kiểm toán','',
f"- Manifest SHA: `{manifest['manifest_sha256']}`.",
f"- Đủ 40 mẫu duy nhất, 110 rotations; mỗi cell đủ 5. Native replay max error={summary['max_native_replay_error']}.",
f"- So với native lịch sử R3: {len(prior_diffs)} predictions đổi; max score drift={max_native_score_drift:.9g}. Chi tiết `{json.dumps(prior_diffs)}`.",
f"- Cả 8 micro rotation0 v5 predictions khớp Stage-0 v5. Max probability difference≈{micro_probability_error:.9g}.",
f"- Parity checks không đạt: {summary['parity_failures']}/110; zero-gate rotations: {summary['zero_gate_rotations']}/110; mean gate={summary['mean_gate']:.6f}. Các checks được ghi log, không loại mẫu theo kết quả kiểm tra.",
f"- Mean TV giữa native/v5 mỗi rotation={statistics.mean(rotation_tvs):.6f}, max={max(rotation_tvs):.6f}; {summary['probability_audit']['changed_rotation_argmax']}/110 rotations đổi argmax riêng. Gate min/max={summary['probability_audit']['min_gate']:.6f}/{summary['probability_audit']['max_gate']:.6f}.",
'- Raw logits/logprobs và aggregation được tính lại trên server và local. Hai archives và từng entry SHA đã kiểm tra; không extract. Source gốc, runner hash, raw banks/vectors và từng rotation được lưu.',
'- Hooks đã được tháo; CUDA khỏe ở cả hai server. Không đọc reserve400 hoặc thay nhãn benchmark.','',
'## Nhận xét','',
f"Gain đo được là {v['correct']-n['correct']:+d}/40 câu ({(v['correct']-n['correct'])*2.5:+.1f} điểm phần trăm). Đây là kết quả trực tiếp trên QA40; các accuracy 67,7/59,9% ở M2 Stage-0 là bài phân loại proxy khác, không thể thay thế score QA này.",'',
'Chỉ một câu CMTB được cứu, đúng mẫu 1898 có native gần như hòa A/C; TA và TF không có câu nào đổi prediction sau aggregation. TF/Past vẫn 0/5. Kết quả này chưa xác nhận giảm temporal hallucination có tính tổng quát. Ở thứ tự lựa chọn gốc, v5 giảm 5 câu đúng; lợi ích của cấu hình hiện tại chưa ổn định theo thứ tự lựa chọn. Không dùng gain của cyclic aggregation để bỏ qua tổn thất này.','',
'Đánh giá tiếp cần tập trung vào các câu được cứu/làm hỏng và quan hệ event/time/source thật. Không thể suy bảo toàn cú pháp, temporal correctness hoặc CVPR acceptance chỉ từ parity, cosine hoặc sensitivity. Không chỉnh strength/gate trên bảng 40 này rồi gọi kết quả mới là held-out confirmation.','',
'Các tệp chính: summary.json, per_sample.csv, per_sample.json, completion_audit.json, protocol_s*.json, raw_artifacts_s*.tar.gz.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
