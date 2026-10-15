"""Summarize archived Stage-0 results, paired controls, and provenance checks."""
import collections
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/pcebr_stage0'
def read(name):return json.loads((FOLDER/name).read_text(encoding='utf-8'))
def dump(name,value):(FOLDER/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

first=read('devicefix_results_s1.json');second=read('devicefix_results_s2.json')
assert {k:v for k,v in first.items() if k!='elapsed_sec'}=={k:v for k,v in second.items() if k!='elapsed_sec'}
original=(FOLDER/'stage0_pcebr_mechanism_check_original.py').read_text(encoding='utf-8')
fixed=(FOLDER/'stage0_pcebr_mechanism_check_devicefix.py').read_text(encoding='utf-8')
assert original.replace('torch.randperm(len(m_idx), generator=g_shuf)',
    'torch.randperm(len(m_idx), generator=g_shuf, device=device)')==fixed
audits=[read(f'audit_results_s{s}.json') for s in (1,2)]
baselines=[read(f'baseline_s{s}.json') for s in (1,2)]
for i,a in enumerate(audits):
    main=first[('M1_native_pairs_factorial','M1_pooled_pairs_factorial')[i]]
    for short,long in [('uncentered','3_pcebr_uncentered_r24'),('dual_center_original','4_pcebr_deconfounded_r24'),('dual_center_shuffle','5_pcebr_shuffled_time_ctrl')]:
        for metric in ('global_hits','within_rec_hits'):
            assert a['results'][f'{short}_seed20261015'][metric]==main[long][metric]
rows=[]
lines=['PC-EBR — kết quả Cổng 0 (Stage-0), 2026-10-15','',
    '**Kết luận: H1 không đạt tiêu chí đề xuất; M2 chỉ qua kiểm tra đại số; H3 chưa được hiện thực. Chưa đủ điều kiện chuyển A0–A7.**','',
    'Hai server T4 chạy cùng cache và cùng script cho toàn bộ số đếm giống nhau. Đây là replicas, không phải hai tập dữ liệu độc lập. '
    'Bản gốc lỗi CUDA-generator/CPU-randperm trên cả hai server. Bản devicefix chỉ thêm `device=device` cho randperm; giữ seed, split, architecture, loss và budget. '
    'Logs lỗi và cả hai source được lưu nguyên. Mỗi exact run sửa device hoàn tất trong khoảng 13–14 giây.','',
    '252 pairs: 202 train/16 recording origins; 50 held-out/4 origins. Held-out này đã được xem ở R14/R18, không phải final test mới. '
    'Nhãn là pseudo descriptions do model sinh, không phải annotation sự kiện hoặc interval truth độc lập. Native cache là post-media last-token state tại block index27 (tầng28), '
    'pooled cache là media-token mean cùng tầng, không phải một extraction mới ở tầng22.','',
    '| Cache | Cấu hình | Global đúng/tổng | Within-recording đúng/tổng |',
    '|---|---|---:|---:|']
for i,(key,label) in enumerate([('M1_native_pairs_factorial','Native post-media readout'),('M1_pooled_pairs_factorial','Pooled media')]):
    b=baselines[i]
    lines.append(f"| {label} | Archived {('R14','R18')[i]} replay | {b['metrics']['global_hits']}/50 | {b['metrics']['within_rec_hits']}/50 |")
    for name,value in first[key].items():
        lines.append(f"| {label} | {name} | {value['global_hits']}/50 | {value['within_rec_hits']}/50 |")
        rows.append(dict(cache=label,method=name,**value))
lines += ['',
    'Native centered r24 chỉ6/50 global so R14=7/50 và r24 uncentered=13/50; within-rec16/50 so cả hai baseline=13/50 (+6 điểm phần trăm). '
    'So native raw chưa học9/50, mức tăng within là14 điểm phần trăm, chưa đạt15 điểm đề xuất. '
    'Pooled centered=5/50 global và11/50 within; R18=1/50 và11/50, tức global+8 điểm phần trăm, within không tăng. '
    'Pooled shuffle=10/50 global,11/50 within. Không thể so within-rec với global7/50 hoặc1/50 như cùng metric.','',
    '**Đối chứng exploratory ba fixed seeds — không thay tiêu chí sau kết quả**','',
    'Seeds 20261015/20261016/20261017;384 steps, cùng r24, lr/loss/hard-neg boost. '
    'Media-only center giữ nguyên query text khi inference, giúp kiểm tra phụ thuộc vào batch pseudo queries held-out. '
    'Các số dưới là từng seed và mean; mean của3 seed không phải150 event labels độc lập. '
    'Audit v2 giữ nguyên AST training gốc; các nhánh seed20261015 khớp toàn bộ mainrun. '
    'Audit v1 viết lại loss tương đương toán học gây đổi optimization trajectory được giữ riêng audit_v1 và không dùng cho bảng chính. '
    'Một lỗi truyền OUT của wrapper v2 cũng được lưu ở audit_wrapper_error trước khi sửa; không có sample nào được thay.','',
    '| Cache | Variant | Global đúng/50 từng seed | Within-rec đúng/50 từng seed | Mean global % | Mean within % |',
    '|---|---|---|---|---:|---:|']
allqueries=[];paired=[]
def mcnemar(w,l):
    n=w+l
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1.0
for i,a in enumerate(audits):
    group=collections.defaultdict(list)
    for name,value in a['results'].items():
        group[value['variant']].append(value)
        for q in value['per_query']:allqueries.append(dict(cache=a['prefix'],method=name,**q))
    for variant,values in group.items():
        values.sort(key=lambda v:v['seed'])
        gh=[v['global_hits'] for v in values];wh=[v['within_rec_hits'] for v in values]
        lines.append(f"| {a['prefix']} | {variant} | {gh} | {wh} | {statistics.mean(gh)*2:.1f} | {statistics.mean(wh)*2:.1f} |")
    candidate=a['results']['dual_center_original_seed20261015']['per_query']
    for control in ('uncentered_seed20261015','dual_center_shuffle_seed20261015'):
        other=a['results'][control]['per_query']
        for metric in ('global_correct','local_correct'):
            w=sum(x[metric] and not y[metric] for x,y in zip(candidate,other))
            l=sum(y[metric] and not x[metric] for x,y in zip(candidate,other))
            gd=collections.defaultdict(int)
            for x,y in zip(candidate,other):gd[x['recording']]+=int(x[metric])-int(y[metric])
            actual=abs(sum(gd.values()));perms=list(itertools.product((-1,1),repeat=len(gd)))
            p=sum(abs(sum(s*d for s,d in zip(sign,gd.values())))>=actual for sign in perms)/len(perms)
            paired.append(dict(cache=a['prefix'],candidate='dual_center_original_seed20261015',control=control,
                metric=metric,wins=w,losses=l,mcnemar_p=mcnemar(w,l),source_signflip_p=p,origin_differences=dict(gd)))
lines += ['',
    'Các comparisons paired bên dưới là exploratory, chưa hiệu chỉnh multiple tests; chỉ4 held-out origins nên không thể xác nhận generalization từ p-value.','',
    '| Cache | Control | Metric | Cứu/làm hỏng | Exact McNemar p | Source sign-flip p |',
    '|---|---|---|---:|---:|---:|']
for p in paired:lines.append(f"| {p['cache']} | {p['control']} | {p['metric']} | {p['wins']}/{p['losses']} | {p['mcnemar_p']:.4f} | {p['source_signflip_p']:.4f} |")
lines += ['','**Candidate sets và tính mơ hồ của pseudo labels**','']
for a in audits:
    lines.append(f"{a['prefix']}: group sizes={[g['n'] for g in a['groups']]}; random uniform local top1 chance={a['uniform_random_local_chance']:.1%}; "
        f"{len(a['duplicate_content_groups'])} nhóm content trùng ở nhiều interval cùng recording/source.")
    for g in a['duplicate_content_groups']:
        lines.append(f"- {g['recording']}/{g['source']}, indices={g['indices']}: {g['content']}")
lines += ['','Nếu cùng mô tả/query feature lặp ở nhiều interval, không có thông tin trong query để chỉ đúng instance start; '
    'hard exact-start top1 có thể phạt một alternative hợp lệ. Cần multi-positive interval labels hoặc instance-disambiguating queries. '
    'Shuffle ở mã gốc tráo media-text matching trong group, không sử dụng intervals trong objective, nên là alignment corruption control chứ chưa chứng minh học đại số thời gian.','',
    '**Kiểm tra M2 và giới hạn**','',
    '```json',json.dumps(first['M2_interval_algebra_invariance'],indent=2),'```','',
    'Đây là kiểm tra hàm Phi trên intervals biết trước; không có audio/video được dịch, không có event retrieval, two-hop reasoning hoặc LLM QA. '
    'Phép cộng interval bằng nhau khi đảo hai hạng là identity đại số, không chứng minh attention tiêu chuẩn mù thứ tự. '
    'Counterexample với content-dependent weights [.9,.1] và centers[3,11]: weighted time3.8, swap times10.2. '
    'Pooling giao hoán theo thứ tự liệt kê cặp (content,time) khác việc tráo gán time giữa hai content.','',
    'Translation invariance error=0 trong toy check. Nhưng common scale x2 làm full Phi đổi0.646627; '
    'signed log gaps và duration ratio có offset0.1 không scale-invariant. Order categories có thể giữ nguyên dưới positive affine map, '
    'không suy full latent/readout invariant. Full Phi không antisymmetric; center-offset channel thì có. '
    'Coverage dùng inter/min(duration), là overlap coefficient, không phải coverage của anchor; anchor[0,10],target[2,4] trả1 thay vì0.2 nếu muốn đo anchor coverage.','',
    '**H3 / M3**','',
    'Không có implementation microprobe4 mẫu, middle/late routing, TV_swap, learned two-hop bank hay source shift QA trong attachment. '
    'Không ghi H3 pass/fail hoặc tự thay bằng một heuristic khác.','']

manifest=json.loads((ROOT/'results/research40_manifest.json').read_text(encoding='utf-8'))
origins={x['video_path'].replace('\\','/').split('/')[-2] for x in manifest['selected']}
ia=read('input_audit_s1.json');ib=read('input_audit_s2.json')
pair_origins=set(ia['avbind_native_pairs']['train_origins'])|set(ia['avbind_native_pairs']['heldout_origins'])
assert not pair_origins&origins
cache_checks={}
for name,record in ia['caches'].items():
    local=hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest()
    cache_checks[name]=dict(local_sha256=local,servers_match=local==record['sha256']==ib['caches'][name]['sha256'])
    assert cache_checks[name]['servers_match']
assert not ia['avbind_native_pairs']['overlap']
exclusions=set()
for item in ia['caches'].values():
    selection=item['metadata'].get('selection',{})
    if 'excluded_recording_ids' in selection:exclusions.update(selection['excluded_recording_ids'])
for b in baselines:exclusions.update(b['metadata']['excluded_recording_ids'])
assert not pair_origins&exclusions
lens_rows=[]
for s in (1,2):
    path=ROOT/f'results/chronodesmear_followup40/results_s{s}.jsonl'
    lens_rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
lens_agreement={}
for layer in ('22','30','34','35'):
    changes=[r['idx'] for r in lens_rows if r['layer_predictions'][layer]!=r['predictions']['N0_native']]
    lens_agreement[layer]=dict(agreement=40-len(changes),different_ids=changes)
dump('paired_controls.json',paired)
dump('completion_audit.json',dict(replica_counts_equal=True,only_device_fix=True,caches=cache_checks,
    research40_origin_count=len(origins),pair_origins=len(pair_origins),research40_overlap=sorted(pair_origins&origins),
    pair_overlap_with_recorded_exclusions=sorted(pair_origins&exclusions),recorded_exclusion_origins=len(exclusions),
    parameters=audits[0]['params'],audit_seed0_matches_main=True,audit_version=2,lens_agreement=lens_agreement,backbone_trained=False,
    QA40_run=False,H1='does_not_meet_proposed_gate',H2='toy_algebra_only',H3='not_implemented'))
lines += ['**Tái lập và provenance**','',
    'Bốn cache local có SHA khớp cả hai server. Pair origins không trùng38 origins của research40; '
    f"không trùng {len(exclusions)} origins trong danh sách loại trừ được lưu. Đây là kiểm tra metadata đã archive; không mở reserve400 hoặc chạy QA. "
    f"Projector r24 có {audits[0]['params']:,} parameters, không phải~98K; R14/R18 có397,440.",
    'Block30 native lens thực tế đồng ý '+str(lens_agreement['30']['agreement'])+'/40 với final (IDs khác: '+str(lens_agreement['30']['different_ids'])+'). '
    'Block34 đồng ý40/40. Không được gọi quan sát agreement là quyết định đã khóa cứng về mặt nhân quả.','',
    'Không chạy A0–A7, không huấn luyện backbone, không chỉnh hyperparameters theo QA40. '
    'Đối chứng3seeds là extension exploratory riêng sau mainrun, không thay H1 threshold hậu kiểm. '
    'Raw main results/audits/job logs, per-query predictions, similarity matrices, sources và cache/checkpoint hashes lưu trong folder này.']
(FOLDER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
dump('table.json',rows)
with (FOLDER/'audit_per_query.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(allqueries[0]));writer.writeheader();writer.writerows(allqueries)
print(json.dumps(dict(report=str(FOLDER/'report.md'),main_rows=len(rows),audit_query_rows=len(allqueries),
    lens_agreement=lens_agreement,cache_checks=cache_checks),indent=2))
