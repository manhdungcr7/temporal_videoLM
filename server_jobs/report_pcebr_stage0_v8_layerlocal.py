"""Aggregate frozen40 paired development results, including origin-cluster uncertainty."""
import hashlib,json,math,random,statistics,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
D=R/'results/pcebr_stage0_v8_layerlocal'
manifest=json.loads((R/'results/research40_manifest.json').read_text(encoding='utf-8'))
rows=[]
completions=[]
for s in (1,2):
    index=json.loads((D/f'archive_index_s{s}.json').read_text())
    archive=D/f'results_s{s}.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==index['sha256']
    with tarfile.open(archive) as tar:
        completion=json.load(tar.extractfile('completion.json'))
        audit=json.load(tar.extractfile('audit.json'))
        source=tar.extractfile('effective_source.py').read()
        assert hashlib.sha256(source).hexdigest()==completion['effective_source_sha256']
        for row in completion['samples']:
            blob=tar.extractfile(f"artifacts/{row['idx']}.pt").read()
            assert hashlib.sha256(blob).hexdigest()==row['artifact_sha256']
    assert audit['all_inputs_and_artifacts_verified']
    assert completion['backbone_versions_unchanged']
    completions.append(completion)
    rows.extend(completion['samples'])
assert len(rows)==40 and len({r['idx'] for r in rows})==40
assert {r['idx'] for r in rows}=={r['idx'] for r in manifest['selected']}
assert completions[0]['effective_source_sha256']==completions[1]['effective_source_sha256']
byid={r['idx']:r for r in manifest['selected']}
micro=set(manifest['server1'][i]['idx'] for i in range(4))|set(manifest['server2'][i]['idx'] for i in range(4))
names=list(rows[0]['conditions'])
def correct(r,k):return r['conditions'][k]['pred']==r['gold']
def paired(subset,k,ref='native'):
    rescue=[r['idx'] for r in subset if correct(r,k) and not correct(r,ref)]
    harm=[r['idx'] for r in subset if correct(r,ref) and not correct(r,k)]
    n=len(rescue)+len(harm)
    p=min(1.,2*sum(math.comb(n,j) for j in range(min(len(rescue),len(harm))+1))/(2**n)) if n else 1.
    return dict(correct=sum(correct(r,k) for r in subset),wrong=sum(not correct(r,k) for r in subset),rescue=sorted(rescue),harm=sorted(harm),mcnemar_exact_two_sided=p)
summary={k:dict(all40=paired(rows,k),outside_micro32=paired([r for r in rows if r['idx'] not in micro],k)) for k in names}
# Identify source grouping from the authoritative manifest, never assume40 videos.
origin_field=next((k for k in ('video_id','origin','youtube_id','video','path','video_path') if k in manifest['selected'][0]),None)
assert origin_field is not None,manifest['selected'][0].keys()
groups={}
for r in rows:
    origin=str(byid[r['idx']][origin_field])
    if origin_field=='video_path':origin=origin.replace('\\','/').split('/')[-2]
    groups.setdefault(origin,[]).append(r)
rng=random.Random(20261017)
keys=list(groups)
boot={k:[] for k in names}
for _ in range(5000):
    sample=[r for _ in keys for r in groups[keys[rng.randrange(len(keys))]]]
    for k in names:boot[k].append(sum(int(correct(r,k))-int(correct(r,'native')) for r in sample)/len(sample))
for k in names:
    b=sorted(boot[k]);summary[k]['origin_cluster_bootstrap_delta_95pct']=[b[124],b[4874]]
bins=sorted(set(f"{r['task']}/{r['gold']}" for r in rows))
buckets={b:{k:paired([r for r in rows if f"{r['task']}/{r['gold']}"==b],k) for k in names} for b in bins}
layer_cos={str(ll):statistics.mean(r['contrast_audit'][str(ll)]['local_vs_shared_cosine'] for r in rows) for ll in (14,16,18,20,22)}
out=dict(n=40,n_origins=len(groups),origin_field=origin_field,scores=summary,buckets=buckets,mean_local_vs_shared_cosine=layer_cos,rows=sorted(rows,key=lambda r:r['idx']))
(D/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
lines=['# Vòng layer-local CLC, 2026-10-17','',
       f"**Nhánh chính layerlocal_native đạt {summary['layerlocal_native']['all40']['correct']}/40 so native {summary['native']['all40']['correct']}/40; cứu {len(summary['layerlocal_native']['all40']['rescue'])}, hại {len(summary['layerlocal_native']['all40']['harm'])}.** Đổi sang contrast theo từng tầng chưa cải thiện QA ở protocol này. shared22_native đạt {summary['shared22_native']['all40']['correct']}/40, cứu {len(summary['shared22_native']['all40']['rescue'])}/hại {len(summary['shared22_native']['all40']['harm'])}; bỏ bivector giảm tác hại so công thức cũ nhưng chưa hơn baseline.",'',
       'Đã chạy đủ40 câu cùng frozen manifest, rotation0, chia hai shard khác mẫu. Cùng hệ số .042×gamma và năm tầng14/16/18/20/22; không train, không tìm hệ số theo gold. Mọi nhánh cuối bỏ CAOR/CT, nhưng vẫn dùng CAOR để xây contrast. Đây là dữ liệu development đã dùng nhiều vòng.','',
       '| Cấu hình | Đúng/40 | Ngoài micro8 đúng/32 | Cứu/hại so native | McNemar p chưa chỉnh đa kiểm định |','|---|---:|---:|---:|---:|']
for k in names:
    a=summary[k]['all40'];o=summary[k]['outside_micro32']
    lines.append(f"| {k} | {a['correct']} | {o['correct']} | {len(a['rescue'])}/{len(a['harm'])} | {a['mcnemar_exact_two_sided']:.4f} |")
lines+=['','shared22_native và layerlocal_native dùng cùng .45 evidence + .35 order đã normalize; chỉ khác tầng lấy hướng. shared22_mixed_replay tái lập công thức cũ có .20 bivector. layerlocal_ev là ablation bỏ order; negated và random là đối chứng. Không chọn nhánh thắng sau khi xem số liệu làm claim xác nhận.','',
        '| Task/gold | N | '+' | '.join(names)+' |','|---|---:|'+'---:|'*len(names)]
for b in bins:
    sub=[r for r in rows if f"{r['task']}/{r['gold']}"==b]
    lines.append('| '+b+' | '+str(len(sub))+' | '+' | '.join(str(buckets[b][k]['correct']) for k in names)+' |')
lines+=['','Cosine trung bình giữa hướng local và hướng22: '+json.dumps(layer_cos)+'. Khác hướng không tự chứng minh khác ý nghĩa temporal.','',
        f"Kiểm toán: {len(groups)} origins; SHA archive/source/từng artifact khớp; source effective hai shard trùng; backbone versions không đổi; native/CAOR/CLC cũ replay sai số xác suất tối đa {max(max(r['prior40_replay_errors'].values()) for r in rows)}. No-op tối đa {max(r['kernel_noop_err'] for r in rows)}. Có {sum(not r['cache_used'] for r in rows)} mẫu giữ full-pass fallback theo ngưỡng .02 cũ.",'',
        'Các khoảng bootstrap trong summary.json lấy mẫu theo recording origin (5000 draws, seed20261017), chỉ diễn tả bất định nội bộ tập40; không khắc phục việc tái sử dụng development set hay đa kiểm định.','',
        '| ID | Gold | '+' | '.join(names)+' |','|---|---|'+'---|'*len(names)]
for r in sorted(rows,key=lambda r:r['idx']):lines.append('| '+str(r['idx'])+' | '+r['gold']+' | '+' | '.join(r['conditions'][k]['pred'] for k in names)+' |')
lines+=['','Nguồn phương pháp: [Tuned Lens](https://arxiv.org/html/2303.08112v4) nêu representation drift và bias của raw logit lens trên các LM được khảo sát; không chứng minh đây là nguyên nhân lỗi Qwen. [CAA](https://aclanthology.org/2024.acl-long.828/) đã dùng residual contrasts để steering, nên phép đổi này chưa có claim novelty.','',
        'Phạm vi: chưa chạy option rotations mới, chưa chấm free-form độc lập, chưa có event/interval gold cho40, chưa chứng minh grounding hoặc triển khai tổng quát.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(n=40,n_origins=len(groups),scores={k:summary[k]['all40']['correct'] for k in names},cosines=layer_cos),indent=2))
