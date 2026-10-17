import hashlib,json,math,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/pcebr_stage0_v8_readout'
manifest=json.loads((R/'results/research40_manifest.json').read_text(encoding='utf-8'))
parent=json.loads((R/'results/pcebr_stage0_v8_layerlocal/summary.json').read_text(encoding='utf-8'))
old={r['idx']:r for r in parent['rows']}
rows=[];sources=[]
for s in (1,2):
    ix=json.loads((D/f'archive_index_s{s}.json').read_text())
    archive=D/f'results_s{s}.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==ix['sha256']
    with tarfile.open(archive) as tar:
        c=json.load(tar.extractfile('completion.json'));a=json.load(tar.extractfile('audit.json'))
        route=tar.extractfile('route_source.py').read()
        assert hashlib.sha256(route).hexdigest()==c['route_source_sha256']
    assert a['all_inputs_sources_replays_verified'] and c['backbone_versions_unchanged']
    sources.append(c['route_source_sha256']);rows+=c['samples']
assert sources[0]==sources[1]
assert len(rows)==40 and {r['idx'] for r in rows}=={r['idx'] for r in manifest['selected']}
for r in rows:
    assert r['source_artifact_sha256']==old[r['idx']]['artifact_sha256']
    assert r['input_cache_sha256']==old[r['idx']]['input_cache_sha256']
micro={r['idx'] for r in manifest['server1'][:4]+manifest['server2'][:4]}
names=list(rows[0]['conditions'])
def corr(r,k):return r['conditions'][k]['pred']==r['gold']
def stats(sub,k):
    rescue=[r['idx'] for r in sub if corr(r,k) and not corr(r,'native')]
    harm=[r['idx'] for r in sub if corr(r,'native') and not corr(r,k)]
    n=len(rescue)+len(harm)
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(len(rescue),len(harm))+1))/2**n) if n else 1.
    return dict(correct=sum(corr(r,k) for r in sub),n=len(sub),rescue=sorted(rescue),harm=sorted(harm),mcnemar_exact_two_sided=p)
scores={k:dict(all40=stats(rows,k),outside_micro32=stats([r for r in rows if r['idx'] not in micro],k)) for k in names}
buckets={b:{k:stats([r for r in rows if f"{r['task']}/{r['gold']}"==b],k) for k in names} for b in sorted({f"{r['task']}/{r['gold']}" for r in rows})}
out=dict(n=40,scores=scores,buckets=buckets,rows=sorted(rows,key=lambda r:r['idx']),route_source_sha256=sources[0])
(D/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
lines=['# Vòng giới hạn CLC vào token trả lời cuối, 2026-10-17','',
    f"**Không nhánh last_ nào hơn native {scores['native']['all40']['correct']}/40.** Nhánh last_shared22_native đạt {scores['last_shared22_native']['all40']['correct']}/40 và last_layerlocal_native đạt {scores['last_layerlocal_native']['all40']['correct']}/40; cả hai không cứu câu nào. Giảm phạm vi can thiệp chưa là giải pháp cho QA temporal.",'',
    'Đủ40 cùng inputs/directions của vòng layer-local; giữ hệ số .042×gamma, năm tầng, không CAOR/CT ở decode cuối. Chỉ đổi phạm vi từ toàn bộ question/options/assistant sang token assistant cuối. Không train, không chọn hệ số theo gold. Cùng38 origins, development đã tái sử dụng.','',
    '| Nhánh | Đúng/40 | Ngoài micro8 đúng/32 | Cứu/hại native | McNemar p chưa chỉnh |','|---|---:|---:|---:|---:|']
for k in names:
    a=scores[k]['all40'];o=scores[k]['outside_micro32'];lines.append(f"| {k} | {a['correct']} | {o['correct']} | {len(a['rescue'])}/{len(a['harm'])} | {a['mcnemar_exact_two_sided']:.4f} |")
lines+=['','Các nhánh last_ dùng cùng directions đã lưu trước vòng này. Negated và random có cùng rows/layers/strength. Phạm vi token ít hơn cũng giảm tổng can thiệp, vì vậy không được quy mọi khác biệt cho việc sửa representation của option.','',
    '| Task/gold | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
for b,ss in buckets.items():lines.append('| '+b+' | '+' | '.join(str(ss[k]['correct']) for k in names)+' |')
lines+=['',f"Kiểm toán: hai route sources trùng SHA; archives/input hashes/parent direction hashes khớp; native và CLC toàn-tail replay sai số xác suất tối đa {max(max(r['prior_replay_errors'].values()) for r in rows)}; backbone versions không đổi. 17/40 mẫu giữ cache full-pass fallback của vòng trước.",'',
    'Wrapper đầu tiên dùng gamma trước làm tròn thay vì gamma sáu chữ số trong profile của lượt gốc. Parity đã phát hiện sai khác; không nhận prediction từ lượt lỗi. Đã sửa wrapper dùng đúng profile gamma và chạy lại, giữ ngưỡng replay1e-6; log initial_replay_error.json nằm trong raw archives. Đây là lỗi wrapper đã sửa, không phải thay đổi phương pháp để nâng điểm.','',
    'Chưa có independent test, option rotations mới, interval ground truth, hoặc free-form semantic scoring. Không chọn nhánh tốt nhất trong bảng làm bằng chứng xác nhận.','',
    '| ID | Gold | '+' | '.join(names)+' |','|---|---|'+'---|'*len(names)]
for r in sorted(rows,key=lambda r:r['idx']):lines.append('| '+str(r['idx'])+' | '+r['gold']+' | '+' | '.join(r['conditions'][k]['pred'] for k in names)+' |')
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(n=40,scores={k:scores[k]['all40']['correct'] for k in names}),indent=2))
