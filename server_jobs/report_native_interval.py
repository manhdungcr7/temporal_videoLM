import hashlib,json,math,statistics,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'results/native_interval_verification'
M=json.loads((D/'manifest.json').read_text(encoding='utf-8'));pairs={p['pair_id']:p for p in M['pairs']}
records=[];provenance=[]
for s in (1,2):
    ix=json.loads((D/f'archive_index_s{s}.json').read_text())
    f=D/f'results_s{s}.tar.gz';assert hashlib.sha256(f.read_bytes()).hexdigest()==ix['sha256']
    with tarfile.open(f) as tar:
        c=json.load(tar.extractfile('completion.json'));a=json.load(tar.extractfile('audit.json'))
        assert c['backbone_versions_unchanged'] and c['remaining_hooks']==0 and a['all_crop_artifact_input_contract_checks_passed']
        assert hashlib.sha256(tar.extractfile('manifest.json').read()).hexdigest()==hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest()
        for r in c['records']:
            assert hashlib.sha256(tar.extractfile(f"crops/{r['pair_id']}_{r['window']}.mp4").read()).hexdigest()==r['crop_sha256']
            assert hashlib.sha256(tar.extractfile(f"artifacts/{r['pair_id']}_{r['window']}_{r['mode']}.pt").read()).hexdigest()==r['artifact_sha256']
    records+=c['records'];provenance.append({k:v for k,v in c.items() if k!='records'})
assert len(records)==len(pairs)*6
lookup={(r['pair_id'],r['window'],r['mode']):r for r in records}
def score(r,method):
    if method=='native_odds':return r['presence']['yes_no_log_odds']
    if method=='description_ll':return r['description_score']['description_mean_logprob']
    if r['lens'] is None:return None
    return r['lens']['pooled_l22_manifold_cosine' if method=='pooled_lens_cosine' else 'pooled_l22_label_gap_mean']
methods=('native_odds','description_ll','pooled_lens_cosine','pooled_lens_gap')
by_pair=[]
for pid,p in pairs.items():
    entry=dict(pair_id=pid,video_id=p['video_id'],modality=p['modality'],split=p['split'],label=p['label'],positive_overlap_sec=p['positive_overlap_sec'],conditions={})
    for mode in ('av','audio','vision'):
        pos=lookup[pid,'positive',mode];neg=lookup[pid,'negative',mode]
        out={}
        for method in methods:
            a,b=score(pos,method),score(neg,method)
            if a is not None and b is not None:out[method]=dict(positive=a,negative=b,delta=a-b,rank_correct=a>b,rank_tie=a==b)
        out['binary_presence']=dict(positive_yes=pos['presence']['yes_no_log_odds']>0,negative_no=neg['presence']['yes_no_log_odds']<0,
            strict_pair=pos['presence']['yes_no_log_odds']>0 and neg['presence']['yes_no_log_odds']<0)
        entry['conditions'][mode]=out
    by_pair.append(entry)
def summarize(sub,modekind):
    out={}
    for method in methods:
        vals=[];binary=[]
        for p in sub:
            mode='av' if modekind=='av' else p['modality'] if modekind=='source_only' else 'vision' if p['modality']=='audio' else 'audio'
            c=p['conditions'][mode]
            if method in c:vals.append(c[method])
            binary.append(c['binary_presence'])
        n=len(vals)
        if not n:continue
        success=sum(v['rank_correct'] for v in vals);ties=sum(v['rank_tie'] for v in vals);nt=n-ties
        # One-sided sign test of above-chance ranking; exploratory and unadjusted.
        pval=sum(math.comb(nt,j) for j in range(success,nt+1))/2**nt if nt else 1.
        out[method]=dict(n=n,rank_correct=success,ties=ties,mean_delta=statistics.mean(v['delta'] for v in vals),sign_test_p_gt_chance=pval)
    out['binary_presence']=dict(n=len(binary),positive_yes=sum(v['positive_yes'] for v in binary),negative_no=sum(v['negative_no'] for v in binary),strict_pair=sum(v['strict_pair'] for v in binary))
    return out
summary={group:{mode:summarize(sub,mode) for mode in ('av','source_only','opposite_only')} for group,sub in
    [('all',by_pair),('audio',[p for p in by_pair if p['modality']=='audio']),('vision',[p for p in by_pair if p['modality']=='vision']),
     ('validation',[p for p in by_pair if p['split']=='validation']),('diagnostic',[p for p in by_pair if p['split']=='diagnostic'])]}
out=dict(n_pairs=len(pairs),n_origins=len({p['video_id'] for p in pairs.values()}),n_cases=len(records),summary=summary,by_pair=by_pair,records=records,
    provenance=provenance,max_prefix_query_change_mean_error=max(max(r['prefix_mean_query_change_max_errors'].values()) for r in records))
(D/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
lines=['# Native interval verification, 2026-10-17','',
    f"Đã chạy {len(pairs)} cặp, {out['n_origins']} recording origins, {len(records)} cases; mỗi cặp có đoạn positive2s và annotation-negative2s, ba chế độ AV/audio/vision. Nhãn public độc lập với SVHalluc QA, nhưng16 origins là development đã dùng trước; đây không phải test mới.",'',
    '| Nhóm | Evidence | N | Native odds xếp đúng | Description LL xếp đúng | Lens cosine xếp đúng | Positive Yes | Negative No | Đúng cả cặp |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for group,ss in summary.items():
    for mode,v in ss.items():
        b=v['binary_presence'];lc=v.get('pooled_lens_cosine')
        lines.append(f"| {group} | {mode} | {b['n']} | {v['native_odds']['rank_correct']} | {v['description_ll']['rank_correct']} | {str(lc['rank_correct']) if lc else 'N/A'} | {b['positive_yes']} | {b['negative_no']} | {b['strict_pair']} |")
lines+=['','Ranking đếm positive score > negative score, ties báo riêng trong summary.json; mốc random-ranking50%. ConstantYes/No đạt50% item accuracy trên mỗi cặp cân bằng nhưng0 strict-pair. Binary threshold là native logodds0, không fit. Các sign-test p trong JSON là exploratory chưa chỉnh đa kiểm định và xem các cặp như độc lập; nhiều cặp có chung origin, nên không dùng p đó làm claim xác nhận.','',
    'Opposite-only là đối chứng bỏ nguồn evidence được hỏi; không coi nó như accuracy thông thường có đủ sensory inputs. Lens không tồn tại khi nguồn bị bỏ. Description target cùng chuỗi cho hai đoạn nên prior văn bản không phụ thuộc media triệt tiêu trong paired delta; vẫn có ảnh hưởng ngữ cảnh/cách đặt prompt.','',
    'Negative chỉ có union overlap0 với annotation mục tiêu và guard.15s; chưa được người chấm lại. Positive có ít nhất.25s annotation support trong2s. Một số action là động tác/state có thể còn mơ hồ ngoài khoảng nhãn; một số sound labels mô tả rất rộng. Không diễn giải bảng này là deployment FPR hoặc benchmark arbitrary-question QA.','',
    'Pooled lens là comparator đơn giản trên mean media post-L22 qua norm/lm_head/top48/input embeddings, không phải exact v8 extraction. Các ontology label có namespace được bỏ bằng quy tắc format chung; không thêm từ khóa task SVHalluc.','',
    f"Audit: crop/artifact/manifest SHA khớp; modalities có token counts đúng; backbone versions không đổi; hooks còn0. Prefix IDs giữa presence và description prompts trùng từng case. Media-mean hidden sai khác tối đa khi thay tail query: {out['max_prefix_query_change_mean_error']}; đây là sai số tính toán quan sát được, không được gọi exact hidden parity nếu khác0.",'',
    '| Pair | Label | AV odds Δ | Source odds Δ | Opposite odds Δ | Source description Δ |',
    '|---|---|---:|---:|---:|---:|']
for p in by_pair:
    src=p['modality'];opp='vision' if src=='audio' else 'audio';c=p['conditions']
    lines.append(f"| {p['pair_id']} | {p['label']} | {c['av']['native_odds']['delta']:.4f} | {c[src]['native_odds']['delta']:.4f} | {c[opp]['native_odds']['delta']:.4f} | {c[src]['description_ll']['delta']:.4f} |")
lines+=['','Nguồn nhãn: [Perception Test official repository](https://github.com/google-deepmind/perception_test). VideoMind có verifier được huấn luyệnSFT ([method](https://arxiv.org/html/2503.13444v1)); không thể giả định hỏi mô hình đóng băng là phép thay thế tương đương. Nghiên cứu [temporal consistency](https://arxiv.org/html/2411.12951v1) cũng phân biệt năng lực grounding và verification.']
(D/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(n_pairs=len(pairs),n_origins=out['n_origins'],summary=summary['all'],max_hidden_mean_error=out['max_prefix_query_change_mean_error']),indent=2))
