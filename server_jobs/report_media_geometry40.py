"""Archive native pooled features and compare within/across-recording cosine."""
import argparse,io,json,tarfile,time,concurrent.futures
from pathlib import PurePosixPath
import numpy as np
from argon_remote import ROOT,call

FOLDER=ROOT/'results/media_geometry40'

def download(server):
    blob=call(server,f'/file?path=media_geometry40_raw_s{server}.tar.gz',raw=True,timeout=180)
    (FOLDER/f'raw_s{server}.tar.gz').write_bytes(blob)
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    ids={s['idx'] for s in manifest[f'server{server}']};count=0
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
        for member in tar.getmembers():
            parts=PurePosixPath(member.name).parts
            assert len(parts)==2 and parts[0]=='media_geometry40' and member.isfile()
            filename=parts[1]
            if filename.startswith('index_s'):
                assert filename==f'index_s{server}.json'
            else:
                stem,extension=filename.rsplit('.',1)
                assert int(stem) in ids and extension in ('json','npz')
                if extension=='npz': count+=1
            (FOLDER/filename).write_bytes(tar.extractfile(member).read())
    assert count==20
    print('MEDIA_GEOMETRY_ARCHIVE',server,len(blob),count,flush=True)

def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def describe(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
        p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95)),n=len(values))

def report():
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    ids=[s['idx'] for s in manifest['selected']]
    origins=[s['video_path'].replace('\\','/').split('/')[-2] for s in manifest['selected']]
    unique_origins=sorted(set(origins));origin_codes=np.array([unique_origins.index(o) for o in origins])
    meta=[json.loads((FOLDER/f'{idx}.json').read_text()) for idx in ids]
    assert len({x['runtime_sha256'] for x in meta})==1
    result={};table=['|Block|Source|Within-recording cosine|Between-recording cosine|LOO-centered within|LOO-centered between|',
        '|---|---|---|---|---|---|']
    for li in (0,8,16,22,30,35):
        for source in ('video','audio'):
            vectors=[]
            for idx in ids:
                with np.load(FOLDER/f'{idx}.npz',allow_pickle=False) as data:
                    values=data[f'{source}_l{li}'].astype(np.float64)
                assert np.isfinite(values).all() and values.ndim==2
                vectors.append(values)
            means=np.stack([v.mean(0) for v in vectors])
            origin_means=np.stack([means[origin_codes==i].mean(0) for i in range(len(unique_origins))])
            total=origin_means.sum(0)
            centered=[v-(total-origin_means[origin_codes[i]])/(len(unique_origins)-1) for i,v in enumerate(vectors)]
            owners=np.concatenate([np.full(len(v),i) for i,v in enumerate(vectors)])
            labels=owners[:,None]==owners[None,:];upper=np.triu(np.ones(labels.shape,dtype=bool),k=1)
            different_origins=origin_codes[owners,None]!=origin_codes[owners][None,:]
            raw=unit(np.concatenate(vectors));shifted=unit(np.concatenate(centered))
            cosine=raw@raw.T;shifted_cosine=shifted@shifted.T
            record=dict(within=describe(cosine[labels&upper]),between=describe(cosine[different_origins&upper]),
                centered_within=describe(shifted_cosine[labels&upper]),centered_between=describe(shifted_cosine[different_origins&upper]),
                recordings=40,source_origins=len(unique_origins),chunk_vectors=len(owners))
            result[f'{source}_{li}']=record
            table.append('|'+ '|'.join([str(li),source]+[f"{record[k]['mean']:.4f}" for k in ('within','between','centered_within','centered_between')])+'|')
    lines=['Native media geometry40 — 2026-10-15','',
        'Hoàn tất40 recordings, một native forward/recording, rotation0. Không đổi inputs/model/answers. Các vector cùng recording pooled theo2s, riêng nguồn. Đây là diagnostic, không phải một method QA mới.',
        '',*table,'',
        'Raw cosine giữa clips cao sẽ cho thấy shared representation/anisotropy là giải thích khả dĩ; cosine nội clip cao không tự đo được phần trăm static background. Bộ40 có38 nguồn gốc; between cosine loại cặp cùng nguồn, centering leave-one-origin-out với trọng số bằng nhau mỗi nguồn. Chỉ phân tích, không tạo đáp án hoặc fit theo gold.',
        '', 'Không có event/time annotations độc lập, nên không dùng cosine để tuyên bố grounded đúng sự kiện, xác nhận audio/video liên kết đúng thời gian, hoặc chứng minh một thành phần background đã bị loại sạch. Native decoder media states có thể chứa cross-source context.',
        '', f"Native replay max={max(x['replay_error'] for x in meta):.8f}. Features40 file npz, metadata40 file json; source/runtime SHA được lưu; protocol trước chạy ở protocol.md."]
    (FOLDER/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (FOLDER/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(table),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        jobs=json.loads((FOLDER/'jobs.json').read_text());downloaded=set();previous={}
        while len(downloaded)<2:
            for job in jobs:
                server=job['server']
                if server in downloaded: continue
                state=call(server,'/job/'+job['job_id'])
                (FOLDER/f'job_s{server}.json').write_text(json.dumps(state,indent=2))
                if state['status']!=previous.get(server):
                    print('GEOMETRY_JOB',server,state['status'],flush=True);previous[server]=state['status']
                if state['status']=='error': raise RuntimeError(state.get('error'))
                if state['status']=='done': download(server);downloaded.add(server)
            if len(downloaded)<2: time.sleep(25)
    elif args.download:
        for server in (1,2): download(server)
    report()
