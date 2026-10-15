"""Archive and re-sum every completed rotation without rerunning inference."""
import argparse,concurrent.futures,json,time,urllib.error
import numpy as np
from argon_remote import ROOT,call

REMOTE={'latent_reopen40':'latent_reopen40','argon_ccld40':'argon_ccld40','symmetric_latent40':'symmetric_latent40_v2',
        'query_latent40':'query_latent40','latent_slot_opt40':'latent_slot_opt40','latent_slot_read15':'latent_slot_read15',
        'presentation_resample40':'presentation_resample40'}

def archive(item):
    phase,server,row,audit=item
    rotation=audit['rotation'];idx=row['idx']
    folder=ROOT/'results'/phase/f'rotations_s{server}';folder.mkdir(exist_ok=True)
    path=folder/f'{idx}_rot{rotation}.json'
    if not path.exists():
        for attempt in range(4):
            try:
                blob=call(server,f'/file?path={REMOTE[phase]}/{idx}_rot{rotation}.json',raw=True,timeout=45)
                break
            except (urllib.error.URLError,TimeoutError,ConnectionResetError):
                if attempt==3:raise
                time.sleep(.5*2**attempt)
        parsed=json.loads(blob)
        assert parsed['manifest_sha256']==row['manifest_sha256']
        path.write_bytes(blob)
    parsed=json.loads(path.read_text())
    assert parsed['idx']==idx and parsed['rotation']==rotation
    assert parsed['manifest_sha256']==row['manifest_sha256']
    return phase,server,idx,parsed

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phases',nargs='*',default=['argon_ccld40'])
    parser.add_argument('--server',type=int,choices=[1,2],help='Archive only a reachable worker; record partial coverage explicitly')
    args=parser.parse_args();items=[];rows={}
    for phase in args.phases:
        for s in ((args.server,) if args.server else (1,2)):
            path=ROOT/'results'/phase/f'results_s{s}.jsonl'
            if not path.exists():continue
            for row in [json.loads(x) for x in path.read_text().splitlines() if x.strip()]:
                rows[(phase,s,row['idx'])]=row
                items += [(phase,s,row,audit) for audit in row['audit']]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:all_rotations=list(pool.map(archive,items))
    grouped={}
    for phase,s,idx,rotation in all_rotations:grouped.setdefault((phase,s,idx),[]).append(rotation)
    max_error=0.
    for key,sample_rotations in grouped.items():
        row=rows[key];totals={m:np.zeros(len(score)) for m,score in row['scores'].items()}
        for rotation in sorted(sample_rotations,key=lambda r:r['rotation']):
            for m,score in rotation['scores'].items():
                for slot,opt in enumerate(rotation['perm']):totals[m][opt]+=score[slot]
        for m,score in totals.items():
            error=float(np.max(np.abs(score-np.array(row['scores'][m]))));max_error=max(max_error,error)
            assert error<1e-10,(key,m,error)
    summary=dict(samples=len(rows),rotations=len(all_rotations),max_reaggregate_error=max_error,
        phases={p:dict(samples=sum(key[0]==p for key in rows),rotations=sum(r[0]==p for r in all_rotations)) for p in args.phases})
    for phase in args.phases:
        (ROOT/'results'/phase/'rotation_archive_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))
