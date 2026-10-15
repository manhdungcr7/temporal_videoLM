"""Download raw rotations and verify their aggregation without trusting tar paths."""
import argparse,io,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
from argon_remote import ROOT,call

def archive(suite,server):
    folder=ROOT/'results'/suite;folder.mkdir(exist_ok=True)
    blob=call(server,f'/file?path={suite}_raw_s{server}.tar.gz',raw=True,timeout=180)
    (folder/f'raw_s{server}.tar.gz').write_bytes(blob)
    manifest=json.loads((ROOT/'results/research40_manifest.json').read_text())
    selected={sample['idx']:sample for sample in manifest[f'server{server}']}
    rotations={};records=None
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
        for member in tar.getmembers():
            parts=PurePosixPath(member.name).parts
            assert len(parts)==2 and parts[0]==suite and member.isfile(),member.name
            data=tar.extractfile(member).read()
            if parts[1]==f'results_s{server}.jsonl':
                assert records is None
                records=[json.loads(line) for line in data.splitlines() if line.strip()]
                (folder/parts[1]).write_bytes(data)
            else:
                name=parts[1];idx,rotation=name.removesuffix('.json').split('_rot')
                idx,rotation=int(idx),int(rotation)
                assert idx in selected and rotation in range(2 if selected[idx]['task']=='TA' else 3)
                assert (idx,rotation) not in rotations
                result=json.loads(data);assert result['idx']==idx and result['rotation']==rotation
                assert result['manifest_sha256']==manifest['manifest_sha256']
                rotations[idx,rotation]=result
                target=folder/f'rotations_s{server}';target.mkdir(exist_ok=True)
                (target/name).write_bytes(data)
    assert records is not None and len(records)==20 and {r['idx'] for r in records}==set(selected)
    assert len(rotations)==55
    max_error=0.;tie_ids={};runtime_shas=set()
    for row in records:
        idx=row['idx'];k=2 if row['task']=='TA' else 3
        for method,stored in row['scores'].items():
            summed=np.zeros(k)
            for rotation in range(k):
                result=rotations[idx,rotation];runtime_shas.add(result['runtime_sha256'])
                for slot,option in enumerate(result['perm']): summed[option]+=result['scores'][method][slot]
            error=float(np.max(np.abs(summed-np.array(stored))));max_error=max(max_error,error)
            assert error<1e-6,(idx,method,error)
            assert chr(65+int(summed.argmax()))==row['predictions'][method]
            if float(np.sort(summed)[-1]-np.sort(summed)[-2])<1e-6:
                tie_ids.setdefault(method,[]).append(idx)
    assert len(runtime_shas)==1
    audit=dict(server=server,samples=20,rotations=55,raw_aggregate_max_error=max_error,
        runtime_sha256=next(iter(runtime_shas)),tie_ids=tie_ids,archive_bytes=len(blob))
    (folder/f'archive_audit_s{server}.json').write_text(json.dumps(audit,indent=2))
    return audit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('suite');parser.add_argument('server',type=int)
    args=parser.parse_args();print(json.dumps(archive(args.suite,args.server)))
