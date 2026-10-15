"""Checkpoint all queued suites and archive complete raw rotations as soon as ready."""
import concurrent.futures,json,time,urllib.error
from argon_remote import ROOT,call
from chronodesmear_manage import summary
from archive_chronodesmear40 import archive

SUITES={'chronodesmear40':'R5_C0_baseline','chronodesmear_followup40':'N0_native',
        'late_path40':'P0_native_reference'}
previous={};announced=set()

def checkpoint(entry):
    suite,job=entry;server=job['server'];folder=ROOT/'results'/suite
    state=call(server,'/job/'+job['job_id'])
    (folder/f'job_s{server}.json').write_text(json.dumps(state,indent=2))
    count=0;download_error=None
    if state['status'] in ('running','done','error'):
        try:
            blob=call(server,f'/file?path={suite}/results_s{server}.jsonl',raw=True)
            parsed=[json.loads(line) for line in blob.splitlines() if line.strip()]
            local=folder/f'results_s{server}.jsonl'
            old=[json.loads(line) for line in local.read_text().splitlines()] if local.exists() else []
            assert len(parsed)>=len(old)
            local.write_bytes(blob);count=len(parsed)
        except urllib.error.HTTPError as exc:
            if exc.code!=404: download_error=repr(exc)
    return dict(suite=suite,server=server,status=state['status'],count=count,
        seconds=state['seconds'],error=state.get('error'),download_error=download_error)

def main():
    entries=[(suite,job) for suite in SUITES for job in json.loads((ROOT/'results'/suite/'jobs.json').read_text())]
    transient=0
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(6) as pool: states=list(pool.map(checkpoint,entries))
            transient=0
        except Exception as exc:
            transient+=1;print('WATCH_RETRY',transient,repr(exc),flush=True)
            if transient>=6: raise
            time.sleep(15);continue
        errors=[s for s in states if s['status']=='error']
        for suite,baseline in SUITES.items():
            local=ROOT/'results'/suite
            result=summary(local,baseline)
            key=(result['completed'],tuple((s['server'],s['status']) for s in states if s['suite']==suite))
            if previous.get(suite)!=key:
                print('WATCH_PROGRESS',suite,json.dumps(result),flush=True);previous[suite]=key
            if result['completed']==40 and suite not in announced:
                print('SUITE_COMPLETE',suite,flush=True)
                print((local/'table.md').read_text(),flush=True);announced.add(suite)
            for server in (1,2):
                if (local/f'archive_audit_s{server}.json').exists(): continue
                # A suite's tar becomes available after its final sample is saved.
                rows=local/f'results_s{server}.jsonl'
                if not rows.exists() or len(rows.read_text().splitlines())!=20: continue
                try:
                    audit=archive(suite,server)
                    print('RAW_ARCHIVE_VERIFIED',suite,json.dumps(audit),flush=True)
                except urllib.error.HTTPError as exc:
                    if exc.code!=404: print('ARCHIVE_RETRY',suite,server,repr(exc),flush=True)
        if errors:
            print('WATCH_JOB_ERRORS',json.dumps(errors),flush=True)
            return 2
        if all(s['status']=='done' for s in states) and all((ROOT/'results'/suite/f'archive_audit_s{server}.json').exists()
                for suite in SUITES for server in (1,2)):
            print('ALL_THREE_SUITES_AND_RAW_ARCHIVES_COMPLETE',flush=True);return 0
        time.sleep(25)

if __name__=='__main__': raise SystemExit(main())
