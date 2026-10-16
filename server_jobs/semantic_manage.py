"""Evaluate the friend's Round-5 ChronoDeSmear methods (R5_C0..C6) on the SVHalluc SEMANTIC tasks, unchanged.
The friend's cache + runner are reused by text substitution (no copy of the method code): manifest, cache and output paths are redirected.

Order (two servers, remote_config.json = server 1, remote_config2.json = server 2; each server needs fix_ffmpeg first):
    python server_jobs/semantic_manage.py upload                 # patched copies of the friend's files -> OUT/uploads on both servers
    python server_jobs/semantic_manage.py manifest               # build the balanced semantic manifest on server 1, copy it to both
    python server_jobs/semantic_manage.py cache --server 1       # encode every rotation once (reusable)   (same for --server 2)
    python server_jobs/semantic_manage.py launch --server 1      # run R5_C0..C6                           (same for --server 2)
    python server_jobs/semantic_manage.py checkpoint             # download results, print the summary
Protocol (fixed before running): results/semantic_chrono/protocol.md."""
import argparse, concurrent.futures, csv, hashlib, json, math, re
from pathlib import Path
from argon_remote import ROOT, call, upload, submit

HERE = Path(__file__).resolve().parent
FOLDER = ROOT / 'results/semantic_chrono'
CELLS = [f'{t}/{g}' for t in ('GSA', 'FGSA', 'CSMB') for g in 'AB']
TASKS = ('GSA', 'FGSA', 'CSMB')


def patched():
    cache = (HERE / 'chronodesmear_cache40.py').read_text(encoding='utf-8')
    cache = cache.replace('research40_manifest.json', 'semantic_manifest.json')
    cache = re.sub(r"\n\s*assert CD_MANIFEST\['total'\]==40\n", '\n', cache)
    cache = cache.replace("argon_latent40/decoder_cache", "semantic_chrono/decoder_cache")
    run = (HERE / 'chronodesmear40.py').read_text(encoding='utf-8')
    run = run.replace("uploads/chronodesmear_cache40.py", "uploads/semantic_cache.py")
    run = run.replace("uploads/chronodesmear40.py", "uploads/semantic_chrono.py")
    run = run.replace("OUT/'chronodesmear40'", "OUT/'semantic_chrono'")
    run = run.replace("chronodesmear40/results_s", "semantic_chrono/results_s")
    assert 'semantic_manifest.json' in cache and 'semantic_chrono/decoder_cache' in cache and "assert CD_MANIFEST['total']==40" not in cache
    assert "OUT/'semantic_chrono'" in run and 'uploads/semantic_cache.py' in run
    return {'semantic_cache.py': cache, 'semantic_chrono.py': run}


def do_upload():
    files = patched()
    tmp = FOLDER / '_patched'; tmp.mkdir(parents=True, exist_ok=True)
    plain = {'chronodesmear_math.py': HERE / 'chronodesmear_math.py', 'avbind_sdpa_compat.py': HERE / 'avbind_sdpa_compat.py',
             'chronodesmear_proposal_original.py': ROOT / 'results/chronodesmear40/proposal_original.py',
             'semantic_manifest_job.py': HERE / 'semantic_manifest_job.py'}
    for name, text in files.items(): (tmp / name).write_text(text, encoding='utf-8'); plain[name] = tmp / name
    out = {}
    for server in (1, 2):
        for name, path in plain.items(): out[f's{server}/{name}'] = upload(server, path, name).get('bytes', 'ok')
    return out


def manifest_launch():
    code = "SEM_PER_CELL=10\nexec(compile((OUT/'uploads/semantic_manifest_job.py').read_text(),'semantic_manifest_job.py','exec'),globals())"
    return submit(1, code, 'semantic_manifest')


def push_manifest():
    blob = call(1, '/file?path=semantic_manifest.json', raw=True)
    FOLDER.mkdir(parents=True, exist_ok=True); (FOLDER / 'semantic_manifest.json').write_bytes(blob)
    return upload(2, FOLDER / 'semantic_manifest.json', 'semantic_manifest.json') | {'sha': json.loads(blob)['manifest_sha256']}


def run_code(server, kind, limit=None):
    head = f'CD_SERVER={server}\nCD_LIMIT={limit!r}\n'
    if kind == 'cache':
        head += "CD_CACHE_RUN=True\nexec(compile((OUT/'uploads/semantic_cache.py').read_text(),'semantic_cache.py','exec'),globals())"
        name = 'semantic_cache'
    else:
        head += "exec(compile((OUT/'uploads/semantic_chrono.py').read_text(),'semantic_chrono.py','exec'),globals())"
        name = 'semantic_chrono'
    return dict(server=server, **submit(server, head, name))


def checkpoint(job):
    server = job['server']; state = call(server, '/job/' + job['job_id'])
    (FOLDER / f'job_s{server}_{job["kind"]}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    error = None
    try:
        blob = call(server, f'/file?path=semantic_chrono/results_s{server}.jsonl', raw=True)
        local = FOLDER / f'results_s{server}.jsonl'
        old = [json.loads(l) for l in local.read_text().splitlines()] if local.exists() else []
        parsed = [json.loads(l) for l in blob.splitlines() if l.strip()]
        assert len(parsed) >= len(old), 'refuse shrinking checkpoint'; local.write_bytes(blob)
    except Exception as exc: error = repr(exc)
    return dict(server=server, kind=job['kind'], status=state['status'], error=state.get('error'), download_error=error,
                tail=state.get('out', '')[-250:])


def wilson(k, n, z=1.96):
    if n == 0: return (0., 1.)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - w) / d, (c + w) / d)


def summary(baseline='R5_C0_baseline', paths=None):
    rows = []
    for path in ([Path(x) for x in paths] if paths else [FOLDER / f'results_s{server}.jsonl' for server in (1, 2)]):
        if path.exists(): rows.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    assert len({r['idx'] for r in rows}) == len(rows)
    methods = list(rows[0]['scores']) if rows else []
    pred = {m: {r['idx']: r['predictions'][m] for r in rows} for m in methods}
    if rows and 'shuffled_seed_scores' in rows[0]:
        for seed in ('42', '1337', '2027'):
            m = 'R5_C6_seed_' + seed; methods.append(m)
            pred[m] = {r['idx']: chr(65 + max(range(len(r['shuffled_seed_scores'][seed])), key=lambda i: r['shuffled_seed_scores'][seed][i])) for r in rows}
    report = {}
    for m in methods:
        ok = lambda r: pred[m][r['idx']] == r['gold']
        resc = sum(ok(r) and pred[baseline][r['idx']] != r['gold'] for r in rows)
        dam = sum((not ok(r)) and pred[baseline][r['idx']] == r['gold'] for r in rows); n = resc + dam
        p = min(1., 2 * sum(math.comb(n, i) for i in range(min(resc, dam) + 1)) / 2 ** n) if n else 1.
        cells = {}
        for c in CELLS:
            S = [r for r in rows if r['task'] + '/' + r['gold'] == c]
            cells[c] = dict(correct=sum(ok(r) for r in S), wrong=sum(not ok(r) for r in S), n=len(S))
        bal = {t: sum(cells[f'{t}/{g}']['correct'] / max(1, cells[f'{t}/{g}']['n']) for g in 'AB') / 2 for t in TASKS}
        hist = {t: [sum(r['task'] == t and pred[m][r['idx']] == L for r in rows) for L in 'AB'] for t in TASKS}
        report[m] = dict(correct=sum(ok(r) for r in rows), n=len(rows), cells=cells, balanced_by_task=bal,
                         balanced_mean=sum(bal.values()) / 3, pred_AB_by_task=hist, rescued=resc, damaged=dam, paired_p=p)
    tests = sorted((v['paired_p'], m) for m, v in report.items() if m != baseline); run = 0.
    for rank, (p, m) in enumerate(tests): run = max(run, min(1., (len(tests) - rank) * p)); report[m]['holm_p'] = run
    FOLDER.mkdir(parents=True, exist_ok=True); (FOLDER / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['|Method|' + '|'.join(CELLS) + '|Correct/n|Balanced acc per task GSA/FGSA/CSMB (mean)|Predictions A/B per task|Rescued/damaged (p, Holm)|', '|' + '---|' * (len(CELLS) + 5)]
    for m, e in report.items():
        lines.append('|' + '|'.join([m] + [f"{e['cells'][c]['correct']}/{e['cells'][c]['wrong']}" for c in CELLS] +
                     [f"{e['correct']}/{e['n']}", '/'.join(f"{100 * e['balanced_by_task'][t]:.0f}" for t in TASKS) + f" ({100 * e['balanced_mean']:.1f})",
                      ' ; '.join('-'.join(map(str, e['pred_AB_by_task'][t])) for t in TASKS),
                      f"{e['rescued']}/{e['damaged']} (p={e['paired_p']:.3f}, Holm={e.get('holm_p', 1):.3f})"]) + '|')
    (FOLDER / 'table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (FOLDER / 'per_sample.csv').open('w', newline='', encoding='utf-8') as h:
        w = csv.writer(h); w.writerow(['idx', 'task', 'gold', 'method', 'prediction', 'correct'])
        for r in rows:
            for m in methods: w.writerow([r['idx'], r['task'], r['gold'], m, pred[m][r['idx']], int(pred[m][r['idx']] == r['gold'])])
    return dict(completed=len(rows), methods={m: v['correct'] for m, v in report.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['upload', 'manifest', 'pushmanifest', 'cache', 'launch', 'checkpoint', 'summary'])
    ap.add_argument('--server', type=int); ap.add_argument('--limit', type=int); ap.add_argument('--files', nargs='*')
    a = ap.parse_args(); FOLDER.mkdir(parents=True, exist_ok=True); jobs_path = FOLDER / 'jobs.json'
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    if a.action == 'upload': print(json.dumps(do_upload()))
    elif a.action == 'manifest': print(json.dumps(manifest_launch()))
    elif a.action == 'pushmanifest': print(json.dumps(push_manifest()))
    elif a.action in ('cache', 'launch'):
        job = run_code(a.server, 'cache' if a.action == 'cache' else 'run', a.limit); job['kind'] = a.action
        jobs.append(job); jobs_path.write_text(json.dumps(jobs, indent=2)); print(json.dumps(job))
    elif a.action == 'checkpoint':
        last = {}
        for j in jobs: last[(j['server'], j['kind'])] = j
        with concurrent.futures.ThreadPoolExecutor(2) as pool: print(json.dumps(list(pool.map(checkpoint, last.values()))))
        print(json.dumps(summary()))
    else: print(json.dumps(summary(paths=a.files))); print((FOLDER / 'table.md').read_text())
