"""Server job: build a balanced SVHalluc SEMANTIC manifest (GSA / FGSA / CSMB, all Yes/No) for the ChronoDeSmear evaluation.
Runs inside the kaggle_rootcause namespace (fetch, cut, decode_audio, SR, BS come from there). Writes OUT/uploads/semantic_manifest.json
with the same schema as the frozen temporal manifest used by chronodesmear40 (total, selected, server1, server2, manifest_sha256).
Cells: (task, gold) for task in GSA, FGSA, CSMB and gold in A (Yes), B (No)  -> 6 cells, SEM_PER_CELL samples each, clip <= SEM_MAXDUR s.
Selection uses only the seed and the clip duration, never the model or the labels' difficulty.  Parameters (globals): SEM_PER_CELL, SEM_SEED, SEM_MAXDUR."""
import hashlib, json, random

SEM_PER_CELL = int(globals().get('SEM_PER_CELL', 10))
SEM_SEED = int(globals().get('SEM_SEED', 31337))
SEM_MAXDUR = float(globals().get('SEM_MAXDUR', 40.0))
rows = json.loads(fetch('samples.json').read_text(encoding='utf-8'))
cells = {}
for i, r in enumerate(rows):
    if r['data_type'] == 'semantic': cells.setdefault((r['task_type_abbr'], r['answer']), []).append(i)

selected = []
for key in sorted(cells):
    ids = cells[key][:]; random.Random(f'{SEM_SEED}-{key}').shuffle(ids); got = []; tried = 0
    for i in ids:
        r = rows[i]; tried += 1
        try:
            path = r['video_path'].replace(BS, '/')
            dur = len(decode_audio(cut(fetch(path), 0, 10 ** 4, 'full'))) / SR
        except Exception as e:
            print('skip', i, repr(e)[:80], flush=True); continue
        if dur > SEM_MAXDUR or dur < 4.0: continue
        got.append(dict(idx=i, task=key[0], gold=key[1], video_path=path, question=r['question'],
                        duration=round(dur, 3), decoded_audio_duration=round(dur, 3)))
        if len(got) == SEM_PER_CELL: break
    print(key, 'selected', len(got), 'after', tried, 'tried', flush=True)
    selected.extend(got)

# alternate samples between the two servers inside every cell, so both servers see all cells
server1, server2 = [], []
for key in sorted(cells):
    cell = [s for s in selected if (s['task'], s['gold']) == key]
    server1.extend(cell[0::2]); server2.extend(cell[1::2])
manifest = dict(total=len(selected), per_cell=SEM_PER_CELL, seed=SEM_SEED, max_duration=SEM_MAXDUR,
                selected=selected, server1=server1, server2=server2)
manifest['manifest_sha256'] = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
(OUT / 'uploads').mkdir(exist_ok=True)
(OUT / 'uploads' / 'semantic_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
(OUT / 'semantic_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
RESULT = dict(total=len(selected), server1=len(server1), server2=len(server2), sha=manifest['manifest_sha256'])
print(RESULT, flush=True)
