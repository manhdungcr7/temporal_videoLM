"""SEM-CHRONO — the friend's ChronoDeSmear (Round 5, R5_C0..C6) evaluated UNCHANGED on the SVHalluc SEMANTIC tasks, inside one Kaggle notebook.
Protocol fixed beforehand: results/semantic_chrono/protocol.md.   Needs the files written by the notebook (see `extra` in build_notebooks.py):
chronodesmear_cache40.py, chronodesmear40.py, chronodesmear_math.py, avbind_sdpa_compat.py, chronodesmear_proposal_original.py, semantic_manifest_job.py.
Steps: (1) copy the helpers to OUT/uploads and patch the friend's cache/runner by text substitution (manifest, cache and output paths only);
(2) build the balanced manifest (GSA/FGSA/CSMB x Yes/No = 6 cells, SEM_PER_CELL each, clip <= SEM_MAXDUR s, seed SEM_SEED);
(3) encode every answer rotation once (cache, FP16 inputs_embeds); (4) run R5_C0..C6 on this shard (SEM_SHARD 0 or 1 = half of every cell).
Parameters (globals): SEM_SHARD, SEM_PER_CELL, SEM_SEED, SEM_MAXDUR; LIMIT (set by the smoke test) -> 1 sample per cell, 2 samples run.
Writes OUT/sem_chrono_s<shard+1>.jsonl (+ OUT/semantic_chrono/*: per-rotation JSON, errors) ."""
import json, re, shutil

SEM_SHARD = int(globals().get('SEM_SHARD', 0))
SMOKE = globals().get('LIMIT') is not None
SEM_PER_CELL = 1 if SMOKE else int(globals().get('SEM_PER_CELL', 10))
SEM_SEED = int(globals().get('SEM_SEED', 31337)); SEM_MAXDUR = float(globals().get('SEM_MAXDUR', 40.0))
U = OUT / 'uploads'; U.mkdir(parents=True, exist_ok=True)

# (1) helpers + patched copies of the friend's cache and runner (only paths change)
for name in ('chronodesmear_math.py', 'avbind_sdpa_compat.py', 'chronodesmear_proposal_original.py'):
    (U / name).write_text((ROOT / name).read_text(encoding='utf-8'), encoding='utf-8')
cache = (ROOT / 'chronodesmear_cache40.py').read_text(encoding='utf-8')
cache = cache.replace('research40_manifest.json', 'semantic_manifest.json')
cache = re.sub(r"\n\s*assert CD_MANIFEST\['total'\]==40\n", '\n', cache)
cache = cache.replace('argon_latent40/decoder_cache', 'semantic_chrono/decoder_cache')
run = (ROOT / 'chronodesmear40.py').read_text(encoding='utf-8')
run = run.replace('uploads/chronodesmear_cache40.py', 'uploads/semantic_cache.py').replace('uploads/chronodesmear40.py', 'uploads/semantic_chrono.py')
run = run.replace("OUT/'chronodesmear40'", "OUT/'semantic_chrono'")
assert "assert CD_MANIFEST['total']==40" not in cache and 'semantic_manifest.json' in cache and "OUT/'semantic_chrono'" in run
(U / 'semantic_cache.py').write_text(cache, encoding='utf-8'); (U / 'semantic_chrono.py').write_text(run, encoding='utf-8')

# (2) balanced semantic manifest (deterministic: seed + durations only)
exec(compile((ROOT / 'semantic_manifest_job.py').read_text(encoding='utf-8'), 'semantic_manifest_job.py', 'exec'), globals())
CD_SERVER = 1 if SMOKE else SEM_SHARD + 1
if SMOKE:                                                    # smoke: 2 samples only
    man = json.loads((U / 'semantic_manifest.json').read_text()); man['server1'] = man['server1'][:2]
    (U / 'semantic_manifest.json').write_text(json.dumps(man), encoding='utf-8')
print('shard', SEM_SHARD, 'server list size', len(json.loads((U / 'semantic_manifest.json').read_text())[f'server{CD_SERVER}']), flush=True)

# (3) encode every rotation once
CD_LIMIT = None
CD_CACHE_RUN = True
exec(compile((U / 'semantic_cache.py').read_text(), 'semantic_cache.py', 'exec'), globals())

# (4) run the seven configurations
exec(compile((U / 'semantic_chrono.py').read_text(), 'semantic_chrono.py', 'exec'), globals())
src = OUT / 'semantic_chrono' / f'results_s{CD_SERVER}.jsonl'
if src.exists(): shutil.copy(src, OUT / f'sem_chrono_s{CD_SERVER}.jsonl')
