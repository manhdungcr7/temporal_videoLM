"""Root-cause report for SVHalluc temporal. Usage: python rootcause_report.py DIR...  (reads rootcause_shard*.jsonl, hidden/*.npz, synthetic_relation.json)"""
import json, sys, glob, os, re, random
from pathlib import Path
import numpy as np

dirs = sys.argv[1:] or ['.']
recs = {}
for d in dirs:
    for p in glob.glob(os.path.join(d, '**', 'rootcause_shard*.jsonl'), recursive=True):
        for l in Path(p).read_text(encoding='utf-8').splitlines():
            if l.strip(): r = json.loads(l); recs[r['idx']] = r
R = [r for r in recs.values() if 'error' not in r]
print(f'items {len(recs)} | valid {len(R)} | skipped/errors {len(recs) - len(R)}')
am = lambda v: int(np.argmax(v))
ok = lambda r, c: am(r[c]) == r['gold']
TASKS = ('TA', 'TF', 'CMTB')


def pct(x): return f'{100 * np.mean(x):5.1f}%' if len(x) else '  n/a'


print('\n=== #12 question wording, #13 content-free prior, modality reliance ===')
for t in TASKS:
    rs = [r for r in R if r['task'] == t]
    if not rs: continue
    k = len(rs[0]['base'])
    line = f'{t:5s} n={len(rs):3d} | base {pct([ok(r, "base") for r in rs])} para1 {pct([ok(r, "para1") for r in rs])} para2 {pct([ok(r, "para2") for r in rs])}'
    agree = np.mean([am(r['base']) == am(r['para1']) == am(r['para2']) for r in rs])
    line += f' | all 3 wordings agree {100 * agree:.0f}%'
    print(line)
    print(f'      vision_only {pct([ok(r, "vision_only") for r in rs])} audio_only(black video) {pct([ok(r, "audio_only") for r in rs])} prior(no content) {pct([ok(r, "prior") for r in rs])}')
    prior_pick = [am(r['prior']) for r in rs]
    print(f'      content-free prior picks: {[round(prior_pick.count(j) / len(rs), 2) for j in range(k)]} | base answer == prior answer: {pct([am(r["base"]) == am(r["prior"]) for r in rs])}'
          f' | among base ERRORS: {pct([am(r["base"]) == am(r["prior"]) for r in rs if not ok(r, "base")])}')
    print(f'      base predicted share {[round(float(np.mean([am(r["base"]) == j for r in rs])), 2) for j in range(k)]} recall/label {[round(float(100 * np.mean([ok(r, "base") for r in rs if r["gold"] == j])), 1) if any(r["gold"] == j for r in rs) else None for j in range(k)]}')
    wrong = [r for r in rs if not ok(r, 'base')]
    if wrong:
        print(f'      base errors that equal the audio_only answer {pct([am(r["base"]) == am(r["audio_only"]) for r in wrong])}, equal the vision_only answer {pct([am(r["base"]) == am(r["vision_only"]) for r in wrong])}')

print('\n=== #9/#10 per-error decomposition (CMTB, base wrong) ===')
cm = [r for r in R if r['task'] == 'CMTB']
w = [r for r in cm if not ok(r, 'base')]
if w:
    cats = {}
    for r in w:
        g, a, b, c, cr = ok(r, 'gate'), ok(r, 'audio_block'), ok(r, 'gate_audio'), ok(r, 'crop'), None
        if g: key = 'fixed by time gate alone'
        elif a: key = 'fixed by audio block alone'
        elif b: key = 'needs BOTH time gate and audio block'
        elif ok(r, 'crop'): key = 'only crop fixes (attention-level fix insufficient)'
        else: key = 'nothing fixes (perception / option confusion)'
        cats[key] = cats.get(key, 0) + 1
    for kk, v in sorted(cats.items(), key=lambda x: -x[1]): print(f'  {kk:52s} {v:3d} ({100 * v / len(w):.0f}%)')
    print(f'  (CMTB n={len(cm)}, base wrong {len(w)}; gate_audio overall {pct([ok(r, "gate_audio") for r in cm])}, crop {pct([ok(r, "crop") for r in cm])}, correct->wrong by gate_audio {sum(ok(r, "base") and not ok(r, "gate_audio") for r in cm)})')

print('\n=== #4 speech content: model "hear" vs Whisper reference (analysis only) ===')
STOP = set('the a an and to of in on with it is i we you this that for then some just now will be going so into your our my at as are'.split())
def words(s): return {w for w in re.findall(r'[a-z]+', (s or '').lower()) if w not in STOP and len(w) > 2}
hw = [r for r in R if r.get('whisper')]
if hw:
    ov = [len(words(r['hear']) & words(r['whisper'])) > 0 for r in hw]
    print(f'  items with a Whisper transcript {len(hw)}; model "hear" shares >=1 content word with the transcript: {pct(ov)}')
    for r in hw[:8]: print(f'    [{r["task"]}] hear={r["hear"]!r} | whisper={r["whisper"][:90]!r}')
else:
    print('  no Whisper transcripts')

print('\n=== #11 relation reasoning from text facts (no media) ===')
for d in dirs:
    for p in glob.glob(os.path.join(d, '**', 'synthetic_relation.json'), recursive=True):
        S = json.loads(Path(p).read_text(encoding='utf-8'))
        for tense in sorted({s.get('tense', 'future') for s in S}):
            Z = [s for s in S if s.get('tense', 'future') == tense]
            tf = [am(s['tf']) == s['rel'] for s in Z]; ta = [am(s['ta']) == s['ta_gold'] for s in Z]
            tfp = [am(s['tf_plain']) == s['rel'] for s in Z if 'tf_plain' in s]
            print(f'  speech tense {tense:8s} n={len(Z)} | Past/Present/Future wording {pct(tf)} per relation {[pct([am(s["tf"]) == s["rel"] for s in Z if s["rel"] == j]) for j in range(3)]}'
                  f' | before/during/after wording {pct(tfp)} | same-time Yes/No {pct(ta)}')

print('\n=== #14 representation-action gap: linear probe on hidden states (5-fold CV) vs model output ===')
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    for t in TASKS:
        rs = [r for r in R if r['task'] == t]
        X, y, base_ok = {}, [], []
        for r in rs:
            f = None
            for d in dirs:
                g = glob.glob(os.path.join(d, '**', 'hidden', f'h_{r["idx"]}.npz'), recursive=True)
                if g: f = g[0]; break
            if not f: continue
            z = np.load(f); layers = list(z['layers'])
            for li, l in enumerate(layers):
                X.setdefault(('last', l), []).append(z['last'][li].astype('float32'))
                X.setdefault(('qmean', l), []).append(z['qmean'][li].astype('float32'))
            y.append(r['gold']); base_ok.append(ok(r, 'base'))
        if len(y) < 20 or len(set(y)) < 2: print(f'  {t}: not enough items ({len(y)})'); continue
        print(f'  {t} n={len(y)} model output accuracy {pct(base_ok)} | chance (majority) {100 * max(np.bincount(y)) / len(y):.1f}%')
        for kind in ('last', 'qmean'):
            line = f'    {kind:5s} '
            for l in layers:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
                s = cross_val_score(clf, np.stack(X[(kind, l)]), np.array(y), cv=5).mean()
                line += f'L{l}:{100 * s:.0f}% '
            print(line)
except Exception as e:
    print('  probe skipped:', repr(e)[:150])

print('\n=== #8 attention of the question rows (eager), mean over items ===')
at = [r for r in R if 'attention' in r]
if at:
    for lo, hi in ((0, 12), (12, 24), (24, 36)):
        def m(g, norm):
            vals = []
            for r in at:
                c = r['attention']['counts']; pl = r['attention']['per_layer'][lo:hi]
                v = np.mean([x[g] for x in pl]); vals.append(v / max(c[g], 1) if norm else v)
            return np.mean(vals)
        print(f'  layers {lo}-{hi - 1}: per-token mass video first/second half ratio {m("v_second_half", 1) / max(m("v_first_half", 1), 1e-12):.2f} | inside/outside speech window {m("v_in", 1) / max(m("v_out", 1), 1e-12):.2f} | total to audio {m("audio", 0):.3f}, to video {m("v_in", 0) + m("v_out", 0):.3f}  (n={len(at)})')

print('\n=== where the error enters: activation patching (CMTB, base wrong & gate_audio right) ===')
pt = [r for r in R if 'patching' in r]
if pt:
    first_fix, first_break = [], []
    for r in pt:
        P = r['patching']; g = r['gold']
        fl = sorted(int(k) for k in P['fwd'])
        ff = next((l for l in fl if am(P['fwd'][str(l)] if str(l) in P['fwd'] else P['fwd'][l]) == g), None)
        fb = next((l for l in fl if am(P['rev'][str(l)] if str(l) in P['rev'] else P['rev'][l]) != g), None)
        first_fix.append(ff); first_break.append(fb)
    print(f'  n={len(pt)} | earliest layer where inserting the CLEAN question states into the plain run fixes the answer: {first_fix}')
    print(f'  earliest layer where inserting the PLAIN question states into the clean run breaks it: {first_break}')
    print('  (a narrow band of layers in which both happen = the evidence-reading window of the question tokens)')
