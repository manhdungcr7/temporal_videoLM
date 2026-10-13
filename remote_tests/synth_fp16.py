import json, time
t = time.time()
S = synthetic_relation()
(OUT / 'synthetic_fp16.json').write_text(json.dumps(S))
RESULT = {'n': len(S), 'sec': round(time.time() - t)}
