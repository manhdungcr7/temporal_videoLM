"""Load Qwen2.5-Omni-3B once into the namespace of kaggle_rootcause.py, then exec job_utils.py and one job file in it.
Usage: python nb_runner.py ROOT OUT JOB.py '{"PARAM": value, "HOURS": 10.5}'
The job sees every helper of kaggle_rootcause.py (cut, fetch, scores, masked_forward, hear, whisper_text, ...), the model
(MODEL_, PROC, LETTERS), OUT, the params as globals, and time_left() (seconds before the self-imposed deadline)."""
import sys, os, time, json
from pathlib import Path

ROOT, OUT, JOB = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PARAMS = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
t_start = time.time()
sys.argv = ['kaggle_rootcause.py', str(ROOT), str(OUT), '1', '0', '1']
NS = {'__name__': 'rc_namespace', '__file__': str(ROOT / 'kaggle_rootcause.py')}
exec(compile((ROOT / 'kaggle_rootcause.py').read_text(encoding='utf-8'), 'kaggle_rootcause.py', 'exec'), NS)
NS['MODEL_'], NS['PROC'] = NS['load_model']()
NS['LETTERS'] = [NS['PROC'].tokenizer.convert_tokens_to_ids(L) for L in 'ABC']
hours = float(PARAMS.pop('HOURS', 10.5))
NS.update(OUT=OUT, ROOT=ROOT, DEADLINE=t_start + hours * 3600, **PARAMS)
NS['time_left'] = lambda: NS['DEADLINE'] - time.time()
print('model ready in', round(time.time() - t_start), 's; params', PARAMS, '; deadline in', hours, 'h', flush=True)
exec(compile((ROOT / 'job_utils.py').read_text(encoding='utf-8'), 'job_utils.py', 'exec'), NS)
exec(compile(JOB.read_text(encoding='utf-8'), JOB.name, 'exec'), NS)
print('RESULT', json.dumps(NS.get('RESULT'), default=str, indent=1), flush=True)
