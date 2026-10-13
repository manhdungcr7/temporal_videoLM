import subprocess
def ffmpeg(*args):
    r = subprocess.run([FF, '-hide_banner', '-y', *args], capture_output=True)
    if r.returncode != 0 and b'fps_mode' in r.stderr:
        args = ['-vsync' if a == '-fps_mode' else a for a in args]
        r = subprocess.run([FF, '-hide_banner', '-y', *args], capture_output=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode(errors='replace')[-400:])
globals()['ffmpeg'] = ffmpeg
src = fetch(json.loads(fetch('samples.json').read_text(encoding='utf-8'))[1917]['video_path'].replace(BS, '/')) if 'json' in globals() else None
import json
src = fetch(json.loads(fetch('samples.json').read_text(encoding='utf-8'))[1917]['video_path'].replace(BS, '/'))
p = cut(src, 0, 10 ** 4, 'full'); print('cut ok', p, p.stat().st_size)
(OUT / 'tf_framing.jsonl').unlink(missing_ok=True); (OUT / 'tense_intervention.jsonl').unlink(missing_ok=True)
RESULT = {'ok': True}
