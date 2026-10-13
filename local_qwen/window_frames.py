import json, sys, av
from pathlib import Path
from PIL import Image
base = Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark')
R = json.loads((base / 'samples.json').read_text(encoding='utf-8'))
idx, t0, t1, tag = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
path = base / R[idx]['video_path'].replace(chr(92), '/')
frames = []
with av.open(str(path)) as c:
    s = c.streams.video[0]
    for f in c.decode(s):
        t = float(f.pts * s.time_base)
        frames.append((t, f.to_image()))
sel = []
for k in range(4):
    target = t0 + (t1 - t0) * (k + 0.5) / 4
    sel.append(min(frames, key=lambda x: abs(x[0] - target))[1])
w, h = sel[0].size; w2, h2 = w // 2, h // 2
im = Image.new('RGB', (w2 * 2, h2 * 2))
for k, s_ in enumerate(sel): im.paste(s_.resize((w2, h2)), ((k % 2) * w2, (k // 2) * h2))
out = Path(__file__).parent / f'win_{idx}_{tag}.jpg'; im.save(out); print(out, im.size)
