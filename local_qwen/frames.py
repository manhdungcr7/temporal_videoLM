import json,sys,av
from pathlib import Path
from PIL import Image
base=Path(r'F:\A_star_paper_NLP_ACL\data\svhalluc_benchmark')
R=json.loads((base/'samples.json').read_text(encoding='utf-8'))
i=int(sys.argv[1]);fr=[]
with av.open(str(base/R[i]['video_path'].replace('\\','/'))) as c:
    for f in c.decode(video=0):fr.append(f.to_image())
n=len(fr);sel=[fr[int(n*k/8)] for k in (1,3,5,7)]
w,h=sel[0].size;sel=[s.resize((w//2,h//2)) for s in sel]
im=Image.new('RGB',(w,h))
for k,s in enumerate(sel):im.paste(s,((k%2)*(w//2),(k//2)*(h//2)))
im.save(Path(__file__).parent/('f%d.jpg'%i));print(n,w,h)
