import json
from pathlib import Path
import decord
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'local_qwen/temporal_preprocess_dryrun.json').read_text(encoding='utf-8'))
for idx in (1961,2081,2310):
    selected=[r for r in rows if r['idx']==idx]
    panels=[]
    for rec in selected:
        vr=decord.VideoReader(rec['normalized_path'])
        frames=vr.get_batch(rec['selected_frame_indices']).asnumpy()
        times=[(a+b)/2 for a,b in rec['selected_frame_pts_intervals_s']]
        thumbs=[]
        for frame,t in zip(frames,times):
            img=Image.fromarray(frame).convert('RGB'); img.thumbnail((280,170))
            tile=Image.new('RGB',(300,205),'white'); tile.paste(img,((300-img.width)//2,5))
            ImageDraw.Draw(tile).text((8,180),f"{t:.2f}s",fill='black')
            thumbs.append(tile)
        cols=4; rows_n=(len(thumbs)+cols-1)//cols
        sheet=Image.new('RGB',(cols*300,rows_n*205),(220,220,220))
        for j,tile in enumerate(thumbs):sheet.paste(tile,((j%cols)*300,(j//cols)*205))
        draw=ImageDraw.Draw(sheet); draw.rectangle((0,0,sheet.width,24),fill='white')
        draw.text((8,5),f"sample {idx} | target {rec['target_fps']} fps | Qwen input frame times",fill='black')
        out=ROOT/f"local_qwen/temporal_debug_media/preprocess_{idx}_{int(rec['target_fps'])}fps.jpg"
        out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out,quality=92)
        print(out)
