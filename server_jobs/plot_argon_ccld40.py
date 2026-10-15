"""Export development-only per-cell recall and TF collapse figures."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/argon_ccld40'
sys.path.insert(0,str(FOLDER/'.plot_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(FOLDER/'.plot_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

summary=json.loads((FOLDER/'summary.json').read_text())
assert len(summary)==8 and all(v['n']==40 for v in summary.values())
methods=list(summary)
names=['C0 native','C1 M9 thought','C2 clean M8','C3 visual delta',
       'C4 shuffled delta','C5 global + visual','C6 CCLD','C7 CCLD + audio damp']
cells=['TA/A','TA/B','TF/A','TF/B','TF/C','CMTB/A','CMTB/B','CMTB/C']
matrix=np.array([[summary[m]['cells'][c]['correct']/5 for c in cells] for m in methods])
fig,(ax,bar)=plt.subplots(1,2,figsize=(13.5,5.4),gridspec_kw={'width_ratios':[1.45,1]})
im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(8),cells,rotation=35,ha='right');ax.set_yticks(range(8),names)
ax.set_title('Correct answers per cell (five examples each)',fontsize=12)
for i in range(8):
    for j in range(8):
        ax.text(j,i,f'{int(round(matrix[i,j]*5))}/5',ha='center',va='center',
                color='white' if matrix[i,j]>.65 else 'black',fontsize=10)
fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Recall')
counts=np.array([summary[m]['tf_pred_ABC'] for m in methods]);left=np.zeros(8)
for j,(label,color) in enumerate(zip(['Past','Present','Future'],['#3979b4','#e68736','#51a176'])):
    bar.barh(np.arange(8),counts[:,j],left=left,label=label,color=color,height=.72)
    for i,count in enumerate(counts[:,j]):
        if count:bar.text(left[i]+count/2,i,str(count),ha='center',va='center',fontsize=10,color='white')
    left+=counts[:,j]
bar.set_yticks(range(8),names);bar.invert_yaxis();bar.set_xlim(0,15)
bar.set_xticks([0,5,10,15]);bar.set_xlabel('Predicted class count (all 15 TF examples)')
bar.set_title('TF prediction distribution',fontsize=12);bar.legend(loc='lower right',bbox_to_anchor=(1,-.26),ncol=3,frameon=False)
bar.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen2.5-Omni-3B: Argon CCLD on fixed development40',fontsize=14)
fig.text(.5,.015,'Repeated development selection; five examples per task/label. This is not a held-out test.',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.07,1,.93])
target=FOLDER/'figures';target.mkdir(exist_ok=True)
for extension in ('png','svg','pdf'):fig.savefig(target/f'ccld_development40.{extension}',dpi=220,bbox_inches='tight')
print('Saved ccld_development40.png/.svg/.pdf')
