"""Render measured training curves; never infer, fit, or select a checkpoint."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sensegraph.io import read_json

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/adaptation_v2.json'); args=p.parse_args()
cfg=read_json(args.config); base=Path(cfg['experiment']); profile=read_json(base/'profile.json')
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for arm,color in [('control','#536878'),('realistic','#b54b30')]:
    path=base/f'{arm}_history.json'
    if not path.exists(): continue
    rows=read_json(path); epochs=[r['epoch'] for r in rows]
    for ax,key in zip(axes,['train_loss','synthetic_ap50','real_recall']):
        ax.plot(epochs,[r[key] for r in rows],marker='o',label=arm,color=color)
axes[0].set_yscale('log'); axes[0].set_title('Training loss (different data)')
axes[1].axhline(profile['reference_synthetic_ap50'],ls='--',color='black',label='A')
axes[1].axhline(profile['reference_synthetic_ap50']-.05,ls=':',color='gray',label='retention floor')
axes[1].set_title('Synthetic validation AP50')
axes[2].axhline(profile['reference_real_recall'],ls='--',color='black',label='A')
axes[2].set_title('Real validation supported recall50')
for ax in axes:
    ax.set_xlabel('Complete epoch'); ax.set_xticks(range(1,6)); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('Matched single-seed pilot; validation reuse disclosed; no QA accuracy claim',fontsize=11)
fig.savefig(base/'learning_curves.png',dpi=180); plt.close(fig)
print(base/'learning_curves.png')
