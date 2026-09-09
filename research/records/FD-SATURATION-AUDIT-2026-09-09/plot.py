"""Export the research figure from reviewed local summary data; no model imports."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-gpu-analysis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root/'layers.csv').open()))
summary = json.loads((root/'analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#222222',
    'axes.labelcolor':'#222222','axes.spines.top':False,'axes.spines.right':False,
    'svg.hashsalt':'fd-saturation-audit-2026-09-09'})
fig, axes = plt.subplots(1,2,figsize=(12,5.8))
fig.subplots_adjust(left=.08,right=.97,bottom=.24,top=.73,wspace=.28)
fig.suptitle('Excursion proxy and FD / AD norm ratio',x=.08,y=.97,ha='left',fontsize=19)
fig.text(.08,.89,'Observed association and a mathematical counterexample; these do not identify the cause.',fontsize=11)
a,b=axes
x=[100*float(r['excursion_over_proxy']) for r in rows]
y=[float(r['fd_over_exact_norm']) for r in rows]
a.scatter(x,y,s=35,color='#3773B8',edgecolor='white',linewidth=.6)
a.set(title='Device record: 33 layers, one row',xlabel='Aggregate excursion / preceding-layer proxy (%)',ylabel='FD / AD Frobenius norm')
a.text(.96,.93,'Spearman −0.872',transform=a.transAxes,ha='right',fontsize=11)
for idx in (0,4,24,32):
    if idx == 24:
        a.annotate('L25',(x[idx],y[idx]),xytext=(2,.67),textcoords='data',fontsize=9,
            arrowprops={'arrowstyle':'-','color':'#777777','lw':.7})
    else:
        a.annotate('L'+rows[idx]['repo_layer'],(x[idx],y[idx]),xytext=(5,5),textcoords='offset points',fontsize=9)
a.set(xlim=(0,60),ylim=(0,1.06))
s=[1+i*15/200 for i in range(201)]
b.plot(s,[1/q for q in s],color='#3773B8',linewidth=1.5)
values=summary['linear_counterexample']['a']['value']
b.scatter(values,[1/q for q in values],s=40,facecolor='white',edgecolor='#3773B8',linewidth=1.4,zorder=3)
b.set(title='Synthetic: perfectly linear functions',xlabel='True derivative scale, s (arbitrary units)',ylabel='Reported / true norm, 1 / s',xlim=(0,17),ylim=(0,1.06))
b.text(.96,.93,'Spearman −1.000',transform=b.transAxes,ha='right',fontsize=11)
b.text(.45,.60,'True map = sI\nFaulty readout = I\nNo saturation',transform=b.transAxes,fontsize=11,linespacing=1.6)
for ax in axes:
    ax.grid(axis='y',color='#E7E7E7',linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines['left'].set_color('#B5B5B5')
    ax.spines['bottom'].set_color('#B5B5B5')
fig.text(.08,.11,'Left: derived from WSD-GOLDEN-4B at 13bce4e, 2026-09-09. Right: an algebraic counterexample, not model data.',fontsize=9)
fig.text(.08,.065,'The true-map size raises the horizontal quantity while dividing the vertical one. Nonmonotone reversals do not remove this coupling.',fontsize=9)
for ext in ('png','svg'):
    fig.savefig(root/f'correlation.{ext}',dpi=170,metadata={'Date':None} if ext=='svg' else {})
svg = root/'correlation.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
