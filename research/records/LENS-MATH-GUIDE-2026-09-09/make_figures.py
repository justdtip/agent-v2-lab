"""Educational diagrams and typeset equations; no model or experiment execution."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/codex-gpu-analysis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mathtext, font_manager
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'tmp/pdfs/lens-math-guide/assets'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'stix','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'text.color':'#20313B',
    'axes.labelcolor':'#20313B'})
E={
'vector':r'$x=(A,B)=(2,3)\ \mathrm{mM}$',
'delta':r'$\Delta x=(0.1,-0.2)\ \mathrm{mM}$',
'rates':r'$r_1=0.5AB,\qquad r_2=0.2B$',
'jacobian':r'$J_{ij}=\frac{\partial y_i}{\partial x_j}$',
'local':r'$\Delta y\ \approx\ J\,\Delta x$',
'row1':r'$\Delta r_1\approx 1.5(0.1)+1(-0.2)=-0.05$',
'row2':r'$\Delta r_2\approx 0(0.1)+0.2(-0.2)=-0.04$',
'partial':r'$\frac{\partial r_1}{\partial A}=0.5B=1.5$',
'cross':r'$0.5\,\Delta A\,\Delta B=-0.01$',
'central':r'$D_hF(x)=\frac{F(x+h)-F(x-h)}{2h}$',
'mm':r'$f(z)=\frac{z}{1+z},\quad f^{\prime}(1)=\frac{1}{4}$',
'mmdiff':r'$D_hf(1)=\frac{1}{4-h^2}$',
'error':r'$E(h)\ \lesssim\ Ah^2+\frac{B}{h}+C$',
'unitstep':r'$x\longmapsto x+h\,v$',
'jv':r'$F(x+h\,v)-F(x)\ \approx\ h\,Jv$',
'norm':r'$\|x\|_2=\sqrt{x_1^2+x_2^2+\cdots+x_d^2}$',
'frob':r'$\|J\|_F=\sqrt{\sum_{i,j}J_{ij}^{\,2}}$',
'rel':r'$r=\frac{\|\widehat J-J\|_F}{\|J\|_F}$',
'cos':r'$c=\frac{\langle\widehat J,J\rangle}{\|\widehat J\|_F\,\|J\|_F}$',
'step':r'$\frac{h}{\mathrm{RMS}(H)}=0.01\sqrt{128\cdot2560}\approx5.724$',
'bf16':r'$1.0000000\ \longrightarrow\ 1.0078125$',
'mean':r'$\overline{J}=\frac{1}{2}\left(J_{8\to8}+J_{8\to127}+J_{127\to127}\right)$',
'cancel':r'$\frac{(+2)+(-2)}{2}=0$',
'couple':r'$X=\frac{h\,m}{T},\qquad Y=\frac{f}{a}$',
'inverse':r'$X\propto s,\qquad Y=\frac{1}{s}$',
'linear':r'$\mathrm{true:}\ F(x)=s\,x\qquad\mathrm{reported:}\ \widehat J=1$',
'actual':r'$v_{\mathrm{actual}}=\frac{x_+-x_-}{2h}$',
}
assert not set(E) & {"enzyme", "error-curve", "coupling"}
metrics={}
for key,expr in E.items():
    mathtext.math_to_image(expr,OUT/(key+'.png'),prop=font_manager.FontProperties(size=19),dpi=260,color='#20313B')
    metrics[key]={'latex':expr}
(OUT/'equations.json').write_text(json.dumps(metrics,indent=2)+'\n')

fig,ax=plt.subplots(figsize=(5.0,2.4))
z=np.linspace(0,3,300)
ax.plot(z,z/(1+z),color='#287B8E',lw=2,label='Saturating response')
ax.plot([.45,1.55],[.5-.25*.55,.5+.25*.55],color='#C18A30',lw=2,label='Local slope at z = 1')
ax.scatter([1],[.5],color='#20313B',s=20,zorder=5)
ax.set(xlabel='Substrate concentration / Km',ylabel='Rate / Vmax',xlim=(0,3),ylim=(0,.85))
ax.legend(frameon=False,fontsize=9,loc='lower right')
fig.tight_layout()
fig.savefig(OUT/'enzyme.png',dpi=220)
plt.close(fig)

fig,ax=plt.subplots(figsize=(5,2.5))
h=np.geomspace(.0001,1,350)
trunc=h*h; rnd=1e-6/h
ax.loglog(h,trunc,color='#287B8E',lw=1.5,label='Finite-step error: A h²')
ax.loglog(h,rnd,color='#C18A30',lw=1.5,label='Rounding term: B / h')
ax.loglog(h,trunc+rnd,color='#20313B',lw=2,label='Illustrative sum')
ax.set(xlabel='Step h (arbitrary units)',ylabel='Error model (arbitrary units)',ylim=(1e-5,2))
ax.legend(frameon=False,fontsize=8.5,loc='upper center')
fig.tight_layout()
fig.savefig(OUT/'error-curve.png',dpi=220)
plt.close(fig)

fig,ax=plt.subplots(figsize=(5,2.25))
s=np.linspace(1,4,100)
ax.plot(s,1/s,lw=2,color='#287B8E')
ax.scatter([1,2,4],[1,.5,.25],facecolor='white',edgecolor='#287B8E',s=50,zorder=4)
ax.set(xlabel='True slope, s',ylabel='Reported / true slope',ylim=(0,1.1),xlim=(.8,4.2))
ax.text(2.15,.82,'All three assays are linear.',fontsize=10)
fig.tight_layout()
fig.savefig(OUT/'coupling.png',dpi=220)
plt.close(fig)
print('Typeset',len(E),'equations and 3 educational figures.')
