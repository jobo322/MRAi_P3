"""Contour plot of folder-51 spectrum (2rr) with system-determined peaks.
x = F2 1H chemical shift (ppm, high->low left-to-right, NMR convention)
y = F1 J-axis in Hz (converted from ppm via OBS), 0 at centre.
Overlays .tab peak positions (X_PPM, Y_PPM->Hz) as red crosses.
"""
import numpy as np, nmrglue as ng
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FT2 = '/tmp/p51/51.ft2'
TAB = '/tmp/p51_run/51.tab'
OUT = '/tmp/p51_run/51_contour.png'
S_CONTOUR = 0.005

dic, S = ng.pipe.read(FT2)
Sn = S / np.max(np.abs(S))
uc_x = ng.pipe.make_uc(dic, S, dim=1)   # F2 direct 1H
uc_y = ng.pipe.make_uc(dic, S, dim=0)   # F1 indirect (J)
x0, x1 = uc_x.ppm_limits()              # high, low
y0_ppm, y1_ppm = uc_y.ppm_limits()
obs_y = dic['FDF1OBS']
y0, y1 = y0_ppm * obs_y, y1_ppm * obs_y  # ppm -> Hz

# peaks from .tab
xp, yp = [], []
if __import__('os').path.isfile(TAB):
    vl = None; rows = []
    for l in open(TAB):
        if l.startswith('VARS   '):
            vl = l.strip().replace('VARS   ', '').split()
        elif l.strip() and l.strip()[0].isdigit() and vl and len(l.split()) == len(vl):
            rows.append(l.split())
    if vl:
        ix, iy = vl.index('X_PPM'), vl.index('Y_PPM')
        for r in rows:
            xp.append(float(r[ix])); yp.append(float(r[iy]) * obs_y)

N, C = 10, 1.3
cl = S_CONTOUR * C ** np.arange(N)
fig, ax = plt.subplots(figsize=(14, 8))
ax.contour(Sn, cl, colors='black', linewidths=0.6, extent=(x0, x1, y0, y1))
if xp:
    ax.scatter(xp, yp, s=60, marker='x', c='red', linewidths=1.2,
               label=f'system peaks ({len(xp)})', zorder=5)
    ax.legend(loc='upper right')
ax.set_xlabel(f"{dic['FDF2LABEL']} (ppm)")
ax.set_ylabel(f"{dic['FDF1LABEL']} J (Hz)")
ax.set_title(f"Folder 51 J-resolved spectrum (2rr) + P3 peaks", pad=12)
ax.invert_xaxis()
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print('saved', OUT, f'with {len(xp)} peaks')
