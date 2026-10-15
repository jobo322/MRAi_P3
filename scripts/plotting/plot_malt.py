"""Static contour plot: Malt HSQC spectrum + system-determined P3 peaks.
Companion to /tmp/p51_run/plot_p51.py, adapted for the 15N-HSQC axes
(x = F2 1H ppm, y = F1 15N ppm).
"""
import numpy as np, nmrglue as ng
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FT2 = '/Users/abolanos/git/mrai_p3/Spectra/2D/Malt.ft2'
TAB = '/tmp/malt_run/Malt.tab'
OUT = '/tmp/malt_run/Malt_contour.png'
S_CONTOUR = 0.005

dic, S = ng.pipe.read(FT2)
Sn = S / np.max(np.abs(S))
uc_x = ng.pipe.make_uc(dic, S, dim=1)   # F2 direct 1H
uc_y = ng.pipe.make_uc(dic, S, dim=0)   # F1 indirect 15N
x0, x1 = uc_x.ppm_limits()
y0, y1 = uc_y.ppm_limits()

xp, yp = [], []
vl = None; rows = []
for l in open(TAB):
    s = l.strip()
    if not s:
        continue
    parts = s.split()
    if parts[0] == 'VARS':
        vl = parts[1:]
    elif vl and s[0].isdigit() and len(parts) == len(vl):
        rows.append(parts)
ix, iy = vl.index('X_PPM'), vl.index('Y_PPM')
for r in rows:
    xp.append(float(r[ix])); yp.append(float(r[iy]))

N, C = 10, 1.3
cl = S_CONTOUR * C ** np.arange(N)
fig, ax = plt.subplots(figsize=(12, 9))
ax.contour(Sn, cl, colors='black', linewidths=0.6, extent=(x0, x1, y0, y1))
ax.scatter(xp, yp, s=60, marker='x', c='red', linewidths=1.2,
           label=f'system peaks ({len(xp)})', zorder=5)
ax.legend(loc='upper right')
ax.set_xlabel(f"{dic['FDF2LABEL']} (ppm)")
ax.set_ylabel(f"{dic['FDF1LABEL']} (ppm)")
ax.set_title('Malt HSQC spectrum + P3 peaks', pad=12)
ax.invert_xaxis(); ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print('saved', OUT, f'with {len(xp)} peaks')
