# Fix: NMRPipe-free header fallback and 2D axis order in the peak-list converters

**Date**: 2026-10-01
**Scope**: the `showhdr`-less nmrglue fallback shipped in wheel `MRAi_P3-1.1.1-py3-none-any.whl`
(commit `fdead0e`) in `P5toPeaks.py` and `PeakstoTab.py`, plus a pre-existing X/Y axis-order defect
those two files disagreed about.
**Out of scope**: wheel repack/versioning, the `StoP3.py` SavedModel path, 3D runtime verification.

## TL;DR

Two independent defects made the `.tab` output numerically wrong. Both are fixed:

1. The nmrglue fallback read Pipe header fields as if they were ppm, and its one correct line was
   dead code, so the `.tab` header carried Hz labelled as ppm and peak point indices landed far
   outside the dataset.
2. `P5toPeaks` and `PeakstoTab` disagreed about which column is which axis: `P5toPeaks` emits the
   direct (1H) coordinate first, `PeakstoTab` calibrated that column with the 15N axis.

The repository carries no package source, so the sources were extracted from the wheel into
`src/MRAi_P3/` and fixed there. **The wheel itself has not been rebuilt** — see Follow-ups.

## Root causes

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| 1 | `ng.pipe.make_uc(dic, None, dim=…)` always raises `AttributeError: 'NoneType' object has no attribute 'ndim'` because `make_uc` reads `data.shape`; a bare `except Exception` swallowed it, so the correct branch never ran | nmrglue 0.11 `make_uc` source: `size = float(data.shape[dim])` | `headers.read_headers` keeps the array from `ng.pipe.read` and passes it |
| 2 | `FDF{i}ORIG` / `FDF{i}SW` are **Hz**, used directly as ppm | `Azurin.ft2`: `FDF1ORIG`=8952.56 Hz, true ppm range 138.012→98.154 | `pfst, plst = uc.ppm_limits()`; `sw = FDF{i}SW / FDF{i}OBS` (ppm/point) |
| 3 | `sz` taken from `FDF2FTSIZE`, which is stale | `FDF2FTSIZE`=4096 vs `FDSIZE`=`data.shape[1]`=1279 | `sz[i] = float(data.shape[i-1])` |
| 4 | `ND` guessed from `FDF2LABEL` / `FDF3FTSIZE` | heuristic mislabels non-folded/quadrature 3D files | `ND = data.ndim` |
| 5 | ~20 lines of header parsing duplicated across the two scripts, already diverging | diff 1.1.0 → 1.1.1 | one `MRAi_P3/headers.py::read_headers(spec_path) -> (ND, sz, sw, pfst, plst, nam)` |
| 6 | **X/Y axis order.** `.P5` rows are `(FDF1=15N ppm, FDF2=1H ppm, prob, intensity)` (P3toP5 overwrites the point-index columns with ppm). `P5toPeaks` unpacks `F1, F2` then emits `{F2} {F1}` — 1H first, the CYANA/xeasy convention — but `#INAME 1`/`#SPECTRUM` were labelled in dimension order, and `PeakstoTab` read `X = parts[1]` (a 1H value) while calibrating it with the 15N axis | old `Azurin.tab`: `X1=817` (axis is 256 pts), `Y1=-31654` | decision below |

## Decision (human-owned)

**x is the 1H (direct) axis.** `P5toPeaks`' emitted column order was kept; the reader and the labels
were fixed. CYANA/xeasy `.peaks` and NMRPipe/nmrDraw `.tab` both put the direct dimension on x, and
this single choice also yields the standard 3D HNCO order `X=dim2 (1H), Y=dim1 (15N), Z=dim3 (13C)`.

- `P5toPeaks`: `#INAME 1 {nam[2]}`, `#INAME 2 {nam[1]}`, `#SPECTRUM {type} {nam[2]} {nam[1]}` (3D adds
  `{nam[3]}` last). Data row formats untouched.
- `PeakstoTab`: `x_pnt` uses dim 2, `y_pnt` uses dim 1, `z_pnt` uses dim 3; the `DATA X_AXIS` /
  `Y_AXIS` lines now declare those axes with their own `sz`/`pfst`/`upper_ppm`. `VARS`/`FORMAT`
  column order is unchanged.

## Before / after

```
BEFORE  Spectra/2D/Azurin.tab (commit fdead0e)
  DATA  X_AXIS N            1   258   8952.557ppm 12573.679ppm     <- Hz labelled ppm
  DATA  Y_AXIS HN           1  4098   5399.669ppm  9901.160ppm     <- stale FDF2FTSIZE
      1  -623.784 -4786.076 ... X1 X3 = -623 -622  Y1 Y3 = -4786 -4785

AFTER   Spectra/2D/Azurin.tab (regenerated)
  DATA  X_AXIS HN           1  1281     10.998ppm     5.991ppm
  DATA  Y_AXIS N            1   258    138.012ppm    97.842ppm
      1   139.098    21.345 ... 139  140   21   22
```

Per-spectrum regeneration (fallback forced, no NMRPipe installed):

```
Azurin      rows=387  X(1H, 1..1279)=119..1270   Y(15N, 1..256)=21..229
Tau         rows=469  X(1H, 1..1233)=607..1231   Y(15N, 1..256)= 6..256
Ubiquitin   rows=191  X(1H, 1..1279)=382..1259   Y(15N, 1..256)=32..229
Malt        rows=414  X(1H, 1..805) =140..730    Y(15N, 1..256)=14..232
```

Header values cross-checked against nmrglue ground truth, e.g. `sw[1] = 40.014 ppm = 3649.635 Hz /
91.209 MHz` and `sw[2] = 5.0034 = 4503.69 / 900.13`.

## Independent verification

`python3 -m pytest tests -q` → **11 passed** (Python 3.13.2, pytest 9.1.1, nmrglue 0.11).

Beyond the suite, a 40-peak random sample per spectrum was checked against the `.ft2` grid: for
85–97% of sampled peaks the dominant local maximum is a strict local maximum on **both** axes, and the
offset from the reported `X1`/`Y1` is within ±4 points (median 1) — consistent with the sub-pixel
centroid accumulation in `P3toP5` and the linear ppm→index rounding in `point_from_ppm`. Before the
fix the same indices were `-31654`, i.e. meaningless.

## Files

```
src/MRAi_P3/headers.py        NEW  read_headers(); showhdr primary, nmrglue fallback
src/MRAi_P3/P5toPeaks.py      CHANGED  six-value unpack; axis labels follow emitted columns
src/MRAi_P3/PeakstoTab.py     CHANGED  per-axis calibration; DATA lines declare the right axes
src/MRAi_P3/{P3toP5,P5toXml,StoP3,Tab3Dto2Dproj,__init__}.py  NEW (extracted unmodified)
tests/test_headers.py         NEW  11 tests, fallback forced, no NMRPipe/TensorFlow needed
Spectra/2D/*.peaks, *.tab      REGENERATED (8 files)
```

## Known gaps / follow-ups

1. **The wheel was not rebuilt.** Consumers still installing `MRAi_P3-1.1.1-py3-none-any.whl` get
   the broken code. The wheel additionally ships `dist-info` = `MRAi_P3-1.0.0.dist-info` with
   `Version: 1.1.0` and stale `RECORD` hashes, so its filename/metadata already disagree.
2. **The 3D path is unverified at runtime** — no 3D spectra exist under `Spectra/`. The `[2,1,3]`
   label order and dim-3 calibration are applied by inspection only.
3. **`P3_Master.py:65`** uses `params.get('Spec_type','None')`, yielding the truthy string `'None'`,
   so `.peaks` can get `#SPECTRUM None HN N`. Pre-existing, untouched.
4. **The `showhdr` primary route is unchanged and unexercised here** (no NMRPipe on this machine).
   It was already verified correct in production, so the fix only re-aligns the fallback with it.
5. **The regenerated `.peaks` files carry `#SPECTRUM HSQC …`** rather than the basename, preserving
   the committed label; `spec_type=None` would have rewritten it to `#SPECTRUM Azurin HN N`.
6. Tests run against nmrglue 0.11 while the package requires `>=0.12,<1`; re-run in a clean env to
   confirm no behavioural difference.

## Reproducing

```bash
cd /Users/abolanos/git/mrai_p3
python3 -m pytest tests -q
```
