# Fix: nmrglue header fallback in P5toPeaks / PeakstoTab (v1.1.1 patch)

## Goal

The `showhdr`-less fallback added in commit `fdead0e` (wheel `MRAi_P3-1.1.1-py3-none-any.whl`)
never converts Hz to ppm and uses a stale size field, so the `.tab` files it writes carry
garbage axis metadata and out-of-range peak point indices. Fix the conversion and cover it
with tests against real repository spectra.

## Context / evidence

Measured on `Spectra/2D/Azurin.ft2` (nmrglue 0.11 reference values):

| dim | label | `FDF{i}ORIG` (Hz) | true `ppm_limits()` | current fallback (`ORIG+SW`) |
|-----|-------|--------------------|---------------------|-------------------------------|
| 1   | 15N   | 8952.56            | (138.012, 98.154)   | 12602.19                      |
| 2   | 1H    | 5399.67            | (10.998, 5.999)     | 9903.36                       |

Root causes:

1. `ng.pipe.make_uc(dic, None, dim=i-1)` always raises `AttributeError: 'NoneType' object has no
   attribute 'ndim'` because `make_uc` dereferences `data.shape`. The bare `except Exception`
   swallows it, so the correct path is dead code and `plst = ORIG + SW` always wins.
2. `FDF{i}ORIG` / `FDF{i}SW` are Hz, not ppm: no division by `FDF{i}OBS`.
3. `sz` uses `FDF2FTSIZE` (4096, stale) instead of the real direct-dimension size `FDSIZE` (1279).
4. `ND` is guessed from `FDF2LABEL` / `FDF3FTSIZE` although `ng.pipe.read` returns the array whose
   `ndim` is authoritative.
5. The ~20 duplicated lines live in two scripts and only one computes `sz`/`sw`.

Observed damage in files committed by `fdead0e` (`Spectra/2D/Azurin.tab`):

```
DATA  X_AXIS N            1   258   8952.557ppm 12573.679ppm
    1  -623.784 -4786.076 ... X1 X3 = -623 -622  Y1 Y3 = -4786 -4785
```

## Non-goals

- Do not rebuild/repack the wheel (see "Follow-up" below); do not bump versions.
- Do not touch `StoP3.py`, the SavedModel loading path, or the trained models.
- Do not change the CLI surface of `P5toPeaks.py` / `PeakstoTab.py`.

## Tasks

- [x] T1 Extract the package sources from `MRAi_P3-1.1.1-py3-none-any.whl` into `src/MRAi_P3/`
      (the repository ships no package source; the wheel is the only carrier) — delegate.
- [x] T2 Add `src/MRAi_P3/headers.py` with one `read_headers(spec_path)` that returns
      `(ND, sz, sw, pfst, plst, nam)` in real ppm and real array sizes, keeping the `showhdr`
      route as primary and nmrglue as fallback — delegate.
- [x] T3 Rewire `P5toPeaks.py` and `PeakstoTab.py` to `read_headers` — delegate.
- [x] T4 Tests against real spectra in `Spectra/2D/*.ft2` asserting ppm values, sizes and that
      generated `.tab` point indices fall inside `1..size` — delegate.
- [x] T5 Parent verification (independent run of tests + regeneration comparison) and summary file
      -> `docs/peak-list-fallback-fix.md`

## Allowed edit surfaces

- `src/MRAi_P3/**`
- `tests/**`

## Status

Complete. T1–T4 delegated to `gentle-ai-worker` (rounds 1–3, session `mrai-fallback-fix`); T5 verified
by the parent.

Two rounds of defects were found by parent verification, not by the writer's own report:
- round 1 shipped a six-vs-two value unpack in `P5toPeaks` (ValueError on every `.peaks`
  conversion) and a test file that did not import;
- round 2 surfaced a pre-existing X/Y axis-order disagreement between `P5toPeaks` and `PeakstoTab`,
  escalated to the human, who decided x = 1H (direct).

Parent verification: 11/11 tests green; 40-peak-per-spectrum probe against the `.ft2` grids shows the
regenerated indices bracket real local maxima (offset <= 4 points) versus the previous -31654.

## Evidence (commits)

- none yet — committing requires explicit human approval

## Follow-up (not in scope)

- rebuild/repack the wheel: `dist-info` is `MRAi_P3-1.0.0.dist-info` with `Version: 1.1.0` and stale
  `RECORD` hashes, so consumers installing `MRAi_P3-1.1.1-py3-none-any.whl` still get the broken code
- verify the 3D label order and dim-3 calibration (no 3D spectra in the repo)
- `P3_Master.py:65` `params.get('Spec_type','None')` yields the truthy string `'None'`