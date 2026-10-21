# Training the model locally — pre-extraction + full retraining (step by step)

End-to-end guide to reproduce the **Phase 2** retraining of the 2D-US model for
elongated peaks **on a local machine** (no Google Colab): pre-extract the
background pool once, then run the complete 210-epoch fine-tune.

* **Repo/branch:** `mrai_p3`, `install_in_mac_m1`
* **Validated:** 2026-10-15 on a MacBook Air M1 (8 cores, 16 GB, macOS 26.6.2),
  CPU only (`GPUs: []`). Every command below was executed successfully on that
  machine (1-epoch full smoke: 35/35 steps, 62/62 weights copied from the
  fine-tune base **by role**, `val_loss 0.675`).
* **Companion docs:** `colab/COLAB.md` (same pipeline on Colab GPU),
  `docs/ring_and_picking_metrics.md` (what the ring and the peak-list chain
  actually do on the dataset, measured: precision/recall against the labels,
  the `msep` knob, and the comparison with the base model),
  `docs/session_checkpoint_2026-10-14.md` (Phase 1 state), and
  `odd/tasks/PLAN_split_rows.md` (measurements behind the row-wise split).
* **Pinned runbook:** `docs/runs/2026-10-21-aniso-60e9ce4.md` — the exact
  anisotropic A/B commands and flags at commit `60e9ce4`. Use it when defaults
  or options may have changed since this guide was written.
* **Executable version:** `docs/local_training.ipynb` — the same steps as a
  runnable Jupyter notebook, with each stage checking its own result. It finds
  a Python that can `import tensorflow` by itself, so it works even from the
  Jupyter env that has no TensorFlow installed (see §1).

---

## 0. What each stage does

| Stage | Script | Cost | When |
|---|---|---|---|
| **Pre-extraction** | `scripts/preextract.py` | minutes, once | whenever `--pool-k`/`--seed`/`--hard-frac` change |
| **Training** | `scripts/train_model.py` | ~1–1.5 h for 210 epochs | every retrain |

**Why pre-extract?** The training set is built from cross-sections ("rows") of
each 256×2048 spectrum. The expensive `split_2d` work is *deterministic*; only
the background sampling is random. `preextract.py` runs the split **once** per
spectrum and stores a pool with ALL positives plus the background budget
(`pool_k × n_pos` rows, fixed `--seed`). Training then subsamples that pool every
epoch (`bg_ratio` background per positive, fresh rng), so the background still
varies per epoch while the split is never recomputed. See `colab/COLAB.md` §4 for
the full rationale and pool-size table.

**The background budget is not uniform.** The label is a sub-pixel marker (1/2/4
px per peak, 2.26 on average) while a peak spans ~20–50 px, so the zeros *inside*
a peak are where the false positives live. Sampling the background uniformly over
all zeros put them at 0.48 % of the pool: per epoch and spectrum, 578 positives
against ~11 negatives 1 px away and ~26 within 2 px, with 96.7 % of the
background ≥5 px from any peak (where the model is already almost perfect). The
measured consequence on the trained model: **81 % of the zeros right next to a
peak got P > 0.5**, the learned boundary sat at ~2–3 px instead of the label's
~0.5–1 px, and the peak list came out ~7.7× too long (1 966 picks where there
were 256). `--hard-frac` reserves a share of the background for that ring
(`d <= --hard-px`), and every pool row stores its distance to the nearest peak in
`d` so the per-epoch share can be tuned at training time without regenerating the
pool.

> **Did it work, and what limits the peak list now?** Measured against the
dataset labels: the ring P > 0.5 fell from 84 % to 7.7 % of the ring pixels and
the peak list from 7.7× to 1.22× the truth — but about half of the peaks that
remain are duplicates on the F1 ridge of ONE real peak, and the candidate mask
(`--spe-the`) throws away the weak peaks. The numbers, the `msep` knob that
removes the duplicates, and the comparison against the base model are in
`docs/ring_and_picking_metrics.md`.

---

## 1. Requirements / environment

* The dataset (Phase 1 output), e.g.:
  `/Users/abolanos/git/cheminfo/spectrum-generator/dataset` —
  `spectra/*.bin` (float32 LE) + `labels/*.bin` (uint8) + `manifest.jsonl`
  (1000 spectra, ~2.5 GB).
* A Phase 1 manifest with the 800/200 train/val split:
  `data/train_2d_elong/manifest.json` (produced by `scripts/jres_ingest.py`;
  see `colab/COLAB.md` §3 if you need to regenerate it).
* A TensorFlow environment. **Any Python 3.10–3.13 environment with
  TensorFlow ≥ 2.15 works** (arm64-native wheel) — the `mrai_p3` env from the
  README *Installation* section qualifies. The `MRAi_P3` wheel is **not**
  required for training: `scripts/*.py` only need TensorFlow + NumPy and this
  checkout. On this Mac the working env is the dedicated **pyenv virtualenv
  `tflow`** (Python 3.13.2, TF 2.21.0, Keras 3.15.1). If you don't have it
  yet:

```bash
pyenv install 3.13                 # if that Python isn't installed yet
pyenv virtualenv 3.13 tflow        # needs the pyenv-virtualenv plugin
                                   # (brew install pyenv-virtualenv)
pyenv activate tflow
python -m pip install --upgrade pip
python -m pip install tensorflow   # >=2.15, native arm64
```

Then verify the environment **and** the training scripts, from the repository
folder:

```bash
cd /Users/abolanos/git/mrai_p3
pyenv activate tflow                # or: conda activate mrai_p3
python -c "import tensorflow as tf; print(tf.__version__)"   # -> 2.21.0
python scripts/preextract.py --help    # prints usage = ready
python scripts/train_model.py --help   # prints usage = ready
```

> **Pitfall:** on this machine the default env (`jres`) has **no
> TensorFlow**. Always activate the TF env explicitly (`pyenv activate
> tflow`, or `conda activate mrai_p3` if you built it per the README) and
> invoke **`python`**, not `python3` (the shim resolves elsewhere).

---

## 2. Step 1 — Pre-extraction (once)

```bash
cd /Users/abolanos/git/mrai_p3
pyenv activate tflow

python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/pre_elong \
  --seed 0 --pool-k 10 --hard-frac 0.25 --hard-px 2 --fast
```

* `--hard-frac` (default **0.25**) reserves that share of the pool's background
  for negatives within `--hard-px` px of a peak; `--hard-frac 0` reproduces the
  old uniform background byte for byte (the pinned pool in
  `tests/test_split_rows.py`).
* `--hard-band2` / `--hard-frac2` add the isotropic middle band
  (`--hard-px < d <= --hard-band2`) with its own share of the background. With
  `--hard-frac2 0` the draw is byte-identical to the old one.
* `--hard-aniso F1LO F1HI F2LO F2HI` replaces `--hard-band2` with a rectangular
  band measured per axis: `F1LO <= d1 <= F1HI` (rows) **and**
  `F2LO <= d2 <= F2HI` (columns), where `d1`/`d2` are the row/column distance to
  the nearest positive. It requires `--hard-frac2 > 0`, is mutually exclusive
  with `--hard-band2`, and stores `d1`/`d2` in the npz so
  `train_model.py --hard-aniso` can resample the band each epoch. Measured
  starting point: `--hard-aniso 3 10 1 3 --hard-frac2 0.25` (F1 tail 3–10 px,
  F2 1–3 px; see `docs/ring_and_picking_metrics.md` §14).

* `--pair-px` (default **0** = off) marks **tight pairs** from
  `<dataset>/manifest.jsonl`: a peak is marked when another, *larger* peak sits
  at euclidean distance <= `--pair-px` px (6 in the measured recipe) with
  `amps[own]/amps[other] <= --pair-ratio` (default 0.5; amplitude ties never
  count). The pair cannot be derived from the cross profile (a diagonal
  neighbour leaves no trace on either arm — measured sensitivity 14 %, 24 of
  176 buried peaks), which is why it comes from the manifest. With the flag > 0
  the npz gains `pair` (uint8, one byte per pool row) plus `pair_px`/
  `pair_ratio`, and `preextract.log` gains a `pairs` column; with **0** the npz
  and log are exactly as before. `--pair-frac` in Step 2 consumes those marks
  (`docs/ring_and_picking_metrics.md` §11).

* `--fast` is a **no-op** now (the row-wise `split_2d_rows` is the default
  path); it is kept so the Colab notebook keeps working. `--slow-split` forces
  the reference full-matrix path (only for verification).
* **`pool_k` must be greater than the training `--bg-ratio`** (here 10 > 4) so
  each epoch draws a different background subset. With `pool_k == bg_ratio` the
  whole pool is consumed every epoch (fixed sampling; `tests/test_pool_equiv.py`).

**Output** (`data/pre_elong/`):

| File | Content |
|---|---|
| `<id>.npz` | `X` (M,128,1) float32, `y` (M,1), `n_pos`, `d` (uint8, distance to the nearest peak of each background row), `hard_px`, `H`, `W`; with `--hard-aniso` also `d1`/`d2` (uint8, row/column distance per background row) — ~3.3 MB each, **~3.0 GB total** for 1000 spectra |
| `manifest.json` | `{source, seed, pool_k, hard_frac, hard_px, hard_band2, hard_frac2, hard_aniso, max_pts, shape, train, val}` — this is what `train_model.py --data` reads |
| `preextract.log` | CSV: `id, split, M, pos, hard, band2, seconds` (+`pairs` with `--pair-px`) |

> **Pools written before the ring existed have no `d`.** `train_model.py` refuses
to run with `--hard-frac > 0` on such a pool (it would silently train with the
uniform background); regenerate it, or pass `--hard-frac 0`.

M (pool rows) = `n_pos + pool_k × n_pos` ≈ 5600–7000 (measured avg **6334**).

**Timing (measured):** each spectrum takes well under 0.1 s on the row-wise
path (the whole 1000-spectrum run wrote `0.0 s` per row in `preextract.log`) —
**roughly half a minute to a couple of minutes** for all 1000 depending on disk
cache. For reference, the old full-matrix path took 219 s
(`odd/tasks/PLAN_split_rows.md` §1).

> **Skip this step** if `data/pre_elong/` already contains 1000 `.npz` +
> `manifest.json` with the `--pool-k`/`--seed` you want (it does in this
> checkout: seed 0, pool_k 10). The pool is *derived data* — it can always be
> regenerated from the `.bin` dataset.

---

## 3. Step 2 — Full training (210 epochs)

```bash
cd /Users/abolanos/git/mrai_p3
pyenv activate tflow

nohup python scripts/train_model.py \
  --data data/pre_elong --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --epochs 210 --patience 20 \
  --out Trained/Probability_2D_Elongated/DNN_0_local \
  > /tmp/retrain_fase2.log 2>&1 &

tail -f /tmp/retrain_fase2.log          # or: tail -f <out>/training.log
```

Key flags:

| Flag | Meaning |
|---|---|
| `--preextracted` | read the `<id>.npz` pools from `--data` instead of `.bin` + on-demand split; `--bg-ratio` controls the per-epoch subsampling |
| `--init .../DNN_0` | fine-tune from the 2D-US base SavedModel. The weights are copied **by role** (`scripts/weight_roles.py`, derived from the base's graph): 62/62, and copying them reproduces the base to `maxdiff < 1e-6` (`tests/test_init_weights.py`). Omit `--init` and pass `--lr 1e-3` to train from scratch |
| `--hard-frac` | share of each epoch's background drawn from the ring (`d <= --hard-px`) instead of uniformly; default 0.25, 0 = old uniform background |
| `--hard-px` | radius in px of the ring treated as a hard negative (default 2) |
| `--hard-band2` | outer radius of the isotropic middle band in px (0 = off); the pool must be generated with the same flag |
| `--hard-frac2` | share of each epoch's background drawn from the middle band; 0 = off |
| `--hard-aniso F1LO F1HI F2LO F2HI` | rectangular band per axis (F1 = rows, F2 = columns) replacing `--hard-band2`; requires `--hard-frac2 > 0` and a pool generated with `preextract.py --hard-aniso` (with `d1`/`d2`); measured start `3 10 1 3` |
| `--pair-frac` | per-epoch oversampling of **tight pairs** (peaks buried in a bigger neighbour's flank): adds `round(pair_frac × n_pos)` rows drawn with replacement from the pool rows marked `pair > 0`. Default 0.0 = off (byte-identical to before). Requires `--preextracted` and a pool written with `preextract.py --pair-px > 0` — it refuses to start otherwise; the background budget (`n_bg`) stays computed on the original `n_pos`, so the chosen negatives do not change |
| `--bg-ratio 4` | paper-like background: M ≈ 581×5 ≈ **2905** points/spectrum → **35 train / 8 val steps** per epoch with `--batch 65536` |
| `--patience 20` | early stopping on `val_loss` (the 15-oct run never triggered it — it was still improving) |

> **Recommended today (measured 2026-10-17, 8 dataset spectra):** train on the
> **two-band** pool and run the chain with `--spe-the 0.2 --pl-the 0.5 --msep 3`.
> That is F1 **0.836** (precision 0.92, recall 0.76, 1690 peaks for 2048 real)
> against 0.615 for the previous model with the chain as it was. Why each knob:
> `--msep` collapses the F1 ridge of one peak (`P3toP5.py`), `--pl-the` removes
> the far false positives (the model gives them P ≈ 0.23 vs 0.99 for real peaks),
> `--spe-the 0.2` lets the weak peaks into the candidate mask. Full numbers, the
> train/val split and the failed experiments: `docs/ring_and_picking_metrics.md`
> §4, §9 and §10. The defaults are still the old ones (1 / 0.2 / 0.5) so nothing
> changes silently.

**Timing (measured on the M1, CPU):** ~0.485 s/step × 35 steps ≈ **~17 s per
epoch** → 210 epochs ≈ **1–1.5 h** (consistent with `colab/COLAB.md` §4:
14.7–15 s/epoch on the earlier local run).

**Outputs** in `--out`:

* `ckpt.keras` — best model by `val_loss` (`ModelCheckpoint(save_best_only)`)
* `training.log` — CSV, one row per epoch
* `history.npy` / `history.txt` — loss/val_loss arrays
* `<out>_savedmodel/` — `model.export()` SavedModel (what StoP3 consumes),
  written at the end of training
* plus a final `predict smoke` line as an inference sanity check

### Looking at the result on the training data

`scripts/plotting/run_train_2d.py` runs the same inference and peak chain as
the folder-51 workflow (`run_51.py`: `P3toP5` -> `P5toPeaks` -> `PeakstoTab`),
but on one of the training spectra, so you can see what the model does on data
it has already seen:

```bash
python scripts/plotting/run_train_2d.py --id 1 \
    --model Trained/Probability_2D_Elongated/DNN_0_local_savedmodel
node scripts/plotting/view2d.mjs --out /tmp/train_p3/1/view2d
```

It writes the peak list, a PNG (spectrum + P3 + ground truth + detections) and
a match report against `dataset/manifest.jsonl`. Measured on spectrum 1 with
the elongated model: **recall 0.617 / precision 0.735**, and the candidate
mask dominates recall — `--spe-the 0.2` reaches recall 0.828 but drops
precision to 0.640. The model also calls **84 % of the candidate pixels in the
2-pixel ring around a label a peak** (`P > 0.5`), which is the over-detection
`--hard-frac` was introduced to fight.

---

## 4. Step 3 — Monitor and (if needed) resume

The run writes `ckpt.keras` + `training.log` on every improvement/epoch, so a
crash or reboot costs at most the current epoch. To continue **exact weights**:

```bash
cd /Users/abolanos/git/mrai_p3 && pyenv activate tflow
# move the partial log aside so the new CSVLogger starts clean
mv Trained/Probability_2D_Elongated/DNN_0_local/training.log \
   Trained/Probability_2D_Elongated/DNN_0_local/training_part1.log

python scripts/train_model.py \
  --data data/pre_elong --preextracted \
  --resume Trained/Probability_2D_Elongated/DNN_0_local/ckpt.keras \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --epochs 210 --patience 20 \
  --out Trained/Probability_2D_Elongated/DNN_0_local
```

> Use **`--resume`**, never `--init`, to continue a run: `--init` would
> re-seed from the base SavedModel and restart from scratch.

---

## 5. Smoke tests (fast validation before spending an hour)

A 1-epoch run over the **full** dataset (~30 s including startup) is the best
check of the exact production command:

```bash
python scripts/train_model.py \
  --data data/pre_elong --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --epochs 1 --out /tmp/smoke_full
```

For a ~10 s mini smoke with a subset, **shrink `--batch` too**:

```bash
python scripts/train_model.py \
  --data data/pre_elong --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 4096 --bg-ratio 4 \
  --epochs 1 --n-train 8 --n-val 4 --out /tmp/smoke_mini
```

> **`_StepGuard` caveat:** an epoch must fill **at least one full batch**.
> Partial trailing batches are discarded by design, so a tiny subset with a
> huge batch (e.g. `--n-train 8 --batch 65536` → ~23k samples < 65536) yields
> 0 batches and training aborts with
> `RuntimeError: epoca 0: 0 batches de entrenamiento…`. That is the safety net
> for the 15-oct silent empty-epoch failure (`colab/COLAB.md` §8) doing its
> job — not a pipeline bug. Always pair `--n-train` reductions with a
> proportionally smaller `--batch`.

---

## 6. Pitfalls checklist

1. **Wrong env** — `pyenv activate tflow`; use `python`, not `python3`.
2. **Reusing an existing `--out`** — `CSVLogger` overwrites `training.log` and
   `ModelCheckpoint` overwrites `ckpt.keras` in place. Point `--out` at a fresh
   directory for a new run (here: `DNN_0_local`, keeping the 15-oct artifacts
   in `Trained/out_elong/` untouched).
3. **`--scratch DIR` / `--sync-every N` are Colab-only** — they keep variable
   Google Drive I/O out of the epoch loop (`colab/COLAB.md` §7). Locally
   `--out` is already on disk; omit them.
4. **Why re-launch instead of resuming the 15-oct Colab run**
   (`Trained/out_elong/`) — that run declared 210 epochs but only **105
   trained** (Keras 3 `from_generator`/`steps_per_epoch` empty-epoch bug, since
   fixed with `.repeat()` + `_StepGuard`). `colab/COLAB.md` §9 recommends
   re-launching the full run from scratch with the fix.
5. **Disk** — the pool is ~3 GB; keep enough free space (it is derived data and
   can be regenerated at any time).

---

## 7. Quick reference (copy-paste)

```bash
cd /Users/abolanos/git/mrai_p3 && pyenv activate tflow

# 1) pre-extraction (only if data/pre_elong is missing or params changed)
python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/pre_elong --seed 0 --pool-k 10 --fast

# 2) full 210-epoch fine-tune
nohup python scripts/train_model.py \
  --data data/pre_elong --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --epochs 210 --patience 20 \
  --out Trained/Probability_2D_Elongated/DNN_0_local \
  > /tmp/retrain_fase2.log 2>&1 &
tail -f /tmp/retrain_fase2.log
```

The finished model for inference is the exported SavedModel at
`<out>_savedmodel/` (plus the best-weights `ckpt.keras` inside `<out>/`).

### 7.1 The measured recipe (two bands + chain tuning)

This is the configuration with the best numbers so far (§9–§10 of
`docs/ring_and_picking_metrics.md`): the background pool reserves a second band
for the 3–6 px ring (where the shoulder false positives live, sampled at 0.5 %
before) and the peak list uses `msep` plus a higher P threshold.

```bash
cd /Users/abolanos/git/mrai_p3 && pyenv activate tflow

# 1) two-band pool (~6.9 GB for 1000 spectra; the old one is 3.3 GB)
python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json --out data/pre_elong_2b \
  --seed 0 --pool-k 24 --hard-frac 0.55 --hard-px 2 --hard-band2 6 --hard-frac2 0.25

# 2) fine-tune (the flags must match the pool's bands)
nohup python scripts/train_model.py \
  --data data/pre_elong_2b --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --hard-frac 0.55 --hard-px 2 --hard-band2 6 --hard-frac2 0.25 \
  --epochs 210 --patience 20 --out Trained/out_elong_2b \
  > /tmp/retrain_2b.log 2>&1 &

# 3) the peak list (this is where F1 0.615 -> 0.836 comes from)
python scripts/plotting/run_train_2d.py --id 1 \
  --model Trained/out_elong_2b_savedmodel \
  --spe-the 0.2 --pl-the 0.5 --msep 3
python scripts/bruker_2d/run_51.py --msep 3 --pl-the 0.5
```

Tight pairs (§11 of `docs/ring_and_picking_metrics.md`): add `--pair-px 6` to
the pool command and `--pair-frac 0.25` to the fine-tune — that is the lever
against the false negatives buried in a bigger peak's flank (measured ceiling:
F1 0.836 → 0.853 recovering half of the 110 strong FNs, 0.869 recovering all,
at ~+5 % per epoch and ~14 KB/spectrum of pool).

### 7.2 Batch inference — replicate this

`scripts/plotting/infer_batch.py` runs the production chain on **many spectra
with the model loaded once**: ~1.4 s per spectrum instead of ~7 s (one TF import
+ model load per id). It is the same maths as `run_train_2d.py` (`predict_P3` →
`p3_tail` → `P3toP5` → `P5toPeaks` → `PeakstoTab`), so the `.tab` it writes for
an id and a chain is the same file `run_train_2d.py` would write.

**Requirements**: the TF env (`~/.pyenv/versions/tflow/bin/python`), the dataset,
the split manifest (`data/train_2d_elong/manifest.json`) and a SavedModel dir.
Nothing else — no pool, no training.

```bash
cd /Users/abolanos/git/mrai_p3

# 100 held-out spectra: the honest number
~/.pyenv/versions/tflow/bin/python scripts/plotting/infer_batch.py \
  --n 100 --split val --model Trained/out_elong_2b_savedmodel \
  --out /tmp/inf100_val

# 100 spectra seen in training (optimistic), pair-oversampled model
~/.pyenv/versions/tflow/bin/python scripts/plotting/infer_batch.py \
  --n 100 --split train --model Trained/out_pair_savedmodel \
  --out /tmp/inf100_pares

# the whole train split with the defaults (800 spectra, ~20 min, 1.6 GB of maps)
~/.pyenv/versions/tflow/bin/python scripts/plotting/infer_batch.py --n 800 --out /tmp/inf_train
```

The defaults are already the measured chain (`--spe-the 0.2`, `--pl-the 0.6`,
`--msep 3 1`, §12); pass them explicitly if you want the command to be
self-documenting. `--help` is the always-current flag reference.

**What it leaves in `--out`**:

```
<out>/0001.tab                peak list, VARS format (what PeakstoTab writes)
<out>/0001/view2d/p3.bin      P3 map, float32 LE (F1 rows x F2 cols), 2 MB
<out>/0001/view2d/meta.json   {"shape": [256, 2048], "p3": "p3.bin", ...}
<out>/batch_report.json       chain used, aggregate metrics, per-id counts,
                              p3_shape (median F1/F2 decay at 3/6 px and
                              their ratio), failures and elapsed seconds
```

Intermediates (`ft2`, `P3`, `P5`, `peaks`) are removed per id; `--keep` retains
them. `--no-p3-maps` skips the 2 MB maps (only if you do not need the viewer's
P3 layer — see below).

| flag | default | what |
|---|---|---|
| `--n` | 100 | how many spectra |
| `--split` | `train` | `train` = seen in training (optimistic), `val` = held out |
| `--model` | first `MODEL_CANDIDATES` | SavedModel dir |
| `--out` | required | output dir (tabs + maps + report) |
| `--dataset` | the local dataset | spectra/labels root |
| `--ids-file` | `data/train_2d_elong/manifest.json` | split manifest |
| `--spe-the` | 0.2 | candidate mask, `|S| >= the * std(S)` (needs the network) |
| `--pl-the` | 0.6 | P3toP5 threshold (picking only) |
| `--msep` | `3 1` | minimum separation per axis (picking only) |
| `--spec-type` | `JRES` | passed to `P5toPeaks` |
| `--match-px` | 2.0 | matching tolerance for the reported metrics |
| `--keep` | off | keep `ft2/P3/P5/peaks` |
| `--no-p3-maps` | off | do not write `<out>/<id>/view2d/p3.bin` |
| `--fail-fast` | off | abort on the first failing spectrum |

**Timing** (M1, measured): 100 spectra ≈ **165 s** with `out_elong_2b` and
**137 s** with `out_pair`; the maps add ~0.02 s and 2 MB per spectrum. Disk for
100 spectra: ~4 MB of tabs + 200 MB of maps.

**Look at it** — one server per inference dir, side by side:

```bash
node scripts/plotting/view2d.mjs \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --inference /tmp/inf100_bandas --port 8787    # -> http://localhost:8787
node scripts/plotting/view2d.mjs \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --inference /tmp/inf100_pares  --port 8788    # -> http://localhost:8788
```

* the sidebar has a **«solo con inferencia»** checkbox (checked by default): it
  lists only the ids that have a `.tab` in `--inference`;
* the map selector offers **Espectro / P3 probability** — the second one needs
  `<out>/<id>/view2d/p3.bin`, so it is dead if you ran with `--no-p3-maps`;
* blue circles = dataset labels, red crosses = picks; the info panel shows the
  per-spectrum counts (truth / labels px / inference).

**Reference numbers** (first 100 spectra of `train`, 25 600 real peaks, chain
`0.2 / 0.6 / 3 1`):

| model | peaks | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|
| `out_elong_2b_savedmodel` | 20 813 | 0.81x | 0.944 | 0.767 | **0.846** |
| `out_pair_savedmodel` | 22 421 | 0.88x | 0.896 | 0.784 | 0.836 |

The ordering matches the 200 held-out ids (§13 of
`docs/ring_and_picking_metrics.md`). **Use `--split val` before drawing
conclusions: `--split train` always looks better.**

**Troubleshooting**:

| symptom | cause |
|---|---|
| crosses only on one spectrum, the rest empty | `--inference` points at a single-id dir (e.g. `/tmp/v_out_pair_1`), not at a batch dir |
| «P3 probability» disabled for every spectrum | no `<out>/<id>/view2d/p3.bin` (ran with `--no-p3-maps`), or a *flat* dir where `findP3` needs `report.json` with that `id` |
| the viewer serves an old page | restart the server (it reads the manifest and caches `hasP3` at startup) and hard-reload (Cmd+Shift+R) |
| ids look non-contiguous (`1,2,…,7,9,…`) | val ids are not in the `train` list — that is the split, not a bug |
| a `.tab` looks shifted by 1 px | read it with `run_train_2d.read_tab` (it inverts `X_PPM`/`Y_PPM`, lands at 0.00 px); the `X1/Y1` columns are 1 px off |

**Where the rest of the documentation lives**: this file is the how-to
(`§7.1` the measured training recipe, `§7.2` this batch flow);
`docs/ring_and_picking_metrics.md` holds the measurements and the decisions
(§2 ring, §4 levers, §9 two bands, §11 tight pairs, §12 chain sweep, §13 the
pair experiment and its verdict); `colab/COLAB.md` mirrors the training flow for
Colab.