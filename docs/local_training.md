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
  `docs/session_checkpoint_2026-10-14.md` (Phase 1 state), and
  `odd/tasks/PLAN_split_rows.md` (measurements behind the row-wise split).
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

* `--fast` is a **no-op** now (the row-wise `split_2d_rows` is the default
  path); it is kept so the Colab notebook keeps working. `--slow-split` forces
  the reference full-matrix path (only for verification).
* **`pool_k` must be greater than the training `--bg-ratio`** (here 10 > 4) so
  each epoch draws a different background subset. With `pool_k == bg_ratio` the
  whole pool is consumed every epoch (fixed sampling; `tests/test_pool_equiv.py`).

**Output** (`data/pre_elong/`):

| File | Content |
|---|---|
| `<id>.npz` | `X` (M,128,1) float32, `y` (M,1), `n_pos`, `d` (uint8, distance to the nearest peak of each background row), `hard_px`, `H`, `W` — ~3.3 MB each, **~3.0 GB total** for 1000 spectra |
| `manifest.json` | `{source, seed, pool_k, hard_frac, hard_px, max_pts, shape, train, val}` — this is what `train_model.py --data` reads |
| `preextract.log` | CSV: `id, split, M, pos, hard, seconds` |

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
| `--bg-ratio 4` | paper-like background: M ≈ 581×5 ≈ **2905** points/spectrum → **35 train / 8 val steps** per epoch with `--batch 65536` |
| `--patience 20` | early stopping on `val_loss` (the 15-oct run never triggered it — it was still improving) |

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