# Training the model locally — pre-extraction + full retraining (step by step)

End-to-end guide to reproduce the **Phase 2** retraining of the 2D-US model for
elongated peaks **on a local machine** (no Google Colab): pre-extract the
background pool once, then run the complete 210-epoch fine-tune.

* **Repo/branch:** `mrai_p3`, `install_in_mac_m1`
* **Validated:** 2026-10-15 on a MacBook Air M1 (8 cores, 16 GB, macOS 26.6.2),
  CPU only (`GPUs: []`). Every command below was executed successfully on that
  machine (1-epoch full smoke: 35/35 steps, 62/62 weights copied from the
  fine-tune base, `val_loss 0.675`).
* **Companion docs:** `colab/COLAB.md` (same pipeline on Colab GPU),
  `docs/session_checkpoint_2026-10-14.md` (Phase 1 state), and
  `odd/tasks/PLAN_split_rows.md` (measurements behind the row-wise split).

---

## 0. What each stage does

| Stage | Script | Cost | When |
|---|---|---|---|
| **Pre-extraction** | `scripts/preextract.py` | minutes, once | whenever `--pool-k`/`--seed` change |
| **Training** | `scripts/train_model.py` | ~1–1.5 h for 210 epochs | every retrain |

**Why pre-extract?** The training set is built from cross-sections ("rows") of
each 256×2048 spectrum. The expensive `split_2d` work is *deterministic*; only
the background sampling is random. `preextract.py` runs the split **once** per
spectrum and stores a pool with ALL positives plus `pool_k × n_pos` background
pixels (fixed `--seed`). Training then subsamples that pool every epoch
(`bg_ratio` background per positive, fresh rng), so the background still varies
per epoch while the split is never recomputed. See `colab/COLAB.md` §4 for the
full rationale and pool-size table.

---

## 1. Requirements / environment

* The dataset (Phase 1 output), e.g.:
  `/Users/abolanos/git/cheminfo/spectrum-generator/dataset` —
  `spectra/*.bin` (float32 LE) + `labels/*.bin` (uint8) + `manifest.jsonl`
  (1000 spectra, ~2.5 GB).
* A Phase 1 manifest with the 800/200 train/val split:
  `data/train_2d_elong/manifest.json` (produced by `scripts/jres_ingest.py`;
  see `colab/COLAB.md` §3 if you need to regenerate it).
* A TensorFlow environment. On this Mac the working one is the **pyenv
  virtualenv `tflow`** (Python 3.13.2, TF 2.21.0, Keras 3.15.1):

```bash
cd /Users/abolanos/git/mrai_p3
pyenv activate tflow
python -c "import tensorflow as tf; print(tf.__version__)"   # -> 2.21.0
```

> **Pitfall:** the default env (`jres`) has **no TensorFlow**. Always
> `pyenv activate tflow` and invoke **`python`**, not `python3` (the shim
> resolves elsewhere).

---

## 2. Step 1 — Pre-extraction (once)

```bash
cd /Users/abolanos/git/mrai_p3
pyenv activate tflow

python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/pre_elong \
  --seed 0 --pool-k 10 --fast
```

* `--fast` is a **no-op** now (the row-wise `split_2d_rows` is the default
  path); it is kept so the Colab notebook keeps working. `--slow-split` forces
  the reference full-matrix path (only for verification).
* **`pool_k` must be greater than the training `--bg-ratio`** (here 10 > 4) so
  each epoch draws a different background subset. With `pool_k == bg_ratio` the
  whole pool is consumed every epoch (fixed sampling; `tests/test_pool_equiv.py`).

**Output** (`data/pre_elong/`):

| File | Content |
|---|---|
| `<id>.npz` | `X` (M,128,1) float32, `y` (M,1), `n_pos`, `H`, `W` — ~3.3 MB each, **~3.0 GB total** for 1000 spectra |
| `manifest.json` | `{source, seed, pool_k, max_pts, shape, train, val}` — this is what `train_model.py --data` reads |
| `preextract.log` | CSV: `id, split, M, pos, seconds` |

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
| `--init .../DNN_0` | fine-tune from the 2D-US base SavedModel (weights copied **by shape**, 62/62 on this dataset). Omit `--init` and pass `--lr 1e-3` to train from scratch |
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