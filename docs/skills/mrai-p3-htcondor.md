---
name: mrai-p3-htcondor
description: Use this skill for any work on the MRAi_P3 2D NMR spectrum project on NMRbox HTCondor: pre-extracting background pools, training 2D models, running batch inference, tuning hard-negative geometry (ring/band/aniso/axis quota), bg-ratio, and verifying pools. Covers paths, parameters, commands, gotchas, and where to read runbooks.
---

# MRAi_P3 — HTCondor workflow (NMRbox)

## 0. Entorno

- Repo: `/home/nmrbox/0070/abolanos/git/mrai_p3` (rama `install_in_mac_m1`, HEAD ~`40f85a5`).
- Python: `~/vtflow/bin/python` (3.12, TF 2.21, Keras 3.15, numpy 2.5, scipy 1.18). `scipy` es **obligatorio** para `preextract.py --hard-aniso`.
- Dataset: `/home/nmrbox/0070/abolanos/git/spectrum-generator/dataset` (1000 spectra + labels + `manifest.jsonl`). **No** existe `git/cheminfo/...` (bug histórico, ya corregido en `htcondor/preextract.sub`).
- HTCondor: `condor-ap-lts.nmrbox.org`, CPU-only, máx. 5 días.
- Init model: `Trained/Probability_2D/US/DNN_0` (SavedModel).
- Identidades: train 800 / val 200 en `data/train_2d_elong/manifest.json`.

## 1. Mapa de archivos clave

- `htcondor/preextract.sub` — template de pre-extracción. `Dataset` debe ser `.../git/spectrum-generator/dataset`.
- `htcondor/train.sub` — template de entrenamiento (`RunName Pool Extra`).
- `htcondor/submit.sh <file.sub> RunName=<nombre> [VAR=valor ...]` — inyecta overrides antes de `queue`, deja copia en `htcondor/runs/`.
- `htcondor/run_train.sh RUN_NAME [POOL] [extra args...]` — lee geometría del `manifest.json` del pool.
- `scripts/preextract.py` — genera el pool (`.npz` + `manifest.json` + `preextract.log`).
- `scripts/train_model.py` — entrena (`--preextracted --data <pool>`).
- `scripts/plotting/infer_batch.py` — inferencia en lote → `batch_report.json`.
- `scripts/pool_flags.sh <pool>` — deriva flags del manifest.
- `scripts/band_breakdown.py` — desglose de banda sobre labels completos.
- `docs/runs/2026-10-26-cupo-eje-band2-4e22c41.md` — runbook del experimento a00/a20/a25.
- `htcondor/README.md` — manual de HTCondor.

## 2. Geometría de negativos duros

Parámetros (viven en el manifest del pool, única fuente de verdad):

- `--hard-frac` (default 0.25): fracción del fondo reservada al anillo `d <= hard_px`.
- `--hard-px` (default 2): radio del anillo.
- `--hard-frac2` (default 0): fracción reservada a la banda media.
- `--hard-aniso F1LO F1HI F2LO F2HI`: banda rectangular anisótropa (reemplaza banda isotrópica). Requiere `hard_frac2 > 0`.
- `--hard-axis-frac A`: cupo del eje F1 puro (`d2 == 0`) dentro de la banda. Requiere `hard_aniso` con `F2LO = 0` y `hard_frac2 > 0`. 0 = natural (~0.16), 0.20 = 1:4, 0.25 = 1:3. Techo efectivo ≈ 0.384.
- `--pool-k` (default 10): pool de fondo = pool-k × positivos. Debe ser > `--bg-ratio` de entrenamiento.
- `--bg-ratio` (default 4, en train): negativos por época = bg_ratio × positivos.
- `--seed` (default 0).

Invariantes: `hard_frac + hard_frac2 <= 1`. Con `0.32 + 0.68 = 1.0` no queda fondo fácil.

## 3. Pre-extracción de un pool

### Comando local (smoke / verificación)
```bash
cd /home/nmrbox/0070/abolanos/git/mrai_p3
/home/nmrbox/0070/abolanos/vtflow/bin/python scripts/preextract.py \
  --dataset /home/nmrbox/0070/abolanos/git/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/<pool> --seed 0 --pool-k 24 \
  --hard-frac 0.32 --hard-px 2 --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.25 --fast
```

### Por HTCondor
```bash
htcondor/submit.sh preextract.sub RunName=<nombre> \
  Pool=data/<pool> PoolK=24 \
  Extra="--pool-k 24 --hard-frac 0.32 --hard-px 2 --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.25"
```

### Verificación de un pool (obligatoria antes de entrenar)
- `ls <pool>/*.npz | wc -l` → 1000.
- Existe `<pool>/manifest.json` y `<pool>/preextract.log` con 1000 filas.
- `scripts/pool_flags.sh <pool>` devuelve los flags esperados.
- `.err` del job vacío, exit 0.
- Capacidad por época: anillo disponible ≥ 0.32×bg_ratio×npos por espectro; banda ≥ 0.68×bg_ratio×npos. Si no, fallback silencioso.

## 4. Entrenamiento

### Por HTCondor
```bash
htcondor/submit.sh train.sub RunName=<run> Pool=<pool_sin_data/> Extra="--bg-ratio 8"
```
`Pool` va **sin** `data/`. La geometría la lee `run_train.sh` del manifest.

### Dry-run (ver comando final)
```bash
MRAI_DRY_RUN=1 htcondor/run_train.sh <run> <pool> --bg-ratio 8
```

### Defaults del wrapper
batch 65536, bg-ratio 4, epochs 210, patience 20, sync-every 5, init `Trained/Probability_2D/US/DNN_0`, scratch `/tmp/run_<run>`. `Extra` se agrega al final y pisa defaults.

### Regla crítica
Pasar `--hard-frac` y `--hard-frac2` **juntos** si se sobrescriben: si solo se pasa uno, el otro queda en el default del wrapper (0.45) y `band_sections` falla en la primera época. `train_model.py` no valida la suma al arrancar.

## 5. Inferencia en lote

```bash
/home/nmrbox/0070/abolanos/vtflow/bin/python scripts/plotting/infer_batch.py \
  --n 100 --split val \
  --dataset /home/nmrbox/0070/abolanos/git/spectrum-generator/dataset \
  --model Trained/<run>_savedmodel \
  --out Trained/infer_out_<run> \
  --spe-the 0.2 --pl-the 0.6 --msep 3 1
```
- `--dataset` es **obligatorio** en esta VM (default apunta a Mac).
- `--split val` para juzgar; `train` es optimista.
- ~3 min/100 espectros en CPU. Escribe `batch_report.json` con aggregate, p3_shape, per_id.

## 6. Métricas clave (runbook)

- `aggregate`: precision, recall, F1, det_over_gt (1.0 ideal), n_detected, n_gt.
- `p3_shape`: `f1@3px`, `f1@6px`, `f1_over_f2` (cola). Criterio del runbook: si el cupo mejora `f1@6`, alcanza; si solo mueve `f1@3`, no.
- Desglose por clase (MIX/PG/PL) desde `per_id` + `manifest.jsonl` del dataset.

## 7. Gotchas

1. `htcondor/preextract.sub` línea 16: `Dataset` debe ser `.../git/spectrum-generator/dataset` (no `git/cheminfo/...`).
2. `scipy` debe estar instalado en `~/vtflow` (preextract --hard-aniso).
3. `htcondor/runs/` está untracked y no gitignored.
4. Symlink trackeado `Trained/.../DNN_0_local_serving/DNN_0` aparece borrado en git status.
5. Logs viejos (clusters fallidos 2222668-70) quedan en `htcondor/logs/`.
6. `--pool-k` duplicado (PoolK + Extra) es inocuo.
7. `train.sub` default `RunName=out_nmrbox_454510` / `Pool=pre_aniso_f2lo0_454510` se pisan con overrides antes de `queue`.
8. Cupo techo efectivo ≈ 0.384; arriba de eso se completa con diagonal en silencio.

## 8. Dónde leer más

- `docs/runs/2026-10-26-cupo-eje-band2-4e22c41.md` — runbook del experimento a00/a20/a25 (geometría, comandos, verificación, resultados esperados).
- `htcondor/README.md` — manual HTCondor NMRbox.
- `scripts/hard_negatives.py` — `choose_background`, `aniso_band_mask`, `axis_sections`.
- `tests/test_hard_axis.py`, `tests/test_band_breakdown.py` — 28 tests.
