# NMRbox / HTCondor (`jobs/`)

Entrenamiento de `scripts/train_model.py` en el pool de NMRbox, siguiendo el
tutorial oficial: <https://github.com/NMRbox/htcondor-tutorial>.

| File | What it is |
|---|---|
| `run_train.sh` | entry point: resuelve el intérprete, lee la geometría del pool, arma el comando y reanuda si hay `ckpt.keras` |
| `train.sub` | submit file de HTCondor (recursos, logs, cola) |
| `logs/` | un `.log` / `.out` / `.err` por proceso (`$(Cluster).$(Process)`) |

> `htcondor/` es la variante anterior (templates por etapa + `submit.sh`).
> `jobs/` la reemplaza para entrenamiento: un solo wrapper, geometría leída del
> `manifest.json` y reanudación automática. No mezcles las convenciones de
> salida: acá es `Trained/<RUN_NAME>`, allá `Trained/out_<RunName>`.

## Una vez

```bash
# 1. entorno: Python 3.12 + TF >= 2.16 (Keras 3). Ya existe en ~/vtflow.
~/vtflow/bin/python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"

# si hubiera que recrearlo
python3.12 -m venv ~/vtflow
~/vtflow/bin/pip install --upgrade pip
~/vtflow/bin/pip install "tensorflow>=2.16,<3" numpy

# 2. USERDIR en train.sub ya apunta a este checkout
#    (/home/nmrbox/0070/abolanos/git/mrai_p3); cambialo si movés el repo.

# 3. chequeo sin entrenar (imprime el comando exacto)
MRAI_DRY_RUN=1 ./jobs/run_train.sh out_test pre_aniso_f2lo0_454510

# 4. humo real: con batch 65536 hacen falta ~23 espectros para que una época
#    produzca al menos un batch (con menos, el step guard aborta a propósito)
./jobs/run_train.sh out_smoke pre_aniso_f2lo0_454510 --n-train 30 --n-val 5 --epochs 1
```

Intérprete que usa `run_train.sh`, en orden:

1. `MRAI_PYTHON` (path absoluto o nombre en el `PATH`), si está definido.
2. `~/vtflow/bin/python`.
3. `python3` del `PATH`, con un `WARN` (puede no tener TensorFlow).

Como `train.sub` usa `getenv = True`, alcanza con exportar la variable antes de
enviar el job:

```bash
export MRAI_PYTHON=/otro/venv/bin/python
condor_submit jobs/train.sub
```

Nota: `train_model.py` y `preextract.py` solo necesitan `numpy` y
`tensorflow`; `nmrglue`, `scipy` y `matplotlib` los usan `src/MRAi_P3`
(cabeceras, P3toP5), `scripts/plotting/` e inferencia.

## Cada vez que querés mandar un trabajo

```bash
condor_q                                   # ¿hay algo corriendo?
condor_submit jobs/train.sub               # imprime el cluster id
condor_q                                   # estado
condor_q -better 12.0                      # si quedó Idle, por qué
condor_ssh_to_job 12.0                     # shell en el nodo, para debug
condor_rm 12                               # cancelar
```

Para otro experimento, no dupliques archivos: cambiá `arguments` (o agregá
otro bloque `queue`) y volvé a hacer `condor_submit`. Con más de un `queue`,
cada proceso escribe sus propios logs porque llevan `$(Process)`.

## Qué garantiza el wrapper

- **Intérprete**: falla al arrancar si `MRAI_PYTHON` no es ejecutable, en vez
  de morir más tarde en `import tensorflow`.
- **Geometría del pool**: `--hard-frac`, `--hard-px`, `--hard-frac2`,
  `--hard-aniso` / `--hard-band2` salen del `manifest.json` del pool. No se
  pueden desincronizar pool y training.
- **Reanudación**: si `Trained/<run>/ckpt.keras` existe, agrega `--resume`.
  Si existe `.done`, se niega a correr (elegí otro `RUN_NAME`). Al reanudar,
  `training.log` se siembra con la historia previa y `CSVLogger` appendea
  (`append=args.resume`), así que no se pierden las épocas anteriores.
- **Checkpoints**: `--scratch` apunta al disco local del nodo y sincroniza a
  `Trained/<run>` cada 5 épocas. Si HTCondor evicta el job, el reenvío reanuda.
- **Staging**: copia el pool (7.0 GB) a `$_CONDOR_SCRATCH_DIR` si hay al menos
  1 GB extra libre; si no, lee desde `$HOME`. Se puede apagar con
  `MRAI_STAGE=0`. Si `_CONDOR_SCRATCH_DIR` no estuviera definido, usa `/tmp`.
- **`TMPDIR`**: `getenv = True` hereda el `TMPDIR` de la VM de submit (acá
  `/scratch/0070/abolanos`, que no existe en los workers). Si el heredado no es
  escribible, el wrapper usa el scratch del job.

## Recursos y política

- `Threads = 8` en `train.sub` alimenta a la vez `request_cpus` y
  `MRAI_THREADS`; un solo valor, sin desincronización. El tutorial recomienda
  jobs de 1 core (los multicore quedan detrás de la cola single-core), pero TF
  rinde mucho mejor con varios hilos: es un trade-off consciente.
- `request_memory = 8GB`. Si ves `Abnormal termination (signal 9)` en el
  `.log`, subila. Medí con `condorize --timeout 120 -- python ...`.
- `request_disk = 16GB`: el pool stageado son 7.0 GB + checkpoints.
- `+Production = True` es necesario: `~/vtflow` **no** es autocontenido
  (`pyvenv.cfg: home = /usr/bin`, stdlib del sistema).
- 48 h máximo por job; checkpointing obligatorio arriba de 48 h (acá ya lo hay
  cada 5 épocas).
- Jobs grandes van **solo** por HTCondor, no por VNC/SSH.
- Máximo 3 conexiones VNC/SSH simultáneas a las VMs.
