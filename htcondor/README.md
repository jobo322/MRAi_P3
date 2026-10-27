# Enviar extracciones y entrenamientos a HTCondor (NMRbox)

Guía corta y repetible para esta VM. El tutorial oficial está en
<https://github.com/NMRbox/htcondor-tutorial>; acá queda solo lo que aplica a
`mrai_p3` y el checklist de cada envío.

---

## 0. Estado de esta VM (comprobado 2026-10-26, `antimony`)

| Cosa | Estado |
|---|---|
| Cliente HTCondor | **instalado y funcionando** (`condor_q` conecta a `condor-ap-lts.nmrbox.org`); además hay un cliente **portátil** en `~/htcondor-portable` (§1.1) |
| `sudo` | pide contraseña; no hay regla en `/etc/sudoers.d` para el usuario |
| `/home/nmrbox/0070/abolanos` | NFS **compartido con todo el pool** → acá van código, datos y salidas |
| `/scratch/0070/abolanos` | disco **local de esta VM** → no lo ve ningún worker |
| Dataset | sí: `git/cheminfo/spectrum-generator/dataset` |
| `~/vtflow` | Python 3.12, **TF 2.21 / Keras 3.15** → entorno que usan los `.sub` |
| `python3.12` | instalado en `/usr/bin`; `pip` llega a PyPI (probado: baja TF 2.20) |

Consecuencia: el cliente y el venv ya están listos; solo queda elegir el
experimento y enviar el job. El cliente portátil (§1.1) sigue disponible como
fallback.

---

## 1. La solicitud (una sola vez)

El cliente no se instala a mano: el paquete de NMRbox
(`nmrbox-condor` + `nmrbox-condor-lts`) baja la configuración del pool desde
`/nmrbox-system/condor/lts/` y arranca el servicio. Sin `sudo` eso lo tiene que
hacer soporte.

Mandá esto a **support@nmrbox.org** (o `support@nmrhub.org`):

> Asunto: Enable HTCondor on my NMRbox VM (beryllium)
>
> Hola,
> estoy trabajando en la VM `beryllium.nmrbox.org` con el usuario `abolanos` y
> quiero enviar jobs al pool de HTCondor. El cliente no está instalado
> (`condor_q: command not found`). ¿Podrían instalar/habilitar
> `nmrbox-condor` y `nmrbox-condor-lts` en esta VM, o indicarme el trámite para
> una VM con HTCondor habilitado?
> Gracias.

Si en cambio tenés root en tu VM, el equivalente es:

```bash
sudo apt update
sudo apt install nmrbox-condor nmrbox-condor-lts
# NO instalar el paquete `condor` pelado: la config del pool la pone nmrbox-condor-lts
sudo systemctl status condor
```

Verificación (cualquiera de las dos vías):

```bash
condor_q                       # tu cola
condor_status | tail -5        # nodos del pool
```

Si `condor_q` anda pero la cola está vacía y querés GPU, pedila en el mismo
correo (el pool tiene T4/V100/A100; ver §7).

### 1.1 Cliente portátil ya instalado (pool **local**, no el de NMRbox)

Probado el 2026-10-26 sin root con el tarball oficial de HTCondor:

| | |
|---|---|
| Binarios | `~/htcondor-portable/condor-23.10.29-1+ubu20-x86_64_Ubuntu20-stripped/` (182 MB) |
| Config | `~/htcondor-portable/condor_config` |
| Estado (log/spool/execute) | `/scratch/0070/abolanos/condor-local/` |
| Arrancar | `source ~/htcondor-portable/activate.sh` |
| Parar | `condor_off -master` |

Es un *personal condor*: `MASTER, COLLECTOR, NEGOTIATOR, SCHEDD, STARTD` en
`127.0.0.1`. Sirve para encolar y correr jobs **en esta VM** (65 cores,
629 GB) con `condor_q`, logs, `condor_rm`, `condor_ssh_to_job`, etc. Verificado:
un job con `/bin/hostname` y otro con `scripts/preextract.py --help` terminaron
con exit 0; `condor_status` ve `slot1@beryllium` con toda la RAM.

**No es el pool de NMRbox.** La config pública del pool
(`/nmrbox-system/condortest/10_pool.conf`) apunta a
`CONDOR_HOST = htcondor-cm.nmrbox.org`, que **no resuelve** desde esta VM (ni
por el DNS interno), y la config viva en `/nmrbox-system/condor/lts` es
solo-root. Sin esa dirección (y sin que acepten un schedd externo en el pool),
el cliente portátil no puede anunciarse. Cuando soporte instale
`nmrbox-condor`, este cliente se descarta; si preferís el camino rootless,
pediles en el mismo correo el `CONDOR_HOST` actual y si aceptan un access point
sin `startd`.

> **Versión:** 23.10.29. El `--download` del script oficial falla con **404**
> para `25.x`/Ubuntu 20.04: el canal nuevo ya no publica tarballs para focal.
> `23.x` es el más nuevo con build `ubu20` (protocolo compatible con el pool
> 24 LTS). Por eso el tarball se bajó de
> `.../tarball/23.x/current/condor-x86_64_Ubuntu20-stripped.tar.gz`.

Mientras tanto, los `.sub` de este directorio funcionan igual contra el pool
local: `+Production = True` queda como atributo sin uso (el startd local tiene
`use POLICY : ALWAYS_RUN_JOBS`).

---

## 2. Entorno Python (ya instalado)

Los scripts necesitan **Keras 3** (`keras.ops`), o sea **TF ≥ 2.16** (un venv
con TF 2.13 falla). El entorno vigente está en el FS compartido, que es lo que
ven todos los nodos:

```bash
~/vtflow/bin/python -c "import tensorflow as tf, keras; print('tf', tf.__version__, 'keras', keras.__version__)"
```

Si hubiera que recrearlo:

```bash
cd ~/git/mrai_p3

python3.12 -m venv --copies ~/vtflow
~/vtflow/bin/pip install --upgrade pip
~/vtflow/bin/pip install "tensorflow>=2.16,<3" numpy

# comprobación (las tres tienen que pasar)
~/vtflow/bin/python -c "import tensorflow as tf, keras; print('tf', tf.__version__, 'keras', keras.__version__)"
~/vtflow/bin/python scripts/preextract.py --help   >/dev/null && echo preextract OK
~/vtflow/bin/python scripts/train_model.py --help  >/dev/null && echo train OK
```

El path `~/vtflow/bin/python` es el que usan los `.sub` de `htcondor/`.
Si preferís otro nombre, editá la línea `executable` de cada template.

> `--copies` hace que el binario del venv sea una copia y no un symlink a
> `/usr/bin/python3.12`. El entorno actual se creó sin `--copies` (usa
> symlinks), así que depende de que el worker tenga `/usr/bin/python3.12`. Ojo:
> el venv **no** es autocontenido — el stdlib sale de `/usr/lib/python3.12`, así
> que el worker necesita esa versión. Todos los nodos de producción vienen de la
> misma imagen NMRbox (que trae 3.10–3.13), así que en la práctica está; si un
> job muere con `No such file or directory`, `condor_ssh_to_job` y comprobá
> `ls /usr/bin/python3*` en el nodo.

---

## 3. Qué dice un `.sub` (lo mínimo del tutorial)

```
universe      = vanilla
executable    = /ruta/absoluta/al/binario     # obligatorio absoluto
arguments     = lo que se le pasa             # admite $(Variables)
getenv        = True                          # hereda tu entorno (PATH, etc.)
should_transfer_files = NO                    # usa el FS compartido, sin copias
transfer_executable   = FALSE
request_cpus   = 1
request_memory = 16GB                         # sin esto el límite es 2 GB
+Production    = True                         # solo nodos con software NMRbox
queue                                         # crea el job
```

Cada bloque `Variable = valor` + `queue` crea un job. `queue 100` crea 100
procesos del mismo cluster, con `$(PROCID)` distinto en cada uno — eso es lo
que sirve para paralelizar, no para un entrenamiento suelto.

Reglas prácticas del pool (del tutorial):

- jobs de **menos de 8 h**; el entrenamiento de 210 epochs entra (~1–1.5 h);
- preferir **1 core por job** antes que `request_cpus = 48` (los jobs multicore
  esperan a que se vacíe la cola de jobs de 1 core);
- `request_memory` realista: pedir de más baja tu prioridad;
- no pedir `transfer_*`: todo el pool ve `/home`, y así no se copian GB de pool.

---

## 4. Enviar — el checklist de cada vez

```bash
ssh beryllium.nmrbox.org          # la VM donde esté habilitado Condor
cd ~/git/mrai_p3
source ~/htcondor-portable/activate.sh   # cliente portátil (§1.1); con el de soporte, no hace falta
```

1. **Código y datos al día.** El job corre el código que está en este
   checkout: si tocaste `scripts/`, commiteá o al menos guardá los cambios.
2. **Nombre de corrida nuevo.** `RunName` nombra logs, `--out` y el pool.
   No reutilices uno: `CSVLogger` y `ModelCheckpoint` **sobrescriben**
   `training.log` y `ckpt.keras`.
3. **Verificá el insumo.** Tiene que existir `data/<pool>/manifest.json` y el
   `Extra` de entrenamiento tiene que repetir las bandas del pool. Los que hay
   hoy en esta VM:

   | pool | flags de `Extra` |
   |---|---|
   | `data/pre_aniso_f2lo0_454510` | `--hard-frac 0.45 --hard-px 2 --hard-aniso 3 10 0 3 --hard-frac2 0.45` |
   | `data/pre_aniso_f2lo0_552520` | `--hard-frac 0.55 --hard-px 2 --hard-aniso 3 10 0 3 --hard-frac2 0.25` |
   | `data/pre_aniso_f2lo0_306010` | **incompleto** (523 npz, sin manifest) → no usar |

4. **Enviá** con el wrapper (deja copia del `.sub` final en `htcondor/runs/`):

   Entrenamiento (los defaults de `train.sub` ya son la receta de 454510):

   ```bash
   htcondor/submit.sh train.sub RunName=2026-10-26_aniso
   ```

   Variante sobre el otro pool:

   ```bash
   htcondor/submit.sh train.sub RunName=2026-10-26_552520 \
       Data=data/pre_aniso_f2lo0_552520 \
       Extra="--hard-frac 0.55 --hard-px 2 --hard-aniso 3 10 0 3 --hard-frac2 0.25"
   ```

   Pre-extracción (si cambiaron los flags del pool):

   ```bash
   htcondor/submit.sh preextract.sub RunName=2b \
       Pool=data/pre_elong_2b PoolK=24 \
       Extra="--pool-k 24 --hard-frac 0.55 --hard-px 2 --hard-band2 6 --hard-frac2 0.25"
   ```

   Inferencia por lote:

   ```bash
   htcondor/submit.sh infer.sub RunName=val100 Split=val N=100 \
       Model=Trained/out_elong_2b_savedmodel
   ```

   El wrapper imprime el cluster:

   ```
   Submitting job(s).
   1 job(s) submitted to cluster 1234.
   ```

5. **Seguí el job** (§5).
6. **Recogé el resultado**: ya está escrito en `$HOME` (`Out`/`--out`), no hay
   nada que transferir.

Sin el wrapper también se puede (`cd ~/git/mrai_p3 && condor_submit
htcondor/train.sub`), pero entonces el `RunName` queda vacío y los paths
salen mal: el wrapper es el camino recomendado.

---

## 5. Monitorear

```bash
condor_q                              # resumen por cluster
condor_q -nobatch                     # un renglón por proceso
condor_q -better 1234                 # POR QUE no arranca (idle/held)
tail -f htcondor/logs/2026-10-26_aniso.*.out
tail -f htcondor/logs/2026-10-26_aniso.*.log   # eventos de Condor
```

Estados: `I` idle, `R` running, `H` held (algo falló: `condor_q -better <id>` y
el `.log`).

Depurar adentro del nodo donde corre:

```bash
condor_ssh_to_job 1234.0      # shell en el worker, en el directorio del job
```

Cancelar: `condor_rm 1234` (todo el cluster) o `condor_rm 1234.0` (un proceso).
Si reenviás después de cancelar, usá otro `RunName`.

---

## 6. Dónde queda cada cosa

| Etapa | Salida |
|---|---|
| `preextract.py` | `Pool/manifest.json`, `<id>.npz`, `preextract.log` |
| `train_model.py` | `Out/ckpt.keras`, `training.log`, `history.*`, `Out_savedmodel/` |
| `infer_batch.py` | `Out/<id>.tab`, `<id>/view2d/p3.bin`, `batch_report.json` |

El modelo que consume la cadena de picking es el **SavedModel**
(`<Out>_savedmodel/`), no `ckpt.keras`.

---

## 7. Gotchas

- **`--out` nuevo en cada corrida** (ver §4.2). Para reanudar una corrida
  cortada usar `--resume`, nunca `--init`.
- **Nada en `/scratch`**: los workers no lo ven. Todo en `$HOME`.
- **Memoria**: el pool con `--pool-k 24` pesa ~7 GB en disco; el job lo lee por
  archivo, así que 16 GB de `request_memory` alcanzan. Si ves `Abnormal
  termination (signal 9)` en el `.log`, subí `request_memory`.
- **`+Production = True`** es lo recomendado: el venv **no** es autocontenido
  (el stdlib sale de `/usr/lib/python3.12`), así que conviene que el worker sea
  la misma imagen NMRbox. Si el venv fuera autocontenido, `+Production = False`
  daría acceso a más nodos.
- **GPU** (si algún día entrenás en GPU):
  ```
  request_gpus = 1
  gpus_minimum_memory = 8000
  # requirements = GPUs_Capability >= 8    # opcional: excluye T4/V100
  ```
  Hoy esta VM no tiene GPU (`nvidia-smi` no existe) y el modelo entrena en CPU.
- **Un entrenamiento = un job de 1 core.** Condor no acelera una corrida
  sola; sirve para lanzar varias en paralelo (A/B de `--hard-aniso`,
  `--pair-frac`, etc.) y para no ocupar tu VM.

---

## 8. Mientras esperás el alta de Condor

Esta VM tiene 64 cores y ~630 GB de RAM: el entrenamiento puede correr acá
mismo, sin Condor, con las **mismas flags** del `.sub`:

```bash
cd ~/git/mrai_p3
mkdir -p htcondor/logs
RUN=$(date +%Y%m%d_%H%M)
nohup ~/vtflow/bin/python scripts/train_model.py \
  --data data/pre_aniso_f2lo0_454510 --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --hard-frac 0.45 --hard-px 2 --hard-aniso 3 10 0 3 --hard-frac2 0.45 \
  --epochs 210 --patience 20 \
  --out "Trained/out_$RUN" \
  > "htcondor/logs/$RUN.out" 2>&1 &

tail -f "htcondor/logs/$RUN.out"
```

Es exactamente lo que el tutorial desaconseja para el pool compartido (mejor
1 core por job y dejar que Condor reparta), pero para una corrida única de
1–1.5 h en tu propia VM es lo más directo y ya deja la Mac libre. Cuando
tengas Condor, el mismo comando va como `train.sub`.
