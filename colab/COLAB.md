# MRAi_P3 — Entrenamiento Fase 2 en Google Colab (GPU NVIDIA)

Guía para retomar: fine-tune del modelo 2D-US de picos elongados en F1
con los scripts del repo, sin reescribir nada.

## 0. Resumen de decisiones

- **Subir a Drive el dataset con los `.bin`**, NO el resultado de
  `prepare_training_data.py`:
  - `train_model.py:BinSequence` solo lee `spectra/<id>.bin` +
    `labels/<id>.bin` vía `memmap` + `split_2d` bajo demanda.
  - Los `.npz` de `prepare_training_data.py` no los lee ningún
    `train_*.py` (formato huérfano).
  - Tamaño: `.bin` = 2.5 GB vs `X` pre-expandido ~= 268 GB.
- La ingesta válida para Colab es **`jres_ingest.py`**.
- `colab/scripts/*.py` son **copias exactas** de `scripts/` (§6).

## 1. Qué subir a Google Drive (una vez, desde el Mac)

En `MyDrive/mrai_p3/` deja dos zips:

| Zip | Contenido (desde el Mac) | Tamaño |
|-----|--------------------------|--------|
| `dataset.zip` | `cheminfo/spectrum-generator/dataset/`: `spectra/*.bin` + `labels/*.bin` + `dataset.json` + `manifest.jsonl` | ~2.5 GB |
| `DNN_0.zip` | `mrai_p3/Trained/Probability_2D/US/DNN_0/` (SavedModel `--init`) | pequeño |

```bash
cd /Users/abolanos/git/cheminfo/spectrum-generator/dataset
zip -r ~/dataset.zip spectra labels dataset.json manifest.jsonl
cd /Users/abolanos/git/mrai_p3/Trained/Probability_2D/US
zip -r ~/DNN_0.zip DNN_0
# subir ambos a MyDrive/mrai_p3/ desde drive.google.com
```

El `manifest.json` de `data/train_2d_elong/` NO se sube: apunta a
`/Users/abolanos/...` y se **regenera en Colab** (celda 3).

## 2. Abrir el notebook en Colab

1. Sube `colab/` a `MyDrive/mrai_p3/colab/`.
2. Abre `MRAi_Colab_Train.ipynb` → *Abrir con Colab*.
3. Entorno → Tipo de ejecución → **GPU** (T4 gratis o A100 en Pro).
4. Ejecuta las celdas en orden 0→6. Si Colab se desconecta, usa
   la celda **5b — Reanudar** (continúa desde `ckpt.keras` en Drive).
   No re-ejecutes la celda 5 tras un corte: empezaría de cero.
5. **`/content` es efímero**: al desconectarse la VM se vacía y solo
   sobrevive Drive. Tras un corte hay que re-ejecutar **1 → 2 → 3 → 4b**
   (~5 min) antes de la celda 5b, porque esta necesita `/content/pre`.
   Por eso `--out` apunta a Drive: `ckpt.keras` y `training.log` sí
   sobreviven y permiten reanudar sin perder épocas.
## 3. Celdas y comandos (los ejecuta el notebook)

| Celda | Qué hace |
|-------|----------|
| 0 — GPU check | `tools/verify_gpu.py`: si `GPUs: []`, cambiar runtime antes de seguir. |
| 1 — Setup | `pip install -r requirements-colab.txt`, montar Drive, `BASE/DATA/OUT` en Drive (sobrevive al reciclaje de la VM). |
| 2 — Stage data | Descomprime `dataset.zip` → `/content/dataset/`, `DNN_0.zip` → `/content/init/DNN_0`. |
| 3 — Ingesta | `jres_ingest.py --validate-only` y luego ingesta real → `/content/staged/manifest.json` con `source=/content/dataset` (800 train / 200 val). |
| 4 — Smoke test | `build_model().summary()` + 1 época `--max-pts 4096 --epochs 1` para validar I/O antes de gastar horas. |
| 4b — Pre-extracción | `preextract.py --pool-k 10 --fast` → `/content/pre` (pool = positivos + 10×positivos de fondo). **~11-16 s medido**, ~3,0 GB. Quita el `split_2d` de cada época y el fondo sigue variando. |
| 5 — Fine-tune | `train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 --scratch /content/out --sync-every 5 ...` (~2 s/época de cómputo; el `--scratch` quita el I/O variable de Drive, ver §7). Ruta literal sin pre-extraer: `--data /content/staged` (~18 s/época en el Mac con el split por filas; más lento en la CPU del runtime de Colab). |
| 5b — Reanudar | Solo tras corte. **`/content` se vacía al desconectarse**: re-ejecuta antes 1→2→3→4b. Renombra `training.log` → `training_part1.log` y relanza con `--data /content/pre --preextracted --resume $OUT/ckpt.keras` (pesos exactos). No re-ejecutes la celda 5: `--init` empezaría de cero. |
| 6 — Export | Empaqueta `$OUT` (`ckpt.keras`, `*_savedmodel`, `history.*`, `training.log`) a zip en Drive para bajar al Mac y usar con `StoP3` sin cambios. |

```bash
python tools/verify_gpu.py
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged --validate-only
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged
python scripts/train_model.py --data /content/staged --max-pts 4096 --epochs 1 --batch 4096 --out /content/smoke
# pre-extraccion: pool de fondo (celda 4b), una vez, ~11-16 s
python scripts/preextract.py --dataset /content/dataset --ids /content/staged/manifest.json --out /content/pre --seed 0 --pool-k 10 --hard-frac 0.25 --hard-px 2 --fast
# fine-tune sobre pre-extraido (celda 5). --scratch: escribe en local y
# sincroniza a $OUT cada 5 epocas (ver §7)
python scripts/train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 --epochs 210 --patience 20 --out $OUT --scratch /content/out --sync-every 5
# reanudar tras corte (celda 5b): ckpt + log ya están en Drive
python scripts/train_model.py --data /content/pre --preextracted --resume $OUT/ckpt.keras --lr 1e-4 --batch 65536 --bg-ratio 4 --epochs 210 --patience 20 --out $OUT --scratch /content/out --sync-every 5
```

## 4. Estimación de tiempos (medida real, 14-15 oct 2026)

Medido sobre el dataset real (`spectra/0001.bin`, 256×2048, 581 positivos):

- Modelo: **6.831 params**. Datos: 800 train / 200 val.
- **Puntos por espectro M ≈ 2905**, NO 65536: el muestreo toma todos los
  positivos (581) + `bg_ratio`(4)× positivos de fondo = 581×5 = 2905.
  `max_pts=65536` nunca se alcanza. Con `--bg-ratio 2` (config del Mac)
  M = 1743: **M depende de `bg_ratio`, nunca de `batch`**. Confundir ambos
  es lo que desincronizó los comentarios de las celdas 4b/5.
- Por eso **steps/época = 35 train / 8 val** (con `--batch 65536`), no
  800/200. `__len__` se corrigió para reflejarlo (`_points_per_spectrum`).
- `split_2d`: loop 0.83 s vs `cross_extract_fast` **0.13 s** (6×,
  paridad `maxdiff == 0.0`, ver `tests/test_cross_fast.py`).

Medido en el **Mac** (M1 CPU, sin GPU; `--batch 16384 --bg-ratio 2`,
581 positivos/espectro, `M` de entrenamiento = 1743 puntos):

| Ruta | Coste/época | 30 épocas | ¿Fondo varía? |
|------|-------------|-----------|---------------|
| On-demand bucle (`--slow-split`) | ~12,7 min | ~6,4 h | ✓ |
| **On-demand (default, split por filas)** | **~18 s** | **~9 min** | ✓ |
| **Pre-extraído pool `--pool-k 10`** | **~15 s** | **~7,5 min** (+11-16 s de pre-extracción) | ✓ |

El `~15 s/época` está medido en el run real de 15-oct (100 épocas en 24:32
= 24 min 32 s → 14,7 s/época; consistente con los mtimes del run, ~56 min
para ~171 épocas). Proyectado en **Colab** (GPU NVIDIA; `--batch 65536
--bg-ratio 4` → `M` = 2905 puntos/espectro, 35 steps train / 8 val):

| Ruta | Coste/época | 30 épocas | ¿Fondo varía? |
|------|-------------|-----------|---------------|
| On-demand (`--data /content/staged`) | ~18 s o más (CPU del runtime) | ~9 min+ | ✓ |
| **Pre-extraído pool `--pool-k 10`** | **~5-10 s** | **~4-8 min** (+11-16 s) | ✓ |

**Dónde está el coste (medido, 15-oct-2026).** El split por filas deja el
prep de datos en ~3,4 s/época *on-demand* frente a ~0,8 s con el pool: ~20 %
frente a ~5 % de una época de ~15 s. El cuello de botella pasó del prep al
cómputo del modelo, así que la preextracción **ya no es obligatoria** — pero
sigue pagándose sola: el *break-even* son ~4 épocas y en un run completo
(171 épocas) ahorra ~7,6 min. El pool acota el fondo al 1,1 % de los píxeles
(ver más abajo); si eso importa más que los ~2,7 s/época, usa on-demand.

> `M` sale del **primer espectro de cada split** (`_points_per_spectrum` lee
> el primer id): train → `0001`, 581 positivos → **2905**; val → 584 → 2920.
> La media del dataset son 576 positivos → 2880. Los 35/8 steps salen iguales
> con cualquiera de los tres, y es lo que verás en la barra de progreso.

### Diseño del pool de fondo (por qué no se congela el fondo)

`split_2d` es determinista; el muestreo de fondo es lo aleatorio. Se separan:

* **Caro y determinista** (una vez): `preextract.py` guarda por espectro
  TODOS los positivos (filas `0:n_pos`) + el presupuesto de fondo
  (`pool_k × n_pos` filas), reservando `--hard-frac` (0.25 por defecto) al anillo
del pico (`d <= --hard-px`) en vez de muestrear uniforme, y guarda en `d` la
distancia de cada fila de fondo a su pico más cercano.
* **Barato y aleatorio** (cada época): `train_model.subsample_pool` toma
  todos los positivos + `bg_ratio × n_pos` filas del pool con un `rng` que
  avanza por época.

Con `pool_k > bg_ratio` cada época ve un subconjunto de fondo distinto →
aumentación preservada sin repagar el split. Con `pool_k == bg_ratio` se
toma el pool entero (equivalente al muestreo fijo); verificado en
`tests/test_pool_equiv.py` (3 tests).

El fondo no es uniforme: `--hard-frac` (0.25 por defecto, en `preextract.py` para
el pool y en `train_model.py` para cada época) lo reserva al anillo del pico. La
etiqueta es un marcador sub-píxel (1/2/4 px) sobre picos de ~20-50 px, así que el
fondo uniforme dejaba los ceros de dentro del pico al 0,48 %: el modelo entrenado
daba P > 0,5 al 81 % de los vecinos inmediatos de un pico y la lista de picos
salía ~7,7× más larga. Los pools anteriores a esto no guardan `d`, así que
`train_model.py` se niega a arrancar con `--hard-frac > 0` y pide regenerarlos.

La banda media puede ser isotrópica (`--hard-band2 6 --hard-frac2 0.25`) o
**anisótropa** (`--hard-aniso F1LO F1HI F2LO F2HI --hard-frac2 0.25`, p.ej.
`3 10 1 3`): esta última selecciona por eje (`d1` filas, `d2` columnas) y guarda
`d1`/`d2` en el npz; `train_model.py --hard-aniso` la remuestrea cada época y
falla claro si el pool no los trae. Medición y estado en
`docs/ring_and_picking_metrics.md` §14.

Lo medido de ese anillo —y el `msep` de la lista de picos, que es lo que limita la
precisión una vez arreglado el fondo— está en `docs/ring_and_picking_metrics.md`
(con la comparación frente al modelo base).

### Tamaño del pool (medido, float32 npz sin comprimir)

| `--pool-k` | M pool | MB/espectro | 1000 espectros | Cobertura del fondo |
|---|---|---|---|---|
| 2 | 1743 | 0.90 | 0.90 GB | 0.2% |
| 5 | 3486 | 1.80 | 1.80 GB | 0.6% |
| **10** | **6391** | **3.30** | **3.30 GB** | 1.1% |
| 20 | 12201 | 6.30 | 6.30 GB | 2.2% |
| 50 | 29631 | 15.29 | 15.29 GB | 5.7% |

Subir `pool_k` = más variedad de fondo y más disco. Referencias: disco del
Mac 28 GB libres; Drive gratis 15 GB; Colab `/content` ~100 GB efímeros.
El pool es **derivado**: se regenera desde `dataset.zip` en ~15 s, así que
no hace falta subirlo a Drive.

**Límite honesto**: con `pool_k 10` el pool cubre 1.1% de los 523.707
píxeles de fondo; la variedad combinatoria por época es enorme, pero el
universo está acotado. Cubrir el fondo completo exigiría el X íntegro =
268 GB (inviable).

Defaults efectivos (verificado con `parity X maxdiff: 0.0`):

- `train_model.py` on-demand → **split vectorizado por defecto**;
  `--slow-split` fuerza el bucle (referencia). Si falta
  `cross_extract_fast.py`, cae al bucle solo (con aviso).
- `preextract.py` → el flag `--fast` es **opt-in** (por defecto bucle).
  La celda 4b del notebook lo pasa explícitamente, así que el flujo
  del notebook sí usa el vectorizado.
- `--preextracted` + `--pool-k` → el split no interviene: ya se aplicó al
  pre-extraer; `--bg-ratio` sí, y controla el submuestreo por época.
- Formato antiguo (npz sin `n_pos`) → se trata como muestra fija, sin
  submuestreo (compatibilidad hacia atrás).

Pre-extracción (medido, 15-oct-2026): 1000 espectros en **219 s (3,6 min)**
→ **3,0 GB** en disco con `--pool-k 10` (media 6334 filas/espectro, 0,22 s
por espectro con `--fast`). El pool es `pool_k ×` más grande que la muestra
fija de las versiones previas de esta guía, así que el "~1,5 GB" que figuraba
antes ya no aplica. npz sin comprimir: float32 no comprime, y comprimir cuesta
26× al escribir y 12× al leer — medido.

**Con `split_2d_rows` ese coste baja a ~11-16 s** (16,4 / 11,5 / 11,1 ms por
espectro en 3 pasadas): antes se construía la matriz completa de 524288 filas
y se conservaba el 1,2 %, ahora solo se calculan las filas del pool. La RAM
pico por espectro baja de 1,39 GB a 0,17 GB.

**Descartado y por qué (para no reabrir el debate):**

| Alternativa | Resultado | Motivo |
|---|---|---|
| `multiprocessing` (2/4 workers) | 218 / 338 ms/esp vs 183 con 1 | *memory-bound*: cada worker escribe 537 MB y saturan el bus |
| torch CPU fp64 | 241 ms/esp vs 143 de numpy | más lento; fp32 rompe la paridad (`maxdiff 2.98e-08`) |
| torch MPS fp32 / fp64 | 240 ms/esp / **no existe** | PyTorch no implementa float64 en MPS |
| GPU en Colab (T4) | peor en fp64 | ratio 1/32 frente a fp32 |

## 5. Estructura de `colab/`

```
colab/
  COLAB.md                   # esta guía
  MRAi_Colab_Train.ipynb     # celdas 0-6 (incl. 4b pre-extracción, 5b reanudar)
  requirements-colab.txt     # tensorflow>=2.16,<2.22, numpy, scipy, nmrglue
  scripts/
    train_model.py           # copia exacta de scripts/train_model.py
    cross_extract.py         # copia exacta (loop, referencia/paridad)
    cross_extract_fast.py    # split_2d vectorizado (6x, maxdiff 0.0)
    preextract.py            # pre-extrae el pool de fondo (positivos + pool_k× fondo)
    jres_ingest.py           # copia exacta (ingesta válida Fase 1)
  tools/
    verify_gpu.py            # check GPU pre-vuelo
```

`prepare_training_data.py`, `run_51.py`, `plotting/`, `bruker_2d/` quedan
fuera a propósito: Colab es solo train; inferencia y plots siguen en local.

## 6. Bug corregido: steps/época inflados

`BinSequence.__len__` calculaba `(len(ids) * max_pts) // batch` = 800,
pero el generador produce `(len(ids) * M) // batch` = **35** batches por
pasada (M≈2905, no 65536). Keras declaraba 800 steps y la época terminaba
a los 35, con barra de progreso engañosa. Se añadió
`_points_per_spectrum()` (lee el primer espectro/npz) para que
`steps_per_epoch` sea real. Verificado: declarado 10 = producido 10.

**Implicación para replicar el paper**: la época efectiva es 35 steps
(no 800). Si buscas un total de steps concreto, escala `--epochs`
(p.ej. 35 steps × 30 épocas = 1.050 steps).

## 7. Épocas de duración variable (1-20 s): el I/O de Drive

Con `--batch 65536` la época son **35 steps** y el step es constante, pero
la duración de época oscila entre 1 s y 20 s. La causa no es el batch: son
los callbacks escribiendo en **Drive**.

Drive es una API de red montada por FUSE, no un disco: cada escritura cuesta
entre ~0,3 s y >20 s según el momento, y Colab la estrangula. Por época:

- `CSVLogger` → abre/escribe/cierra `training.log` en Drive, **cada época**.
- `ModelCheckpoint(save_best_only=True)` → escribe `ckpt.keras` (**273 KB**,
  medido en `Trained/Probability_2D_Elongated/DNN_0/ckpt.keras`) en Drive
  **cada vez que mejora `val_loss`**.

Lo que se mide es entonces `cómputo (~2 s, estable) + I/O a Drive (0-20 s,
variable)`. Predicción comprobable: la varianza debería **encogerse** según
avanza el entrenamiento, porque `save_best_only=True` deja de escribir
cuando `val_loss` se estanca.

**Arreglo: `--scratch DIR --sync-every N`.** El entrenamiento escribe en un
directorio local y un callback (`_Syncer` / `make_drive_sync`) copia
`ckpt.keras` + `training.log` al destino cada N épocas (5 por defecto) y
siempre al terminar. La copia es atómica (fichero temporal + `os.replace`)
para que un corte a mitad de escritura no deje un checkpoint truncado en
Drive.

- Quita el I/O variable de las 210 épocas.
- Conserva la red de seguridad: pierdes como mucho N épocas en un corte,
  coherente con `--patience 20`.
- Sin `--scratch` el comportamiento es el de siempre (escribir en `--out`).

**Otras fuentes de varianza** (menores, no arregladas): warm-up de la 1ª
época (tracing de `@tf.function`, autotuning de cuDNN, lectura en frío de
los `.npz`), la T4 compartida de Colab free, y el generador Python en CPU
que alimenta a la GPU (`.prefetch(2)` solo amortigua 2 batches).

**Nota de paso:** `BinSequence.__iter__` concatenaba su buffer 4 veces por
batch (2 para X, 2 para y); ahora concatena una vez. Refactor sin cambio de
salida (fijado por `tests/test_train_sync.py`), solo quita copias en CPU.

## 8. Bug corregido: épocas vacías (se perdió la mitad del run)

El run del 15-oct en `Trained/out_elong` declaró **210 épocas y solo entrenó
105**: todas las impares tienen `acc = auc = loss = 0.0` y su `val_loss` se
mueve ±5e-5 (solo ruido del remuestreo del fondo), frente a caídas de 0.2 en
las pares.

Causa, en Keras 3 (`keras/src/trainers/epoch_iterator.py`):

```python
if self._current_iterator is None or self.steps_per_epoch is None:
    self._current_iterator = iter(self._get_iterator())
```

El iterador **solo** se recrea cuando es `None`, y el bloque que lo resetea al
final de una época exige `self._num_batches` — que es `None` para un
`from_generator` (cardinalidad desconocida). Con `steps_per_epoch=35`
exactamente igual al número de batches que entrega el generador, el agotamiento
cae justo en el límite de época: la siguiente arranca con el iterador muerto,
hace 0 batches, emite `_interrupted_warning()` y **solo entonces** pone
`_current_iterator = None`. De ahí que alterne en vez de morir del todo.

El aviso de TF lo dice literalmente (visible al inicio del log de Colab):

> UserWarning: Your input ran out of data; interrupting training. Make sure
> that your dataset or generator can generate at least `steps_per_epoch *
> epochs` batches. You may need to use the `.repeat()` function when building
> your dataset.

**Por qué el run viejo no lo tenía:** con `steps_per_epoch=800` (≫ 35 batches
disponibles) Keras *tenía* que reiniciar el iterador a mitad de época, así que
nunca llegaba al caso "agotado en el límite". Su `training.log` tiene 0 ceros
en 171 épocas. El bug lo introdujo el arreglo de `_points_per_spectrum()`
(§6): al corregir `steps_per_epoch` de 800 a 35 se alineó con el agotamiento.

**Arreglo:**

1. `BinSequence.to_dataset()` aplica `.repeat()` **antes** de `.prefetch(2)`.
   El dataset pasa a ser infinito y `steps_per_epoch` gobierna la época en
   solitario; cada vuelta de `repeat()` vuelve a llamar a `gen()`, así que
   `__iter__` rebaraja con el rng y el fondo sigue variando.
2. `make_step_guard(tf)` — callback que cuenta batches por época y **aborta**
   con un `RuntimeError` claro si alguna hace 0. Va primero en la lista de
   callbacks para que ningún otro escriba nada antes de fallar.

**Verificado:** 12 tests en `tests/test_train_epoch_boundary.py` (el
`.repeat()` se comprueba con un `tf` falso que registra las transformaciones;
el guard es lógica pura sin TF).

## 9. Resultado del run del 15-oct (`Trained/out_elong`)

| | Viejo (on-demand) | Nuevo (`--preextracted`) |
|---|---|---|
| Épocas con entrenamiento real | 171 / 171 (100 %) | **105 / 210 (50 %)** |
| Mejor `val_loss` | 0.052119 | 0.054100 |
| `val_acc` / `val_auc` | 0.983332 / 0.997033 | 0.982519 / 0.995428 |

`val_loss` por bloques de 42 épocas: 0.176 → 0.079 → 0.065 → 0.059 → 0.0557.
Descenso monótono hasta el final (**mejor época 206 de 209**), así que el run
quedó corto: `--patience 20` nunca disparó porque seguía mejorando.
`history.npy` / `history.txt` son consistentes con `training.log`.

La calidad es equiparable al run viejo con ~60 % de las épocas de datos
distintos, lo que dice bien del pipeline del pool — pero **hay que re-lanzar**
con el arreglo de §8 para aprovechar las 210 épocas completas.


---

## 10. Ajuste recomendado de la cadena (medido 2026-10-17)

El fondo de dos bandas (`--hard-band2 6 --hard-frac2 0.25`, pool con `--pool-k 24
--hard-frac 0.55`) más la cadena afinada dan el mejor resultado medido hasta
ahora sobre el dataset: **F1 0,836** (precisión 0,92, recall 0,76, 1690 picos
para 2048 reales) frente a 0,615 del modelo anterior con la cadena de entonces.

```bash
# pool (una vez; ~7 GB) y fine-tune
python scripts/preextract.py --dataset /content/dataset --ids /content/staged/manifest.json \
  --out /content/pre_2b --seed 0 --pool-k 24 --hard-frac 0.55 --hard-px 2 \
  --hard-band2 6 --hard-frac2 0.25 --fast

python scripts/train_model.py --data /content/pre_2b --preextracted \
  --init /content/Trained/Probability_2D/US/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --hard-frac 0.55 --hard-px 2 --hard-band2 6 --hard-frac2 0.25 \
  --epochs 210 --patience 20 --scratch /content/out_2b --sync-every 5 --out $OUT

# lista de picos
python scripts/plotting/run_train_2d.py --id 1 --model $OUT/../out_elong_2b_savedmodel \
  --spe-the 0.2 --pl-the 0.5 --msep 3
```

Por qué cada ajuste (con las mediciones y los experimentos descartados):
`docs/ring_and_picking_metrics.md` §4, §9 y §10. Los valores por defecto de los
scripts siguen siendo los antiguos (1 / 0,2 / 0,5) para no cambiar nada en
silencio. La variante anisótropa (`--hard-aniso 3 10 1 3`) está implementada
pero **todavía no reentrenada**; el A/B pendiente está en §14.
