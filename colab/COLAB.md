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
| 4b — Pre-extracción | `preextract.py --pool-k 10 --fast` → `/content/pre` (pool = positivos + 10×positivos de fondo). ~3,6 min medido, ~3,0 GB. Quita el `split_2d` de cada época y el fondo sigue variando. |
| 5 — Fine-tune | `train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 ...` (~5-10 s/época). Ruta literal sin pre-extraer: `--data /content/staged` (~3,5 min/época en el Mac; más lento en la CPU del runtime de Colab). |
| 5b — Reanudar | Solo tras corte. **`/content` se vacía al desconectarse**: re-ejecuta antes 1→2→3→4b. Renombra `training.log` → `training_part1.log` y relanza con `--data /content/pre --preextracted --resume $OUT/ckpt.keras` (pesos exactos). No re-ejecutes la celda 5: `--init` empezaría de cero. |
| 6 — Export | Empaqueta `$OUT` (`ckpt.keras`, `*_savedmodel`, `history.*`, `training.log`) a zip en Drive para bajar al Mac y usar con `StoP3` sin cambios. |

```bash
python tools/verify_gpu.py
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged --validate-only
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged
python scripts/train_model.py --data /content/staged --max-pts 4096 --epochs 1 --batch 4096 --out /content/smoke
# pre-extraccion: pool de fondo (celda 4b), una vez, ~3,6 min
python scripts/preextract.py --dataset /content/dataset --ids /content/staged/manifest.json --out /content/pre --seed 0 --pool-k 10 --fast
# fine-tune sobre pre-extraido (celda 5)
python scripts/train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 --epochs 210 --patience 20 --out $OUT
# reanudar tras corte (celda 5b): ckpt + log ya están en Drive
python scripts/train_model.py --data /content/pre --preextracted --resume $OUT/ckpt.keras --lr 1e-4 --batch 65536 --bg-ratio 4 --epochs 210 --patience 20 --out $OUT
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
| **On-demand (default, vectorizado)** | ~3,5 min | ~1,8 h | ✓ |
| **Pre-extraído pool `--pool-k 10`** | **~15 s** | **~7,5 min** (+3,6 min de pre-extracción) | ✓ |

El `~15 s/época` está medido en el run real de 15-oct (100 épocas en 24:32,
85 steps/época). Proyectado en **Colab** (GPU NVIDIA; `--batch 65536
--bg-ratio 4` → `M` = 2905 puntos/espectro, 35 steps train / 8 val):

| Ruta | Coste/época | 30 épocas | ¿Fondo varía? |
|------|-------------|-----------|---------------|
| On-demand (`--data /content/staged`) | ~3,5 min o más (CPU del runtime) | ~1,8 h+ | ✓ |
| **Pre-extraído pool `--pool-k 10`** | **~5-10 s** | **~4-8 min** (+3,6 min) | ✓ |

> `M` sale del **primer espectro de cada split** (`_points_per_spectrum` lee
> el primer id): train → `0001`, 581 positivos → **2905**; val → 584 → 2920.
> La media del dataset son 576 positivos → 2880. Los 35/8 steps salen iguales
> con cualquiera de los tres, y es lo que verás en la barra de progreso.

### Diseño del pool de fondo (por qué no se congela el fondo)

`split_2d` es determinista; el muestreo de fondo es lo aleatorio. Se separan:

* **Caro y determinista** (una vez): `preextract.py` guarda por espectro
  TODOS los positivos (filas `0:n_pos`) + `pool_k × n_pos` píxeles de fondo.
* **Barato y aleatorio** (cada época): `train_model.subsample_pool` toma
  todos los positivos + `bg_ratio × n_pos` filas del pool con un `rng` que
  avanza por época.

Con `pool_k > bg_ratio` cada época ve un subconjunto de fondo distinto →
aumentación preservada sin repagar el split. Con `pool_k == bg_ratio` se
toma el pool entero (equivalente al muestreo fijo); verificado en
`tests/test_pool_equiv.py` (3 tests).

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
El pool es **derivado**: se regenera desde `dataset.zip` en ~3,6 min, así que
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

