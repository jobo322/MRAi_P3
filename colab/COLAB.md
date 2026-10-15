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
## 3. Celdas y comandos (los ejecuta el notebook)

| Celda | Qué hace |
|-------|----------|
| 0 — GPU check | `tools/verify_gpu.py`: si `GPUs: []`, cambiar runtime antes de seguir. |
| 1 — Setup | `pip install -r requirements-colab.txt`, montar Drive, `BASE/DATA/OUT` en Drive (sobrevive al reciclaje de la VM). |
| 2 — Stage data | Descomprime `dataset.zip` → `/content/dataset/`, `DNN_0.zip` → `/content/init/DNN_0`. |
| 3 — Ingesta | `jres_ingest.py --validate-only` y luego ingesta real → `/content/staged/manifest.json` con `source=/content/dataset` (800 train / 200 val). |
| 4 — Smoke test | `build_model().summary()` + 1 época `--max-pts 4096 --epochs 1` para validar I/O antes de gastar horas. |
| 4b — Pre-extracción | `preextract.py --fast` → `/content/pre` (`(X,y)` fijos por semilla). ~6 min, ~1.5 GB. Quita el `split_2d` de cada época. |
| 5 — Fine-tune | `train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 ...` (~5-10 s/época). Ruta literal sin pre-extraer: `--data /content/staged` (~15 min/época). |
| 5b — Reanudar | Solo tras corte: renombra `training.log` → `training_part1.log` y relanza con `--resume $OUT/ckpt.keras` (pesos exactos). No re-ejecutes la celda 5: `--init` empezaría de cero. |
| 6 — Export | Empaqueta `$OUT` (`ckpt.keras`, `*_savedmodel`, `history.*`, `training.log`) a zip en Drive para bajar al Mac y usar con `StoP3` sin cambios. |

```bash
python tools/verify_gpu.py
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged --validate-only
python scripts/jres_ingest.py --dataset /content/dataset --out /content/staged
python scripts/train_model.py --data /content/staged --max-pts 4096 --epochs 1 --batch 4096 --out /content/smoke
# pre-extraccion (celda 4b): una vez, ~6 min
python scripts/preextract.py --dataset /content/dataset --ids /content/staged/manifest.json --out /content/pre --seed 0 --fast
# fine-tune sobre pre-extraido (celda 5)
python scripts/train_model.py --data /content/pre --preextracted --init /content/init/DNN_0 --lr 1e-4 --batch 65536 --epochs 210 --patience 20 --out $OUT
# reanudar tras corte (celda 5b): ckpt + log ya están en Drive
python scripts/train_model.py --data /content/pre --preextracted --resume $OUT/ckpt.keras --lr 1e-4 --out $OUT
```

## 4. Estimación de tiempos (medida real, 14-oct-2026)

Medido sobre el dataset real (`spectra/0001.bin`, 256×2048, 581 positivos):

- Modelo: **6.831 params**. Datos: 800 train / 200 val.
- **Puntos por espectro M ≈ 2905**, NO 65536: el muestreo toma todos los
  positivos (581) + `bg_ratio`(4)× positivos de fondo = 581×5 = 2905.
  `max_pts=65536` nunca se alcanza.
- Por eso **steps/época = 35 train / 8 val** (con `--batch 65536`), no
  800/200. `__len__` se corrigió para reflejarlo (`_points_per_spectrum`).
- `split_2d`: loop 0.83 s vs `cross_extract_fast` **0.13 s** (6×,
  paridad `maxdiff == 0.0`, ver `tests/test_cross_fast.py`).

| Ruta | Coste/época | 30 épocas |
|------|-------------|-----------|
| On-demand loop (setup original) | ~15 min | ~7.5 h |
| On-demand `--fast` | ~5 min | ~2.5 h |
| **Pre-extraído** | ~5-10 s | **~10-15 min** (+6 min de pre-extracción) |

Pre-extracción: 1000 espectros × 0.36 s ≈ 6 min, ~1.5 GB en disco
(npz sin comprimir: float32 no comprime, y comprimir cuesta 26× al
escribir y 12× al leer — medido).

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
    preextract.py            # pre-extrae (X,y) fijos por semilla
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

