# Plan: `split_2d_rows` — calcular solo las filas que se guardan

**Estado:** **IMPLEMENTADO** y verificado (§9, §14).
**Repo:** `mrai_p3`, rama `install_in_mac_m1`.
**Baseline:** commit `a163563` (el refactor *pool* ya está commiteado; la base quedó
limpia con `74e3cd1` + `34ec4b5` + `a163563`).
**Fecha mediciones:** 2026-10-15 · **Máquina:** MacBook Air M1, 8 cores, 16 GB, macOS 26.6.2
**Dataset:** `/Users/abolanos/git/cheminfo/spectrum-generator/dataset` — 1000 espectros `256×2048` (`spectra/*.bin` 2.0 MB + `labels/*.bin` 0.5 MB)
**Pool existente:** `data/pre_elong/` — 1000 `.npz`, 3.0 GB, generado con `--seed 0 --pool-k 10 --fast`

---

## 1. Punto de partida (lo que hay hoy)

> **Nota de baseline:** este §1 describe el árbol de trabajo **sin commitear**, no
> `efd2bcc`. Ver cabecera.

`preextract.py` y `train_model.py` (ruta on-demand) hacen:

```python
X, _ = split_2d(S.astype(np.float64), CROSS)   # X = (524288, 128, 1) float64
X = np.asarray(X, dtype=np.float32)
X[~np.isfinite(X)] = 0.0
Xs, ys, n_pos = build_pool(X, L, pool_k, ...)  # X[idx]  -> solo ~6391 filas
```

Es decir: se construye la **matriz completa** de 524288 filas para conservar
`n_pos + pool_k*n_pos ≈ 6391` filas (**1,2 %**). El `norm` de `split_2d` es
**por fila** (`np.linalg.norm(S_data, axis=(1,2))`), así que las filas son
independientes y la matriz entera no es necesaria.

### Coste medido del código actual

| Concepto | Medido |
|---|---|
| `split_2d_fast` por espectro | 143–195 ms (variante con carga) |
| Asignación `np.empty(H*W,128,1) float64` | **537 MB** por espectro |
| Desperdicio de cómputo | 524288 / 6391 = **82×** |
| RAM pico por espectro (`ru_maxrss`) | **1,39 GB** con X completo vs **0,17 GB** solo filas |
| `preextract` 1000 espectros (run real) | **219 s (3,6 min)** |
| `preextract` benchmark 64 esp | 166,7 ms/esp → **2,8 min**/1000 |
| `on-demand` 800 esp (una época) | 175 ms/esp → **140 s/época** |

Desglose por paso (ms/espectro): `astype fp64` 0,2 · `np.pad` 0,3 ·
**alloc 537 MB + copia 2 brazos ≈ dominante** · `linalg.norm` · divide.

---

## 2. Hipótesis descartadas (medidas, no asumidas)

### 2.1 `multiprocessing` por espectro → **empeora**

```python
# split_2d_fast completo, 8/16/24 espectros, fork Pool
workers=1: 183 ms/esp   (3,1 min/1000)
workers=2: 218 ms/esp
workers=4: 338 ms/esp
```

**Causa:** cada worker escribe 537 MB → saturan el ancho de banda de memoria.
Más cores reparten el mismo bus; el trabajo es *memory-bound*, no compute-bound.

### 2.2 GPU (torch) → **más lenta que numpy**

| backend | ms/espectro | paridad vs numpy fp64 |
|---|---|---|
| numpy fp64 (actual) | **143** | — |
| torch cpu fp64 | 241 | `maxdiff 0.00e+00` (idéntico) |
| torch cpu fp32 | 273 | `2.98e-08` |
| torch **mps fp32** | 240 | `2.98e-08` |
| torch **mps fp64** | — | **no existe: PyTorch no implementa float64 en MPS** |

Además, en Colab la T4 es aún más lenta en fp64 (ratio 1/32 sobre fp32).
Y `fp32` rompe la paridad byte-a-byte de los `.npz` (`maxdiff 2.98e-08 ≠ 0`).

**Conclusión de 2.1 y 2.2:** ninguna de las dos acelera el código actual.

---

## 3. La mejora propuesta: `split_2d_rows(S, idx, Cross_Size)`

Calcular **solo** las filas de `idx`:

```python
def split_2d_rows(S, idx, Cross_Size):
    S = np.asarray(S).astype(np.float64)      # mismo cast que preextract/_points
    idx = np.asarray(idx)                     # idx // W exige ndarray (no listas)
    c0, c1 = int(Cross_Size[0]), int(Cross_Size[1])
    H, W = S.shape
    p0a, p0b = c0 // 2, int(c0 / 2 - 1 / 2)
    p1a, p1b = c1 // 2, int(c1 / 2 - 1 / 2)
    S_pad = np.pad(S, ((p0a, p0b), (p1a, p1b)), "wrap")
    col = np.lib.stride_tricks.sliding_window_view(S_pad[:, p1a:p1a+W], c0, axis=0)
    row = np.lib.stride_tricks.sliding_window_view(S_pad[p0a:p0a+H, :], c1, axis=1)
    ii, jj = idx // W, idx % W                # fancy index sobre vistas
    out = np.empty((len(idx), c0 + c1), dtype=np.float64)
    out[:, :c0] = col[ii, jj]
    out[:, c0:] = row[ii, jj]
    norm = np.linalg.norm(out, axis=1)        # ANTES de dividir, como split_2d
    out /= norm[:, None]
    return out.reshape(len(idx), c0 + c1, 1), norm
```

Notas de implementación:

- **`Norm` se calcula ANTES de dividir**, igual que `split_2d`
  (`cross_extract.py:30`) y `split_2d_fast` (`cross_extract_fast.py:38`). Si se
  calcula *después* de dividir, todas las filas devuelven `1.0` y se rompe el
  contrato de §5.1 y el test 1 de §7. *(Bug detectado en revisión; corregido aquí.)*
- **No** hacer `col.reshape(H*W, c0)` (eso materializa la matriz entera): usar
  `col[ii, jj]` directamente (fancy indexing sobre la vista, sin copia).
  Verificado bit-idéntico vs `col.reshape(H*W, c0)[idx]` en `(256,2048)`,
  `(256,512)`, `(64,65)` e `idx` ordenado/desordenado/vacío (`maxdiff 0.0` en `X`
  y en `Norm`).
- `ii`/`jj` **sin** `% H` / `% W`: las vistas ya están acotadas a `H×W`.
- `idx` vacío → devuelve `(0,128,1)` y `(0,)`. `idx` desordenado → ok.
  `idx` fuera de rango → `IndexError` (igual que `split_2d(S)[idx]`).
- Filas de norma 0 → `NaN` en `X`; el llamante las limpia con
  `X[~np.isfinite(X)] = 0.0` (idéntico a hoy).
- Si `len(idx) == H*W` → delegar a `split_2d_fast` (evita regresión). Ojo:
  `split_2d_fast` **no** hace `astype(np.float64)` (asigna a un buffer float64);
  numéricamente idéntico para `S` float32/float64, pero no para `S` entero.
- Firma compatible: `(X_rows, Norm_rows)` con `X_rows` `(len(idx),128,1)` fp64.

### 3.1 Coste medido (prototipo)

| Ruta | actual | con filas | speedup |
|---|---|---|---|
| `preextract` (split) | 166,7 ms/esp | **2,5 ms/esp** | ~66× |
| `preextract` (e2e `.bin`+split+`savez`) | 194,9 ms/esp | **4,6–6,1 ms/esp** | **~36×** |
| → 1000 espectros | 2,8–3,6 min | **≈ 5–6 s** | |
| `on-demand` (e2e) | 175 ms/esp | **2,7 ms/esp** | **64×** |
| → época 800 esp | 140 s | **2,2 s** + cómputo | |

RAM pico: **1,39 GB → 0,17 GB** por espectro.

---

## 4. Paridad (lo que hace el cambio seguro)

| Verificación | Resultado |
|---|---|
| **`.npz` regenerados vs `data/pre_elong/` existentes (1000 esp.)** | **X 1000/1000, y 1000/1000, n_pos 1000/1000 — byte-idénticos** |
| Ruta on-demand, 16 espectros reales (`bg_ratio 4`, `max_pts 65536`) | `maxdiff 0.0` |
| 6 espectros aleatorios | `maxdiff 0.0` |
| Variante *chunked* (5 espectros, descartada) | `maxdiff 0.0` |
| torch cpu fp64 (referencia externa) | `maxdiff 0.00e+00` |

**Nota sobre `Norm`:** los `.npz` **no** guardan `Norm` (`preextract` lo descarta
con `X, _ = ...`), así que la byte-identity de la tabla **no** cubre `Norm`. Lo
cubre el test 1 de §7, que compara directamente contra `split_2d`.

**Por qué no cambia nada:**

1. El `rng` se consume en el **mismo orden** (`choice` → `shuffle`); `split_2d`
   no toca `rng`, así que reordenar *plan de índices* antes del split no altera
   la muestra. Verificado: mismo `idx` antes/después.
2. `X[~np.isfinite(X)] = 0.0` es elementwise → idéntico aplicado a filas.
3. El `norm` es por fila → cada fila se calcula igual esté sola o en la matriz.

**Consecuencia operativa:** los 3 GB de `data/pre_elong/` **siguen siendo
válidos; no hay que re-extraer.**

---

## 5. Cambios por fichero

### 5.1 `scripts/cross_extract_fast.py`

- Nueva `split_2d_rows(S, idx, Cross_Size) -> (X_rows, Norm_rows)`.
- Documentar contrato: misma semántica que `split_2d(S)[idx]` y
  `split_2d(S)[idx]` en `Norm`.
- Avisar en el docstring del módulo que `StoP3.split_2D` sigue siendo la
  referencia y cualquier cambio debe reflejarse en `cross_extract.py` **y aquí**.

### 5.2 `scripts/preextract.py`

- Nueva función `plan_pool_idx(L, pool_k, max_pts, rng) -> (idx, n_pos)`
  con la lógica **exacta** de `build_pool` (sin tocar `X`): conserva
  `n_bg = min(len(neg), int(n_pos*pool_k))`, el clamp de `max_pts` y consume
  `rng.choice` **una sola vez** en el mismo orden.
- `build_pool(X, L, ...)` pasa a ser wrapper sobre `plan_pool_idx` + `X[idx]`
  (mantiene compatible `tests/test_pool_equiv.py`, fuente única de verdad).
- `main()`:
  ```python
  S, L = load_pair(...)
  idx, n_pos = plan_pool_idx(L, pool_k, max_pts, rng)   # rng aquí (mismo orden)
  if args.slow_split:                                   # referencia
      X, _ = split_2d(S.astype(np.float64), CROSS)
      Xs = np.asarray(X, np.float32)[idx]
  else:                                                 # solo esas filas
      Xs, _ = split_2d_rows(S, idx, CROSS)
  Xs = np.asarray(Xs, np.float32); Xs[~np.isfinite(Xs)] = 0.0
  ys = L.reshape(-1, 1)[idx].astype(np.float32)
  np.savez(..., X=Xs, y=ys, n_pos=np.int32(n_pos), ...)
  ```
  (La selección `split_2d`/`split_2d_fast` de hoy se conserva solo para
  `--slow-split`; la ruta por defecto ya no la necesita.)
- Flags (`argparse` `mutually_exclusive_group` para fijar precedencia):
  - `--slow-split` → referencia: `split_2d` completo + `X[idx]`. **Gana** si se
    pasa junto a `--fast`.
  - `--fast` → **no-op documentado** (hoy es `store_true` y la celda 12 del
    notebook lo pasa; quitarlo rompería el notebook). Actualizar su `help`
    para que diga que ya no selecciona nada.
- Sin `--workers`, sin `--gpu` (explícitamente fuera de alcance, ver §6).

### 5.3 `scripts/train_model.py`

- `_points(S, L)`: reordenar a `idx = choice(...)` → `shuffle(idx)` →
  `X, _ = split_2d_rows(S, idx, CROSS)`, y devolver **también `y[idx]`**
  (hoy devuelve `X[idx], y[idx]`). **Mismo orden de consumo de `rng`** que hoy
  (hoy: `split_2d` (sin rng) → `choice` → `shuffle`).
- Con `--slow-split`: mantener `split_2d` completo + `X[idx]` (referencia).
- `__init__` ya expone `fast_split`; reutilizar ese eje.
- No tocar `_load_pre` / `subsample_pool` / `subsample_pool` ni `__len__`.

### 5.4 Ficheros **NO** tocados (y por qué)

| Fichero | Motivo |
|---|---|
| `scripts/cross_extract.py` | referencia *loop*, se queda tal cual |
| `src/MRAi_P3/StoP3.py` | referencia de inferencia; tocarla rompe train↔inferencia |
| `scripts/jres_ingest.py:98` | `X, Norm = split_2d(...)`: asigna `Norm` pero **no lo usa** (la sonda solo usa `X`); además es Fase 1, una vez |
| `scripts/prepare_training_data.py:65` | **sí** consume `Norm` (`return X, y, Norm`), pero vía `split_2D` de `StoP3`, no `split_2d_rows` → fuera de alcance |
| `scripts/bruker_2d/run_51.py` | inferencia con `TFSMLayer`, no usa split |
| `data/pre_elong/**` | byte-idéntico → no regenerar |
| `Trained/**` | run completado (171 épocas); fuera de alcance |

---

## 6. Alcance explícito: qué queda FUERA

- **No GPU.** Medida 1,7× más lenta que numpy (§2.2); MPS sin `float64` →
  rompe paridad byte-a-byte; T4 de Colab peor aún en fp64.
- **No `multiprocessing`/`ThreadPool`.** Medido negativo en el código actual
  (§2.1). Tras el cambio el trabajo total son ~5 s, así que el paralelismo
  sobre ~5 s no aporta nada práctico.
- No se cambia el formato de `.npz`, ni `manifest.json`, ni hiperparámetros.
- No se re-escribe `prepare_training_data.py` (ruta sintética Fase 1).
- No se migra a MLX ni a TF-Metal (planes anteriores, descartados aquí).

> **Dato para el revisor:** tras este cambio la ruta *on-demand* queda en
> ~3,4 s de datos por época (≈18 s/época con cómputo) frente a ~0,8 s de la
> pre-extraída, y su RAM pico baja de 1,39 GB a **0,17 GB** (§3). Es decir:
> tras el cambio **ni la velocidad ni la RAM** justifican pre-extraer (el
> argumento de 1,39 GB correspondía al split completo antiguo). Se mantiene
> por consistencia con el flujo Colab y por ~2,7 s/época. **Punto legítimo a
> discutir** — ver §6.1.

### 6.1 ¿Sigue siendo necesaria la preextracción? (medido, 2026-10-15)

> Corrección de encuadre: el cambio acelera **la preextracción** y **el
> entrenamiento *on-demand***. **No** acelera el entrenamiento *pre-extraído*:
> en esa ruta `split_2d` nunca se ejecuta durante el entrenamiento.

Coste de datos por espectro (medido, 60 espectros, caché caliente):

| Ruta | ms/espectro | 800 esp/época |
|---|---:|---:|
| Pre-extraída (`npz` 3,32 MB + `subsample_pool`) | **0,97** | **0,78 s** |
| On-demand con `split_2d_rows` (nuevo) | **4,30** | **3,44 s** |
| On-demand con `split_2d_fast` completo (viejo) | ~175 | ~140 s |

Preextracción de 1000 espectros: **219 s** (log real `preextract.log`,
mediana 0,200 s/esp) → **11–16 s** medidos (16,4 / 11,5 / 11,1 ms/esp en 3
pasadas). ~**15–20×**, no 5–6 s: el coste lo domina escribir 3,32 MB/esp.

Época real del run completado: **~15 s** (`COLAB.md`: "100 épocas en 24:32"
= 24 min 32 s → 14,7 s/época; consistente con los mtimes del run: ~56 min
para ~171 épocas).

Reparto del tiempo de época:

| Ruta | datos/época | % de la época |
|---|---:|---:|
| Pre-extraída | 0,78 s | **~5 %** |
| On-demand (nuevo) | 3,44 s | **~20 %** |
| On-demand (viejo) | ~140 s | ~90 % |

**El cuello de botella cambió de sitio.** Antes el `split_2d` dominaba
(140 s de ~155 s): trabajo 82× redundante y pipeline *data-bound*. Ahora
manda el **cómputo del modelo**.

**Sobre el solapamiento (`prefetch`).** `BinSequence.to_dataset()` usa
`.prefetch(2)`, así que en teoría el prep podría esconderse tras el cómputo.
Evidencia indirecta de que **no** lo hace en Mac: el run on-demand viejo
tardaba ~210 s/época ≈ 140 s de prep + ~15 s de cómputo → es **aditivo**.
El log muestra `TF 2.21.0 GPUs: []` (Mac CPU-only): el hilo generador compite
por los mismos 8 cores y el mismo bus de memoria.

> ⚠️ **Medición pendiente (la única que zanja la pregunta):** si el `prefetch`
> sí solapara bien, on-demand = pre-extraída y el ahorro sería 0. Requiere TF
> instalado: `--n-train 200 --epochs 3` con y sin `--preextracted`, comparando
> tiempo de pared. No se pudo hacer en el entorno de este plan.

**Veredicto: no es obligatoria, pero sigue siendo rentable.**

- **Break-even:** ~11 s de preextracción ÷ ~2,7 s ahorrados por época ≈
  **4 épocas**.
- **Run completo (171 épocas):** ahorra ~2,7 s × 171 ≈ **7,6 min** (13–18 %
  de un run de 42–57 min), costando ~15 s y 3,0 GB de disco.
- **Run corto (≤4 épocas):** on-demand gana (no escribe 3 GB).

| Escenario | Recomendación | Por qué |
|---|---|---|
| Mac CPU, run completo (171 ép.) | **Pre-extraer** | ahorra ~7,6 min; coste 15 s |
| Mac CPU, experimento (≤5 ép.) | On-demand | no se alcanza el break-even |
| Colab GPU | **Pre-extraer** | compute más barato → el prep pesa relativamente más |
| Colab, reanudar sesión | **Pre-extraer** | `/content` es efímero: se re-extrae cada sesión (antes ~6 min, ahora ~20 s) |

**Único argumento restante a favor de on-demand (científico, no de
velocidad):** el pool con `pool_k=10` tapa el fondo a `10 × n_pos ≈ 5 760`
píxeles de los ~523 700 disponibles (**1,1 %**); on-demand muestrea de los
523 700. El *soporte* del pool está acotado para siempre. Si importa la
diversidad máxima de fondo, on-demand es estrictamente mejor **y ahora es
barato**.

---

## 7. Tests

Nuevo `tests/test_split_rows.py` (estilo `unittest`, como el resto):

1. **Paridad vs `split_2d` (bucle)** con `idx` ordenado / desordenado / vacío
   y formas edge `(256,512)`, `(256,2048)`, `(64,65)` → `maxdiff 0.0` en `X` y
   en `Norm` **pre-división**. Este test es el que caza el bug de §3 si `Norm`
   se calcula después de dividir (daría `1.0` en todas las filas).
2. `idx = arange(H*W)` ≡ `split_2d_fast`.
3. **Regresión real:** pipeline de `preextract` vs `data/pre_elong/<id>.npz`
   → `np.array_equal` (con `skipUnless(os.path.isdir(...))`; el dataset local
   son ~3,0 GB y `data/pre_elong/` está *untracked*, así que el skip es
   necesario y suficiente).
4. `_points` con el mismo `seed` → misma muestra que la ruta full-split.
5. No más lento que `split_2d_fast` (assert de tiempo **laxo**, con margen
   amplio; precedente en `tests/test_cross_fast.py::test_speedup`, que ya es
   timing-dependent).

Verificado: `tests/test_headers.py` valida parseo NMR (nmrglue), **no** impone
convenciones sobre ficheros de test nuevos.

Tests existentes que deben seguir verdes: `test_cross_fast.py` (2),
`test_pool_equiv.py` (3), `test_headers.py` (15 `def test`).

---

## 8. Sincronía y documentación

- `scripts/{cross_extract_fast,cross_extract,jres_ingest,preextract,train_model}.py`
  → `colab/scripts/`, `diff` = 0 en los 5 (verificado hoy: sin diferencias).
- `py_compile` en `scripts/`, `colab/scripts/`, `colab/tools/`, `tests/`.
- `colab/COLAB.md`:
  - §4: sustituir "219 s (3,6 min), ~3,0 GB" por **~11–16 s** medidos.
  - §4: añadir nota **"GPU y `--workers` medidos y descartados"** con las
    tablas de §2 para que no se reabra el debate.
  - §4: la ruta on-demand ya no es lenta (~3,5 min/época → ~18 s) →
    actualizar la comparativa y el encuadre de §6.1.
- `colab/MRAi_Colab_Train.ipynb`: celda 13 (markdown) dice hoy
  "en el Mac son ~3,5 min/época con el split vectorizado" → ~18 s/época.
  Celda 4b (pre-extracción) → actualizar tiempo/ETA (~11–16 s).
- JSON del notebook re-parseado tras editar.

---

## 9. Validación final (gate de aceptación) — RESULTADOS

- [x] `python3 -m pytest tests/ -q` → **28 passed, 3 skipped** (los 3 skips son
      `test_headers.py` por falta de `showhdr`/NMRPipe en el PATH). 12 tests
      nuevos en `tests/test_split_rows.py`, 23 subtests.
- [x] Byte-identity **1000/1000 espectros** vs `data/pre_elong/` (X, y y
      `n_pos`) en 9,9 s → 9,9 ms/espectro. Además, CLI real sobre 20
      espectros: default ≡ `--fast` ≡ `--slow-split` ≡ pool guardado.
- [x] `diff scripts/ ↔ colab/scripts/` → idénticos (5/5)
- [x] `py_compile` sin errores en `scripts/`, `colab/scripts/`, `colab/tools/`,
      `tests/`
- [x] Tiempos re-medidos (CLI real, 20 espectros): **default 12,2 ms/espectro**
      (~12 s/1000) vs **`--slow-split` 950,9 ms/espectro** (~16 min/1000) →
      **78×**. On-demand: 3,4 s de datos/época (~20 % de una época de ~15 s).
- [ ] **Medición pendiente (requiere TF):** 3 épocas con y sin
      `--preextracted` sobre 200 espectros → decide si la preextracción se
      mantiene (§6.1). Sin ella, el ahorro de 2,7 s/época es una estimación,
      no un hecho medido. **No bloquea la entrega.**
- [x] Notebook JSON válido, 19 celdas
- [x] `git status` limpio respecto a `Trained/` y `data/` (`data/`,
      `Trained copy/` y los PDFs están en `.gitignore`)

---

## 10. Orden de ejecución propuesto

0. **Commitear el refactor *pool*** → baseline limpio (ver cabecera).
1. `tests/test_split_rows.py` **primero** (falla: aún no existe la función).
2. `split_2d_rows` en `cross_extract_fast.py` → paridad verdes.
3. `plan_pool_idx` + `main()` en `preextract.py`.
4. `_points()` en `train_model.py`.
5. Byte-identity 1000 espectros + benchmark final.
6. Sincronizar `colab/`, actualizar `COLAB.md` y notebook.
7. Gate completo de §9.

**Riesgo residual principal:** diferencias numéricas en `col[ii,jj]` vs
`col.reshape(H*W,c0)[idx]`. Mitigado por el test 1 + byte-identity 1000 esp.

---

## 11. Preguntas para el revisor

1. ¿Mantener `preextract.py` si tras el cambio on-demand rinde igual?
   (§6, decisión de diseño abierta.)
2. ¿Es aceptable que `--fast` quede como no-op en `preextract.py` para no
   romper la celda 12 del notebook, o prefieres actualizar el notebook y
   eliminar el flag?
3. ¿El `skipUnless` del test de regresión (requiere dataset local de 2,5 GB)
   es suficiente, o quieres un fixture pequeño versionado?
4. ¿Confirmas que `jres_ingest.py` puede quedarse en el split completo aunque
   su `Norm` no se use en el probe? (Lo he dejado fuera por prudencia.)
5. ¿Alguna duda sobre por qué se descartaron GPU y `multiprocessing`?
   §2 tiene las mediciones y §6 el razonamiento.

---

## 12. Cambios de esta revisión (r2)

Aplicados tras la revisión del plan:

1. **Bug bloqueante corregido:** `Norm` se calcula **antes** de dividir (§3).
   Verificado bit-idéntico en `(256,2048)`, `(256,512)`, `(64,65)` e `idx`
   ordenado/desordenado/vacío.
2. **Baseline fijado:** cabecera y §1 aclaran que el punto de partida es el
   árbol de trabajo sin commitear sobre `efd2bcc`; se añade el paso 0 de §10.
3. **§6 corregido:** el argumento de RAM para mantener `preextract` era
   contradictorio con §3 (0,17 GB tras el cambio).
4. **§5.4 corregido:** `jres_ingest.py` **no** usa `Norm`.
5. **§9/§7:** conteo de tests corregido (20 actuales + 5 nuevos = 25).
6. **§5.2:** precedencia `--slow-split` > `--fast` con
   `mutually_exclusive_group`; `main()` completo (rama slow, `y[idx]`).
7. **§5.3:** `_points` debe devolver `y[idx]`.
8. **§8:** enumerados los 5 ficheros sincronizados.
9. **§3:** `idx = np.asarray(idx)`; nota sobre `split_2d_fast` y `astype`.
10. **§7:** nota de que la byte-identity **no** cubre `Norm`; tamaño real del
    dataset (3,0 GB) y `skipUnless`; aviso de test de tiempo laxo.

**Riesgo residual principal (sin cambios):** diferencias numéricas en
`col[ii,jj]` vs `col.reshape(H*W,c0)[idx]`. Mitigado por el test 1 + la
byte-identity de 1000 espectros; la paridad de `Norm` la cubre el test 1.

---

## 13. Cambios de esta revisión (r3 — análisis de necesidad)

Añadido tras medir el coste real de ambas rutas con los datos del repo:

1. **§6.1 nuevo:** tabla de coste medido por ruta (0,97 / 4,30 / ~175 ms por
   espectro), reparto del tiempo de época, y la corrección de encuadre — el
   cambio **no** acelera el entrenamiento pre-extraído.
2. **Número real de preextracción:** 219 s → **11–16 s** medidos (el plan
   decía 5–6 s).
3. **Tabla de decisión por entorno** (Mac CPU / experimento corto / Colab GPU
   / Colab reanudar).
4. **Break-even explícito:** ~4 épocas.
5. **Medición pendiente documentada** en §9 y §6.1: el efecto del `prefetch`
   sobre el solapamiento no se pudo medir sin TF; la evidencia indirecta
   (run viejo aditivo) sugiere que no solapa.
6. **Único argumento restante para on-demand:** soporte de fondo 1,1 % vs
   100 % (científico, no de velocidad).

**Correcciones de datos colaterales:** `COLAB.md` y la celda 13 del notebook
arrastran el número viejo "on-demand ~3,5 min/época" (pre-`split_2d_rows`);
con el cambio son **~18 s/época**. Y
`Trained/Probability_2D_Elongated/DNN_0/stdout_part1.log` es un run viejo
(on-demand sobre `dataset/`, 800 steps/época, ~830 s/época) — **no** es
representativo del config actual.

---

## 14. Registro de implementación

### 14.1 Base limpia (3 commits, antes de tocar el split)

| Commit | Contenido |
|---|---|
| `74e3cd1` | `chore: ignore generated artifacts and stop tracking them` — `.gitignore` para `.DS_Store`, `__pycache__`, `.pytest_cache`, `data/`, `Trained copy/`, PDFs; destrackeados los 17 artefactos que ya estaban versionados |
| `34ec4b5` | `fix(headers): map showhdr columns to NMRPipe dims via ORDER` — fix independiente que estaba sin commitear |
| `a163563` | `feat(pool): pre-extract a per-spectrum background pool` — el refactor *pool* que este plan asume como punto de partida |

### 14.2 Cambios implementados

| Fichero | Cambio |
|---|---|
| `scripts/cross_extract_fast.py` (+`colab/scripts/`) | nueva `split_2d_rows(S, idx, Cross_Size)`; docstring del módulo actualizado |
| `scripts/preextract.py` (+`colab/scripts/`) | nueva `plan_pool_idx(L, pool_k, max_pts, rng)`; `build_pool` pasa a wrapper; `main()` planifica `idx` antes del split; `--slow-split` (referencia) y `--fast` (no-op) en `mutually_exclusive_group` |
| `scripts/train_model.py` (+`colab/scripts/`) | `_points()` calcula `idx` antes del split y usa `split_2d_rows`; `self.fast_split`; la rama lenta conserva `split_2d` + `X[idx]` |
| `tests/test_split_rows.py` | **nuevo**, 12 tests / 23 subtests |
| `colab/COLAB.md`, `colab/MRAi_Colab_Train.ipynb` | tiempos actualizados, tabla de decisión y nota "GPU/`--workers` descartados" |

### 14.3 Evidencia de verificación

```
pytest tests/ -q          -> 28 passed, 3 skipped (los 3 por falta de showhdr)
py_compile (4 dirs)       -> OK
diff scripts/ colab/      -> 5/5 idénticos
notebook                  -> JSON válido, 19 celdas
byte-identity 1000 esp    -> 1000/1000 (X, y, n_pos) en 9,9 s
CLI 20 esp: default       -> 12,2 ms/espectro
CLI 20 esp: --slow-split  -> 950,9 ms/espectro  (78x)
```

El test de regresión de `tests/test_split_rows.py` reproduce el pool guardado
desde `data/pre_elong/` (mismo `seed`, mismo orden de ids → misma secuencia
`rng.choice`) y lo compara byte a byte; el chequeo de 1000 espectros de arriba
es ese mismo test extendido a todo el dataset.

### 14.4 Desviaciones respecto al plan

1. **`--slow-split` vs `--fast`:** el plan decía "`--slow-split` gana". Se
   implementó `mutually_exclusive_group`, así que pasar ambos es un error de
   argparse (más claro que una precedencia silenciosa).
2. **`--fast` se conserva** como no-op documentado (la celda 12 del notebook
   lo sigue pasando), con `help` que lo dice explícitamente.
3. **`split_2d_rows` devuelve `(len(idx), c0+c1, 1)`** en lugar de
   `(len(idx), 128, 1)` fijo: es la forma general y coincide con la
   referencia para `CROSS = [64, 64]`.
4. **Guardas de import** en `preextract.py` y `train_model.py`
   (`HAVE_FAST_SPLIT`), siguiendo el patrón que ya usaba `train_model.py`:
   sin `cross_extract_fast` el CLI cae al split completo con un aviso.
