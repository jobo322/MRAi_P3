# WNN Architecture Audit — `WNN_2D` (MRAi_P3, Fase 2)

**Tipo de documento:** auditoría de solo lectura. No se ha modificado código.
**Fecha de análisis:** generado a partir del árbol de trabajo actual.
**Fuentes inspeccionadas:**

| Fichero | Qué se tomó de él |
|---|---|
| `scripts/train_model.py` | `build_model()` (L67), `_weight_roles()` (L134), `subsample_pool()` (L209), `BinSequence` (L318), `main()` (L667) |
| `scripts/cross_extract_fast.py` | `split_2d_fast()` (L24), `split_2d_rows()` (L49) — geometría del cross |
| `scripts/weight_roles.py` | claves de rol y derivación desde el grafo del SavedModel |
| `scripts/preextract.py` | pool de fondo y distancias `d`/`d1`/`d2` |
| `src/MRAi_P3/StoP3.py`, `src/MRAi_P3/model_output.py` | contrato de inferencia (`split_2D`, `single_output`) |
| `docs/sciadv.ady7995_sm.pdf` p. 2 | listado de referencia `WNN_2D` del material suplementario |
| `docs/training_dataset_requirements.md`, `docs/local_training.md`, `docs/ring_and_picking_metrics.md` | semántica de ejes, etiquetas y métricas medidas |

Los conteos de parámetros y MACs de este documento se derivan analíticamente de las formas declaradas y
se han recalculado con un script; el total coincide con el que el propio repo afirma
(`scripts/train_mixed_precision.py`: *«6 831 parámetros y los 62 pesos»*).

---

## 1. Resumen ejecutivo

| Propiedad | Valor |
|---|---|
| Nombre | `WNN_2D` (Weighted Neural Network, 2D) |
| Entrada | `(batch, 128, 1)` — cross 1D de 64 puntos F1 + 64 puntos F2 |
| Salida | `(batch, 1)` — probabilidad escalar en `(0, 1)` |
| Parámetros | **6 831** (todos entrenables) |
| Tensores de peso | **62** (kernel + bias por capa) |
| Capas con pesos | **31** (26 `Conv1D` + 5 `Dense`) |
| Normalización / Dropout | ninguna |
| Conexiones residuales | ninguna |
| Coste | ≈ 29 780 MACs/muestra (≈ 59,6 kFLOPs) |
| Pérdida | `BinaryCrossentropy` (probabilidades, `from_logits=False`) |
| Optimizador | `Adam` (lr `1e-3` desde cero, `1e-4` en fine-tune) |
| Reparto de parámetros | brazos WNN **63,24 %**, DNN **30,16 %**, fusión **6,44 %**, cabeza **0,16 %** |

La arquitectura es un **clasificador global sobre el perfil en cruz**: dos ramas convolucionales con
*gating* multiplicativo reducen cada brazo de 64 puntos a un descriptor de 10 dimensiones, se fusionan,
y un DNN con *gating* produce la probabilidad del píxel central.

---

## 2. Entradas

### 2.1 Tensor de entrada

```text
x ∈ R^{B × 128 × 1}          B = tamaño de lote (por defecto 65 536)
```

No es un espectro 2D. Es un **vector 1D de 128 muestras** por píxel candidato, con un único canal:

```text
x[n] = [ F1(0..63) ‖ F2(0..63) ]        ← 64 puntos del brazo F1, luego 64 del brazo F2
```

### 2.2 Geometría del cross (`split_2d_rows`, `cross_extract_fast.py:49`)

Para el píxel `(i, j)` de un espectro `S ∈ R^{256 × 2048}` (con `S[k][l]`, `k = F1` eje corto/indirecto,
`l = F2` eje largo/directo, según `training_data_contract.py`):

```text
F1 arm :  x[i, j, 0:64]  = S_pad[i + 0 .. i + 63, j + 32]        (ventana vertical, eje 0)
F2 arm :  x[i, j, 64:128] = S_pad[i + 32, j + 0 .. j + 63]        (ventana horizontal, eje 1)
```

con `S_pad = np.pad(S, ((32, 31), (32, 31)), "wrap")`. Es decir: **dos ventanas de 64 puntos centradas
en el píxel**, una por eje, con `padding` periódico en los bordes. La primera mitad es el brazo F1 (el
eje elongado en este trabajo).

### 2.3 Normalización (obligatoria, previa al modelo)

Cada vector se normaliza por su **norma L2 sobre las 128 muestras**:

```text
x ← x / ‖x‖₂ ,     ‖x‖₂ = sqrt( Σ_{n=0}^{127} x[n]² )
```

En `split_2d_rows` (`cross_extract_fast.py:83-85`). Consecuencias auditadas:

* La red **nunca ve la amplitud absoluta**, solo la *forma* del perfil. Toda la información de
  intensidad relativa entre brazos queda codificada en cómo se reparte la norma.
* Si `‖x‖₂ = 0` la fila sale `NaN`; el llamante la sanea con `X[~np.isfinite(X)] = 0.0`
  (`train_model.py:479`). En inferencia, `Production_2D` filtra antes por umbral (`Spe_The`), así que
  en la práctica solo se predicen píxeles con señal.

### 2.4 Transformación de entrada (`train_model.py:109`)

```text
u = softsign(10 · x) ,     softsign(z) = z / (1 + |z|) = 10x / (1 + 10|x|)
```

* Acota la entrada a `(-1, 1)` de forma suave (sin saturación dura como `tanh`).
* La ganancia `10` está calibrada para la entrada ya normalizada: un pico que reparte su norma entre
  ~20 muestras da `x ≈ 0,22`, luego `softsign(2,2) ≈ 0,69`. Sin esa ganancia la red trabajaría en la
  zona casi lineal de `softsign`.

---

## 3. Etiquetas

| Propiedad | Valor |
|---|---|
| Tensor | `y ∈ {0, 1}^{B × 1}` (`float32`) |
| Origen | `labels/<id>.bin`, `uint8` binario, misma forma que el espectro |
| Semántica | marca **sub-píxel** del pico: 1/2/4 px por pico (mediana 2, media **2,26**) |
| Densidad | 256 picos/espectro; 575 835 positivos en 1000 espectros (≈ **576/espectro**) |
| Fondo | el resto; el pico real abarca ~20–50 px, así que **los ceros dentro del pico son fondo** |

Punto crítico que la auditoría subraya: **el target es el marcador, no la cresta**. Sobre un pico de
~10–50 px de ancho se etiquetan 1–4 píxeles. Las consecuencias están medidas en
`docs/ring_and_picking_metrics.md:82`:

* el modelo aprende una **cresta en F1**, no un punto: a ±1 px la probabilidad es `0,259` en F1 y
  `0,030` en F2;
* 97 picos reales generan 249 detecciones (2,6× por pico, separadas ~4,0 px de mediana);
* el muestreo de fondo debe compensarse con negativos duros (`--hard-frac`), porque muestrear
  uniforme deja los ceros intra-pico al 0,48 % del fondo y el modelo da `P > 0,5` al 81 % de los
  vecinos inmediatos.

`docs/ring_and_picking_metrics.md:283` y `scripts/ridge_target.py` documentan la alternativa
(target de cresta) que **no** está implementada en el modelo auditado.

---

## 4. Arquitectura real, paso a paso

### 4.0 Notación

Convolución 1D `'valid'`, canales-last (`(batch, steps, channels)`):

```text
y[n, c_out] = b[c_out] + Σ_{c_in=1}^{C_in} Σ_{k=0}^{K-1} W[k, c_in, c_out] · x[n·S + k, c_in]
```

`σ(·)` es la sigmoide, `⊙` el producto elemento a elemento.

### 4.1 Preprocesado y split (`train_model.py:104-118`)

```text
size = [64, 64] ,   nlev = [log2(64), log2(64)] = [6, 6]

u    = softsign(10 · x)                                    u ∈ R^{B × 128 × 1}
u_0  = u[:, 0:64, :]      (brazo F1)                       u_0 ∈ R^{B × 64 × 1}
u_1  = u[:, 64:128, :]    (brazo F2)                       u_1 ∈ R^{B × 64 × 1}
```

En Keras 3 el `tf.split` del SM se sustituye por slicing con `Lambda` y `output_shape` estático
(`train_model.py:113-118`): `K.split` perdía la longitud estática y la conv colapsaba a 0.

### 4.2 Bloque *gated* de convolución (el ladrillo del WNN)

Por cada nivel `ℓ = 1..6` y cada brazo, dos convoluciones **en paralelo** sobre la *misma* entrada,
combinadas multiplicativamente:

```text
t_ℓ = tanh( W^{t}_ℓ *_{k=2, s=2} a_{ℓ-1} + b^{t}_ℓ )
s_ℓ = σ   ( W^{s}_ℓ *_{k=2, s=2} a_{ℓ-1} + b^{s}_ℓ )
a_ℓ = t_ℓ ⊙ s_ℓ
```

* `t` es la rama **valor** (tanh, salida en `(-1,1)`).
* `s` es la rama **compuerta** (sigmoide, salida en `(0,1)`): modula cuánto de `t` pasa.
* Es un *gating* multiplicativo análogo a un GRU/LSTM simplificado, aplicado espacialmente.

`k=2, s=2` ⇒ cada nivel **divide la longitud por 2**:

```text
64 → 32 → 16 → 8 → 4 → 2 → 1
```

Tras los 6 niveles cada brazo colapsa a `a_6 ∈ R^{B × 1 × 10}`: un descriptor global de 10 números.

**Campo receptivo:** con 6 niveles de stride 2 sobre 64 puntos, la salida depende de **todo el brazo**.
Aunque la implementación sea convolucional, no hay localidad espacial efectiva: el WNN es un extractor
de características globales del perfil.

### 4.3 Fusión (`train_model.py:119-122`)

```text
g_0 = W^{f}_0 *_{k=1} a_6^{(F1)} + b^{f}_0        10 → 20 canales, lineal (sin activación)
g_1 = W^{f}_1 *_{k=1} a_6^{(F2)} + b^{f}_1        10 → 20 canales, lineal
```

### 4.4 Concatenación y *squeeze* (`train_model.py:124`)

```text
z = squeeze( concat([g_0, g_1], axis=-1), axis=1 )        (B,1,20)‖(B,1,20) → z ∈ R^{B × 40}
```

### 4.5 Bloque *gated* denso (el ladrillo del DNN, `train_model.py:93-103`)

Dos bloques con la misma estructura que el WNN pero sobre vectores:

```text
Bloque 1 (units = 2·filter = 20):   p = tanh(W^{t}_1 z + b^{t}_1) ,  q = σ(W^{s}_1 z + b^{s}_1) ,  z_1 = p ⊙ q
Bloque 2 (units =   filter = 10):   r = tanh(W^{t}_2 z_1 + b^{t}_2) , v = σ(W^{s}_2 z_1 + b^{s}_2) , z_2 = r ⊙ v
```

### 4.6 Cabeza (`train_model.py:126`)

```text
p̂ = σ( w_o · z_2 + b_o ) ∈ (0, 1)
```

Capa nombrada **`dense_14`** para conservar el nombre del SavedModel original de Keras 2.

### 4.7 Forward completo en una línea

```text
p̂(x) = σ( w_o · DNN( concat( fus_F1( WNN_6(u_0) ), fus_F2( WNN_6(u_1) ) ) ) + b_o )
        con u = softsign(10 x) ,  x = cross(i,j) / ‖cross(i,j)‖₂
```

---

## 5. Parámetros por capa

`filter = 10`. `K` = tamaño de kernel, `b` = sesgo. Formas en formato `(longitud, canales)`.

| # | rol | capa | forma | kernel | bias | params |
|---|---|---|---|---|---|---|
| 1 | `conv\|F1\|1\|t` | `Conv1D(10,k=2,s=2,tanh)` | (64,1)→(32,10) | 20 | 10 | **30** |
| 2 | `conv\|F1\|1\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (64,1)→(32,10) | 20 | 10 | **30** |
| 3 | `conv\|F1\|2\|t` | `Conv1D(10,k=2,s=2,tanh)` | (32,10)→(16,10) | 200 | 10 | **210** |
| 4 | `conv\|F1\|2\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (32,10)→(16,10) | 200 | 10 | **210** |
| 5 | `conv\|F1\|3\|t` | `Conv1D(10,k=2,s=2,tanh)` | (16,10)→(8,10) | 200 | 10 | **210** |
| 6 | `conv\|F1\|3\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (16,10)→(8,10) | 200 | 10 | **210** |
| 7 | `conv\|F1\|4\|t` | `Conv1D(10,k=2,s=2,tanh)` | (8,10)→(4,10) | 200 | 10 | **210** |
| 8 | `conv\|F1\|4\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (8,10)→(4,10) | 200 | 10 | **210** |
| 9 | `conv\|F1\|5\|t` | `Conv1D(10,k=2,s=2,tanh)` | (4,10)→(2,10) | 200 | 10 | **210** |
| 10 | `conv\|F1\|5\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (4,10)→(2,10) | 200 | 10 | **210** |
| 11 | `conv\|F1\|6\|t` | `Conv1D(10,k=2,s=2,tanh)` | (2,10)→(1,10) | 200 | 10 | **210** |
| 12 | `conv\|F1\|6\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (2,10)→(1,10) | 200 | 10 | **210** |
| 13 | `fus\|F1` | `Conv1D(20,k=1,linear)` | (1,10)→(1,20) | 200 | 20 | **220** |
| 14 | `conv\|F2\|1\|t` | `Conv1D(10,k=2,s=2,tanh)` | (64,1)→(32,10) | 20 | 10 | **30** |
| 15 | `conv\|F2\|1\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (64,1)→(32,10) | 20 | 10 | **30** |
| 16 | `conv\|F2\|2\|t` | `Conv1D(10,k=2,s=2,tanh)` | (32,10)→(16,10) | 200 | 10 | **210** |
| 17 | `conv\|F2\|2\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (32,10)→(16,10) | 200 | 10 | **210** |
| 18 | `conv\|F2\|3\|t` | `Conv1D(10,k=2,s=2,tanh)` | (16,10)→(8,10) | 200 | 10 | **210** |
| 19 | `conv\|F2\|3\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (16,10)→(8,10) | 200 | 10 | **210** |
| 20 | `conv\|F2\|4\|t` | `Conv1D(10,k=2,s=2,tanh)` | (8,10)→(4,10) | 200 | 10 | **210** |
| 21 | `conv\|F2\|4\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (8,10)→(4,10) | 200 | 10 | **210** |
| 22 | `conv\|F2\|5\|t` | `Conv1D(10,k=2,s=2,tanh)` | (4,10)→(2,10) | 200 | 10 | **210** |
| 23 | `conv\|F2\|5\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (4,10)→(2,10) | 200 | 10 | **210** |
| 24 | `conv\|F2\|6\|t` | `Conv1D(10,k=2,s=2,tanh)` | (2,10)→(1,10) | 200 | 10 | **210** |
| 25 | `conv\|F2\|6\|s` | `Conv1D(10,k=2,s=2,sigmoid)` | (2,10)→(1,10) | 200 | 10 | **210** |
| 26 | `fus\|F2` | `Conv1D(20,k=1,linear)` | (1,10)→(1,20) | 200 | 20 | **220** |
| 27 | `dense\|0\|t` | `Dense(20,tanh)` | (40,)→(20,) | 800 | 20 | **820** |
| 28 | `dense\|0\|s` | `Dense(20,sigmoid)` | (40,)→(20,) | 800 | 20 | **820** |
| 29 | `dense\|1\|t` | `Dense(10,tanh)` | (20,)→(10,) | 200 | 10 | **210** |
| 30 | `dense\|1\|s` | `Dense(10,sigmoid)` | (20,)→(10,) | 200 | 10 | **210** |
| 31 | `out` | `Dense(1,sigmoid)` | (10,)→(1,) | 10 | 1 | **11** |
| | | | | | **TOTAL** | **6 831** |

Capas sin parámetros: `Input`, `Lambda` de softsign, dos `Lambda` de slicing, `Lambda` de
`squeeze(concat)`.

---

## 6. Distribución de parámetros: WNN + DNN

### 6.1 Por bloque funcional

| Bloque | Capas | Parámetros | % del total | % acumulado |
|---|---:|---:|---:|---:|
| Brazo WNN **F1** (12 convs) | 12 | 2 160 | 31,62 % | 31,62 % |
| Brazo WNN **F2** (12 convs) | 12 | 2 160 | 31,62 % | 63,24 % |
| **Subtotal WNN** | 24 | **4 320** | **63,24 %** | |
| Fusión F1 (`Conv1D(20,k=1)`) | 1 | 220 | 3,22 % | 66,46 % |
| Fusión F2 (`Conv1D(20,k=1)`) | 1 | 220 | 3,22 % | 69,68 % |
| **Subtotal fusión** | 2 | **440** | **6,44 %** | |
| **DNN** bloque 0 (2×`Dense(20)`) | 2 | 1 640 | 24,01 % | 93,69 % |
| **DNN** bloque 1 (2×`Dense(10)`) | 2 | 420 | 6,15 % | 99,84 % |
| **Subtotal DNN** | 4 | **2 060** | **30,16 %** | |
| Cabeza `Dense(1)` | 1 | 11 | 0,16 % | 100,00 % |
| **TOTAL** | **31** | **6 831** | **100 %** | |

### 6.2 Por tipo de capa

| Tipo | Capas | Parámetros | % |
|---|---:|---:|---:|
| `Conv1D` (brazos + fusión) | 26 | 4 760 | 69,68 % |
| `Dense` (DNN + cabeza) | 5 | 2 071 | 30,32 % |

### 6.3 Por nivel dentro de un brazo WNN (2 160 params)

| Nivel | Longitud | Params (t + s) | % del brazo |
|---|---|---:|---:|
| 1 | 64 → 32 | 60 | 2,78 % |
| 2 | 32 → 16 | 420 | 19,44 % |
| 3 | 16 → 8 | 420 | 19,44 % |
| 4 | 8 → 4 | 420 | 19,44 % |
| 5 | 4 → 2 | 420 | 19,44 % |
| 6 | 2 → 1 | 420 | 19,44 % |

**Lectura del reparto:** el nivel 1 apenas pesa (2,78 %) porque su entrada es de 1 solo canal; a partir
del nivel 2 el coste se estabiliza en 420 por nivel y ya no depende de la longitud. El **DNN concentra
el 30 % de los parámetros en solo 5 capas**, y su bloque 0 (`Dense(20)` sobre 40 entradas) es la capa
más grande de todo el modelo (820 params cada una). Los dos brazos son exactamente simétricos
(2 160 cada uno): la asimetría entre F1 y F2 **no está en los pesos**, solo en los datos.

### 6.4 Coste computacional (MACs, forward, por muestra)

| Bloque | MACs |
|---|---:|
| Brazo F1 (L1..L6) | 13 680 |
| Brazo F2 (L1..L6) | 13 680 |
| Fusión (2×) | 400 |
| DNN (bloque 0 + 1) | 2 000 |
| Cabeza | 20 |
| **TOTAL** | **29 780** (≈ 59,6 kFLOPs) |

Con `batch = 65 536`: ≈ **1,95 GMACs por batch**. Con 35 steps/época: ≈ **68 GMACs/época**.
El cómputo del modelo es despreciable frente al pipeline de datos: por época entran
`800 × 2 905 × 128 ≈ 297 M` valores de entrada.

---

## 7. Los 62 pesos y `--init` por rol

`build_model()` devuelve el modelo con un atributo extra, `model.weight_roles` (`train_model.py:130`,
`_weight_roles()` L134): la lista de roles en el **mismo orden que `model.weights`**. Se autovalida:
si algún peso queda sin rol, lanza `ValueError` en vez de copiar mal.

### 7.1 Esquema de claves (`weight_roles.py`)

```text
conv|<arm>|<level>|<act>|<k|b>       conv|F1|1|t|k        arm ∈ {F1,F2}, level ∈ 1..6, act ∈ {t,s}
fus|<arm>|<k|b>                      fus|F2|b
dense|<block>|<act>|<k|b>            dense|0|t|k          block ∈ {0,1}
out|<k|b>                            out|b
```

24 claves de brazo ×2 = 48, + 4 de fusión, + 8 de DNN, + 2 de cabeza = **62**.

### 7.2 Por qué hace falta (y no vale emparejar por forma)

Documentado y medido en `weight_roles.py`: el SavedModel base de Keras 2 lista las variables
**intercaladas por nivel y con la rama F2 primero**:

```text
(2,1,10) → [F2 L1 Tanh, F2 L1 Sigmoid, F1 L1 Tanh, F1 L1 Sigmoid]
```

mientras `build_model()` crea **F1 entera y luego F2**. Como todas las formas coinciden entre ramas y
niveles, emparejar por forma asignaba a cada brazo los pesos del otro (**48 de 62**), y el fine-tune
arrancaba peor que aleatorio: **BCE 1,0433 frente a 0,3524** del modelo base.

El rol se deduce del **grafo** (puerto del `Split`/`SplitV` o offset del `StridedSlice`, activación que
consume la salida, profundidad de convs aguas arriba), no de los nombres, para funcionar igual con el
SavedModel de Keras 2 y con un re-export de Keras 3.

---

## 8. Flujo de entrenamiento

### 8.1 Datos: dos rutas (`main()`, L667)

| Ruta | Flag | Fuente | Coste dominante |
|---|---|---|---|
| A — bajo demanda | (por defecto) | `spectra/<id>.bin` + `labels/<id>.bin` vía `memmap` (`SHAPE = (256,2048)`) | si `--hard-frac > 0`: la EDT (`distance_to_positive`) **por espectro y época** |
| B — pool pre-extraído | `--preextracted` | `<id>.npz` de `preextract.py`: positivos + fondo + distancias `d`,`d1`,`d2` | lectura del pool completo por época |

Ruta A (`_points`, L413): muestrea todos los positivos + `bg_ratio · n_pos` negativos, construye el
cross **solo de las filas muestreadas** (`split_2d_rows`, ~60× menos trabajo que la matriz completa) y
sanea `NaN → 0`.

Ruta B (`_load_pre`, L483 → `subsample_pool`, L209): lee el pool y submuestrea por época con un `rng`
nuevo, así que **la variación de fondo se conserva sin repagar `split_2d`**.

### 8.2 Muestreo de fondo (no uniforme)

El fondo se reparte en bandas (`choose_background` en `hard_negatives.py`):

```text
n_bg = min(len(X) − n_pos, ⌊n_pos · bg_ratio⌋, max_pts − n_pos)
     = ⌊hard_frac · n_bg⌋      filas del anillo      (d ≤ hard_px)
     + ⌊hard_frac2 · n_bg⌋     filas de la banda 2    (hard_px < d ≤ band2_px, o rectángulo aniso)
     + resto                   uniforme
```

Por defecto `hard_frac = 0.25`, `hard_px = 2`, `bg_ratio = 4.0`. Con `--hard-axis-frac` se reserva parte
del presupuesto de la banda al eje F1 puro (`d2 == 0`). Todos los positivos entran siempre y el tamaño
del lote no cambia. El `rng` se consume en orden fijo, y ese orden está fijado por tests
(`test_hard_bands.py`, `test_pool_equiv.py`).

### 8.3 Batching y época

```text
points_per_spectrum ≈ n_pos + n_bg ≈ 2 905            (n_pos ≈ 581, bg_ratio = 4)
steps_per_epoch     = (n_ids · points_per_spectrum) // batch
                    = (800 · 2 905) // 65 536 ≈ 35
```

El `tf.data` (`to_dataset`, L528) es `from_generator(gen, output_signature=...)` → `.repeat()` →
`.prefetch(2)`:

* **`.repeat()` no es opcional**: Keras 3 solo recrea el iterador si `_current_iterator is None`, y el
  bloque de reset exige `_num_batches`, que es `None` para `from_generator`. Sin `.repeat()`, un
  generador que se agota en el límite de época deja un iterador muerto: la época siguiente hace 0
  batches, todas las métricas salen 0.0 y solo aparece un `UserWarning`. Ocurrió de verdad: **105 de
  210 épocas sin entrenar** (15-oct-2026).
* `.prefetch(2)` solapa la producción (un hilo) con el cómputo; no crea un segundo productor.

`output_signature`:

```text
X : TensorSpec((None, 128, 1), float32)
y : TensorSpec((None, 1),     float32)
```

### 8.4 Objetivo, optimizador y métricas (`train_model.py:800`)

```text
L(y, p̂) = −[ y · log p̂ + (1 − y) · log(1 − p̂) ]        (media sobre el lote; from_logits = False)

optimizer = Adam(learning_rate = lr)                     lr = 1e-3 desde cero, 1e-4 con --init/--resume
metrics   = [BinaryAccuracy("acc"), AUC("auc")]
```

### 8.5 Inicialización

* `--init <SavedModel>`: copia **por rol** con `weight_roles.assign_weights` (ver §7). Informa
  `copiados/total` y `aleatorios`.
* `--resume <ckpt.keras>`: `load_weights` exacto (incluye optimizer si se compiló antes).

### 8.6 Callbacks (`train_model.py:920`)

| Orden | Callback | Función |
|---|---|---|
| 1 | `StepGuard` | aborta si una época hace 0 batches (red de seguridad del fallo del 15-oct) |
| 2 | `EarlyStopping(val_loss, patience=20, restore_best_weights=True)` | parada temprana |
| 3 | `ModelCheckpoint(ckpt.keras, monitor="val_loss", save_best_only=True)` | mejor modelo |
| 4 | `CSVLogger(training.log, append=bool(resume))` | curva de entrenamiento |
| 5 | `DriveSync` (solo con `--scratch`) | copia atómica local → destino cada `--sync-every` épocas |

El `StepGuard` va **primero** para abortar antes de que los demás escriban checkpoint o log.

### 8.7 Llamada a `fit` y post-proceso

```text
model.fit(train_seq.to_dataset(),
          validation_data = val_seq.to_dataset(),
          epochs          = 210,
          callbacks       = cbs,
          steps_per_epoch = len(train_seq),
          validation_steps= max(1, len(val_seq)))
```

Después: `history.npy` (N,2 = [train, val]) y `history.txt`, `model.export(<out>_savedmodel)` y un
smoke test `model.predict(np.zeros((1024,128,1)))`.

---

## 9. Contrato de inferencia

| Aspecto | Detalle |
|---|---|
| Consumidor | `src/MRAi_P3/StoP3.py` → `Production_2D` |
| Carga | `TFSMLayer(<dir>/DNN_0, call_endpoint="serving_default")` dentro de un `Sequential` |
| Preprocesado | `split_2D(S, [64,64])` — **espejo NumPy exacto** de `split_2d_rows`, misma normalización L2 y mismo `pad` `wrap` |
| Filtrado previo | `RI = |S| ≥ Spe_The · std(S)`: solo se predicen los píxeles que pasan el umbral |
| Salida | `single_output(model.predict(...))` — no depende del nombre de la capa |
| Nombre de la capa | `dense_14` (Keras 2) o `output_0` (re-export Keras 3); `single_output` exige **una única salida** y falla si hay más |

El acoplamiento crítico es la **geometría del cross**: si `split_2D` y `split_2d_rows` divergieran, el
modelo entrenado recibiría una entrada distinta a la de entrenamiento. Está pinado por
`tests/test_split_rows.py` (paridad bit a bit).

---

## 10. Desviaciones respecto al material suplementario (p. 2)

El SM lista `WNN_2D` así:

```python
x_0 = WNN(I[0], filter)(x_0)
x_1 = WNN(I[1], filter)(x_1)
x_0 = tf.keras.layers.Conv1D(3*filter, kernel_size=1)(x_0)     # ← SM: 3*filter
x_1 = tf.keras.layers.Conv1D(3*filter, kernel_size=1)(x_1)
output = tf.squeeze(tf.concat([x_0,x_1],-1), 1)
output = DNN(filter)(output)
output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
```

| Aspecto | SM (p. 2) | Implementación auditada | Nota |
|---|---|---|---|
| Fusión `Conv1D` | `3·filter` = 30 canales | **`2·filter` = 20** | La docstring lo declara explícitamente: *«(=20, como en pesos guardados)»*. El SM de 3D usa `2·filter`, así que el `3·` del SM 2D es una inconsistencia del propio SM. Se adopta 20 para ser compatible con los SavedModels base. |
| Split de entrada | `tf.split(x, [Size[0],Size[1]], 1)` | `Lambda` con `output_shape` estático | En Keras 3 `K.split` pierde la longitud estática y la conv colapsa a 0. |
| Activaciones | `tf.keras.activations.softsign` directo | `Lambda(lambda t: K.softsign(10*t), output_shape=...)` | Las ops TF puras no aceptan `KerasTensor` en Keras 3. |
| Trazabilidad de pesos | no existe | `created` + `model.weight_roles` | Añadido en este repo para `--init` por rol. |
| `DNN` | `Dense(2·filter)` y `Dense(filter)` | igual | Coincide. |
| `WNN` | `for i in range(layer)` | `for lvl in range(1, layer+1)` | Misma semántica, con nombre de nivel para el rol. |

**Consecuencia del cambio 30 → 20:** el número de parámetros de fusión baja de 2×(1·10·30+30) = 660 a
440. Un modelo entrenado con 30 canales de fusión **no es cargable** en este `build_model()`
(desajuste de forma en `fus|F1|k` / `fus|F2|k`). Si algún día se reentrena desde el SM literal, hay que
cambiar esa línea y regenerar los roles.

---

## 11. Invariantes y trampas conocidas

1. **`.repeat()` obligatorio** en `to_dataset()`: sin él, época vacía silenciosa (§8.3).
2. **Normalización L2 antes del softsign**: cambiar una sin la otra rompe la calibración de la ganancia
   10. Un vector de norma 0 produce `NaN`; hay que sanearlo.
3. **Orden del `rng` en el muestreo**: es parte del contrato de reproducibilidad
   (`subsample_pool`, `choose_background`). Reordenar el complemento de filas reservadas cambia las
   filas elegidas con la misma semilla — está fijado por tests.
4. **Los brazos son simétricos en parámetros**: cualquier asimetría F1/F2 observada viene de los datos
   o del muestreo, no de la arquitectura.
5. **Campo receptivo global**: cada brazo ve sus 64 puntos completos; no hay localidad espacial. La
   red no puede distinguir "el pico está en la muestra 30" más allá de lo que codifique el descriptor
   de 10 dimensiones.
6. **Target = marcador sub-píxel, no cresta**: explica la cresta de falsos positivos en F1 (§3).
7. **Fusión 20, no 30**: incompatibilidad de pesos con el SM literal (§10).

---

## 12. Apéndice — derivación de los conteos

```text
Conv1D(C_out, k, C_in)  →  k · C_in · C_out + C_out
Dense(C_out, C_in)      →  C_in · C_out + C_out

Brazo WNN (6 niveles, filter=10, entrada 64×1):
  L1 : 2 × (2·1·10 + 10)   = 2 × 30  =   60
  L2 : 2 × (2·10·10 + 10)  = 2 × 210 =  420
  L3..L6: 4 × 420          =         1 680
  ─────────────────────────────────────────
  por brazo                          2 160
  × 2 brazos                         4 320

Fusión: 2 × (1·10·20 + 20) = 2 × 220 =   440

DNN:   2 × (40·20 + 20) = 2 × 820  =  1 640
       2 × (20·10 + 10) = 2 × 210  =    420
                                      ─────
                                      2 060

Cabeza: 10·1 + 1                    =     11
                                     ──────
TOTAL                                 6 831     ✔ coincide con el repo

Tensores: 31 capas × (kernel + bias) = 62      ✔ coincide con el repo
```

---

*Auditoría de solo lectura: no se ha modificado ningún fichero de código, datos o configuración.*
