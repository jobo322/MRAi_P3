# Requerimientos del conjunto de entrenamiento — MR-Ai P3 2D-US elongado en F1

> Alcance: re-entrenamiento / fine-tune del modelo 2D-US (`Trained/Probability_2D/US/DNN_0`,
> entrada `(128,1) = 64+64`) para espectros **2D-US** con picos **elongados en la dimensión
> indirecta (F1)**. Síntesis **directamente en el dominio de la frecuencia** (sin FID ni FT).
> Referencia metodológica: `docs/sciadv.ady7995.pdf` (págs. 12–13) + `docs/sciadv.ady7995_sm.pdf` (págs. 2–4).
> Implementación de inferencia: `src/MRAi_P3/StoP3.py` (`split_2D`, `Production_2D`, `P3_2D`).

---

## 1. Nomenclatura técnica (obligatoria en código y documentación)

Para evitar la ambigüedad entre "ancho" y "forma", se adopta la nomenclatura estándar de
perfil pseudo-Voigt:

| Símbolo | Nombre técnico | Descripción | Unidad / rango | Parámetro de control |
|---|---|---|---|---|
| `Γ_F1`, `Γ_F2` | **FWHM** (*full width at half maximum*, anchura a media altura) | **Parámetro a variar**: anchura del pico por dimensión. Controla la elongación. `Γ_F1 = r · Γ_F2` | Hz (generación) → puntos (§3) | Sí — variable principal del diseño |
| `η_F1`, `η_F2` | **Fracción lorentziana** (*Lorentzian fraction / mixing factor*) | **Parámetro de forma (*shape*)**: `η=0` gaussiana pura, `η=1` lorentziana pura, `0<η<1` pseudo-Voigt | Adimensional `[0,1]` | Sí — secundaria, independiente por dimensión |
| `r` | **Relación de anisotropía** (*anisotropy ratio*) `r = Γ_F1 / Γ_F2` | Cuantifica la elongación en indirecta | Adimensional `≥1` | Derivado, nunca primario de muestreo |
| `A` | **Amplitud** del pico | Altura del máximo | Normalizada `[0.05,1]` (paper, Tabla S1) | Sí |
| `(ν1₀, ν2₀)` | **Posición del centro** (desplazamiento químico) | Centro continuo sub-píxel del pico | Hz o ppm, continuo | Sí — uniforme + pares controlados (§5) |
| `φ` | **Error de fase residual** (mezcla dispersiva) | Fracción de componente dispersiva por fase imperfecta | Ver §6 | Opcional fase 2 |

> Regla: en todo el código, `Gamma_F1_Hz / Gamma_F2_Hz` denotan FWHM en Hz;
> `eta_F1 / eta_F2` denotan forma; `r` es derivado, nunca parámetro primario de muestreo.
> No usar `w` ni `mu` como nombres públicos.

### 1.1 Definición matemática del perfil (FWHM-consistente)

Con `x` = distancia al centro en la unidad de la dimensión, y `Γ` = FWHM en esa misma unidad:

```text
G(x; Γ) = exp( -4·ln(2)·x² / Γ² )          # gaussiana, FWHM = Γ
L(x; Γ) = 1 / ( 1 + 4·x² / Γ² )            # lorentziana, FWHM = Γ
V(x; Γ, η) = η·L(x; Γ) + (1-η)·G(x; Γ)     # pseudo-Voigt, FWHM ≈ Γ, normalizado a 1 en x=0
```

Pico 2D separable (requerido):

```text
S_peak(i,j) = A · V_F1(i − i₀; Γ_F1, η_F1) · V_F2(j − j₀; Γ_F2, η_F2)
```

Ambas componentes comparten el mismo `Γ` dentro de cada dimensión para que `η` module
**solo la forma (colas)**, no el ancho efectivo.

---

## 2. Geometría espectral y rejilla de síntesis

### 2.1 Geometría experimental de referencia

| Dimensión | Eje NMR típico | SW (ejemplo) | N experimental | N tras zero-filling (producción) |
|---|---|---|---|---|
| F2 (directa) | ¹H | `SW_F2` Hz (p. ej. 5000–10000) | hasta 16K (16384) | 16384 (sin ZF adicional) |
| F1 (indirecta) | ¹⁵N | `SW_F1` Hz (p. ej. 2000–3500) | ~40 en t1 | **N1 = 256 o 512** (fijar uno para train = producción) |

### 2.2 Rejilla sintética (parches, no matrices completas)

- **NO** generar matrices de `16384 × 256` (~32 MB c/u → ~20 GB por 640 espectros).
- Generar **parches** de `N1 × N2_synth = 256 × 512` (alternativa: `256 × 1024`)
  con la **misma escala Hz/punto** que producción (§3).
- Posiciones `(i₀, j₀)` continuas sub-píxel, uniformes en el parche.
- Bordes: padding periódico (`wrap`), idéntico a `split_2D()` en
  `src/MRAi_P3/StoP3.py:20-22` (`np.pad(S, pad_width, 'wrap')`).
- Fijar `N1_synth = N1_producción` (256 o 512). No generar a 40 + ZF a posteriori:
  generar directamente a `N1` final con `Γ_Hz` real.

### 2.3 Ventana de la red (restricción dura)

- `split_2D(S, Cross_Size=[64,64])` (`StoP3.py:17-26`): por punto, 64 pts en F1 + 64 pts
  en F2 → vector `(128,1)` normalizado por norma L2. Arquitectura `WNN_2D` (SM p. 2):
  `Input([128,1]) → split[64,64] → 6× Conv1D(k=2,s=2) → 64→…→1` por rama.
- **Los 64 puntos NO deben contener la señal completa hasta línea base.**
  Requisito real: **lóbulo central contenido con margen**:
  `FWHM_pts ≲ 15–20` y primer cruce por cero dentro de `±32` puntos del centro.
- Si `FWHM_F1_pts > ~25`, escalar a ventana asimétrica (opción C del plan,
  rompe compatibilidad de pesos). Ver criterio §3.3.
---

## 3. Conversión Hz → puntos (normativa)

La red aprende en **puntos**; la generación se parametriza en **Hz**. Conversión obligatoria
por dimensión:

```text
Δν (Hz/punto) = SW_Hz / N_freq
Γ_pts         = Γ_Hz / Δν = Γ_Hz · N_freq / SW_Hz
```

Ejemplo ilustrativo (sustituir por valores medidos):

```text
F2: SW=8000 Hz, N=16384 → 0.49 Hz/pt → Γ=15 Hz ⇒ ~31 pts
F1: SW=2500 Hz, N=256   → 9.8 Hz/pt  → Γ=30 Hz ⇒ ~3 pts
```

> "Elongado en Hz" ≠ "elongado en puntos". El diseño (`r`, densidades, overlaps) se fija
> en **puntos**; la síntesis se ejecuta en **Hz**.

### 3.1 Valores a medir en datos reales (entrada requerida antes de generar)

- [ ] `SW_F1_Hz`, `SW_F2_Hz` (cabecera NMRPipe / `nmrglue`)
- [ ] `N1` final (256 o 512), `N2` final (16384)
- [ ] `Γ_F1_Hz`, `Γ_F2_Hz` típicos (ajuste pseudo-Voigt en 5–10 picos aislados reales)
- [ ] `η` aparente (¿colas lorentzianas o perfil gaussianizado por apodización?)
- [ ] Presencia de *sinc-wiggles* por truncamiento t1=40 (§6)

### 3.2 Tabla de diseño (rellenar con valores medidos)

| Parámetro | F2 (directa) | F1 (indirecta) |
|---|---|---|
| `SW_Hz` | … | … |
| `N_freq` | 16384 | 256 / 512 |
| `Δν` (Hz/pt) | … | … |
| `Γ_Hz` típico | … | … |
| `Γ_pts` (= `Γ_Hz/Δν`) | … | … |
| `r_pts = Γ_F1_pts/Γ_F2_pts` | — | … |

### 3.3 Criterio de aceptación de ventana

- [ ] `Γ_F1_pts ≤ 20` y colas pseudo-Voigt (con `η` máx. de diseño) decaen a `<5%` antes
  de `±32` pts → ventana 64 válida (opciones A/B del plan).
- [ ] En caso contrario → ventana asimétrica (opción C), nuevo `Input` + parche de
  `Production_2D()` / `P3_2D()`.

---

## 4. Perfil pseudo-Voigt anisotrópico (especificación de síntesis)

### 4.1 Parámetros por pico

| Parámetro | Distribución requerida | Notas |
|---|---|---|
| `A` | Uniforme `[0.05, 1]` (paper, Tabla S1) | Estratificar ratios en pares (§5.3) |
| `Γ_F2` | Estrecho en puntos: `Γ_F2_pts ≈ 3–5` (calibrar a reales §3.1); convertir a Hz para síntesis | Ancla de la escala |
| `Γ_F1` | `Γ_F1 = r · Γ_F2` (en puntos; convertir a Hz por dimensión), `r ∈ {1, 2, 3, 5, 8}` | `r=1` incluido como control anti-regresión |
| `η_F1`, `η_F2` | Independientes por dimensión. Base uniforme `[0,1]` + extremos puros: ~10–20% con `η=0`, ~10–20% con `η=1` | `η` controla colas; `r` controla elongación. No acoplarlos |
| `(ν1₀, ν2₀)` | Uniforme continuo + inyección de pares/clusters (§5) | Sub-píxel obligatorio para regla 0.25–0.75 |

### 4.2 Procedimiento de pintado (por espectro sintético)

1. Inicializar `S = zeros(N1, N2_synth)`.
2. Para cada pico `k`: `S += A_k · V_F1(·; Γ_F1,k, η_F1,k) · V_F2(·; Γ_F2,k, η_F2,k)`
   con centros continuos `(i₀, j₀)`.
3. Sumar todos los picos (overlaps por superposición lineal).
4. Añadir ruido gaussiano blanco **en frecuencia** (§6).
5. Guardar `S` float32 + lista de centros verdaderos para etiquetado (§7).

---

## 5. Diseño orientado a solapamiento (requisito central)

Base: **256 picos / espectro** (paper §2D-US), **640 espectros**, split **4:1 train/val**.

### 5.1 Composición por espectro (punto de partida)

| Grupo | Fracción | Definición operativa |
|---|---|---|
| Aislados | ~40% | Vecino más cercano a `> 4·FWHM` en ambas dimensiones |
| Pares controlados | ~40% | Separación muestreada según §5.2 |
| Triples / clusters | ~10% | 3 picos en radio `±2·FWHM` |
| Hombros extremos | ~10% | Separación `< 1·FWHM` (máximos fusionados) |

### 5.2 Separación en unidades de FWHM (no en píxeles)

- Normalizar por eje: `d1 = ΔF1 / Γ_F1_pts`, `d2 = ΔF2 / Γ_F2_pts`.
- Muestrear `d` log-uniforme en `[0.3, 5]`, énfasis en `[0.5, 2.5]`
  (transición resuelto/no-resuelto, cf. Fig. 8; zona gris `< 3σCRLB`).
- Orientaciones: mayoría **solo en F1** (alineados en ¹H, distintos en ¹⁵N);
  fracción en ambas; minoría solo en F2.

### 5.3 Amplitudes en overlaps

- Ratio `A_débil / A_fuerte` log-uniforme en `[0.05, 1]`; ~50% de pares en `[0.05, 0.3]`
  (cf. Fig. 2C del paper, ratio 3:1 → P3 ~0.8 en ambos).

### 5.4 Nota sobre densidad vs elongación

- El área por pico crece ~`r` veces → más overlap incidental. **Reportar**
  `% de pares con d<2` por cada `r`. Si se satura: ~150–200 picos para `r` grande
  o F1 sintético a 512.

---

## 6. Ruido, fase y artefactos (caso 2D-US)

| Efecto | Especificación | Justificación |
|---|---|---|
| Ruido base | Gaussiano blanco en **frecuencia**, `SD_ruido = A_min` (paper p. 12) | Para 2D-US ≈ temporal tras FT; correlación por apodización/ZF es 2º orden (documentar) |
| Fase residual | Fase 1: absortivos puros (documentar). Fase 2: dispersiva pequeña aleatoria, `φ ∈ [−0.5, 0.5]` Tabla S1 | El paper entrena con fase aleatoria; empezar sin fase es simplificación consciente |
| Truncamiento t1=40 | Fase 1: ignorar y validar. Si hay *sinc-wiggles* (~`SW/40`): Fase 2a: término `sinc` débil en F1; Fase 2b: híbrido F1 vía FT de FID 1D de 40 pts + F2 Voigt | El truncamiento real no es Voigt puro |
| NUS / CS-IST | **Excluido**: solo 2D-US. Cauchy+gaussiano y doble-gaussiano (pp. 12–13) no aplican aquí | Artefactos NUS irreproducibles en síntesis frecuencial |

---

## 7. Formato y estructura de los labels

### 7.1 Matriz binaria por espectro (paper p. 12)

- `L` del **mismo shape que `S`**: si `S:(H,W)` ⇒ `L:(H,W)`, `float32` con valores `{0,1}`
  (para `BinaryCrossentropy`).
- `L[i,j] = 1` en índices de máximos de pico, `0` en el resto.
- **Regla sub-píxel**: centro entre dos puntos a distancia `0.25–0.75` de cada uno
  → **ambos adyacentes = 1**.

```text
S: (H, W) float32 — pseudo-Voigts elongados + ruido gaussiano normalizado
L: (H, W) {0,1}  — 1 en máximos (doble-1 si centro entre píxeles 0.25–0.75), 0 fuera
```

### 7.2 Pares (X, y) que ve la red

- Por punto `(i,j)`, `split_2D()` (`StoP3.py:17-26`) construye el *cross*:
  primeros 64 valores = corte F1 (columna `j` centrada en `i`),
  últimos 64 = corte F2 (fila `i` centrada en `j`) → `X:(128,1)` normalizado por norma L2.
- Label del vector: **el escalar del punto central**: `y = L[i,j] ∈ {0,1}`.
- Batch: `X:(B,128,1)`, `y:(B,1)` o `(B,)`, con `B = 2**16 = 65536`
  (inferencia `Production_2D()`, `StoP3.py:59`).
- Desbalanceo 2D ~1:100 **sin remuestreo** (paper p. 13; undersampling solo en 3D).

### 7.3 Regla extendida para hombros (`d < ~0.7·FWHM`, máximo único fusionado)

- El máximo de la suma ≠ centros verdaderos. Adoptar **opción (b)** para el 10% de hombros:
  marcar los **dos centros verdaderos** aunque caigan en flanco (enseña conteo `IP3→2`).
- Evaluar hombros con integral de P3 (`IP3` como proxy del nº de picos, cf. Fig. 8).

### 7.4 Extracción

- `X` extraído **obligatoriamente** con `split_2D()` sin modificar → compatibilidad
  train ↔ inferencia (`Production_2D`, `P3_2D`).

---

## 8. Volumen, splits y artefactos

- Volumen base (paper): **640 espectros × 256 picos**, split **4:1**.
- Curvas en formato heredado: `history.npy (N,2)` [train,val] o `history.txt`,
  + curvas estilo Fig. S33 y calibración estilo Fig. S28.
- Export: `Trained/Probability_2D_Elongated/...` (**nueva carpeta, no sobrescribir**),
  verificable con `model.predict(...)['dense_14']` contra `Production_2D()`.
- Arquitectura: `WNN_2D` del SM, `filter=10, Size=[64,64]`, fusión `2*filter` (=20,
  como en pesos guardados), `BCE + Adam` (`lr=1e-3` retrain / `~1e-4` fine-tune).

---

## 9. Evaluación específica (overlap + elongación)

1. Curva `recall vs d` por eje y por `r` (Fig. 8 adaptada).
2. `F1` estratificado: aislados vs pares `d∈[0.5,1]` / `[1,2]` / `>2`, y por ratio de amplitudes.
3. `IP3` en hombros: ¿cuenta 2 donde P3 muestra 1?
4. Comparativa misma batería: base vs fine-tune vs retrain; control `r=1` sin regresión.
5. Validación final en 2–3 espectros reales elongados con lista curada.

---

## 10. Checklist de aceptación del dataset

- [ ] Nomenclatura `Γ` (FWHM) / `η` (forma) / `r` aplicada en código y docs.
- [ ] `N1_synth = N1_producción` con idéntica escala Hz/pt; parches `256×512` (o `256×1024`).
- [ ] `Γ_pts` calibrados a reales; criterio de ventana §3.3 superado.
- [ ] `η_F1 ⊥ η_F2` con extremos puros; `r ∈ {1,2,3,5,8}`.
- [ ] Mezcla 40/40/10/10 con separaciones en FWHM y ratios asimétricos; `% pares d<2` por `r`.
- [ ] Ruido gaussiano `SD=A_min`; limitaciones de fase/truncamiento documentadas.
- [ ] `L` binaria + regla 0.25–0.75 + regla (b) en hombros; `X` vía `split_2D()` intacto.
- [ ] 640 espectros / 256 picos / split 4:1; `history` + calibración generados.

