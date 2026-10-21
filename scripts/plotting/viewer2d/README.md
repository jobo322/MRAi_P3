# view2d — visor web del espectro 2D folder-51 + picos P3

Port del viewer de `cheminfo/spectrum-generator` (`scripts/viewDataset.mjs` +
`viewer/index.html`) a los datos que consume `interactive_2d.py`: el espectro
NMRPipe folder-51, el mapa de probabilidad P3 y los picos del sistema
(`.tab` / `.peaks`).

- **Sistema de control**: idéntico al viewer original — sidebar con lista y
  filtro, toolbar con sliders de niveles, zoom por arrastre + doble click,
  panel inferior con tabla de picos.
- **Datos**: los de folder-51. Python solo se usa **una vez** para convertir
  (`convert_2d.py`, reutiliza los loaders exactos de `interactive_2d.py`);
  el servidor y el visor son Node + navegador puro, sin dependencias npm.

## Ver los datos de ENTRENAMIENTO (no folder-51)

`scripts/plotting/run_train_2d.py` es el mismo workflow que
`scripts/bruker_2d/run_51.py` (inferencia bloqueada -> P3 -> `P3toP5` ->
`P5toPeaks` -> `PeakstoTab`) pero aplicado a un espectro del dataset de
entrenamiento (sintético, J-resolved), para ver qué hace el modelo sobre datos
que ya ha visto:

```bash
/Users/abolanos/.pyenv/versions/3.13.2/envs/tflow/bin/python \
    scripts/plotting/run_train_2d.py --id 1
node scripts/plotting/view2d.mjs --out /tmp/train_p3/1/view2d
```

Detalles que conviene conocer:

- **`.ft2` sintético.** Los espectros de entrenamiento son `.bin` crudos y la
  cadena de picos lee cabeceras NMRPipe, así que se reempaquetan como Pipe con
  una calibración de **1 punto = 1 ppm** (`car = N/2 - 1`, `sw = N`, `obs = 1`),
  elegida para que `P3toP5` y `nmrglue.pipe.make_uc(...).ppm()` coincidan
  exactamente (coinciden a 0.0) y la ida y vuelta sea `índice = (N-1) - round(ppm)`
  sin ambigüedad.
- **Artefactos**: los mismos que produce `convert_2d.py` (`meta.json`,
  `spectrum.bin`, `p3.bin`, `f2ppm.bin`, `f1hz.bin`), **más** `meta.truth` con
  las posiciones reales del `manifest.jsonl`. Ojo: el visor pinta `meta.peaks`
  (las detecciones), **no** `truth`; para ver ambas superpuestas usa el PNG
  `spectrum_P3_peaks.png` que deja el script al lado.
- **Informe de aciertos**: TP/FP/FN, recall y precisión contra el ground truth
  (`--match-px`, 2 px por defecto), guardados en `report.json`.
- **`--plot-only`** redibuja el PNG y el informe reutilizando `<id>.P3` y
  `<id>.tab` (la inferencia es la parte cara).
- **`--spe-the`** (máscara de candidatos `|S| >= the * std(S)`) domina el
  recall: en el espectro 1, 0.5 da recall 0.617 y 0.2 da 0.828 (precisión
  0.735 -> 0.640); solo 177 de las 256 verdades tienen candidato a 0.5.

## Uso

```bash
# 1) Conversión (una vez, o cuando cambien los .ft2/.tab/.P3):
/Users/abolanos/.pyenv/versions/tflow/bin/python scripts/plotting/convert_2d.py

# 2) Visor:
node scripts/plotting/view2d.mjs                    # sirve /tmp/p51_view2d en :8787
node scripts/plotting/view2d.mjs --port 8888 --out /tmp/p51_view2d
node scripts/plotting/view2d.mjs --ft2 X --tab Y --peaks Z --p3 W   # convierte solo si falta meta.json
node scripts/plotting/view2d.mjs --no-convert       # fallar si no hay meta.json, nunca convertir
```

Abre `http://localhost:8787/` (o el puerto elegido). **Si cambias los datos,
reinicia el servidor**: `meta.json` se lee una sola vez al arrancar.

## Sistema de control (igual que el viewer original)

- **Sidebar**: lista de mapas disponibles con filtro ( Spectrum,
  P3 probability). Seleccionar carga el mapa bajo demanda con indicador «cargando».
- **Toolbar**:
  - `niveles` — nº de niveles de contorno (4–32, por defecto 14).
  - `nivel mín ×SD` — nivel más bajo como múltiplo de la SD de ruido
    (estimada por MAD muestreada).
  - `nivel máx %zMax` — nivel más alto como fracción del máximo visible.
  - `espaciado` — log o lineal.
  - `fijar niveles` — congela los niveles al hacer zoom.
  - `picos` — cruces rojas de los picos del sistema.
  - `raster` — underlay en escala de grises normalizado a los mismos niveles.
  - `reset zoom` — vista completa (también doble click).
- **Zoom**: click + arrastre (rectángulo rojo discontinuo) · rueda = zoom sobre
  el cursor · teclas heredadas de `interactive_2d.py`: `+`/`−` niveles,
  `v` picos, `r`/`o` reset.
- **Panel inferior**: título, SD/SNR/zMax del mapa y tabla de picos
  (`INDEX, X_PPM, Y_HZ, HEIGHT, VOL, TYPE` del `.tab`).
- **Ejes**: x = F2 (ppm, alto a la izquierda, convención NMR), y = F1 (Hz).

Nota: los paneles laterales de `interactive_2d.py` (proyección F2 y slice F1)
no forman parte del sistema de control del viewer original y se quitaron;
el modo dual/p3only se sustituye por la selección en la sidebar.

## Optimizaciones de velocidad

El grid folder-51 es 64×32768 (2M celdas); el viewer original hace marching
squares a resolución completa, aquí sería carísimo por redibujado. Medidas:

1. **Muestreo adaptativo**: el recorte visible se remuestrea a ≤420 celdas por
   eje antes de marching squares (los detalles finos reaparecen al hacer zoom,
   donde el recorte es pequeño y ya no necesita remuestreo).
2. **Cache de recortes**: la rejilla remuestreada se cachea por crop
   (LRU de 3) — panear o cambiar niveles no remuestrea.
3. **Salto por rango**: los niveles fuera de `[min, max]` del recorte se
   descartan sin recorrer celdas.
4. **Segmentos planos**: salida como `Float32Array` plano (`x0,y0,x1,y1,…`)
   en vez de arrays de pares (menos GC).
5. **Coalescing con `requestAnimationFrame`**: rueda/zoom/sliders agrupan
   redibujados a 60 fps.
6. **Underlay con `ImageData`**: raster a resolución de pantalla, sin
   operaciones por píxel en JS del hilo principal.
7. **Carga perezosa**: solo se descarga el mapa seleccionado (8 MB por mapa,
   cacheado en memoria).

Si hiciera falta más (paneo fluido a máxima resolución), el siguiente paso
natural sería mover marching squares a un Web Worker — la extracción ya está
aislada en `marchingSquares`/`sampledGrid`.

## Modo B — dataset de entrenamiento (`--dataset`)

```bash
node scripts/plotting/view2d.mjs --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
    [--inference <dir>] [--port 8787]
```

Navega un dataset estilo spectrum-generator (`dataset.json` + `manifest.jsonl` +
`spectra/<id>.bin` + `labels/<id>.bin` uint8): sidebar con los 1000 espectros
(id, clase PG/PL/MIX, r, nº picos), mismos sliders de niveles (SD del manifest),
raster, zoom por arrastre/rueda.

En `viewer2d/dataset.html`:

- **○ círculos azules sin relleno** = labels del dataset de entrenamiento
  (píxeles donde `labels/<id>.bin == 1`, ground truth)
- **× cruces rojas** = picos del peak picking post-inferencia, leídos de
  `<id>.tab` (VARS `X1`/`Y1` = píxel del centro) en el directorio de inferencia:
  por defecto `<dataset>/inference/`, o pásalo con `--inference <dir>`.
  Si no existe, el viewer arranca igual (sin cruces) y la tabla muestra la
  verdad del manifest.
- Ambos overlays tienen toggle propio (`labels ○` / `inferencia ×`).

API modo B: `GET /api/dataset`, `/api/manifest`, `/api/spectrum/:id`,
`/api/labels/:id`, `/api/inference/:id` (0..N-1, validado contra el manifest).

## Artefactos de la conversión (`--out`, por defecto `/tmp/p51_view2d`)

| Archivo | Contenido |
|---|---|
| `meta.json` | shape, obs, labels, picos (x/y + filas con HEIGHT/VOL/TYPE), fuentes |
| `spectrum.bin` / `p3.bin` | `float32` LE, C-order, shape `(F1, F2)` |
| `f2ppm.bin` / `f1hz.bin` | escalas de ejes como `uc.ppm_scale()` / ppm×obs |

## API del servidor

- `GET /` → `viewer2d/index.html`
- `GET /circles` → `viewer2d/index-circles.html` — copia modificada donde los
  labels del dataset se dibujan como **círculos azules sin relleno** (el
  original usa cruces rojas); el toggle se llama «labels»
- `GET /api/meta` → `meta.json`
- `GET /api/bin/spectrum|p3|f2ppm|f1hz` → bytes `float32` LE (whitelist, sin traversal)