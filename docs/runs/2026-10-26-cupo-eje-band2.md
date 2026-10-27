# Cupo del eje F1 en la banda anisótropa — 2026-10-26

Cómo llevar a la máquina de HTCondor y correr los brazos 1:4 y 1:3 de
`eje:diagonal` dentro de la banda de negativos duros.

> **Estado de la rama.** El código del cupo está commiteado en `f2c9565` y el
> trabajo de HTCondor en `6d49cb8`, que dejó **dos** sistemas de jobs (`jobs/` y
> `htcondor/`). En el árbol de trabajo quedó colapsado a uno solo (`htcondor/`):
> ver §2. Lo que falta commitear es el colapso, los tests del cupo, la
> herramienta de medición, `scripts/pool_flags.sh` y el espejo de Colab.
> Cuando commitees, renombrá este archivo a
> `docs/runs/2026-10-26-cupo-eje-band2-<hash>.md` y poné el hash acá arriba
> (convención de `docs/runs/2026-10-21-aniso-60e9ce4.md`).
> Commit de referencia del código base: `6d49cb8` (rama `install_in_mac_m1`).

## 1. Qué cambió

Un cupo nuevo, **`--hard-axis-frac`** (interno: `band2_axis_frac`), que reserva
esa fracción del **presupuesto de la banda** (`--hard-frac2`) a los píxeles del
**eje F1 puro** (`d2 == 0`, `d2 = |Δcolumna|` al positivo más cercano). El resto
de la banda sigue saliendo de la diagonal. Sirve para fijar la relación
`eje:diagonal` en vez de aceptar la natural:

| `--hard-axis-frac` | relación eje:diagonal |
|---|---|
| `0` (default, apagado) | natural (~1:5,4 en 3–8 × 0–4) |
| `0.20` | 1:4 |
| `0.25` | 1:3 |
| `0.33` | 1:2 |

| archivo | cambio |
|---|---|
| `scripts/hard_negatives.py` | `axis_sections(n_band2, frac)`; `choose_background(..., band2_axis_mask, band2_axis_frac)` reparte el sorteo de la banda entre eje y diagonal |
| `scripts/preextract.py` | flag `--hard-axis-frac`; registra `hard_axis_frac` en `manifest.json` |
| `scripts/train_model.py` | flag `--hard-axis-frac`; lo pasa a `subsample_pool` y a `BinSequence`; **corta con `SystemExit`** si el pool se generó con otra fracción |
| `htcondor/run_train.sh` | (era `jobs/run_train.sh`) lee `hard_axis_frac` del manifest y lo pasa al entrenamiento, igual que el resto de la geometría |
| `htcondor/train.sub` | pasa a ejecutar `htcondor/run_train.sh`: la geometría ya no se escribe a mano en `Extra` |
| `scripts/pool_flags.sh` | **nuevo**: deriva los flags de negativos duros (incluido `--hard-axis-frac`) del manifest de un pool |
| `colab/scripts/*` | espejos byte-idénticos de los 3 scripts (se arregló la divergencia que dejó `b3d20f6`, ver §2) |
| `tests/test_hard_axis.py` | 16 tests nuevos (no entraron en `f2c9565`) |

Invariantes que importan:

- **Con `--hard-axis-frac 0` nada cambia.** La secuencia de `rng` y el resultado
  son byte a byte los de antes (verificado contra el código de `HEAD` en 60
  trials × 4 configuraciones). Los pools viejos (`pre_aniso_f2lo0_454510`, etc.)
  siguen entrenando igual: sin la clave en el manifest, el wrapper no pasa el
  flag y `train_model.py` no se queja.
- **Validaciones:** el cupo exige `--hard-aniso` con `F2LO = 0` (si no, no hay
  columna de eje dentro de la banda) y `--hard-frac2 > 0`; `0 <= frac <= 1`.
- **Válvula pool/entrenamiento:** si el manifest del pool trae `hard_axis_frac`
  y el entrenamiento pide otro valor (o pide `> 0` y la clave no está), el
  entrenamiento no arranca y dice qué regenerar. No hay forma de entrenar con
  una relación eje:diagonal distinta a la del pool sin darse cuenta.
- **Techo:** el eje disponible en 3–8 × 0–4 es 3 609 px/espectro y el
  presupuesto de banda del pool es `0.68 × 13 830 = 9 404` filas, así que el
  cupo efectivo máximo es **0,384**. Arriba de eso el faltante se completa con
  diagonal **en silencio** (el tamaño del fondo no cambia, es la regla histórica
  de `choose_background`). Verificado: pedí `0.45` y salió `0.368`.
- `python3 -m pytest tests -q` → **231 passed, 18 skipped** (con el espejo de
  Colab arreglado; sin arreglarlo falla `tests/test_script_mirrors.py`, ver §2).

## 2. Estado de la rama y qué falta commitear

La rama ya está unificada con el trabajo de HTCondor: `install_in_mac_m1` está en
`6d49cb8`. Ese commit traía **dos** sistemas de jobs; se colapsaron en uno:
**todo vive en `htcondor/`** (`submit.sh`, `train.sub`, `preextract.sub`,
`infer.sub`, `run_train.sh`, `README.md`). El wrapper `run_train.sh` se movió
desde `jobs/` con `git mv` (la historia se conserva) y `jobs/` se eliminó.

En el árbol de trabajo quedan sin commitear, de este trabajo:

```bash
# en esta máquina
# `git add -A` sobre htcondor/ y jobs/ registra el colapso (moves y borrados)
git add -A htcondor jobs .gitignore \
        scripts/pool_flags.sh scripts/band_breakdown.py \
        colab/scripts/train_model.py \
        tests/test_hard_axis.py tests/test_band_breakdown.py \
        docs/runs/2026-10-26-cupo-eje-band2.md
git status --short          # revisá que jobs/ quede como borrado
# commit sugerido (dos, si preferís separar el colapso del cupo):
#   refactor: unify HTCondor job system in htcondor/ (jobs/ moved in)
#   feat: axis quota tests, pool_flags helper and HTCondor wiring
git commit -m "refactor: unify HTCondor jobs in htcondor/ and wire the axis quota"
git push
# en la máquina de HTCondor
git pull
```

Para llevarlo sin git:

```bash
# en esta máquina
git diff > /tmp/cupo_eje.patch          # htcondor/, colab/, scripts/, .gitignore
git status --short                      # los ?? nuevos van aparte
tar czf /tmp/cupo_eje_nuevos.tgz scripts/pool_flags.sh scripts/band_breakdown.py \
    tests/test_hard_axis.py tests/test_band_breakdown.py docs/runs/2026-10-26-cupo-eje-band2.md
# en la otra
git apply /tmp/cupo_eje.patch && tar xzf /tmp/cupo_eje_nuevos.tgz
```

### El espejo de Colab ya está arreglado (pero verificá)

`b3d20f6` tocó `scripts/train_model.py` (el `append=bool(args.resume)` del
`CSVLogger`) sin espejar `colab/scripts/train_model.py`, y `f2c9565` commiteó el
cupo en ambos pero **sin** esa línea, así que el espejo quedaba desincronizado y
`tests/test_script_mirrors.py` fallaba en la rama. En el árbol de trabajo ya está
resuelto (`cp scripts/train_model.py colab/scripts/train_model.py`); el `git add`
de arriba lo incluye. Verificación:

```bash
for f in hard_negatives preextract train_model; do cmp scripts/$f.py colab/scripts/$f.py; done
python3 -m pytest tests -q      # 231 passed, 18 skipped
```

## 3. Los dos pools

Misma geometría que el brazo que decidimos medir: **banda 3–8 × 0–4**, `f2lo = 0`,
`hard_frac 0.32 / hard_frac2 0.68`, `pool-k 24`, `seed 0`. Lo único que cambia
entre los dos es el cupo.

```bash
cd ~/mrai_p3   # o donde esté el repo en la máquina de HTCondor

# --- pool 1:4 (cupo 0.20) ---------------------------------------------------
~/.pyenv/versions/tflow/bin/python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/pre_aniso_f2lo0_3268_f2hi4_a20 \
  --seed 0 --pool-k 24 --hard-frac 0.32 --hard-px 2 \
  --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.20 --fast

# --- pool 1:3 (cupo 0.25) ---------------------------------------------------
~/.pyenv/versions/tflow/bin/python scripts/preextract.py \
  --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
  --ids data/train_2d_elong/manifest.json \
  --out data/pre_aniso_f2lo0_3268_f2hi4_a25 \
  --seed 0 --pool-k 24 --hard-frac 0.32 --hard-px 2 \
  --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.25 --fast
```

Notas:

- El nombre codifica `hard_frac`/`hard_frac2`/`F1HI` y agrega `f2hi4` + `a<cupo>`,
  porque la convención vieja (`...454510`) no distinguía `F2HI`.
- `--dataset` apunta al dataset de 1000 espectros 256×2048; en la máquina de
  HTCondor hay que ajustar esa ruta. Los 1000 ids son train 800 + val 200 del
  manifest del repo.
- `--fast` es no-op (compatibilidad). Salida ~7 GB por pool, igual que
  `data/pre_aniso_f2lo0_*`.
- `--hard-frac 0.32` ya **no satura el anillo**: pide 4 426 de los 5 880 px/esp
  que hay, así que el 32 % es efectivo (en los runs `0.45/0.45` el anillo pedía
  6 224 y quedaba en 42,5 %).
- `0.32 + 0.68 = 1.0` ⇒ **cero fondo fácil**: todo negativo queda a ≤8 px en F1 y
  ≤4 px en F2 de un pico. Es deliberado, pero es el riesgo principal del
  experimento (ver §6).

### Verificación de cada pool (antes de gastar 1,5 h de entrenamiento)

```bash
~/.pyenv/versions/tflow/bin/python - <<'PY'
import json, numpy as np
for name in ("pre_aniso_f2lo0_3268_f2hi4_a20", "pre_aniso_f2lo0_3268_f2hi4_a25"):
    m = json.load(open(f"data/{name}/manifest.json"))
    ids = m["train"][:20]
    band = axis = ring = bg = 0
    for sid in ids:
        z = np.load(f"data/{name}/{sid}.npz")
        d, d1, d2 = z["d"], z["d1"], z["d2"]
        b = (d1 >= 3) & (d1 <= 8) & (d2 <= 4)
        band += int(b.sum()); axis += int((b & (d2 == 0)).sum())
        ring += int((d <= 2).sum()); bg += len(d)
    n, share = len(ids), axis / band
    print(f"{name}: aniso={m['hard_aniso']} frac={m['hard_frac']}/{m['hard_frac2']} "
          f"axis_frac={m['hard_axis_frac']} | fondo {bg/n:.0f} banda {band/n:.0f} "
          f"eje {axis/n:.0f} share {share:.3f} (eje:diagonal 1:{1/share-1:.2f}) "
          f"anillo {ring/n:.0f}")
PY
```

Esperado (medido sobre 20 espectros de train): `fondo` **13 830**, `banda`
**9 404**, `eje` **1 881** (a20) / **2 351** (a25), `share` **0,200** / **0,250**
(1:4,00 y 1:3,00), `anillo` **4 426**, y el mismo tamaño de fondo en los dos. Si
el share sale distinto, el pool no es el que creés: compará `hard_axis_frac` del
manifest con el `--hard-axis-frac` que pasaste.

## 4. Las tres corridas

Todo el envío de jobs vive en **`htcondor/`** (el sistema se colapsó ahí: `jobs/`
se eliminó). El entrenamiento pasa por `htcondor/run_train.sh`, que **lee la
geometría del manifest del pool** (incluido el cupo del eje) y agrega resume,
staging del pool a scratch y marca `.done`. `Extra` queda solo para flags sueltos
de `train_model.py` (por ejemplo `--bg-ratio 6`): la geometría no va ahí.

Para encolar las tres, un `condor_submit` por corrida con el wrapper de submit
(deja copia del `.sub` final en `htcondor/runs/`):

```bash
for A in a00 a20 a25; do
  htcondor/submit.sh train.sub RunName=out_aniso_$A Pool=pre_aniso_f2lo0_3268_f2hi4_$A
done
condor_q
```

O las tres en un solo cluster, con un bloque `queue` por corrida en
`htcondor/train.sub` (los `arguments` son `RunName Pool Extra`):

```
arguments = out_aniso_a00 pre_aniso_f2lo0_3268_f2hi4_a00
queue
arguments = out_aniso_a20 pre_aniso_f2lo0_3268_f2hi4_a20
queue
arguments = out_aniso_a25 pre_aniso_f2lo0_3268_f2hi4_a25
queue
```

Antes de encolar, el chequeo sin entrenar que imprime el comando exacto (y que
confirma que el cupo del pool se está pasando):

```bash
MRAI_DRY_RUN=1 ./htcondor/run_train.sh out_aniso_a20 pre_aniso_f2lo0_3268_f2hi4_a20
# cmd: ... --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.2 ...

MRAI_DRY_RUN=1 ./htcondor/run_train.sh out_aniso_a25 pre_aniso_f2lo0_3268_f2hi4_a25
# cmd: ... --hard-aniso 3 8 0 4 --hard-frac2 0.68 --hard-axis-frac 0.25 ...
```

Si en cambio corrés `train_model.py` a mano (sin el wrapper), los flags salen del
helper: `Extra="$(scripts/pool_flags.sh data/<pool>)"`. Y si escribís el `Extra` a
mano y te olvidás del cupo, `train_model.py` corta con un mensaje que dice qué
pool regenerar o qué fracción usar: no hay forma de entrenar con una relación
eje:diagonal distinta a la del pool.

**Corrida 1 — brazo 1:4 (cupo 0.20):** `out_aniso_a20` sobre
`pre_aniso_f2lo0_3268_f2hi4_a20`.

**Corrida 2 — brazo 1:3 (cupo 0.25):** `out_aniso_a25` sobre
`pre_aniso_f2lo0_3268_f2hi4_a25`.

**Corrida 3 — control.** Dos opciones, elegí una:

**(a) Control geométrico (recomendado para aislar el efecto del cupo).** Tercer
pool idéntico con `--hard-axis-frac 0` (mismo comando de §3, `--out
data/pre_aniso_f2lo0_3268_f2hi4_a00` y `--hard-axis-frac 0`) y la corrida
`out_aniso_a00` sobre ese pool.

Así las tres corridas comparten banda, fracciones, `bg_ratio`, épocas y semilla,
y lo único que cambia es la relación eje:diagonal (natural vs 1:4 vs 1:3).

**(b) Control histórico (costo cero).** Ya está entrenado
`Trained/out_aniso_f2lo0_454510` (banda 3–10 × 0–3, `0.45/0.45`, sin cupo). No
hay que entrenar nada: solo evaluarlo (§5). Sirve como referencia contra el run
que ya existe, pero **no** aísla el efecto del cupo del efecto de la banda.

### Si en vez de HTCondor querés el comando pelado

```bash
~/.pyenv/versions/tflow/bin/python scripts/train_model.py \
  --data data/pre_aniso_f2lo0_3268_f2hi4_a20 --preextracted \
  --init Trained/Probability_2D/US/DNN_0 \
  --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --hard-frac 0.32 --hard-px 2 --hard-aniso 3 8 0 4 --hard-frac2 0.68 \
  --hard-axis-frac 0.20 \
  --epochs 210 --patience 20 \
  --scratch /tmp/out_aniso_a20 --sync-every 5 \
  --out Trained/out_aniso_a20
```

(salidas: `ckpt.keras`, `training.log`, `history.*`; al terminar
`Trained/out_aniso_a20_savedmodel/`, que es lo que consume `StoP3`).

**Tiempo:** el runbook midió ~17 s/época en la M1 a `bg_ratio 4` (~1–1,5 h para
210 épocas); en los nodos de 8 CPUs de NMRbox el orden debería ser parecido.
Las tres corridas son independientes y encolan en paralelo (48 h de límite por
job, `request_memory 8GB`, staging del pool a scratch).

## 5. Evaluación (idéntica para las tres)

Mismos 200 val, misma cadena, para que el A/B sea válido:

```bash
for R in a00 a20 a25; do
  ~/.pyenv/versions/tflow/bin/python scripts/plotting/infer_batch.py \
    --n 200 --split val \
    --model Trained/out_aniso_${R}_savedmodel \
    --out /tmp/inf200_${R} \
    --spe-the 0.2 --pl-the 0.6 --msep 3 1
done
```

Y la comparación:

```bash
~/.pyenv/versions/tflow/bin/python - <<'PY'
import json
rows = [("ctl 3-10 0.45/0.45", "/tmp/inf200_f2lo0_454510"),
        ("a00 natural 1:5,4", "/tmp/inf200_a00"),
        ("a20 1:4", "/tmp/inf200_a20"),
        ("a25 1:3", "/tmp/inf200_a25")]
for name, p in rows:
    r = json.load(open(p + "/batch_report.json")); a = r["aggregate"]; s = r["p3_shape"]
    print(f'{name:<20} f1={a["f1"]:.4f} prec={a["precision"]:.4f} rec={a["recall"]:.4f} '
          f'f1@3={s["at_3px"]["f1"]:.4f} f1@6={s["at_6px"]["f1"]:.4f} '
          f'F1/F2@6={s["at_6px"]["f1_over_f2"]:.2f}')
PY
```

Ojo: `f1`/`precision`/`recall` están dentro de `aggregate`, no en la raíz del
JSON. La primera fila solo existe si tenés
`/tmp/inf200_f2lo0_454510/batch_report.json` en esa máquina (o si evaluás el
control histórico, que sí está entrenado).

## 6. Cómo leer el resultado

- **La métrica que decide es `p3_shape`** (`at_3px` y `at_6px`: `f1`, `f2`,
  `f1_over_f2`), no solo el F1 de la lista de picos.
- **Gana el cupo si baja `f1@6` y `F1/F2@6`** sin que el F1 agregado se caiga
  groseramente respecto de ~0,85.
- Dato de los runs previos, para calibrar: al meter `f2lo = 0` en la banda, la
  precisión subió (0,9379 → 0,9501) y el recall bajó (0,7846 → 0,7658); la cola
  a 3 px bajó (0,1078 → 0,0889) pero la de 6 px **subió** (0,0047 → 0,0167) y la
  anisotropía también (1,98 → 3,32). O sea: más negativos cerca de los picos
  corren la cola hacia afuera. Si el cupo 1:3 mejora `f1@6`, eso es lo que
  buscábamos; si solo mueve `f1@3`, no alcanza.
- **Riesgo del cero fácil:** con `0.32/0.68` el modelo nunca ve fondo fácil. El
  síntoma esperable es pérdida de recall/precisión lejos de los picos, que no
  aparece en `p3_shape` sino en `aggregate`. Si las tres corridas pierden contra
  el control en `f1`, el siguiente brazo es el mismo pool con
  `--hard-frac 0.28 --hard-frac2 0.58` (10 % fácil) y el cupo igual.
- **El cupo no cambia el costo por época** (mismo `n_bg`, mismo `batch`), así
  que si 1:3 gana se puede subir la dosis con `--bg-ratio 6` sin tocar la
  geometría: eso sube el eje F1 por época de 392 a 588 puntos.

## 7. Por qué esta banda (medición sobre los 1000 espectros)

`scripts/band_breakdown.py --intensity` (12 s sin intensidad, 17 s con ella):

| región | px/esp | % del fondo | eje `d2=0` | diagonal | snr 1–3σ | snr ≥100σ |
|---|---|---|---|---|---|---|
| 3–8 × 0–4 | 22 657 | 4,326 % | 3 609 (15,9 %) | 19 049 (84,1 %) | 4,4 % | 27,3 % |
| 3–8 × 0–3 | 17 942 | 3,426 % | 3 609 (20,1 %) | 14 333 (79,9 %) | 3,4 % | 29,2 % |
| 3–8 × 4–4 | 4 715 | 0,900 % | 0 | 4 715 (100 %) | 7,9 % | 20,1 % |
| anillo `d<=2` | 5 880 | 1,123 % | — | — | 1,5 % | 34,4 % |

- **El eje F1 absoluto es siempre 3 609 px/esp**: ensanchar `F2` no agrega eje,
  solo lo diluye. Por eso el cupo es la palanca correcta si el objetivo es la
  cresta F1.
- El tramo `d2 = 4` es el más ruidoso de la banda (7,9 % en 1–3σ), y por eso
  3–8 × 0–4 conserva mejor la proporción de eje que 3–8 × 0–5 (15,9 % vs 13,2 %).
- En intensidad: el 96 % de la banda está sobre 3σ y el 27 % sobre 100σ. No hay
  "ruido puro" en esta banda: el 0,0 % de todas las regiones cae por debajo de
  1σ, porque con 256 picos en una grilla de 256×2048 cualquier ventana de 64 px
  cruza un pico. Por clase (`snr >= 3`): PG 92,7 % | MIX 96,5 % | PL 98,2 %.
- **Eje F1 por época** con `bg_ratio 4` (`n_bg ≈ 2 305`, banda 1 567 pts):

| corrida | eje F1/época |
|---|---|
| control 3–10 × 0–3, 0.45/0.45 | 208 |
| a00 natural (0,156) | 244 |
| **a20 (1:4)** | **313** |
| **a25 (1:3)** | **392** |
| (a33, 1:2) | 517 |

1:3 da 1,9× los negativos de eje por época del control, sin cambiar el tamaño
del batch ni el tiempo por época.

## 8. Reproducir las mediciones de este documento

```bash
~/.pyenv/versions/tflow/bin/python scripts/band_breakdown.py --intensity --jobs 8 \
  --band 3 8 0 4 --band 3 8 0 3 --band 3 8 4 4 --band 1 2 0 3 \
  --out /tmp/bb_38_f4.json
```

`--band F1LO F1HI F2LO F2HI` es repetible; `--ids-file data/train_2d_elong/manifest.json`
restringe a un split; `--intensity` agrega el eje de intensidad (lee también
`spectra/<id>.bin`, ~3× más lento). El JSON trae `total`, `by_class`, `by_split`
y `per_id`.
