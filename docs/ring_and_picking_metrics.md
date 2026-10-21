# Anillo, lista de picos y precisión: lo que dicen las mediciones

Nota técnica de medición. Responde a una pregunta concreta: **el modelo
reentrenado con el minado del anillo (`--hard-frac`) ¿arregló lo que tenía que
arreglar, y de qué depende que la lista de picos sea correcta?** Todo se mide
contra el dataset de entrenamiento (`spectra/*.bin` + `labels/*.bin` +
`manifest.jsonl`), que es la fuente de la verdad: las etiquetas son los picos
reales que el generador puso en cada espectro sintético.

* **Modelo medido:** `Trained/Probability_2D_Elongated/DNN_0_local_serving/DNN_0`
  (fine-tune de 210 épocas, `val_loss 0.3029 -> 0.1407`, pool regenerado con
  `--hard-frac 0.5 --hard-px 2`).
* **Modelo de referencia:** `Trained/Probability_2D/US/DNN_0` (base, ver §5 —
  ojo: **no** aísla solo el anillo).
* **Herramienta:** `scripts/plotting/run_train_2d.py` (misma cadena que la
  carpeta 51: `Production_2D` bloqueado -> tail P3 -> `P3toP5` -> `P5toPeaks` ->
  `PeakstoTab`), que además compara con las etiquetas.
* **Documentos hermanos:** `docs/local_training.md` (cómo se entrena en local),
  `colab/COLAB.md` (la misma cadena en Colab). El minado del anillo y su
  motivación están en §0 de `local_training.md`.

---

## 1. Cómo se cuenta (y por qué el conteo importa)

Cada detección se empareja **uno a uno** con un pico real, con tolerancia de
2 px (`--match-px`), y se reportan TP / FP / FN. El emparejamiento uno a uno es
deliberado: si el modelo emite dos picos sobre el mismo pico real, el segundo es
un falso positivo aunque caiga «cerca» — y eso es exactamente lo que hace la
lista de picos que consume el usuario.

Medida clave del anillo: **fracción de píxeles del anillo de 2 px alrededor de
una etiqueta cuya P > 0.5**. Se reporta dos veces, porque la máscara de
candidatos (`|S| >= spe_the * std(S)`) pone P = 0 fuera de ella:

* *sobre todos los píxeles del anillo*, y
* *sobre los candidatos*, que es la cifra independiente de la máscara.

```bash
# un espectro, con artefactos para mirar (PNG + visor)
tflow/bin/python scripts/plotting/run_train_2d.py --id 1
node scripts/plotting/view2d.mjs --out /tmp/train_p3/1/view2d
```

---

## 2. El anillo: arreglado

Motivación (medida antes del reentrenamiento, ver `local_training.md` §0): el
fondo se muestreaba uniforme, así que los ceros *dentro* del pico entraban al
**0,48 %** del pool (~11 negativos a 1 px por cada 578 positivos) y el modelo
daba **P > 0.5 al 81 %** de los píxeles vecinos a una etiqueta; la lista de
picos salía **7,7× demasiado larga** (1966 picos donde había 256).

Tras el fine-tune con el anillo en el pool:

| Medida | Antes | Ahora (8 espectros) |
|---|---|---|
| P > 0.5 en el anillo de 2 px | 81–84 % | **5,5–11,6 %** |
| picos emitidos / picos reales | 7,7× | **1,22×** |

Es decir: el modelo **dejó de disparar sobre el anillo**. Ese objetivo está
cumplido y no es motivo para otro reentrenamiento.

---

## 3. Entonces, ¿de dónde salen los falsos positivos?

Espectro `id 1` (256 picos reales):

| | |
|---|---|
| detecciones | **322** (1,26× la verdad) |
| TP / FP / FN | **161 / 161 / 95** |
| precision / recall | **0,500 / 0,629** |
| distancia mediana al pico real | **0,00 px** |

Los 161 FP **no son ruido**: 137 están a ≤4 px de un pico real, 19 a 5–8 px, 5
son duplicados y **0 están aislados**. Su P mediana es 0,476, frente a 0,997 en
los TP. La causa es geométrica:

* la etiqueta marca el pico con un **marcador de 1/2/4 px** (mediana 2 px), pero
  el pico elongado mide ~10 px de ancho en F1;
* el modelo responde con una **cresta en F1**: a 1 px de distancia la P vale
  **0,259 en F1** y **0,030 en F2** (a 4 px, 0,003 y 0,001);
* detecciones por pico real: 86 picos → ninguna, 73 → una, 43 → dos, 53 → tres.
  O sea **97 picos generan 249 detecciones** (2,6× cada uno), separadas por
  4,0 px de mediana;
* y `P3toP5.py` llamaba a `find_all_thres_fast(..., msep=[1,1])`: **sin
  separación mínima**, así que cada máximo de la cresta se convertía en un pico.

El segundo problema es el complementario:

* 88 de los 95 FN caen **fuera de la máscara de candidatos**;
* y no porque el modelo no los vea: evaluado sin máscara da **P > 0.5 en el
  76 %** de ellos (P > 0.2 en el 89 %, mediana 0,786);
* son sistemáticamente los picos **débiles**: amplitud mediana 0,003–0,006
  frente a 0,03–0,08 en los que sí entran en la máscara.

En resumen: **la precisión la rompe la separación mínima del picker y el recall
lo limita la máscara de candidatos**, no el aprendizaje del anillo.

---

## 4. Las dos palancas (medidas, 8 espectros = 4 train + 4 val, 2048 picos)

### 4.1 `msep`: separación mínima entre picos (implementado)

`P3toP5.peak_points(P3, thres, msep)` acepta ahora `msep` (int o lista por
dimensión; `None` = `[1,1]`, el comportamiento histórico). Expuesto como
`-ms/--msep` en `P3toP5.py`, `--msep` en `scripts/plotting/run_train_2d.py` y
`--msep` en `scripts/bruker_2d/run_51.py`, con **default 1** para no cambiar el
flujo existente sin decirlo.

Con `--spe-the 0.5` (el actual):

| msep | picos | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|
| **1 (histórico)** | 2505 | 1,22× | 0,559 | 0,684 | **0,615** |
| 3 | 1763 | 0,86× | 0,750 | 0,646 | 0,694 |
| 4 | 1479 | 0,72× | 0,869 | 0,627 | 0,729 |
| 5 | 1318 | 0,64× | 0,951 | 0,612 | 0,745 |
| 6 | 1249 | 0,61× | 0,974 | 0,594 | 0,738 |

La precisión se multiplica por ~1,7 y el recall cae 2–7 puntos: **casi todo lo
que quita `msep` son duplicados de un pico real, no picos reales** (los picos de
un solapamiento genuino están a ≤2 px y siguen apareciendo como FN, que es el
precio honesto de no resolver solapamientos).

### 4.2 `--spe-the`: la máscara de candidatos (recall)

| spe_the | etiquetas dentro de la máscara | picos | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0,5 (actual) | 70 % | 2505 | 1,22× | 0,559 | 0,684 | 0,615 |
| 0,2 | 89 % | 4150 | 2,03× | 0,431 | 0,874 | 0,578 |

Bajar la máscara recupera ~390 TP pero dispara los FP aislados (43 → 434): **sola
no sirve**. Combinada con `msep`:

| combinación | picos | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|
| spe 0,5 + msep 1 (hoy) | 2505 | 1,22× | 0,559 | 0,684 | 0,615 |
| spe 0,5 + msep 5 | 1318 | 0,64× | 0,951 | 0,612 | 0,745 |
| spe 0,2 + msep 4 | 2143 | 1,05× | 0,741 | 0,775 | 0,758 |
| **spe 0,2 + msep 5** | **1899** | **0,93×** | **0,810** | **0,751** | **0,779** |

La mejor combinación medida sube el F1 un **27 % relativo** y deja el número de
picos prácticamente exacto (1899 para 2048 reales), sin reentrenar.

> Coste de bajar `spe_the`: se evalúan ~30 % más píxeles candidatos por espectro
> (2433–2543 → 3161–3373 en estos 8). El NMS de `msep` se come después el ruido
> que entra por ahí.

---

## 5. ¿Y el modelo base?

Mismos 8 espectros, misma cadena, misma máscara: `Trained/Probability_2D/US/DNN_0`
frente al fine-tune con anillo.

| modelo | spe | msep | picos | det/GT | precision | recall | F1 | anillo (% cand) |
|---|---|---|---|---|---|---|---|---|
| anillo | 0,5 | 1 | **2505** | **1,22×** | **0,559** | 0,684 | **0,615** | **7,7 %** |
| base | 0,5 | 1 | 4309 | 2,10× | 0,296 | 0,624 | 0,402 | 16,4 % |
| anillo | 0,5 | 5 | 1318 | 0,64× | 0,951 | 0,612 | 0,745 | 7,7 % |
| base | 0,5 | 5 | 1675 | 0,82× | 0,469 | 0,383 | 0,422 | 16,4 % |
| anillo | 0,2 | 1 | 4150 | 2,03× | 0,431 | 0,874 | 0,578 | 13,3 % |
| base | 0,2 | 1 | 5893 | 2,88× | 0,269 | 0,774 | 0,399 | 17,8 % |
| **anillo** | **0,2** | **5** | **1899** | **0,93×** | **0,810** | **0,751** | **0,779** | 13,3 % |
| base | 0,2 | 5 | 2004 | 0,98× | 0,477 | 0,466 | 0,471 | 17,8 % |

Lectura, con los ajustes que hay hoy en producción (spe 0,5 + msep 1):

* **F1 0,615 frente a 0,402** (+53 % relativo al modelo base).
* La mejora es de **precisión** (0,559 frente a 0,296), no de recall (0,684 frente
a 0,624): el fine-tune no encuentra muchos más picos, sino que **deja de
inventarlos** alrededor de los que ya encuentra. 2505 picos frente a 4309,
para los mismos 2048 reales.
* La métrica del anillo lo dice directamente: **16,4 % → 7,7 %** de los píxeles
del anillo con P > 0.5, o sea el modelo con anillo dispara **la mitad** sobre
los vecinos inmediatos de un pico. Es el efecto que se buscaba con
`--hard-frac` y es el que se mide, no el que se supone.
* Con `msep` alto el base se degrada más (recall 0,624 → 0,383): su cresta en
F1 es más ancha y menos centrada, así que colapsarla se lleva picos reales.
  Otra señal de que el fine-tune localiza mejor.

> **Matiz importante:** el base *no* es «el mismo modelo sin el anillo». Es el
> punto de partida del que sale el fine-tune, así que la diferencia mezcla tres
> cosas: el entrenamiento con picos elongados, el minado del anillo y 210
> épocas más. La única cifra atribuible **solo** al anillo es la de la última
> columna (la fracción del anillo que el modelo marca como pico). Para aislarla
del todo haría falta un fine-tune idéntico con `--hard-frac 0`, que no está
hecho.

---

## 6. Datos reales: carpeta 51

Misma cadena sobre la carpeta 51 real (`scripts/bruker_2d/run_51.py --msep N`).
Aquí no hay verdad conocida, así que no hay TP/FP: se mide **qué hace `msep`**.

| msep | picos |
|---|---|
| 1 (histórico) | **1124** |
| 4 | **527** (−53 %) |
| 5 | **424** (−62 %) |

Y lo que quita son duplicados, no picos distintos:

* los 527 picos de `msep 4` **ya existían exactos** en la lista de `msep 1`
  (527/527; y lo mismo con `msep 5`: 424/424) → `msep` **quita, no mueve**;
* los 597 eliminados son más débiles: HEIGHT mediana **36,1** frente a **46,8**
  de los que sobreviven (la lista completa tiene mediana 41,2).

Conclusión honesta: en datos reales `msep 4` recorta **la mitad de la lista sin
introducir ni desplazar un solo pico**, y se queda el representante más intenso
de cada grupo. Lo que no prueba es que ese representante sea el correcto en un
solapamiento real; eso se mira en el visor
(`node scripts/plotting/view2d.mjs --out …`) o contra una lista de picos de
referencia.

---

## 7. Decisión recomendada

1. **No reentrenar por el anillo**: está medido y arreglado (§2).
2. **Adoptar `msep 4–5` en la cadena** (una línea, ya implementada) y evaluar
   `--spe-the 0.2` acompañado de `msep` (§4.2). En datos reales `msep 4` quita
   la mitad de la lista sin quitar picos nuevos (§6); falta la comprobación
   visual en solapamientos antes de fijarlo como default.
3. **Si se reentrena, cambiar la representación del target, no repetir el run.**
   Los datos dicen que el modelo no sabe en qué fila de F1 está el ápice
   (P a ±1 fila = 0,259, a ±4 filas ≈ 0,5) mientras la etiqueta es un marcador
   de 1/2/4 px sobre una cresta de ~10 px. Opciones, por orden de pago/esfuerzo:
   * etiquetar **el segmento completo de la cresta** y colapsar a un pico por
     componente conexa en la cadena;
   * target gaussiano suave (σ ≈ 1 px en F2, más ancho en F1) para calibrar la
     confianza sin perder el máximo único;
   * los picos débiles (amplitud ≈ 0,003) son un asunto del **dataset**
     (`aMin`, ruido), no de la loss.
4. **Dónde duele más**: los espectros con solapamiento (clase `MIX`). En el
   `id 4` el modelo satura la zona de solapamiento (929 detecciones con
   `spe_the 0.2` frente a 256 reales), mientras el `id 2` (clase `PG`) da
   precision 0,79 / recall 0,85. Ese es el objetivo natural del próximo
   entrenamiento.

---

## 8. Dos bandas de fondo, y el target de cresta que se descartó

### 8.1 Dos bandas (implementado)

Medido en §3: el anillo (`d <= 2`) entró al pool al **54 %** de sus píxeles, pero
la banda `3 < d <= 6` —donde caen los hombros que se vuelven falsos positivos—
solo al **0,5 %**, porque el fondo uniforme la diluye frente a las ~500 000
filas a `d > 6`. El presupuesto se reparte ahora explícitamente:

    [ positivos | anillo (d <= hard_px) | banda 2 (hard_px < d <= band2_px) | resto ]

```bash
python scripts/preextract.py --dataset … --ids … --out data/pre_elong_2b \
  --seed 0 --pool-k 10 --hard-frac 0.5 --hard-px 2 --hard-band2 6 --hard-frac2 0.2

python scripts/train_model.py --data data/pre_elong_2b --preextracted \
  --hard-frac 0.5 --hard-px 2 --hard-band2 6 --hard-frac2 0.2 …
```

* `--hard-band2` / `--hard-frac2` existen en `preextract.py` y en
  `train_model.py`; el pool guarda `d`, así que las bandas se eligen también en
  el submuestreo por época sin regenerar nada.
* El formato no cambia (`d` ya estaba); el manifest añade `hard_band2` y
  `hard_frac2`, y `preextract.log` gana la columna `band2`.
* Con `--hard-frac2 0` el sorteo es **byte a byte** el histórico (misma secuencia
  de `rng`): `tests/test_hard_bands.py` lo fija, junto al reparto por banda, la
  absorción de las bandas incompletas y el layout del pool.
* Coste: cubrir dos bandas con el mismo presupuesto baja la cobertura del anillo;
  para mantener las dos hacen falta ~8100 filas duras por espectro
  (`--pool-k` ~24), o sea un pool de ~7 GB.

### 8.2 El target de cresta: medido antes de entrenar y descartado

La idea: etiquetar la **región del pico por encima de `frac · A`** (semialtura)
en vez del marcador de 1/2/4 px, para que el target no tenga la ambigüedad de
fila de §1 de esta nota. Implementado en `scripts/ridge_target.py`
(`ridge_owners`, `count_instances`) con sus tests, y medido **sin entrenar**:

| medida (8 espectros, 2048 picos) | valor |
|---|---|
| área de la cresta por pico (mediana) | 18–39 px (el marcador: 2 px) |
| densidad de etiqueta en ±1 fila | cresta 0,97–0,99 · marcador 0,48–0,51 |
| IoU entre la cresta y la `P > 0.5` del modelo actual | **0,09** |
| techo «un pico por componente», cresta cruda | 2713 componentes (**132 %**) |
| ídem con cierre morfológico / área mínima | 1455 (**71 %**) |
| ídem con el modelo actual (`P > 0.5`, `spe_the 0.5`) | 1693 (**83 %**) |

Lectura: la cresta cruda **se fragmenta** (132 %) porque el contorno de
semialtura sobre el espectro con ruido no es continuo; al limpiarla se
**sobre-fusiona** (71 %), porque las crestas de picos distintos se tocan — el
dataset es denso en solapamientos a propósito. Con ese techo, el target de
cresta no puede superar el recall que ya da el marcador + `msep` (0,87 con
`spe_the 0.2`). **Descartado**: si algún día se quiere atacar la ambigüedad de
fila, tendrá que ser con instancias (centro + offset) o con un target más suave
que el umbral duro, no con un blob.

Queda entonces el plan del §7: `msep` para los duplicados, banda 2 en el pool
para los hombros, y el reentrenamiento con las dos bandas como siguiente paso.

---

## 9. El reentrenamiento con las dos bandas: comparación medida

Pool `data/pre_elong_2b` (`--pool-k 24 --hard-frac 0.55 --hard-px 2
--hard-band2 6 --hard-frac2 0.25`): la banda 3–6 px pasa del **0,5 %** al **25 %**
del fondo. Modelo `Trained/out_elong_2b_savedmodel` (210 épocas, convergido;
`val_loss 0.382 → 0.194`, más alta que el run anterior porque el fondo es más
difícil — no es comparable entre pools).

Mismos 8 espectros, 2048 picos reales, misma medición para los tres modelos:

| modelo | spe | msep | picos | det/GT | precision | recall | F1 | anillo |
|---|---|---|---|---|---|---|---|---|
| base | 0,5 | 1 | 4309 | 2,10× | 0,296 | 0,624 | 0,402 | 26,7 % |
| anillo | 0,5 | 1 | 2505 | 1,22× | 0,559 | 0,684 | 0,615 | 12,6 % |
| **bandas** | 0,5 | 1 | **1929** | **0,94×** | **0,709** | 0,668 | **0,688** | **6,2 %** |
| anillo | 0,5 | 3 | 1763 | 0,86× | 0,750 | 0,646 | 0,694 | 12,6 % |
| **bandas** | 0,5 | 3 | 1570 | 0,77× | 0,832 | 0,638 | **0,722** | 6,2 % |
| anillo | 0,2 | 5 | 1899 | 0,93× | 0,810 | 0,751 | **0,779** | 16,2 % |
| bandas | 0,2 | 5 | 2098 | 1,02× | 0,730 | 0,748 | 0,739 | 8,1 % |

### Qué arregló y cómo

Con la cadena tal cual (`msep 1`, `spe_the 0,5`), el modelo con bandas es
**mejor en todo lo que importaba**: F1 0,688 frente a 0,615 (+12 % relativo),
precisión 0,709 frente a 0,559, y 1929 picos para 2048 reales (0,94×) frente a
2505 (1,22×).

Los falsos positivos por tipo (msep 1) dicen de dónde salió la mejora:

| | duplicado | hombro ≤4 px | 5–8 px | aislado | detecciones/pico |
|---|---|---|---|---|---|
| anillo | 61 | **593** | 407 | 43 | 1,45 |
| bandas | 44 | **248** (−58 %) | 191 (−53 %) | 78 | **1,10** |

Y el mecanismo, que es lo que se predijo: la P mediana en F1 a **1 px** cae de
**0,251 a 0,123** (a 2 px, 0,077 → 0,033; a 3 px, 0,043 → 0,015) y en F2 a 1 px
de 0,059 a 0,025. Es decir: la respuesta de la cresta bajó **por debajo del
umbral 0,2 del picker**, así que los máximos secundarios dejan de existir antes
de que `msep` tenga que quitarlos. La métrica del anillo acompaña: 12,6 % → 6,2 %.

### El precio, y qué configuración usar

* **Suben los FP aislados** (lejos de todo pico): 43 → 78 con `spe_the 0,5`, y
  434 → 736 con 0,2. El modelo se ha vuelto más estricto cerca de los picos y algo
  más nervioso en el fondo lejano.
* Por eso **`msep 5` ya no es la receta** para este modelo: al estar las crestas
  ya separadas, un NMS de 5 px se come picos reales (recall 0,857 → 0,748 con
  `spe_the 0,2`). Con `msep 3` queda en F1 0,722 (`spe 0,5`) / 0,717 (`spe 0,2`),
  y el mejor número absoluto de la tabla sigue siendo el del modelo **con anillo**
  en su ajuste más agresivo (0,779), que compra ese F1 a costa de 1900 picos
  cuando hay 2048 y de una precisión de 0,81.

Recomendación práctica: **modelo con bandas + `msep 3`**, o incluso sin `msep`
(`0,94×` los picos reales, precisión 0,71). El `msep 5` queda como recurso para
el modelo anterior.

---

## 10. Los falsos positivos lejanos: qué son, y el umbral que los resuelve

El modelo con bandas sube los FP **aislados** (a >8 px de todo pico) de 43 a 78
con `spe_the 0,5`. Medido qué son:

| medida | anillo | bandas |
|---|---|---|
| distancia a la etiqueta más cercana (mediana / rango) | 10 px (7–35) | 16 px (8–40) |
| señal / σ del ruido (mediana) | 22 | 16 |
| compatibles con ruido (S < 3σ) | 0 % | 0 % |
| S / máximo local 21×21 (mediana) | 0,23 | 0,23 |
| dentro del contorno al 5 % de un pico | 56 % | 36 % |

No son ruido: son señal de 16–22 σ sobre la **falda** de los picos (y en los
solapes de dos colas), donde la etiqueta es 0. Y viven exactamente donde el pool
no mira:

| d (px) | píxeles disponibles | filas en el pool (`pre_elong_2b`) | cobertura |
|---|---|---|---|
| 1 | 2503 | 2503 | **100 %** |
| 2 | 3350 | 3350 | **100 %** |
| 3 | 4118 | 559 | 13,6 % |
| 4 | 7382 | 973 | 13,2 % |
| 6 | 8427 | 1114 | 13,2 % |
| **8–30** | ~150 000 | ~150 por distancia | **~1 %** |

O sea: sí, son fondo que el pool prácticamente no muestreó. Pero **no hace falta
muestrearlo**: su P es baja.

| | P mediana | P p75 | máximo |
|---|---|---|---|
| FP aislados (n=78) | **0,231** | 0,282 | 0,777 |
| TP (n=1368) | **0,995** | — | — |

Subir el umbral de la cadena (`--pl-the`, hoy 0,2) los elimina casi gratis:

| umbral | sobreviven FP aislados | pierden TP |
|---|---|---|
| 0,3 | 23 % | 2 % |
| 0,4 | 14 % | 4 % |
| 0,5 | 6 % | 5 % |

### El ajuste que sale de todo esto (modelo con bandas)

Barrido completo, 8 espectros / 2048 picos reales:

| spe_the | pl_the | msep | picos | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| 0,5 | 0,2 (hoy) | 1 | 1929 | 0,94× | 0,709 | 0,668 | 0,688 |
| 0,5 | 0,4 | 1 | 1414 | 0,69× | 0,931 | 0,643 | 0,760 |
| 0,5 | 0,5 | 1 | 1357 | 0,66× | 0,959 | 0,636 | 0,765 |
| 0,2 | 0,4 | 3 | 1821 | 0,89× | 0,871 | 0,775 | 0,820 |
| **0,2** | **0,5** | **3** | **1690** | **0,83×** | **0,924** | **0,763** | **0,836** |
| 0,2 | 0,5 | 5 | 1690 | 0,83× | 0,920 | 0,726 | 0,811 |

Y no es sobreajuste a los espectros de train (el mismo ajuste, partido):

| configuración | train (4 ids) | val (4 ids) |
|---|---|---|
| 0,2 / 0,5 / 3 | P 0,904 · R 0,758 · **F1 0,825** | P 0,945 · R 0,768 · **F1 0,847** |

**Recomendación final:** modelo `out_elong_2b_savedmodel` con
`--spe-the 0.2 --pl-the 0.5 --msep 3` → F1 0,836 (0,93× los picos reales,
precisión 0,92), frente al 0,615 del modelo anterior con la cadena de hoy. Los
FP lejanos no se atacan con más muestreo (haría falta un pool ~3× mayor para
llegar al 5 % de cobertura a d≥8) sino con el umbral, porque el modelo ya les da
una P baja.

---

## 11. Los pares apretados: sobremuestrear lo que el perfil en cruz no ve

Qué es un **par apretado**: dos picos a <= 6 px el uno del otro con amplitudes
muy distintas —el pequeño vale la mitad o menos del grande (`ratio 0.5`), y los
empates de amplitud no cuentan—. El pequeño queda enterrado en el flanco del
grande, y ahí viven los falsos negativos que quedan tras el minado del anillo:
de 2048 picos reales hay 486 FN, 110 de ellos FUERTES (A >= 20×noise.sd) y 45
con un pico mayor a <= 6 px (enterrados en el flanco de uno grande).

**Por qué no se puede derivar del perfil.** La fila que la red entrenada ve es
una CRUZ: 64 px del eje F1 y 64 px del eje F2 por el centro de cada píxel. Un
vecino en diagonal a 4 px no cruza ninguna de las dos ramas, así que ninguna
feature de la fila delata al vecino: medido, querer detectar el par desde el
perfil da una sensibilidad del 14 % (24 de 176 picos enterrados). El
`manifest.jsonl` del dataset sí conoce el centro y la amplitud de TODOS los
picos, así que la marca del par se calcula AHÍ, una sola vez, al generar el
pool: `preextract.py --pair-px 6 --pair-ratio 0.5` la escribe en el npz como un
byte por fila del pool (`pair`, marcando también los pixelSet de los picos
enterrados), y con `--pair-px 0` el npz y el log no cambian ni un byte.

**Coste.** ~14 KB por espectro de disco (un byte por fila del pool, ~14 MB por
1000 espectros) y ~+5 % por época con `--pair-frac 0.25` (filas extra por lote:
round(0.25×n_pos) ≈ 145 sobre ~2905).

**Techo medido.** Cuánto recall queda sobre la mesa, sobre el mejor punto de la
cadena hoy (§12: `spe_the 0.2`, `pl_the 0.5`, `msep [3,1]` → F1 0,844 con 449 FN,
100 de ellos fuertes): recuperar la mitad de esos 100 sube el F1 a 0,859;
recuperarlos todos, a 0,874.
El sobremuestreo (`train_model.py --pair-frac`, que añade por época
`round(pair_frac×n_pos)` filas del pool marcadas, con reemplazo, sin tocar el
sorteo del fondo) es la palanca para acercarse a ese techo.


---

## 12. Zona de trabajo de la cadena: barrido de `spe_the` × `pl_the` × `msep`

Mismos 8 espectros, 2048 picos reales, modelo `out_elong_2b_savedmodel`,
emparejamiento 1:1 a 2 px. Las tres columnas de `msep` son la misma matriz P3
re-cosechada (medido: `msep` no toca la P3, solo la lista de picos).

**`pl_the 0,5`** (el que gana):

| `spe_the` | `msep` | picos | det/GT | precision | recall | F1 |
|---|---|---|---|---|---|---|
| **0,2** | **`[3,1]`** | **1741** | **0,85×** | **0,918** | **0,781** | **0,844** |
| 0,2 | `3` | 1690 | 0,83× | 0,924 | 0,763 | 0,836 |
| 0,2 | `[2,1]` | 1793 | 0,88× | 0,901 | 0,789 | 0,841 |
| 0,1 | `[3,1]` | 2155 | 1,05× | 0,774 | 0,815 | 0,794 |
| 0,1 | `3` | 2083 | 1,02× | 0,784 | 0,798 | 0,791 |
| 0,05 | `[3,1]` | 2298 | 1,12× | 0,729 | 0,818 | 0,771 |
| 0,05 | `3` | 2217 | 1,08× | 0,741 | 0,802 | 0,770 |

**`pl_the 0,3`**: con `spe_the 0,2` baja a F1 0,793–0,796 y con `spe_the 0,1`
se hunde a 0,52 (el recuento de picos se triplica: 4383–4865) porque la máscara
laxa mete fondo que el modelo puntúa por encima del umbral. No sirve.

Conclusión: **`--spe-the 0.2 --pl-the 0.5 --msep 3 1`**. Bajar la máscara no es
una palanca de recall: compra +0,04 de recall a cambio de −0,15 de precisión.

### Los FN fuertes no son un problema de máscara

Techo de máscara (picos reales con `|S[centro]|` por debajo del corte, es decir
nunca evaluados) y, al mejor punto de cada `spe_the`, cuántos FN quedan:

| | `spe 0,2` | `spe 0,1` | `spe 0,05` |
|---|---|---|---|
| picos por debajo de la máscara | 229 (11,2 %) | 75 (3,7 %) | 30 (1,5 %) |
| de ellos, FN de verdad (no absorbidos a ≤2 px) | 161 | 52 | 23 |
| FN totales | 449 | 420 | 372 |
| **de ellos, fuertes (A ≥ 20σ)** | **100** | **100** | **100** |

Los mismos **100 picos fuertes** fallan en los tres `spe_the`: no están tapados
por la máscara sino que el modelo los confunde con el flanco de un vecino. Ese
es exactamente el grupo que ataca `--pair-frac` (§11), y ninguna combinación de
`spe_the`/`pl_the`/`msep` lo recupera.

---

## 13. El sobremuestreo de pares, medido: el mecanismo funciona, el balance sale neutro

Run ejecutado: pool `data/pre_elong_2b_pair` (`--pair-px 6 --pair-ratio 0.5`,
1000 npz, 6,9 GB) + `train_model.py --pair-frac 0.25`, 210 épocas, `--init` el
base. `val_loss 0.2544` al final — **no comparable** con el 0.1942 del run sin
pares: el propio val lleva el sobremuestreo, así que la distribución de
validación cambió.

Mismos 8 espectros, 2048 picos reales, misma cadena (`spe_the 0.2`, `msep [3,1]`):

| modelo | `pl_the` | picos | det/GT | precision | recall | F1 | anillo | FN fuertes |
|---|---|---|---|---|---|---|---|---|
| bandas | 0,50 | 1741 | 0,85× | 0,918 | 0,781 | 0,844 | 6,7 % | 100 |
| bandas | 0,60 | 1633 | 0,80× | 0,956 | 0,762 | 0,848 | 6,7 % | 110 |
| bandas | 0,65 | 1593 | 0,78× | 0,970 | 0,754 | **0,849** | 6,7 % | 114 |
| pares | 0,50 | 1892 | 0,92× | 0,875 | **0,808** | 0,840 | 9,8 % | 81 |
| **pares** | **0,60** | 1739 | 0,85× | 0,930 | 0,790 | **0,855** | 9,8 % | 89 |
| pares | 0,65 | 1665 | 0,81× | 0,953 | 0,774 | 0,854 | 9,8 % | 95 |

### Lo que sí se consiguió (el mecanismo)

* **Recall**: 0,762 → 0,790 a igualdad de umbral (0,781 → 0,808 con `pl_the 0,5`).
* **FN fuertes** (A ≥ 20σ): 110 → 89 en sus respectivos óptimos; y de los **100**
  FN fuertes concretos del modelo anterior, **24 quedan recuperados** (24 %), que
  es más de lo que ninguna combinación de `spe_the`/`pl_the`/`msep` lograba (§12).
* Los FN del modelo nuevo siguen teniendo un pico mayor a ≤6 px en 218 de 393
  casos (108 de ellos ≥2× más débiles): el problema está atacado, no resuelto.

### Lo que costó

* **Precisión**: −4,3 puntos a igualdad de umbral (0,918 → 0,875 con `pl_the 0,5`).
* **La métrica del anillo empeora: 6,7 % → 9,8 %.** Era el riesgo anunciado: la
  firma del par («bulto pequeño junto a un pico grande = pico») es la MISMA
  geometría que produce los hombros del anillo. El modelo aprendió a ser más
  generoso en el flanco del pico.

### Veredicto

El mejor punto absoluto sube de **0,849 a 0,855** (+0,006), es decir: el techo
que se había estimado (0,859 recuperando la mitad de los FN fuertes) se rozó,
pero el coste en precisión y en anillo se comió casi toda la ganancia.

**Verificación en 200 espectros** (los de val del split, 51 200 picos reales,
nada de esto se usó para ajustar nada): `out_elong_2b` **F1 0,844** (P 0,939,
R 0,766, anillo 7,1 %) frente a `out_pair` **F1 0,832** (P 0,889, R 0,782,
anillo 10,1 %). Con 25× más datos la ventaja de 8 espectros **desaparece y se
invierte**: el sobremuestreo de pares gana 1,6 puntos de recall y pierde 5 de
precisión.

**Recomendación: `out_elong_2b` sigue siendo el modelo de producción.** El
sobremuestreo de pares queda implementado, probado y documentado como
experimento **negativo** con su medida; si algún día se reintenta, que sea con
`--pair-frac 0.1` (menos presión sobre el flanco) y midiendo en los 200 ids, no
en 8. Lo que sí se queda del trabajo es la cadena: `--pl-the 0.6` es el óptimo
también en los 200 ids (0,844; con 0,55 baja a 0,841 y con 0,65 a 0,842), junto
con `--msep 3 1`.

### Dos bugs reales que salieron por el camino (arreglados)

1. `pair_marks.row_pair_flags` marcaba un **disco de radio `px`** alrededor del
   pico en vez de sus píxeles de etiqueta → 1112 filas marcadas de 14525 en el
   pool real, de las que solo 49 eran positivas (96 % negativos). Con eso el
   «sobremuestreo de positivos» habría añadido fondo. Ahora toma `pixel_sets`.
2. `train_model._points_per_spectrum` leía `z["pair"]` **fuera del
   `with np.load(...)`** → `AttributeError: 'NoneType' object has no attribute
   'open'` al calcular `len(val_seq)`: el entrenamiento no arrancaba. Corregido y
   cubierto con tests (170 pasan, 18 skips).

---

## 14. La cola de la P3 por eje: métrica y banda anisótropa

### El problema, medido

El match F1/precisión dice si el pico aparece, no **cuánto peso deja la P3
alrededor**. Sobre 6 espectros de validación y 297 picos aislados (A >= 0,05) se
midió el perfil en cruz de la P3 (centro normalizado a 1, promedio de las dos
muestras simétricas a ±d px) y el FWHM por eje:

| Mapa | FWHM F1 (px) | FWHM F2 (px) | ratio F1/F2 | decay ±3 px F1 | decay ±3 px F2 |
|---|---:|---:|---:|---:|---:|
| S (señal) | 6,75 | 4,10 | 1,73 | — | — |
| P3 producción (`out_elong_2b`) | 1,97 | 1,98 | **1,02** | **5,9 %** | 0,1 % |
| P3 base isotrópico | 9,68 | 1,79 | **5,60** | **107 %** | — |

Lectura: la S es anisótropa (F1 más ancha que F2, ratio 1,73), pero la P3 de
producción es **casi isótropa** (ratio 1,02) y conserva una cola en F1 (5,9 % a
±3 px) que la S no tiene (0,1 %). El base isotrópico exagera el otro extremo:
107 % a ±3 px en F1, un plateau que `msep` tiene que colapsar. La cola F1 es,
por tanto, la firma que produce la cresta de un pico a lo largo de F1.

### La métrica (implementada)

`shape_values(P, rows, cols)` y `shape_report(values)` en
`scripts/plotting/run_train_2d.py`:

```
F1(d) = 0.5 * (|P[r-d, c]| + |P[r+d, c]|) / |P[r, c]|
F2(d) = 0.5 * (|P[r, c-d]| + |P[r, c+d]|) / |P[r, c]|
```

con `d = 3, 6`, el centro tomado como máximo local en una ventana 3x3 (el GT cae
a <=1 px del máximo real), y mediana sobre los picos. Un pico en el borde no
aporta la distancia que se sale de la imagen; un centro con |P| = 0 se ignora.

Sale en `report.json` (`"p3_shape"`) y, agregado sobre todos los espectros, en
`batch_report.json` (`"p3_shape"`) de `infer_batch.py`:

```json
"p3_shape": {
  "n": 297,
  "at_3px": {"f1": 0.114, "f2": 0.002, "f1_over_f2": 26.3},
  "at_6px": {"f1": 0.050, "f2": 0.001, "f1_over_f2": 17.32}
}
```

Es una comprobación de integración de un solo espectro de val (`0000`, cadena
`--spe-the 0.2 --pl-the 0.6 --msep 3 1`): la dirección coincide con la medición
agregada (cola F1 >> F2), no es una medida nueva de calidad.

### La banda anisótropa (implementada, sin reentrenar todavía)

La banda isotrópica `3 <= d <= 6` no distingue ejes: un negativo a 5 px en F1 y
otro a 5 px en F2 entran por igual. La banda nueva selecciona un rectángulo:

```
F1 (filas)    3 .. 10 px
F2 (columnas) 1 .. 3 px
```

* `hard_negatives.axis_distance_to_positive` descompone la distancia en `d1`
  (fila) y `d2` (columna) usando los índices del **mismo**
  `distance_transform_edt(..., return_indices=True)`, así que `d` sigue siendo
  idéntica a `distance_to_positive`.
* `aniso_band_mask(d1, d2, f1_lo, f1_hi, f2_lo, f2_hi)` construye la máscara.
* `choose_background(..., band2_mask=...)` la usa en lugar de `band2_px`; el
  anillo (`d <= hard_px`) tiene prioridad y una fila nunca cae en dos bandas.
* `preextract.py --hard-aniso 3 10 1 3 --hard-frac2 0.25` guarda `d1`/`d2` en el
  npz. `train_model.py --hard-aniso 3 10 1 3 --hard-frac2 0.25` remuestrea la
  banda cada época y **se niega a arrancar** si el pool no trae `d1`/`d2`.

```bash
# pool (una vez; el tamaño no cambia)
python scripts/preextract.py --dataset <dataset> --ids data/train_2d_elong/manifest.json \
  --out data/pre_aniso --seed 0 --pool-k 24 --hard-frac 0.55 --hard-px 2 \
  --hard-aniso 3 10 1 3 --hard-frac2 0.25 --fast

# fine-tune
python scripts/train_model.py --data data/pre_aniso --preextracted \
  --init Trained/Probability_2D/US/DNN_0 --lr 1e-4 --batch 65536 --bg-ratio 4 \
  --hard-frac 0.55 --hard-px 2 --hard-aniso 3 10 1 3 --hard-frac2 0.25 \
  --epochs 210 --patience 20 --out <out>
```

**Estado**: el mecanismo está implementado, probado (`tests/test_hard_aniso.py`,
`tests/test_p3_shape.py`) y el pool se puede generar. **No hay todavía un modelo
reentrenado con esta banda**, así que no se reclama ninguna ganancia de F1: la
comparación pendiente es un fine-tune idéntico con y sin `--hard-aniso`, mirando
F1/precisión/recall en los 200 de val **y** `p3_shape` (F1/F2 a 3 y 6 px).

**Detalle a barrer**: `F2LO = 1` excluye la cresta pura (`d2 = 0`). Si el
reentrenamiento no mueve la cola, la ablación natural es `3 10 0 3`, que sí
incluye las columnas del pico.
