"""Pares apretados: dos picos que el perfil en cruz no puede distinguir.

Por qué
-------
El error que queda tras el minado del anillo son falsos negativos, y una parte
son picos enterrados en el flanco de uno mayor: de 2048 picos reales, 486 FN,
110 FUERTES (A >= 20*noise.sd) y 45 con un pico mayor a <= 6 px. La fila que la
red ve es una CRUZ (eje F1 + eje F2 por el centro de cada píxel), así que un
vecino en diagonal a 4 px no deja huella en ninguna de las dos ramas: querer
derivar el par desde el perfil da una sensibilidad del 14 % (24 de 176 picos
enterrados). El manifiesto del dataset sí conoce el centro y la amplitud de
TODOS los picos, así que la marca del par se calcula AHÍ, una sola vez, al
generar el pool (preextract.py --pair-px) y viaja en el npz como un byte por
fila del pool (`pair`); el entrenamiento la usa para sobremuestrear esas filas
cada época (train_model.py --pair-frac).

Qué hace este módulo
--------------------
  * `peak_pair_flags`: decide QUÉ picos forman un par apretado. Un pico se
    marca si existe OTRO con amplitud MAYOR (estricto: los empates no cuentan),
    a distancia euclídea <= `px` y con amps[propio]/amps[otro] <= `ratio`. Los
    límites son inclusivos: distancia == px marca y ratio == umbral marca.
  * `row_pair_flags`: propaga la marca a los índices planos del pool. Un píxel
    se marca 1 si está a <= `px` del centro de ALGÚN pico marcado (los
    pixelSet de la etiqueta, de 1/2/4 px, caen dentro de sobra); un píxel junto
    a un pico grande que no forma par con nadie sale 0.

NumPy puro a propósito: preextract.py corre sin TensorFlow y esto no añade
dependencias (fijado en tests/test_pair_marks.py).
"""
import numpy as np


def peak_pair_flags(rows, cols, amps, px=6.0, ratio=0.5):
    """(bool, n): True si el pico i forma un par apretado con otro mayor.

    `rows`/`cols` son los centros (fila = F1, col = F2) y `amps` las amplitudes
    del manifiesto. El pico i se marca si existe j != i con amps[j] > amps[i]
    (sin empates), distancia euclídea <= `px` y amps[i]/amps[j] <= `ratio`.
    Acepta listas, tuplas y arrays de dtypes varios; con n = 0 o 1 no hay par
    posible y devuelve todo False (o un array vacío). Las amplitudes son
    amplitudes de picos (positivas).
    """
    rows = np.asarray(rows, dtype=np.float64).reshape(-1)
    cols = np.asarray(cols, dtype=np.float64).reshape(-1)
    amps = np.asarray(amps, dtype=np.float64).reshape(-1)
    if not (len(rows) == len(cols) == len(amps)):
        raise ValueError(
            f"rows ({len(rows)}), cols ({len(cols)}) y amps ({len(amps)}) "
            f"deben tener la misma longitud")
    n = len(rows)
    if n <= 1:
        return np.zeros(n, dtype=bool)
    dx = cols[:, None] - cols[None, :]
    dy = rows[:, None] - rows[None, :]
    close = (dx * dx + dy * dy) <= float(px) * float(px)
    greater = amps[None, :] > amps[:, None]     # amps[j] > amps[i]: sin empates
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ok = (amps[:, None] / amps[None, :]) <= float(ratio)
    # other[i, j]: j es un vecino mayor, cercano y bastante más fuerte de i
    other = greater & ratio_ok & close
    return other.any(axis=1)


def row_pair_flags(rows, cols, amps, idx_flat, W, px=6.0, ratio=0.5,
                   pixel_sets=None):
    """(uint8, len(idx_flat)): 1 si el píxel es de un pico marcado.

    Con `pixel_sets` (la lista de `peaks[].pixelSet` del manifiesto, un elemento
    por pico) el marcado es EXACTO: solo los píxeles de la etiqueta de los picos
    que `peak_pair_flags` marca. Es la forma correcta, y la única que sirve para
    sobremuestrear positivos: el pool guarda filas de FONDO a esas mismas
    distancias, y marcarlas haría que el sobremuestreo eligiera casi solo
    negativos (medido en el pool real, id 0001: 1112 filas marcadas frente a 49
    positivas, es decir un 96 % de negativos).

    Sin `pixel_sets` se mantiene la aproximación por radio: se marca todo píxel
    a distancia euclídea <= `px` del centro de un pico marcado. Útil cuando solo
    hay centros, pero cuenta fondo y hasta los píxeles del pico grande vecino.

    Acepta 0 picos, 1 pico, listas, arrays vacíos y dtypes varios; con pool
    vacío devuelve un array vacío.
    """
    rows = np.asarray(rows, dtype=np.float64).reshape(-1)
    cols = np.asarray(cols, dtype=np.float64).reshape(-1)
    marked = peak_pair_flags(rows, cols, amps, px=px, ratio=ratio)
    idx_flat = np.asarray(idx_flat).reshape(-1).astype(np.int64, copy=False)
    out = np.zeros(len(idx_flat), dtype=np.uint8)
    if len(idx_flat) == 0 or not marked.any():
        return out
    if pixel_sets is not None:
        etiqueta = []
        for i in np.where(marked)[0]:
            for rc in pixel_sets[i]:
                etiqueta.append(int(rc[0]) * int(W) + int(rc[1]))
        if not etiqueta:
            return out
        out[np.isin(idx_flat, np.asarray(etiqueta, dtype=np.int64))] = 1
        return out
    prow = idx_flat // int(W)
    pcol = idx_flat % int(W)
    # distancia al cuadrado del píxel al centro de cada pico marcado: los picos
    # marcados son pocos (decenas), así que un bucle con mínimo incremental
    # gasta menos memoria que la matriz (len(idx), n_marked) completa
    best2 = np.full(len(idx_flat), np.inf)
    for r, c in zip(rows[marked].tolist(), cols[marked].tolist()):
        dr = prow - r
        dc = pcol - c
        np.minimum(best2, dr * dr + dc * dc, out=best2)
    out[best2 <= float(px) * float(px)] = 1
    return out