"""Target alternativo: la CRESTA del pico en vez del marcador de 1/2/4 px.

Por qué
-------
La etiqueta actual marca 1/2/4 px por pico (mediana 2) sobre picos de ~10 px de
ancho en F1. Medido: a 1 fila de distancia la ventana de entrada del modelo es
97 % idéntica, así que el modelo solo puede devolver la densidad local de
etiqueta (~0.2) — y lo hace bien (P media 0.14–0.21 frente a densidad 0.19–0.20),
pero esos valores intermedios son los que el picker convierte en 2–4 picos por
pico real.

La alternativa: etiquetar **la región del pico por encima de una fracción de su
propia amplitud** (`frac`, 0.5 = semialtura). Entonces:
  * dentro de la cresta la etiqueta es 1 en todas las filas -> sin ambigüedad de
    fila, el target pasa a ser una función de la señal;
  * fuera de la cresta es 0 -> el modelo aprende dónde acaba el pico, que es lo
    que hoy no controla;
  * el picker pasa a ser "un pico por componente conexa" (centroide), no
    "todos los máximos locales".

Dos detalles que costaron un test rojo y están resueltos:
  1. El umbral se toma de `amp` (la amplitud del pico en el manifest), no de
     `S[row, col]`: con un pico débil junto a uno fuerte, `S[row, col]` está
     contaminado y la región del débil se fuga hacia el fuerte.
  2. Las regiones son **exclusivas** (`exclusive=True`): en la zona de solape
     cada píxel se asigna al pico más cercano (`exclusive=False` se lo queda el
     primero que llega). Así dos picos solapados conservan cada uno su trozo y el
     picker «un pico por componente» los puede contar por separado; el techo de
     recall se mide con `count_instances`.

`ridge_labels` + `components` permiten medir el techo de recall ANTES de gastar
un entrenamiento (¿cuántas componentes salen por espectro frente a cuántos picos
hay?).
"""
import numpy as np

RIDGE_FRAC = 0.5
RIDGE_RADIUS = 8


def ridge_region(S, row, col, amp=None, frac=RIDGE_FRAC, radius=RIDGE_RADIUS):
    """Componente conexa (4-vecindad) por encima de `frac * amp`, ventana local.

    Devuelve (máscara local, (r0, c0)). Se usa la señal observada (`S`), no los
    parámetros del generador: la etiqueta es lo que de verdad se ve. `radius`
    acota la ventana (los picos son de ~10 px en F1; 16 px de diámetro sobra).
    """
    S = np.asarray(S)
    H, W = S.shape
    if amp is None:
        amp = S[row, col]
    r0, r1 = max(0, row - radius), min(H, row + radius + 1)
    c0, c1 = max(0, col - radius), min(W, col + radius + 1)
    win = S[r0:r1, c0:c1]
    lab0 = np.zeros(win.shape, dtype=bool)
    if not (amp > 0) or not np.isfinite(amp):
        # amplitud nula (o no finita): el pico no existe, no hay cresta que
        # etiquetar. Sin este guardia `thr = 0` y la región inunda la ventana.
        return lab0, (r0, c0)
    thr = frac * amp
    above = win >= thr
    lab = np.zeros(win.shape, dtype=bool)
    i0, j0 = row - r0, col - c0
    if not above[i0, j0]:
        return lab, (r0, c0)
    stack = [(i0, j0)]
    lab[i0, j0] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < lab.shape[0] and 0 <= b < lab.shape[1] \
                    and above[a, b] and not lab[a, b]:
                lab[a, b] = True
                stack.append((a, b))
    return lab, (r0, c0)


def ridge_owners(S, rows, cols, amps=None, frac=RIDGE_FRAC, radius=RIDGE_RADIUS,
                 exclusive=True):
    """Mapa de propietario: índice del pico dueño de cada píxel (-1 = ninguno).

    `amps`: amplitud de cada pico (manifest `A`). Si falta, se usa `S[row, col]`
    (peor: un pico débil junto a uno fuerte se fuga hacia el fuerte).

    `exclusive=True` (por defecto) asigna cada píxel al pico más cercano: dos
    picos solapados conservan cada uno su trozo, si lo tienen. `exclusive=False`
    es la unión cruda (el primero que llega se queda el píxel), útil para medir
    cuánto se fusionaría sin la corrección.
    """
    S = np.asarray(S)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    own = np.full(S.shape, -1, dtype=np.int32)
    if len(rows) == 0:
        return own
    if amps is None:
        amps = S[rows, cols]
    amps = np.asarray(amps, dtype=float)
    best = np.full(S.shape, np.inf, dtype=np.float64) if exclusive else None
    for i, (row, col, amp) in enumerate(zip(rows, cols, amps)):
        lab, (r0, c0) = ridge_region(S, row, col, amp, frac, radius)
        if not lab.any():
            continue
        h, w = lab.shape
        win = np.s_[r0:r0 + h, c0:c0 + w]
        if exclusive:
            rr = np.arange(r0, r0 + h)[:, None]
            cc = np.arange(c0, c0 + w)[None, :]
            d2 = (rr - row) ** 2 + (cc - col) ** 2
            bs = best[win]
            upd = lab & (d2 < bs)
            sub = own[win]
            sub[upd] = i
            bs[upd] = d2[upd]
        else:
            sub = own[win]
            sub[lab & (sub < 0)] = i
    return own


def ridge_labels(S, rows, cols, amps=None, frac=RIDGE_FRAC, radius=RIDGE_RADIUS,
                 exclusive=True):
    """Etiquetas de cresta (float32 0/1): `ridge_owners(...) >= 0`.

    Es el target que consumiría el entrenamiento; para el conteo por pico se usa
    `ridge_owners` + `count_instances`.
    """
    own = ridge_owners(S, rows, cols, amps=amps, frac=frac, radius=radius,
                       exclusive=exclusive)
    return (own >= 0).astype(np.float32)


def count_instances(own):
    """Techo de detecciones: trozos conexos por pico en el mapa de propietarios.

    Un pico con su región partida en dos daría dos detecciones, así que se cuenta
    cada trozo: es el máximo recall alcanzable por «un pico por componente».
    """
    own = np.asarray(own)
    idx = np.unique(own)
    return int(sum(components(own == i) for i in idx if i >= 0))


def components(mask):
    """Nº de componentes conexas (4-vecindad) de una máscara booleana.

    Es el techo del picker nuevo: un pico por componente, centroide como
    posición. Usa scipy si está; si no, un etiquetado por pila mínimo.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    try:
        from scipy.ndimage import label
        _, n = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        return int(n)
    except ImportError:  # pragma: no cover
        seen = np.zeros(mask.shape, dtype=bool)
        n = 0
        H, W = mask.shape
        for i, j in zip(*np.nonzero(mask)):
            if seen[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and mask[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
        return n
