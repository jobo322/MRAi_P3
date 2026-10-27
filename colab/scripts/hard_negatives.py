"""Fondo del entrenamiento: distancia al pico y minado de negativos duros.

Por qué
-------
El fondo se muestreaba uniforme sobre TODOS los ceros. Pero la etiqueta es un
marcador sub-píxel (1/2/4 px por pico, 2,26 de media) mientras el pico abarca
~20-50 px, así que los ceros *dentro* del pico son la mayoría de los errores
posibles y entraban al 0,48 % del fondo. Medido por época y espectro: 578
positivos frente a ~11 negativos a 1 px y ~26 a <=2 px, con el 96,7 % del fondo a
>=5 px de cualquier pico (donde el modelo ya es casi perfecto). Consecuencia
medida en el modelo entrenado: el 81 % de los ceros inmediatamente al lado de un
pico reciben P > 0,5, la frontera aprendida queda a ~2-3 px en vez de en la del
label, y el pipeline elige ~7,7x picos (1.966 donde hay 256).

Qué hace este módulo
--------------------
  * `distance_to_positive`: distancia (px) de cada píxel al positivo más cercano.
  * `axis_distance_to_positive`: la misma distancia descompuesta por eje (F1 =
    filas, F2 = columnas), para la banda anisótropa.
  * `aniso_band_mask`: banda rectangular `F1 in [f1_lo, f1_hi]` y
    `F2 in [f2_lo, f2_hi]` px del positivo más cercano.
  * `plan_pool_idx` (en preextract) reserva `hard_frac` del presupuesto de fondo
    del pool al anillo; el pool guarda esas distancias en `d`, así el submuestreo
    por época puede pedir cualquier `hard_px` sin volver a generarlo.
  * `choose_background`: por época, `n_bg` negativos de los que `hard_frac` salen
    del anillo (<= `hard_px`). Mantiene el tamaño total y toma todos los
    positivos, que es lo que fijan tests/test_pool_equiv.py.

Sin `hard_frac` (o sin `d`, p.ej. un pool antiguo) todo cae al camino histórico:
un único `rng.choice` sobre todos los negativos, byte a byte igual que antes.
"""
import numpy as np

HARD_PX = 2
HARD_FRAC = 0.25


def distance_to_positive(L, cap=255):
    """Distancia en px de cada píxel al positivo (label = 1) más cercano.

    Euclídea, redondeada a píxel entero (error <= 0,5) y recortada a `cap` para
    caber en uint8. `cap` acota el tamaño en disco del array de distancias del
    pool (~6 kB por espectro con pool_k=10).
    """
    from scipy.ndimage import distance_transform_edt
    L = np.asarray(L)
    d = distance_transform_edt(L <= 0.5)
    return np.rint(np.minimum(d, cap)).astype(np.uint8)


def axis_distance_to_positive(L, cap=255):
    """(d, d1, d2): distancia euclídea y su descomposición por eje.

    `d1` es |Δfila| y `d2` es |Δcolumna| al positivo más cercano (F1 = filas,
    F2 = columnas). Los índices salen del MISMO `distance_transform_edt`, así
    que ambos ejes apuntan al mismo positivo y `d` coincide exactamente con
    `distance_to_positive`. `d1`/`d2` se recortan a `cap` para caber en uint8.

    Por qué: la banda isotrópica `3 <= d <= 6` no distingue la cola F1 (la que
    el modelo aprende y hay que contrarrestar) de la F2. Con esto la banda se
    puede pedir como un rectángulo en (d1, d2).
    """
    from scipy.ndimage import distance_transform_edt
    L = np.asarray(L)
    if L.ndim != 2:
        raise ValueError(f"L debe ser 2D, no {L.shape}")
    d, idx = distance_transform_edt(L <= 0.5, return_indices=True)
    rows = np.arange(L.shape[0])[:, None]
    cols = np.arange(L.shape[1])[None, :]
    d1 = np.abs(rows - idx[0])
    d2 = np.abs(cols - idx[1])
    return (np.rint(np.minimum(d, cap)).astype(np.uint8),
            np.minimum(d1, cap).astype(np.uint8),
            np.minimum(d2, cap).astype(np.uint8))


def aniso_band_mask(d1, d2, f1_lo, f1_hi, f2_lo, f2_hi):
    """Máscara booleana de la banda anisótropa en (d1, d2).

    Incluye los píxeles con `f1_lo <= d1 <= f1_hi` **y** `f2_lo <= d2 <= f2_hi`.
    No sabe nada del anillo (`d <= hard_px`): `choose_background` lo excluye
    para que una fila no caiga en dos bandas a la vez.

    Ejemplo medido (6 val, 297 picos): la P3 de producción tiene FWHM F1/F2 =
    1,97/1,98 y la S 6,75/4,10. Para empujar la cola F1 sin tocar la F2:
    `aniso_band_mask(d1, d2, 3, 10, 1, 3)`.
    """
    vals = (f1_lo, f1_hi, f2_lo, f2_hi)
    if min(vals) < 0:
        raise ValueError(f"la banda anisótropa no admite negativos: {vals}")
    if f1_lo > f1_hi or f2_lo > f2_hi:
        raise ValueError(f"rango invertido en la banda anisótropa: {vals}")
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    if d1.shape != d2.shape:
        raise ValueError(f"d1 {d1.shape} y d2 {d2.shape} no coinciden")
    return ((d1 >= f1_lo) & (d1 <= f1_hi) &
            (d2 >= f2_lo) & (d2 <= f2_hi))


def split_negatives(neg_idx, d_neg, hard_px):
    """(duros, fáciles): los negativos con `d <= hard_px` y el resto."""
    neg_idx = np.asarray(neg_idx)
    d_neg = np.asarray(d_neg)
    if len(neg_idx) != len(d_neg):
        raise ValueError(f"{len(neg_idx)} negativos y {len(d_neg)} distancias")
    return neg_idx[d_neg <= hard_px], neg_idx[d_neg > hard_px]


def pool_sections(n_bg, hard_frac):
    """(n_hard, n_easy) del presupuesto de fondo: cuánto se reserva al anillo."""
    n_bg = max(0, int(n_bg))
    n_hard = min(n_bg, int(round(n_bg * float(hard_frac))))
    return n_hard, n_bg - n_hard


def band_sections(n_bg, hard_frac, hard_frac2=0.0):
    """(n_hard, n_band2, n_easy): el presupuesto de fondo repartido en dos bandas.

    `hard_frac` reserva el anillo (`d <= hard_px`) y `hard_frac2` la banda media
    (`hard_px < d <= band2_px`); el resto se sortea uniforme. Con
    `hard_frac2 = 0` devuelve lo mismo que `pool_sections`.

    Por qué dos bandas: medido sobre 8 espectros, el anillo (d<=2) entró al pool
    al 54 % pero la banda 3-6 px —donde caen los hombros que se vuelven falsos
    positivos— al 0,5 %, porque el fondo uniforme la diluye frente a las ~500 000
    filas a d>6. Repartir el presupuesto es lo que la saca de ahí.
    """
    n_bg = max(0, int(n_bg))
    f1, f2 = float(hard_frac), float(hard_frac2)
    if f1 < 0 or f2 < 0:
        raise ValueError(f"fracciones negativas: {f1}, {f2}")
    if f1 + f2 > 1.0 + 1e-9:
        raise ValueError(
            f"hard_frac ({f1}) + hard_frac2 ({f2}) suman más que el presupuesto "
            f"de fondo: no se puede reservar más de lo que hay")
    n_hard = min(n_bg, int(round(n_bg * f1)))
    n_band2 = min(n_bg - n_hard, int(round(n_bg * f2)))
    return n_hard, n_band2, n_bg - n_hard - n_band2


def axis_sections(n_band2, band2_axis_frac):
    """(n_axis, n_rest): el cupo del eje F1 puro dentro de la banda 2.

    `band2_axis_frac` es la fracción del presupuesto de la banda que se
    reserva a los píxeles del eje F1 (d2 == 0): 0.20 fija la relación
    eje:diagonal 1:4 y 0.25 la 1:3. Sin cupo el sorteo uniforme de la banda
    la deja en la proporción natural (~1:5), porque los píxeles diagonales
    son ~6x más numerosos que los axiales y el fondo hereda ese sesgo.
    """
    n_band2 = max(0, int(n_band2))
    f = float(band2_axis_frac)
    if f < 0 or f > 1:
        raise ValueError(f"band2_axis_frac ({f}) debe estar en [0, 1]")
    n_axis = min(n_band2, int(round(n_band2 * f)))
    return n_axis, n_band2 - n_axis


def choose_background(candidates, d_candidates, n_bg, hard_frac, hard_px,
                      rng, band2_px=None, band2_frac=0.0, band2_mask=None,
                      band2_axis_mask=None, band2_axis_frac=0.0):
    """Posiciones (dentro de `candidates`) del fondo de una época.

    Devuelve `n_bg` posiciones sin repetición: `hard_frac` del anillo
    (`d <= hard_px`), `hard_frac2` de la banda media y el resto uniforme. Si a
    una banda le faltan candidatos se completa con las otras: el tamaño de la
    muestra no cambia, que es lo que mantiene constantes los steps por época.

    La banda media es isotrópica (`hard_px < d <= band2_px`) o anisótropa
    (`band2_mask`, alineada con `candidates`, p.ej. de `aniso_band_mask`). Las
    dos formas son excluyentes. La anisótropa nunca incluye filas del anillo,
    aunque la máscara las cubra: el anillo tiene prioridad y una fila no puede
    contarse dos veces.

    Con `band2_frac = 0` (o `band2_px=None`) el sorteo es EXACTAMENTE el
    histórico: misma secuencia de `rng`, mismo orden de salida.

    `band2_axis_frac` (0 = apagado) reparte el presupuesto de la banda
    anisótropa: esa fracción sale de los píxeles del eje F1 puro
    (`band2_axis_mask`, los de d2 == 0) y el resto de la diagonal. Sin cupo
    la banda sale en la proporción natural eje:diagonal ~1:5; 0.20 la fija
    en 1:4 y 0.25 en 1:3. Si en la banda faltan píxeles de una de las dos
    clases, el faltante lo absorbe el resto uniforme: el tamaño no cambia.
    Con `band2_axis_frac = 0` ni la máscara ni el cupo se miran: la secuencia
    de `rng` y el resultado son byte a byte los del camino histórico.
    """
    candidates = np.asarray(candidates)
    d_candidates = np.asarray(d_candidates)
    n_bg = min(int(n_bg), len(candidates))
    if n_bg <= 0:
        return np.array([], dtype=np.int64)
    if band2_mask is not None and band2_px is not None:
        raise ValueError("band2_px y band2_mask son excluyentes")
    if band2_mask is not None:
        if band2_frac <= 0:
            raise ValueError("band2_mask necesita band2_frac > 0")
        band2_mask = np.asarray(band2_mask)
        if band2_mask.shape != candidates.shape:
            raise ValueError(
                f"band2_mask {band2_mask.shape} != candidates {candidates.shape}")
    hard = np.where(d_candidates <= hard_px)[0]
    use_band2 = band2_frac > 0 and (band2_px is not None or band2_mask is not None)
    axis_mask = None
    if band2_axis_frac > 0:
        if not use_band2:
            raise ValueError(
                "band2_axis_frac > 0 necesita la banda de negativos duros "
                "(band2_frac > 0 con band2_px o band2_mask): el cupo reparte "
                "el presupuesto de la banda, no el del fondo entero")
        if band2_axis_mask is None:
            raise ValueError(
                "band2_axis_frac > 0 necesita band2_axis_mask (los píxeles "
                "del eje dentro de la banda): sin ella el cupo no se puede "
                "aplicar")
        axis_mask = np.asarray(band2_axis_mask)
        if axis_mask.shape != candidates.shape:
            raise ValueError(
                f"band2_axis_mask {axis_mask.shape} != candidates "
                f"{candidates.shape}")
    if use_band2:
        if band2_mask is not None:
            band2 = np.where(band2_mask & (d_candidates > hard_px))[0]
        else:
            band2 = np.where((d_candidates > hard_px) &
                             (d_candidates <= band2_px))[0]
    else:
        band2 = np.array([], dtype=np.int64)

    if not use_band2 and (hard_frac <= 0 or len(hard) == 0):
        # Camino histórico intacto: un único sorteo uniforme sobre todo el fondo.
        return rng.choice(len(candidates), size=n_bg, replace=False)

    n_hard_req, n_band2_req, _ = band_sections(n_bg, hard_frac, band2_frac)
    n_hard = min(len(hard), n_hard_req)
    take_hard = (rng.choice(hard, size=n_hard, replace=False) if n_hard
                 else np.array([], dtype=np.int64))
    if axis_mask is not None:
        # Cupo eje:diagonal dentro de la banda. Rama aparte a propósito: con
        # cupo 0 el sorteo de la banda tiene que seguir siendo la UNICA llamada
        # a rng.choice de abajo, byte a byte como siempre (tests que fijan la
        # secuencia de rng).
        n_axis_req, _ = axis_sections(n_band2_req, band2_axis_frac)
        axis = np.where(axis_mask & (d_candidates > hard_px))[0]
        if band2_mask is not None:
            diag = np.where(band2_mask & (d_candidates > hard_px)
                            & ~axis_mask)[0]
        else:  # banda isotrópica (band2_px): la diagonal es band2 sin el eje
            diag = np.setdiff1d(band2, axis)
        n_axis = min(len(axis), n_axis_req)
        n_diag = min(len(diag), n_band2_req - n_axis)
        parts = []
        if n_axis:
            parts.append(rng.choice(axis, size=n_axis, replace=False))
        if n_diag:
            parts.append(rng.choice(diag, size=n_diag, replace=False))
        take_band2 = (np.concatenate(parts) if parts
                      else np.array([], dtype=np.int64))
    else:
        n_band2 = min(len(band2), n_band2_req)
        take_band2 = (rng.choice(band2, size=n_band2, replace=False) if n_band2
                      else np.array([], dtype=np.int64))
    taken = np.concatenate([take_hard, take_band2])

    # El resto absorbe lo que falte en las bandas, así el tamaño nunca cambia.
    reserved = np.concatenate([hard, band2]) if use_band2 else hard
    rest = np.setdiff1d(np.arange(len(candidates)), reserved)
    n_easy = n_bg - len(take_hard) - len(take_band2)
    if n_easy > len(rest):
        # Solo las filas reservadas que NO salieron: `rest` ya está en
        # `take_easy`, así que volver a sortear sobre todo el fondo las
        # duplicaría y dejaría parte del anillo fuera.
        remaining = np.setdiff1d(reserved, taken)
        extra = rng.choice(remaining, size=n_easy - len(rest), replace=False)
        take_easy = np.concatenate([rest, extra])
    elif n_easy:
        take_easy = rng.choice(rest, size=n_easy, replace=False)
    else:
        take_easy = np.array([], dtype=np.int64)
    return np.concatenate([take_hard, take_band2, take_easy]).astype(np.int64)
