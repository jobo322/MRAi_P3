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


def choose_background(candidates, d_candidates, n_bg, hard_frac, hard_px,
                      rng):
    """Posiciones (dentro de `candidates`) del fondo de una época.

    Devuelve `n_bg` posiciones sin repetición, de las cuales `hard_frac` se
    reservan a candidatos con `d <= hard_px`. Si no hay suficientes duros (o
    fáciles) se completa con los otros: el tamaño de la muestra no cambia, que es
    lo que mantiene constantes los steps por época.
    """
    candidates = np.asarray(candidates)
    d_candidates = np.asarray(d_candidates)
    n_bg = min(int(n_bg), len(candidates))
    if n_bg <= 0:
        return np.array([], dtype=np.int64)
    hard = np.where(d_candidates <= hard_px)[0]
    if hard_frac <= 0 or len(hard) == 0:
        return rng.choice(len(candidates), size=n_bg, replace=False)
    n_hard = min(len(hard), int(round(n_bg * float(hard_frac))))
    take_hard = rng.choice(hard, size=n_hard, replace=False)
    rest = np.setdiff1d(np.arange(len(candidates)), hard)
    n_easy = n_bg - n_hard
    if n_easy > len(rest):
        remaining = np.setdiff1d(hard, take_hard)
        extra = rng.choice(remaining, size=n_easy - len(rest), replace=False)
        take_easy = np.concatenate([rest, extra])
    elif n_easy:
        take_easy = rng.choice(rest, size=n_easy, replace=False)
    else:
        take_easy = np.array([], dtype=np.int64)
    return np.concatenate([take_hard, take_easy]).astype(np.int64)
