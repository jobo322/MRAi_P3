"""Tests del cupo del EJE F1 dentro de la banda de negativos duros.

Qué añade y por qué
-------------------
Dentro de la banda anisótropa los píxeles diagonales (d2 > 0) son ~6x más
numerosos que los axiales (d2 == 0, la columna del pico), así que el sorteo
uniforme de la banda la deja en la proporción natural ~1:5 de eje:diagonal y
el fondo hereda ese sesgo. `band2_axis_frac` reserva una fracción del
presupuesto de la banda al eje: 0.20 fija la relación 1:4 y 0.25 la 1:3.

Reglas que fijan estos tests:

* `axis_sections` reparte el presupuesto de la banda con redondeo y clamping,
  y rechaza fracciones fuera de [0, 1];
* `choose_background` con cupo dibuja primero los axiales pedidos y completa
  con diagonales; el tamaño total no cambia (el faltante lo absorbe el resto);
* con `band2_axis_frac = 0` la secuencia de `rng` y el resultado son byte a
  byte los del camino histórico: una sola `rng.choice(band2, ...)`, aunque la
  máscara del eje venga pasada;
* `plan_pool_idx` exige una banda con F2LO = 0 (sin la columna del eje no hay
  píxeles axiales que reservar) y conserva el tamaño del pool.
"""
import importlib
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from hard_negatives import (aniso_band_mask, axis_distance_to_positive,  # noqa: E402
                            axis_sections, choose_background)
from train_model import subsample_pool  # noqa: E402

# F2LO = 0: la banda contiene la columna del pico (d2 == 0), donde viven los
# píxeles del eje F1 puro. Con F2LO > 0 no hay eje dentro de la banda.
ANISO = (3, 10, 0, 3)


class TestAxisSections(unittest.TestCase):
    def test_reparte_con_redondeo_y_clamping(self):
        self.assertEqual(axis_sections(100, 0.2), (20, 80))
        self.assertEqual(axis_sections(100, 0.25), (25, 75))
        self.assertEqual(axis_sections(100, 0.0), (0, 100))
        self.assertEqual(axis_sections(100, 1.0), (100, 0))
        # clamping: el cupo nunca pide más de lo que hay
        self.assertEqual(axis_sections(7, 1.0), (7, 0))
        self.assertEqual(axis_sections(0, 0.5), (0, 0))

    def test_frac_fuera_de_rango_falla(self):
        with self.assertRaises(ValueError):
            axis_sections(10, -0.1)
        with self.assertRaises(ValueError):
            axis_sections(10, 1.1)


def candidatos(n=3000, seed=0):
    """Fondo sintético con anillo, eje F1 y diagonal dentro de la banda.

    Devuelve (candidatos, d, d1, d2, mask) con las tres clases separadas y
    las proporciones conocidas: 300 de anillo, 1000 del eje (d2 == 0) y 1000
    de la diagonal (d2 > 0), todos dentro de la banda anisótropa ANISO.
    """
    rng = np.random.default_rng(seed)
    d = np.full(n, 30, dtype=np.uint8)
    d1 = np.full(n, 30, dtype=np.uint8)
    d2 = np.full(n, 30, dtype=np.uint8)
    anillo = np.arange(300)
    d[anillo] = rng.integers(0, 3, size=len(anillo))
    d1[anillo] = d[anillo]
    d2[anillo] = rng.integers(0, 3, size=len(anillo))
    eje = np.arange(300, 1300)
    d1[eje] = rng.integers(3, 11, size=len(eje))
    d2[eje] = 0
    d[eje] = d1[eje]
    diag = np.arange(1300, 2300)
    d1[diag] = rng.integers(3, 11, size=len(diag))
    d2[diag] = rng.integers(1, 4, size=len(diag))
    d[diag] = d1[diag]
    mask = aniso_band_mask(d1, d2, *ANISO)
    return np.arange(n), d, d1, d2, mask


class TestChooseBackgroundCupo(unittest.TestCase):
    def test_la_sub_banda_tiene_la_fraccion_de_eje_pedida(self):
        cand, d, d1, d2, mask = candidatos()
        take = choose_background(cand, d, 200, 0.0, 2,
                                 np.random.default_rng(7),
                                 band2_frac=0.5, band2_mask=mask,
                                 band2_axis_mask=(mask & (d2 == 0)),
                                 band2_axis_frac=0.2)
        self.assertEqual(len(take), 200)
        self.assertEqual(len(set(take.tolist())), 200, "sin repeticiones")
        n_band2 = 100                       # round(200 * 0.5)
        self.assertTrue(mask[take[:n_band2]].all(),
                        "las primeras n_band2 filas salen de la banda")
        n_axis = 20                         # round(100 * 0.2)
        self.assertEqual(int((d2[take[:n_band2]] == 0).sum()), n_axis)
        self.assertTrue((d2[take[:n_band2]][n_axis:] > 0).all(),
                        "el resto de la banda sale de la diagonal")

    def test_el_tamano_no_cambia_con_banda_corta(self):
        cand, d, d1, d2, _ = candidatos()
        # solo 5 axiales disponibles para un cupo de 20: el faltante pasa al
        # resto uniforme y el tamaño se mantiene
        d2[300:1295] = 1                    # deja solo 5 filas del eje
        mask = aniso_band_mask(d1, d2, *ANISO)
        take = choose_background(cand, d, 200, 0.0, 2,
                                 np.random.default_rng(7),
                                 band2_frac=0.5, band2_mask=mask,
                                 band2_axis_mask=(mask & (d2 == 0)),
                                 band2_axis_frac=0.2)
        self.assertEqual(len(take), 200)
        self.assertEqual(len(set(take.tolist())), 200, "sin repeticiones")
        n_band2 = 100
        self.assertTrue(mask[take[:n_band2]].all())
        self.assertEqual(int((d2[take[:n_band2]] == 0).sum()), 5)
        self.assertGreater(int((d[take] > 2).sum()), n_band2,
                           "el faltante se completa fuera de la banda")

    def test_sin_mascara_falla(self):
        cand, d, d1, d2, mask = candidatos()
        with self.assertRaises(ValueError):
            choose_background(cand, d, 200, 0.0, 2,
                              np.random.default_rng(1),
                              band2_frac=0.5, band2_mask=mask,
                              band2_axis_frac=0.2)

    def test_frac_mayor_que_uno_falla(self):
        cand, d, d1, d2, mask = candidatos()
        with self.assertRaises(ValueError):
            choose_background(cand, d, 200, 0.0, 2,
                              np.random.default_rng(1),
                              band2_frac=0.5, band2_mask=mask,
                              band2_axis_mask=(mask & (d2 == 0)),
                              band2_axis_frac=1.5)

    def test_mascara_de_tamano_incorrecto_falla(self):
        cand, d, d1, d2, mask = candidatos()
        with self.assertRaises(ValueError):
            choose_background(cand, d, 200, 0.0, 2,
                              np.random.default_rng(1),
                              band2_frac=0.5, band2_mask=mask,
                              band2_axis_mask=np.zeros(len(cand) - 1,
                                                       dtype=bool),
                              band2_axis_frac=0.2)


class TestInvarianteRng(unittest.TestCase):
    """Con cupo 0 el sorteo es byte a byte el histórico, máscara o sin máscara.

    El camino histórico de la banda es UNA sola `rng.choice(band2, ...)`;
    cualquier llamada extra a `rng` (aunque sea con un array vacío) cambiaría
    el fondo de todas las épocas siguientes. Lo fija replicando el sorteo a
    mano y comparando byte a byte.
    """

    def test_cupo_cero_es_el_historico_byte_a_byte(self):
        cand, d, d1, d2, mask = candidatos()
        n = len(cand)
        # réplica manual del camino histórico (cupo 0): una sola choice de
        # banda y después el resto uniforme
        rng_ref = np.random.default_rng(11)
        hard = np.where(d <= 2)[0]
        band2 = np.where(mask & (d > 2))[0]
        n_band2 = min(len(band2), 100)      # round(200 * 0.5)
        take_band2 = rng_ref.choice(band2, size=n_band2, replace=False)
        rest = np.setdiff1d(np.arange(n), np.concatenate([hard, band2]))
        take_easy = rng_ref.choice(rest, size=200 - n_band2, replace=False)
        esperado = np.concatenate([take_band2, take_easy])

        historico = choose_background(cand, d, 200, 0.0, 2,
                                      np.random.default_rng(11),
                                      band2_frac=0.5, band2_mask=mask)
        np.testing.assert_array_equal(historico, esperado)

        con_frac_cero = choose_background(cand, d, 200, 0.0, 2,
                                          np.random.default_rng(11),
                                          band2_frac=0.5, band2_mask=mask,
                                          band2_axis_frac=0.0)
        np.testing.assert_array_equal(con_frac_cero, esperado)

        # la máscara del eje pasada con cupo 0 tampoco puede tocar el sorteo
        con_mascara_cupo_cero = choose_background(
            cand, d, 200, 0.0, 2, np.random.default_rng(11),
            band2_frac=0.5, band2_mask=mask,
            band2_axis_mask=(mask & (d2 == 0)), band2_axis_frac=0.0)
        np.testing.assert_array_equal(con_mascara_cupo_cero, esperado)

    def test_el_cupo_si_cambia_el_resultado(self):
        cand, d, d1, d2, mask = candidatos()
        a = choose_background(cand, d, 200, 0.0, 2,
                              np.random.default_rng(11),
                              band2_frac=0.5, band2_mask=mask)
        b = choose_background(cand, d, 200, 0.0, 2,
                              np.random.default_rng(11),
                              band2_frac=0.5, band2_mask=mask,
                              band2_axis_mask=(mask & (d2 == 0)),
                              band2_axis_frac=0.2)
        self.assertFalse(np.array_equal(a, b),
                         "el cupo debe reordenar la banda")


class TestPlanPoolIdxCupo(unittest.TestCase):
    def setUp(self):
        import os
        self.pre = importlib.import_module("preextract")
        self.L = np.zeros((30, 40), dtype=np.float32)
        self.L[15, 20] = 1.0

    def test_pool_conserva_tamano_y_composicion(self):
        idx, n_pos = self.pre.plan_pool_idx(
            self.L, 100, None, np.random.default_rng(0),
            hard_frac=0.0, hard_px=2, band2_frac=0.5,
            aniso=ANISO, band2_axis_frac=0.2)
        self.assertEqual(n_pos, 1)
        bg = idx[n_pos:]
        self.assertEqual(len(bg), 100, "el tamaño del pool no cambia")
        _, d1, d2 = axis_distance_to_positive(self.L)
        d2_bg = d2.reshape(-1)[bg]
        mask = aniso_band_mask(d1.reshape(-1)[bg], d2_bg, *ANISO)
        n_band2 = round(len(bg) * 0.5)
        self.assertTrue(mask[:n_band2].all(),
                        "las primeras n_band2 filas del pool salen de la banda")
        n_axis = round(n_band2 * 0.2)
        self.assertEqual(int((d2_bg[:n_band2] == 0).sum()), n_axis)
        self.assertTrue((d2_bg[:n_band2][n_axis:] > 0).all(),
                        "el resto de la banda sale de la diagonal")

    def test_cupo_con_f2lo_mayor_que_cero_falla(self):
        with self.assertRaises(ValueError):
            self.pre.plan_pool_idx(
                self.L, 100, None, np.random.default_rng(0),
                hard_frac=0.0, hard_px=2, band2_frac=0.5,
                aniso=(3, 10, 1, 3), band2_axis_frac=0.2)

    def test_cupo_sin_aniso_falla(self):
        with self.assertRaises(ValueError):
            self.pre.plan_pool_idx(
                self.L, 100, None, np.random.default_rng(0),
                hard_frac=0.0, hard_px=2, band2_frac=0.5,
                aniso=None, band2_axis_frac=0.2)

    def test_cupo_sin_frac2_falla(self):
        with self.assertRaises(ValueError):
            self.pre.plan_pool_idx(
                self.L, 100, None, np.random.default_rng(0),
                hard_frac=0.0, hard_px=2, band2_frac=0.0,
                aniso=ANISO, band2_axis_frac=0.2)


def pool_sintetico(n_pos=2, n=2000, seed=0):
    """Pool sintético: 2 positivos y 2000 filas de fondo con d1/d2 conocidas.

    200 filas en la banda (40 del eje con d2 == 0, 160 de la diagonal); fuera
    de la banda NINGUNA fila tiene d2 == 0, así que contar d2 == 0 en el lote
    cuenta exactamente el cupo cumplido.
    """
    rng = np.random.default_rng(seed)
    X = np.arange(n, dtype=np.float32).reshape(-1, 1)
    y = np.zeros((n, 1), dtype=np.float32)
    y[:n_pos] = 1.0
    d = np.full(n - n_pos, 30, dtype=np.uint8)
    d1 = np.full(n - n_pos, 30, dtype=np.uint8)
    d2 = np.full(n - n_pos, 30, dtype=np.uint8)
    eje = np.arange(200)
    d1[eje] = rng.integers(3, 11, size=len(eje))
    d2[eje] = 0
    d[eje] = d1[eje]
    diag = np.arange(200, 400)
    d1[diag] = rng.integers(3, 11, size=len(diag))
    d2[diag] = rng.integers(1, 4, size=len(diag))
    d[diag] = d1[diag]
    return X, y, n_pos, d, d1, d2


class TestSubsamplePoolCupo(unittest.TestCase):
    def test_lote_respeta_el_cupo_del_eje(self):
        X, y, n_pos, d, d1, d2 = pool_sintetico()
        x, _ = subsample_pool(
            X, y, n_pos, 1 << 20, 4.0, np.random.default_rng(7),
            d=d, hard_frac=0.0, band2_frac=0.5,
            d1=d1, d2=d2, aniso=ANISO, band2_axis_frac=0.2)
        got = x[:, 0].astype(np.int64) - n_pos
        self.assertEqual(len(got), n_pos + n_pos * 4)   # positivos + fondo
        n_band2 = round(n_pos * 4 * 0.5)    # 4 filas de banda
        n_axis = round(n_band2 * 0.2)       # round(0.8) = 1
        self.assertEqual(int((d2[got] == 0).sum()), n_axis)
        self.assertEqual(int((aniso_band_mask(d1[got], d2[got],
                                              *ANISO)).sum()), n_band2)

    def test_cupo_sin_aniso_falla(self):
        X, y, n_pos, d, _, _ = pool_sintetico()
        with self.assertRaises(ValueError):
            subsample_pool(
                X, y, n_pos, 1 << 20, 4.0, np.random.default_rng(1),
                d=d, hard_frac=0.0, band2_frac=0.5, band2_axis_frac=0.2)

    def test_cupo_con_f2lo_mayor_que_cero_falla(self):
        X, y, n_pos, d, d1, d2 = pool_sintetico()
        with self.assertRaises(ValueError):
            subsample_pool(
                X, y, n_pos, 1 << 20, 4.0, np.random.default_rng(1),
                d=d, hard_frac=0.0, band2_frac=0.5,
                d1=d1, d2=d2, aniso=(3, 10, 1, 3), band2_axis_frac=0.2)


if __name__ == "__main__":
    unittest.main()
