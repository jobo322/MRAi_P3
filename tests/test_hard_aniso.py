"""Tests de la banda de fondo ANISÓTROPA (negativos duros por eje).

Qué añade y por qué
-------------------
La banda isotrópica `3 <= d <= 6` no distingue ejes: un negativo a 5 px en F1
(la cola que el modelo aprende) y otro a 5 px en F2 entran por igual. Medido en
6 espectros de validación, la P3 de producción tiene FWHM F1/F2 = 1,97/1,98
(ratio 1,02, isotrópica) mientras la S tiene 6,75/4,10 (ratio 1,73): el modelo
no copia la anisotropía de la señal y deja una cola en F1 (5,9 % a ±3 px) que
la S no tiene (0,1 %). La banda nueva selecciona por eje:

    F1 (filas)    3 .. 10 px
    F2 (columnas) 1 .. 3 px

Reglas que fijan estos tests:

* `axis_distance_to_positive` descompone la distancia euclídea por eje usando
  el MISMO positivo más cercano, y su `d` coincide con `distance_to_positive`;
* la máscara es rectangular y no incluye el anillo (`d <= hard_px`), que tiene
  prioridad: nunca se duplican filas entre bandas;
* con `aniso=None` el sorteo sigue siendo el histórico byte a byte;
* el tamaño del pool y del lote no cambia; si la banda no alcanza, se completa.
"""
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from hard_negatives import (aniso_band_mask, axis_distance_to_positive,  # noqa: E402
                            choose_background, distance_to_positive)
from train_model import subsample_pool  # noqa: E402

N_BG = 100


class TestAxisDistance(unittest.TestCase):
    def test_descompone_la_distancia_por_eje(self):
        L = np.zeros((7, 9), dtype=np.float32)
        L[3, 4] = 1
        d, d1, d2 = axis_distance_to_positive(L)
        self.assertEqual(int(d[3, 4]), 0)
        self.assertEqual((int(d1[3, 4]), int(d2[3, 4])), (0, 0))
        # axial en F1 (misma columna) y axial en F2 (misma fila)
        self.assertEqual((int(d1[0, 4]), int(d2[0, 4])), (3, 0))
        self.assertEqual((int(d1[3, 0]), int(d2[3, 0])), (0, 4))
        # diagonal: cada eje cuenta lo suyo
        self.assertEqual((int(d1[5, 6]), int(d2[5, 6])), (2, 2))

    def test_d_coincide_con_distance_to_positive(self):
        rng = np.random.default_rng(0)
        L = (rng.random((40, 60)) > 0.99).astype(np.float32)
        L[10:13, 20:24] = 1.0
        d, _, _ = axis_distance_to_positive(L)
        np.testing.assert_array_equal(d, distance_to_positive(L))

    def test_cap_satura_las_distancias(self):
        L = np.zeros((11, 11), dtype=np.float32)
        L[5, 5] = 1
        d, d1, d2 = axis_distance_to_positive(L, cap=3)
        self.assertEqual(int(d[0, 5]), 3)
        self.assertEqual(int(d1[0, 5]), 3)
        self.assertEqual(int(d2[5, 0]), 3)


class TestAnisoMask(unittest.TestCase):
    def test_banda_rectangular(self):
        d1 = np.array([2, 3, 5, 10, 11, 5, 5], dtype=np.uint8)
        d2 = np.array([2, 1, 3, 3, 2, 0, 4], dtype=np.uint8)
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        np.testing.assert_array_equal(
            mask, [False, True, True, True, False, False, False])

    def test_rango_invalido_falla(self):
        d = np.zeros(4, dtype=np.uint8)
        with self.assertRaises(ValueError):
            aniso_band_mask(d, d, 10, 3, 1, 3)
        with self.assertRaises(ValueError):
            aniso_band_mask(d, d, 3, 10, 3, 1)
        with self.assertRaises(ValueError):
            aniso_band_mask(d, d, -1, 10, 1, 3)


def candidates_aniso(n=4000, seed=0):
    """Fondo sintético con anillo, banda anisótropa y resto."""
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 40, size=n).astype(np.uint8)
    d1 = rng.integers(0, 40, size=n).astype(np.uint8)
    d2 = rng.integers(0, 40, size=n).astype(np.uint8)
    band = np.zeros(n, dtype=bool)
    band[: int(0.10 * n)] = True          # 400 filas en la banda
    d1[band] = rng.integers(3, 11, size=band.sum())
    d2[band] = rng.integers(1, 4, size=band.sum())
    d[band] = rng.integers(4, 12, size=band.sum())  # fuera del anillo
    return np.arange(n), d, d1, d2, band


class TestChooseBackgroundAniso(unittest.TestCase):
    def test_respeta_la_mascara_y_el_tamano(self):
        cand, d, d1, d2, band = candidates_aniso()
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        self.assertGreater(int((mask & band).sum()), N_BG)
        take = choose_background(cand, d, N_BG, 0.0, 2,
                                 np.random.default_rng(7),
                                 band2_frac=0.5, band2_mask=mask)
        self.assertEqual(len(take), N_BG)
        self.assertEqual(len(set(take.tolist())), N_BG, "sin repeticiones")
        n_band2 = round(N_BG * 0.5)
        # el orden de salida es [anillo | banda2 | resto]
        self.assertTrue(mask[take[:n_band2]].all(),
                        "las primeras n_band2 filas salen de la banda")
        self.assertTrue((d[take[:n_band2]] > 2).all(),
                        "la banda no se solapa con el anillo")

    def test_la_banda_no_pisa_el_anillo(self):
        cand, d, d1, d2, _ = candidates_aniso()
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        d[:50] = 1                         # 50 filas de la banda caen en el anillo
        take = choose_background(cand, d, N_BG, 0.2, 2,
                                 np.random.default_rng(3),
                                 band2_frac=0.5, band2_mask=mask)
        self.assertEqual(len(take), N_BG)
        self.assertEqual(len(set(take.tolist())), N_BG, "sin repeticiones")

    def test_banda_vacia_se_completa(self):
        cand, d, d1, d2, _ = candidates_aniso()
        mask = np.zeros(len(cand), dtype=bool)
        take = choose_background(cand, d, N_BG, 0.4, 2,
                                 np.random.default_rng(5),
                                 band2_frac=0.5, band2_mask=mask)
        self.assertEqual(len(take), N_BG)
        self.assertEqual(len(set(take.tolist())), N_BG)

    def test_mascara_y_band2_px_son_excluyentes(self):
        cand, d, d1, d2, _ = candidates_aniso()
        with self.assertRaises(ValueError):
            choose_background(cand, d, N_BG, 0.0, 2,
                              np.random.default_rng(1),
                              band2_px=6, band2_frac=0.2,
                              band2_mask=np.zeros(len(cand), dtype=bool))

    def test_mascara_de_tamano_incorrecto_falla(self):
        cand, d, _, _, _ = candidates_aniso()
        with self.assertRaises(ValueError):
            choose_background(cand, d, N_BG, 0.0, 2,
                              np.random.default_rng(1),
                              band2_frac=0.2,
                              band2_mask=np.zeros(len(cand) - 1, dtype=bool))


class TestPlanPoolIdxAniso(unittest.TestCase):
    def setUp(self):
        import importlib
        self.pre = importlib.import_module("preextract")
        self.L = np.zeros((30, 40), dtype=np.float32)
        self.L[15, 20] = 1.0

    def test_pool_usa_la_banda_por_eje(self):
        idx, n_pos = self.pre.plan_pool_idx(
            self.L, 100, None, np.random.default_rng(0),
            hard_frac=0.0, hard_px=2, band2_frac=0.5,
            aniso=(3, 10, 1, 3))
        self.assertEqual(n_pos, 1)
        bg = idx[n_pos:]
        d, d1, d2 = axis_distance_to_positive(self.L)
        d1, d2 = d1.reshape(-1)[bg], d2.reshape(-1)[bg]
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        n_band2 = round(len(bg) * 0.5)
        self.assertTrue(mask[:n_band2].all())
        self.assertTrue((d.reshape(-1)[bg][:n_band2] > 2).all())

    def test_sin_banda_sigue_siendo_el_historico(self):
        a, na = self.pre.plan_pool_idx(
            self.L, 10, None, np.random.default_rng(42),
            hard_frac=0.4, hard_px=2)
        b, nb = self.pre.plan_pool_idx(
            self.L, 10, None, np.random.default_rng(42),
            hard_frac=0.4, hard_px=2, aniso=None)
        self.assertEqual(na, nb)
        self.assertTrue(np.array_equal(a, b))

    def test_aniso_sin_frac2_falla(self):
        with self.assertRaises(ValueError):
            self.pre.plan_pool_idx(
                self.L, 10, None, np.random.default_rng(0),
                hard_frac=0.4, hard_px=2, band2_frac=0.0,
                aniso=(3, 10, 1, 3))


def pool_sintetico(n_pos=2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.arange(n, dtype=np.float32).reshape(-1, 1)
    y = np.zeros((n, 1), dtype=np.float32)
    y[:n_pos] = 1.0
    d = rng.integers(0, 40, size=n - n_pos).astype(np.uint8)
    d1 = rng.integers(0, 40, size=n - n_pos).astype(np.uint8)
    d2 = rng.integers(0, 40, size=n - n_pos).astype(np.uint8)
    band = np.zeros(n - n_pos, dtype=bool)
    band[:200] = True
    d1[band] = rng.integers(3, 11, size=band.sum())
    d2[band] = rng.integers(1, 4, size=band.sum())
    d[band] = 10
    return X, y, n_pos, d, d1, d2


class TestSubsamplePoolAniso(unittest.TestCase):
    def test_lote_usa_la_banda(self):
        X, y, n_pos, d, d1, d2 = pool_sintetico()
        x, _ = subsample_pool(
            X, y, n_pos, 1 << 20, 4.0, np.random.default_rng(7),
            d=d, hard_frac=0.0, band2_frac=0.5,
            d1=d1, d2=d2, aniso=(3, 10, 1, 3))
        got = x[:, 0].astype(np.int64)
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        n_band2 = round(8 * 0.5)          # bg = 2 positivos * 4
        self.assertEqual(int(mask[got - n_pos].sum()), n_band2)

    def test_aniso_sin_d1_d2_falla(self):
        X, y, n_pos, d, _, _ = pool_sintetico()
        with self.assertRaises(ValueError):
            subsample_pool(
                X, y, n_pos, 1 << 20, 4.0, np.random.default_rng(1),
                d=d, hard_frac=0.0, band2_frac=0.5,
                aniso=(3, 10, 1, 3))


class TestFallbackSinDuplicados(unittest.TestCase):
    """El relleno no puede repetir filas ni perder el anillo.

    Regresión: `choose_background` completaba con
    `setdiff1d(arange(len(candidates)), taken)`, que incluye las filas de
    `rest` ya agregadas; las duplicaba y dejaba fuera parte del anillo.
    También afecta a `subsample_pool` con `aniso=None`.
    """

    @staticmethod
    def _pool():
        # 60 fondos: 33 anillo, 27 fáciles. n_bg=40, hard_frac=.25:
        # sobran 30 fáciles pero solo hay 27 -> salta el relleno.
        d = np.concatenate([np.ones(33, np.uint8),
                            np.full(27, 20, np.uint8)])
        return np.arange(60), d

    def test_historico_relleno_sin_duplicados(self):
        cand, d = self._pool()
        take = choose_background(cand, d, 40, 0.25, 2,
                                 np.random.default_rng(0))
        self.assertEqual(len(take), 40)
        self.assertEqual(len(set(take.tolist())), 40)
        self.assertEqual(int((d[take] <= 2).sum()), 13)
        self.assertEqual(int((d[take] > 2).sum()), 27)

    def test_banda_vacia_relleno_sin_duplicados(self):
        cand, d = self._pool()
        mask = np.zeros(len(cand), dtype=bool)
        take = choose_background(cand, d, 40, 0.25, 2,
                                 np.random.default_rng(0),
                                 band2_frac=0.25, band2_mask=mask)
        self.assertEqual(len(take), 40)
        self.assertEqual(len(set(take.tolist())), 40)
        self.assertEqual(int((d[take] <= 2).sum()), 13)
        self.assertEqual(int((d[take] > 2).sum()), 27)

    def test_banda_llena_pero_faltan_faciles(self):
        n = 60
        d = np.empty(n, np.uint8)
        d1 = np.empty(n, np.uint8)
        d2 = np.empty(n, np.uint8)
        d[:33] = 1; d1[:33] = 0; d2[:33] = 0          # anillo
        d[33:48] = 5; d1[33:48] = 5; d2[33:48] = 2    # banda anisótropa
        d[48:] = 20; d1[48:] = 30; d2[48:] = 30       # fáciles
        mask = aniso_band_mask(d1, d2, 3, 10, 1, 3)
        take = choose_background(np.arange(n), d, 40, 0.25, 2,
                                 np.random.default_rng(0),
                                 band2_frac=0.25, band2_mask=mask)
        self.assertEqual(len(take), 40)
        self.assertEqual(len(set(take.tolist())), 40)
        self.assertEqual(int(((d[take] > 2) & ~mask[take]).sum()), 12)

    def test_subsample_pool_aniso_none_sin_duplicados(self):
        n_pos = 10
        X = np.arange(70, dtype=np.float32).reshape(-1, 1)
        y = np.zeros((70, 1), dtype=np.float32)
        y[:n_pos] = 1.0
        d = np.concatenate([np.ones(33, np.uint8),
                            np.full(27, 20, np.uint8)])
        x, _ = subsample_pool(X, y, n_pos, 1 << 20, 4.0,
                              np.random.default_rng(0), d=d,
                              hard_frac=0.25)
        idx = x[:, 0].astype(np.int64)
        self.assertEqual(len(idx), 50)
        self.assertEqual(len(set(idx.tolist())), 50)


if __name__ == "__main__":
    unittest.main()
