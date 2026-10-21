"""Tests for the two-band background sampling (anillo + banda media).

Qué se añade y por qué
----------------------
Medido sobre 8 espectros del dataset: el fondo con `d <= 2` px entró al pool al
54 %, pero la banda `3 <= d <= 6` —donde viven los hombros que se convierten en
falsos positivos (d_gt mediana 2,2–4,0 px)— entró al **0,5 %**: el fondo uniforme
la diluye porque compite con las ~500 000 filas a `d > 6`. Las dos bandas
reparten el presupuesto explícitamente:

    [ positivos | anillo (d <= hard_px) | banda 2 (hard_px < d <= band2_px) | resto ]

Reglas que fijan estos tests:

* con `band2_frac = 0` el sorteo es **byte a byte** el histórico (misma secuencia
  de `rng`, mismo orden de salida) → los pools ya generados y
  `tests/test_split_rows.py` no cambian;
* el tamaño de la muestra por época nunca cambia (steps constantes);
* si una banda no tiene suficientes candidatos se completa con las otras, sin
  repetir posiciones.
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from hard_negatives import (band_sections, choose_background,  # noqa: E402
                            pool_sections)

N_BG = 600


def candidates_with_bands(n=4000, seed=0):
    """Negativos sintéticos: 15 % en d<=2, 20 % en 3..6, resto lejos."""
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 40, size=n).astype(np.uint8)
    d[: int(0.15 * n)] = rng.integers(1, 3, size=int(0.15 * n))
    d[int(0.15 * n): int(0.35 * n)] = rng.integers(3, 7,
                                                   size=int(0.20 * n))
    return np.arange(n), d


def legacy_choose_background(candidates, d_candidates, n_bg, hard_frac, hard_px,
                             rng):
    """Implementación histórica, copiada tal cual, para el control de regresión."""
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


class TestBandSections(unittest.TestCase):
    def test_suma_el_presupuesto(self):
        for frac2 in (0.0, 0.1, 0.25, 0.5):
            for frac in (0.0, 0.4, 0.5):
                nh, nb2, ne = band_sections(N_BG, frac, frac2)
                self.assertEqual(nh + nb2 + ne, N_BG, (frac, frac2))
                self.assertGreaterEqual(min(nh, nb2, ne), 0)

    def test_sin_banda2_es_pool_sections(self):
        for frac in (0.0, 0.25, 0.5, 1.0):
            nh, nb2, ne = band_sections(N_BG, frac, 0.0)
            self.assertEqual((nh, ne), pool_sections(N_BG, frac))
            self.assertEqual(nb2, 0)

    def test_bandas_que_no_caben_fallan_claro(self):
        with self.assertRaises(ValueError):
            band_sections(N_BG, 0.6, 0.6)


class TestChooseBackgroundBands(unittest.TestCase):
    def test_reparto_por_banda(self):
        cand, d = candidates_with_bands()
        rng = np.random.default_rng(7)
        take = choose_background(cand, d, N_BG, 0.4, 2, rng,
                                 band2_px=6, band2_frac=0.2)
        self.assertEqual(len(take), N_BG)
        self.assertEqual(len(set(take.tolist())), N_BG, "sin repeticiones")
        dd = d[take]
        n_hard = int((dd <= 2).sum())
        n_band2 = int(((dd > 2) & (dd <= 6)).sum())
        self.assertEqual(n_hard, round(N_BG * 0.4), "anillo como se pidió")
        self.assertEqual(n_band2, round(N_BG * 0.2), "banda 2 como se pidió")
        self.assertEqual(len(take) - n_hard - n_band2, N_BG - n_hard - n_band2)

    def test_banda2_apagada_es_el_historico_byte_a_byte(self):
        cand, d = candidates_with_bands()
        for frac in (0.0, 0.25, 0.5, 0.9):
            a = legacy_choose_background(cand, d, N_BG, frac, 2,
                                         np.random.default_rng(11))
            b = choose_background(cand, d, N_BG, frac, 2,
                                  np.random.default_rng(11),
                                  band2_px=6, band2_frac=0.0)
            self.assertTrue(np.array_equal(a, b), f"frac={frac}")

    def test_sin_candidatos_en_la_banda2_se_completa(self):
        cand, d = candidates_with_bands()
        d[cand] = 1                      # todos en el anillo: banda 2 vacía
        take = choose_background(cand, d, N_BG, 0.4, 2,
                                 np.random.default_rng(3),
                                 band2_px=6, band2_frac=0.3)
        self.assertEqual(len(take), N_BG, "el tamaño no cambia")
        self.assertEqual(len(set(take.tolist())), N_BG)

    def test_sin_candidatos_en_el_anillo_tambien_completa(self):
        cand, d = candidates_with_bands()
        d[cand] = 20                     # nada cerca de un pico
        take = choose_background(cand, d, N_BG, 0.4, 2,
                                 np.random.default_rng(5),
                                 band2_px=6, band2_frac=0.2)
        self.assertEqual(len(take), N_BG)
        self.assertEqual(len(set(take.tolist())), N_BG)


class TestPlanPoolIdxBands(unittest.TestCase):
    """`plan_pool_idx` con dos bandas: layout [pos | anillo | banda2 | resto]."""

    def setUp(self):
        import importlib
        self.pre = importlib.import_module("preextract")
        self.L = np.zeros((20, 40), dtype=np.float32)
        # 4 positivos en una fila -> hay anillo, banda 2 y resto
        self.L[10, 5:9] = 1.0

    def test_layout_y_conteo(self):
        idx, n_pos = self.pre.plan_pool_idx(
            self.L, 10, None, np.random.default_rng(0),
            hard_frac=0.4, hard_px=2, band2_px=6, band2_frac=0.2)
        self.assertEqual(n_pos, 4)
        self.assertEqual(idx[:n_pos].min(), np.ravel_multi_index((10, 5), self.L.shape))
        from hard_negatives import distance_to_positive
        d = distance_to_positive(self.L).reshape(-1)[idx[n_pos:]]
        n_bg = len(idx) - n_pos
        self.assertEqual(len(d), n_bg)
        self.assertGreater(int((d <= 2).sum()), 0, "hay anillo en el pool")
        self.assertGreater(int(((d > 2) & (d <= 6)).sum()), 0, "hay banda 2")

    def test_sin_banda2_es_identico_al_historico(self):
        a, na = self.pre.plan_pool_idx(self.L, 10, None,
                                       np.random.default_rng(42),
                                       hard_frac=0.4, hard_px=2)
        b, nb = self.pre.plan_pool_idx(self.L, 10, None,
                                       np.random.default_rng(42),
                                       hard_frac=0.4, hard_px=2,
                                       band2_px=6, band2_frac=0.0)
        self.assertEqual(na, nb)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
