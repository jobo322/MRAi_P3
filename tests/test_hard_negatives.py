"""Tests del minado de negativos duros (el fondo pegado a un pico).

El problema que fijan: el fondo se muestreaba uniforme sobre TODOS los ceros, así
que los ceros dentro del pico (la etiqueta es un marcador sub-píxel de 1/2/4 px,
mientras el pico abarca ~20-50 px) entraban al 0,48 % del fondo. Por época y
espectro: 578 positivos frente a ~11 negativos a 1 px y ~26 a <=2 px. Resultado
medido en el modelo entrenado: el 81 % de los ceros inmediatamente al lado de un
pico reciben P > 0,5 (frontera aprendida a ~2-3 px, no en la del label), y el
pipeline elegía 1.966 picos donde había 256.

Lo que se fija aquí:
  * la distancia al positivo más cercano se mide bien;
  * el pool reserva una fracción del fondo al anillo (y sin `hard_frac` mantiene
    el sorteo histórico intacto, que es lo que fija test_split_rows);
  * la muestra por época lleva esa fracción de negativos duros, conserva TODOS
    los positivos y el tamaño total;
  * sin distancias en el pool (pool antiguo) se cae al fondo uniforme, sin
    inventarse nada.
"""
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from hard_negatives import (choose_background, distance_to_positive,  # noqa: E402
                            pool_sections, split_negatives)
from preextract import build_pool, plan_pool_idx  # noqa: E402
from train_model import subsample_pool  # noqa: E402

HARD_PX = 2


def _spectrum_with_peaks(h=64, w=64):
    """Etiquetas con positivos aislados y bien separados."""
    L = np.zeros((h, w), dtype=np.float32)
    L[10, 10] = 1.0
    L[10, 40] = 1.0
    L[40, 25] = 1.0
    return L


def _spectrum_with_many_peaks(h=64, w=64, step=6):
    """20 positivos separados más que el radio del anillo (no se tocan)."""
    L = np.zeros((h, w), dtype=np.float32)
    n = 0
    for i in range(step, h - step, step):
        for j in range(step, w - step, step):
            L[i, j] = 1.0
            n += 1
    return L, n


class TestDistance(unittest.TestCase):
    def test_distance_is_zero_on_positives(self):
        L = _spectrum_with_peaks()
        d = distance_to_positive(L)
        self.assertEqual(d.dtype, np.uint8)
        self.assertTrue(np.all(d[L > 0.5] == 0))

    def test_distance_is_euclidean_and_capped(self):
        L = np.zeros((8, 8), dtype=np.float32)
        L[4, 4] = 1.0
        d = distance_to_positive(L)
        self.assertEqual(d[4, 5], 1)                      # 1 px
        self.assertEqual(d[5, 5], 1)                      # diagonal: 1,41 -> 1
        self.assertEqual(d[0, 0], 6)                      # 5,66 -> 6
        self.assertLessEqual(d.max(), 255)

    def test_cap_is_respected(self):
        L = np.zeros((64, 64), dtype=np.float32)
        L[0, 0] = 1.0
        d = distance_to_positive(L, cap=3)
        self.assertEqual(int(d.max()), 3)

    def test_split_negatives(self):
        L = np.zeros((8, 8), dtype=np.float32)
        L[4, 4] = 1.0
        d = distance_to_positive(L).reshape(-1)
        neg = np.where(L.reshape(-1) < 0.5)[0]
        hard, easy = split_negatives(neg, d[neg], HARD_PX)
        self.assertEqual(len(hard) + len(easy), len(neg))
        self.assertTrue(np.all(d[hard] <= HARD_PX))
        self.assertTrue(np.all(d[easy] > HARD_PX))
        # anillo: el 5x5 menos las 4 esquinas (2,83 -> 3) y menos el propio pico
        self.assertEqual(len(hard), 20)


class TestChooseBackground(unittest.TestCase):
    def setUp(self):
        # 30 duros (d<=2) y 70 fáciles
        self.d = np.concatenate([np.full(30, 1.0), np.full(70, 30.0)])
        self.cand = np.arange(100) + 1000

    def test_reserves_the_requested_fraction(self):
        rng = np.random.default_rng(0)
        pick = choose_background(self.cand, self.d, 20, 0.5, HARD_PX, rng)
        self.assertEqual(len(pick), 20)
        self.assertEqual(int((self.d[pick] <= HARD_PX).sum()), 10)
        self.assertEqual(len(set(pick.tolist())), 20)     # sin repeticiones

    def test_zero_fraction_is_uniform_over_everything(self):
        rng = np.random.default_rng(0)
        pick = choose_background(self.cand, self.d, 60, 0.0, HARD_PX, rng)
        self.assertEqual(len(pick), 60)
        n_hard = int((self.d[pick] <= HARD_PX).sum())
        # uniforme ~ 30 % de 60 = 18; con hard_frac=0.5 serian 30
        self.assertGreater(n_hard, 5)
        self.assertLess(n_hard, 25)

    def test_size_is_capped_by_availability(self):
        rng = np.random.default_rng(0)
        pick = choose_background(self.cand, self.d, 500, 0.5, HARD_PX, rng)
        self.assertEqual(len(pick), 100)

    def test_hard_fraction_capped_by_available_hard(self):
        """Con pocos duros disponibles se toman todos y se completa con fáciles."""
        rng = np.random.default_rng(0)
        pick = choose_background(self.cand, self.d, 100, 0.9, HARD_PX, rng)
        self.assertEqual(len(pick), 100)
        self.assertEqual(int((self.d[pick] <= HARD_PX).sum()), 30)  # los 30 duros

    def test_pool_sections(self):
        self.assertEqual(pool_sections(100, 0.25), (25, 75))
        self.assertEqual(pool_sections(3, 0.5), (2, 1))
        self.assertEqual(pool_sections(0, 0.5), (0, 0))


class TestPoolReservesTheRing(unittest.TestCase):
    def _flat(self, L, pool_k=10.0, max_pts=1 << 20, seed=0, **kw):
        idx, n_pos = plan_pool_idx(L, pool_k, max_pts,
                                   np.random.default_rng(seed), **kw)
        return idx, n_pos

    def test_legacy_path_is_untouched(self):
        """Sin hard_frac el pool debe ser EXACTAMENTE el de siempre."""
        L = _spectrum_with_peaks()
        idx, n_pos = self._flat(L)
        rng = np.random.default_rng(0)
        neg = np.where(L.reshape(-1) < 0.5)[0]
        n_bg = int(n_pos * 10.0)
        legacy = np.concatenate([np.where(L.reshape(-1) > 0.5)[0],
                                 rng.choice(neg, size=n_bg, replace=False)])
        np.testing.assert_array_equal(idx, legacy)

    def test_hard_frac_puts_the_ring_in_the_pool(self):
        L = _spectrum_with_peaks()
        d = distance_to_positive(L).reshape(-1)
        idx, n_pos = self._flat(L, hard_frac=0.5, hard_px=HARD_PX)
        bg = idx[n_pos:]
        self.assertEqual(len(bg), int(n_pos * 10.0))
        hard_share = float((d[bg] <= HARD_PX).mean())
        self.assertAlmostEqual(hard_share, 0.5, places=2)

    def test_pool_still_has_every_positive(self):
        L = _spectrum_with_peaks()
        idx, n_pos = self._flat(L, hard_frac=0.5)
        self.assertEqual(int(n_pos), int((L > 0.5).sum()))
        self.assertTrue(np.all(L.reshape(-1)[idx[:n_pos]] > 0.5))
        self.assertTrue(np.all(L.reshape(-1)[idx[n_pos:]] < 0.5))

    def test_deterministic_for_a_seed(self):
        L = _spectrum_with_peaks()
        a, _ = self._flat(L, hard_frac=0.5, seed=7)
        b, _ = self._flat(L, hard_frac=0.5, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_build_pool_accepts_hard_frac(self):
        """build_pool es un envoltorio de plan_pool_idx: mismo pool."""
        L = _spectrum_with_peaks()
        X = np.random.default_rng(0).standard_normal((L.size, 128, 1)).astype(np.float32)
        px, py, n_pos = build_pool(X, L, 10.0, 1 << 20,
                                   np.random.default_rng(0),
                                   hard_frac=0.5, hard_px=HARD_PX)
        idx, n_pos_ref = plan_pool_idx(L, 10.0, 1 << 20, np.random.default_rng(0),
                                       hard_frac=0.5, hard_px=HARD_PX)
        self.assertEqual(n_pos, n_pos_ref)
        np.testing.assert_array_equal(px, X[idx])
        np.testing.assert_array_equal(py[:, 0], L.reshape(-1)[idx])


def _pool(L, pool_k=10.0, hard_frac=0.5, seed=0):
    """Pool + las distancias que guarda el .npz + un mapa fila -> distancia."""
    X = np.random.default_rng(0).standard_normal((L.size, 128, 1)).astype(np.float32)
    d_flat = distance_to_positive(L).reshape(-1)
    idx, n_pos = plan_pool_idx(L, pool_k, None, np.random.default_rng(seed),
                               hard_frac=hard_frac, hard_px=HARD_PX)
    px, py = X[idx], L.reshape(-1, 1)[idx]
    d_pool = d_flat[idx]
    by_row = {bytes(row.tobytes()): float(d) for row, d in zip(px, d_pool)}
    return px, py, n_pos, d_pool[n_pos:], by_row


class TestEpochSample(unittest.TestCase):
    def test_epoch_takes_the_requested_hard_share(self):
        L, n_peaks = _spectrum_with_many_peaks()
        px, py, n_pos, d_bg, by_row = _pool(L)
        self.assertEqual(n_pos, n_peaks)
        bg_ratio, hard_frac = 4.0, 0.25
        previous = None
        for seed in range(4):
            sx, sy = subsample_pool(px, py, n_pos, 1 << 20, bg_ratio,
                                    np.random.default_rng(seed), d=d_bg,
                                    hard_frac=hard_frac, hard_px=HARD_PX)
            self.assertEqual(len(sx), n_pos + int(n_pos * bg_ratio))
            d_sampled = np.array([by_row[bytes(r.tobytes())] for r in sx])
            self.assertEqual(int((d_sampled == 0).sum()), n_pos)   # todos los positivos
            bg = d_sampled > 0
            share = float((d_sampled[bg] <= HARD_PX).mean())
            self.assertAlmostEqual(share, hard_frac, places=2)
            # el fondo cambia entre épocas (la aumentación se conserva)
            if previous is not None:
                self.assertFalse(np.array_equal(np.sort(d_sampled), np.sort(previous)))
            previous = d_sampled

    def test_sample_size_is_unchanged(self):
        L, _ = _spectrum_with_many_peaks()
        px, py, n_pos, d_bg, _by_row = _pool(L)
        for max_pts, bg in [(1 << 20, 2.0), (1 << 20, 4.0), (n_pos + 5, 4.0)]:
            sx, _ = subsample_pool(px, py, n_pos, max_pts, bg,
                                   np.random.default_rng(1), d=d_bg,
                                   hard_frac=0.25, hard_px=HARD_PX)
            expected = n_pos + min(len(px) - n_pos, int(n_pos * bg), max_pts - n_pos)
            self.assertEqual(len(sx), expected, (max_pts, bg))

    def test_without_distances_falls_back_to_uniform(self):
        """Pool antiguo (sin `d`): mismo sorteo que antes, sin fallar."""
        L, _ = _spectrum_with_many_peaks()
        px, py, n_pos, _d, _by = _pool(L, hard_frac=0.0)
        a = subsample_pool(px, py, n_pos, 1 << 20, 4.0, np.random.default_rng(3))
        b = subsample_pool(px, py, n_pos, 1 << 20, 4.0, np.random.default_rng(3),
                           d=None, hard_frac=0.25, hard_px=HARD_PX)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_hard_frac_zero_keeps_the_legacy_rows(self):
        L, _ = _spectrum_with_many_peaks()
        px, py, n_pos, d_bg, _by = _pool(L, hard_frac=0.0)
        a = subsample_pool(px, py, n_pos, 1 << 20, 2.0, np.random.default_rng(5))
        b = subsample_pool(px, py, n_pos, 1 << 20, 2.0, np.random.default_rng(5),
                           d=d_bg, hard_frac=0.0, hard_px=HARD_PX)
        np.testing.assert_array_equal(a[0], b[0])

    def test_thin_pool_without_enough_hard(self):
        """Si el pool no tiene suficientes duros, se completa y no se recorta."""
        L, n_peaks = _spectrum_with_many_peaks()
        px, py, n_pos, d_bg, _by = _pool(L, hard_frac=0.0)   # pool sin duros
        sx, _ = subsample_pool(px, py, n_pos, 1 << 20, 4.0,
                               np.random.default_rng(1), d=d_bg,
                               hard_frac=0.5, hard_px=HARD_PX)
        self.assertEqual(len(sx), n_pos + int(n_pos * 4.0))


if __name__ == "__main__":
    unittest.main()
