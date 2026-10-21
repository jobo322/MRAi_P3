"""Tests del sobremuestreo de pares apretados (`--pair-frac`).

Reglas que fijan:

* con `pair_frac = 0` el sorteo es BYTE A BYTE el histórico, con o sin array
  `pair` en el pool y con o sin `d` (misma semilla -> mismos índices); como en
  tests/test_hard_bands.py, la referencia histórica va copiada tal cual aquí,
  así que el control de regresión no depende del código nuevo;
* con `pair_frac > 0` las filas marcadas (`pair > 0`) aparecen exactamente
  `int(round(pair_frac * n_pos))` veces más y el tamaño del lote crece lo mismo;
* el presupuesto de fondo (`n_bg`) se calcula sobre el `n_pos` ORIGINAL: los
  negativos elegidos no cambian al activar el sobremuestreo (solo cambian los
  positivos) y `max_pts` sigue acotando el fondo;
* sin array `pair` con `pair_frac > 0` se falla claro: en `subsample_pool`, en
  la guardia `_pools_missing_pair` (que lista cuántos y cuáles) y en la ruta
  sin preextracción (`BinSequence._points`).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from hard_negatives import choose_background  # noqa: E402
from train_model import (BinSequence, _pools_missing_pair,  # noqa: E402
                         subsample_pool)

N_POS = 10
POOL_K = 6
MAX_PTS = 1 << 20


def legacy_subsample_pool(X, y, n_pos, max_pts, bg_ratio, rng, d=None,
                          hard_frac=0.0, hard_px=2, band2_px=None,
                          band2_frac=0.0):
    """Implementación histórica (antes de --pair-frac), copiada tal cual."""
    n_bg = max(0, min(len(X) - n_pos, int(n_pos * bg_ratio), max_pts - n_pos))
    if d is not None and (hard_frac > 0 or band2_frac > 0) and n_bg > 0:
        cand = np.arange(n_pos, len(X))
        d_bg = np.asarray(d)[:len(cand)]
        bg = cand[choose_background(cand, d_bg, n_bg, hard_frac, hard_px, rng,
                                    band2_px=band2_px, band2_frac=band2_frac)]
    else:
        bg = rng.choice(len(X) - n_pos, size=n_bg, replace=False) + n_pos
    idx = np.concatenate([np.arange(n_pos), bg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _pool(seed=0):
    """Pool sintético: X con la identidad de cada fila en la 1ª componente."""
    rng = np.random.default_rng(seed)
    M = N_POS + N_POS * POOL_K
    X = rng.standard_normal((M, 128, 1)).astype(np.float32)
    X[:, 0, 0] = np.arange(M)
    y = np.zeros((M, 1), dtype=np.float32)
    y[:N_POS] = 1.0
    return X, y


def _pair(marked):
    p = np.zeros(N_POS + N_POS * POOL_K, dtype=np.uint8)
    p[marked] = 1
    return p


def _identidad(X):
    """Índice del pool del que salió cada fila del lote."""
    return X[:, 0, 0].astype(np.int64).tolist()


class TestPairFracCeroEsHistorico(unittest.TestCase):
    def _compara(self, d=None, hard_frac=0.0, pair=None, seeds=(0, 1, 2)):
        X, y = _pool()
        for seed in seeds:
            with self.subTest(seed=seed):
                ref_x, ref_y = legacy_subsample_pool(
                    X, y, N_POS, MAX_PTS, 4.0, np.random.default_rng(seed),
                    d=d, hard_frac=hard_frac)
                new_x, new_y = subsample_pool(
                    X, y, N_POS, MAX_PTS, 4.0, np.random.default_rng(seed),
                    d=d, hard_frac=hard_frac, pair=pair, pair_frac=0.0)
                self.assertTrue(np.array_equal(ref_x, new_x))
                self.assertTrue(np.array_equal(ref_y, new_y))

    def test_sin_d_y_sin_pair(self):
        self._compara()

    def test_con_d_y_anillo(self):
        d = np.random.default_rng(9).integers(
            0, 30, size=N_POS * POOL_K).astype(np.uint8)
        self._compara(d=d, hard_frac=0.4)

    def test_con_pair_presente_pero_frac_0(self):
        self._compara(pair=_pair([2, 5, 14, 33]))


class TestSobremuestreo(unittest.TestCase):
    MARKED = [2, 5, 14, 33]   # 2 positivos + 2 de fondo

    def test_filas_marcadas_aparecen_mas_veces(self):
        X, y = _pool()
        pair = _pair(self.MARKED)
        x0, _ = subsample_pool(X, y, N_POS, MAX_PTS, 4.0,
                               np.random.default_rng(1), pair=pair,
                               pair_frac=0.0)
        x1, _ = subsample_pool(X, y, N_POS, MAX_PTS, 4.0,
                               np.random.default_rng(1), pair=pair,
                               pair_frac=0.5)

        def ocurrencias(x):
            ids = _identidad(x)
            return sum(ids.count(m) for m in self.MARKED)

        self.assertEqual(ocurrencias(x1) - ocurrencias(x0),
                         int(round(0.5 * N_POS)))

    def test_tamano_del_lote_y_todos_los_positivos(self):
        X, y = _pool()
        pair = _pair(self.MARKED)
        x1, _ = subsample_pool(X, y, N_POS, MAX_PTS, 4.0,
                               np.random.default_rng(1), pair=pair,
                               pair_frac=0.5)
        extra = int(round(0.5 * N_POS))
        n_bg = min(N_POS * POOL_K, int(N_POS * 4.0))
        self.assertEqual(len(x1), N_POS + extra + n_bg)
        self.assertTrue(set(range(N_POS)) <= set(_identidad(x1)),
                        "todos los positivos originales siguen en el lote")

    def test_todos_los_marcados_positivos_dan_los_positivos_esperados(self):
        """Si todo marcado es positivo, y==1 sube exactamente `extra`."""
        X, y = _pool()
        pair = _pair([2, 5, 7])
        _, y1 = subsample_pool(X, y, N_POS, MAX_PTS, 4.0,
                               np.random.default_rng(3), pair=pair,
                               pair_frac=0.25)
        self.assertEqual(int(y1.sum()), N_POS + int(round(0.25 * N_POS)))

    def test_max_pts_acota_el_fondo_no_los_marcados(self):
        """n_bg se calcula sobre el n_pos ORIGINAL: el lote puede pasar max_pts."""
        X, y = _pool()
        pair = _pair(self.MARKED)
        max_pts = N_POS + 5
        x1, _ = subsample_pool(X, y, N_POS, max_pts, 4.0,
                               np.random.default_rng(1), pair=pair,
                               pair_frac=0.5)
        n_bg = min(N_POS * POOL_K, int(N_POS * 4.0), max_pts - N_POS)  # = 5
        self.assertEqual(len(x1), N_POS + int(round(0.5 * N_POS)) + n_bg)


class TestGuardias(unittest.TestCase):
    def test_subsample_pool_falla_sin_pair(self):
        X, y = _pool()
        with self.assertRaises(ValueError) as ctx:
            subsample_pool(X, y, N_POS, MAX_PTS, 4.0, np.random.default_rng(1),
                           pair=None, pair_frac=0.5)
        self.assertIn("pair", str(ctx.exception))

    def test_pools_missing_pair_lista_cuantos_y_cuales(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((20, 128, 1)).astype(np.float32)
            y = np.zeros((20, 1), dtype=np.float32)
            y[:5] = 1.0
            np.savez(os.path.join(tmp, "0000.npz"), X=X, y=y,
                     n_pos=np.int32(5), pair=np.zeros(20, dtype=np.uint8))
            np.savez(os.path.join(tmp, "0001.npz"), X=X, y=y,
                     n_pos=np.int32(5))
            np.savez(os.path.join(tmp, "0002.npz"), X=X, y=y,
                     n_pos=np.int32(5))
            self.assertEqual(
                _pools_missing_pair(tmp, ["0000", "0001", "0002", "0003"]),
                ["0001", "0002", "0003"])


class TestNegativosEstables(unittest.TestCase):
    """Activar pair_frac no cambia los negativos elegidos (misma semilla)."""

    def _negativos(self, **kw):
        X, y = _pool()
        pair = _pair([2, 5, 7])          # solo positivos marcados
        a = subsample_pool(X, y, N_POS, MAX_PTS, 4.0, np.random.default_rng(7),
                           pair=pair, pair_frac=0.0, **kw)
        b = subsample_pool(X, y, N_POS, MAX_PTS, 4.0, np.random.default_rng(7),
                           pair=pair, pair_frac=0.5, **kw)
        na = sorted(i for i, yy in zip(_identidad(a[0]), a[1][:, 0])
                    if yy < 0.5)
        nb = sorted(i for i, yy in zip(_identidad(b[0]), b[1][:, 0])
                    if yy < 0.5)
        return na, nb

    def test_fondo_uniforme(self):
        na, nb = self._negativos()
        self.assertEqual(na, nb)

    def test_fondo_con_anillo(self):
        d = np.random.default_rng(9).integers(
            0, 30, size=N_POS * POOL_K).astype(np.uint8)
        na, nb = self._negativos(d=d, hard_frac=0.4)
        self.assertEqual(na, nb)


class TestBinSequencePropaga(unittest.TestCase):
    def _seq(self, pre_dir, pair_frac, seed=1):
        seq = object.__new__(BinSequence)   # sin __init__ (no importa TF)
        seq.rng = np.random.default_rng(seed)
        seq.bg_ratio = 4.0
        seq.max_pts = MAX_PTS
        seq.hard_frac = 0.0
        seq.hard_px = 2
        seq.hard_band2 = 0
        seq.hard_frac2 = 0.0
        seq.pair_frac = pair_frac
        seq.preextracted = True
        seq.pre_dir = pre_dir
        return seq

    def test_load_pre_sobremuestrea(self):
        X, y = _pool()
        marked = [2, 5, 14, 33]
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "0000.npz"), X=X, y=y,
                     n_pos=np.int32(N_POS),
                     d=np.zeros(N_POS * POOL_K, dtype=np.uint8),
                     pair=_pair(marked))
            a = self._seq(tmp, 0.0)._load_pre("0000")
            b = self._seq(tmp, 0.5)._load_pre("0000")
        self.assertEqual(len(b[0]), len(a[0]) + int(round(0.5 * N_POS)))
        ma, mb = _identidad(a[0]), _identidad(b[0])
        self.assertGreater(
            sum(mb.count(m) for m in marked),
            sum(ma.count(m) for m in marked))

    def test_load_pre_sin_pair_y_frac_0_funciona(self):
        X, y = _pool()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "0000.npz"), X=X, y=y,
                     n_pos=np.int32(N_POS),
                     d=np.zeros(N_POS * POOL_K, dtype=np.uint8))
            x, _ = self._seq(tmp, 0.0)._load_pre("0000")
        self.assertEqual(len(x), N_POS + N_POS * 4)

    def test_load_pre_sin_pair_y_frac_mayor_falla(self):
        X, y = _pool()
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "0000.npz"), X=X, y=y,
                     n_pos=np.int32(N_POS))
            with self.assertRaises(ValueError):
                self._seq(tmp, 0.5)._load_pre("0000")

    def test_ruta_sin_preextraccion_falla_con_pair_frac(self):
        seq = object.__new__(BinSequence)
        seq.pair_frac = 0.5
        S = np.zeros((8, 8), dtype=np.float32)
        L = np.zeros((8, 8), dtype=np.float32)
        L[3, 3] = 1.0
        with self.assertRaises(ValueError) as ctx:
            seq._points(S, L)
        self.assertIn("preextra", str(ctx.exception))


class TestPointsPerSpectrumPreextraido(unittest.TestCase):
    """`_points_per_spectrum` lee `pair` y tiene que hacerlo DENTRO del
    `with np.load(...)`. Si se lee fuera, el NpzFile ya está cerrado y el
    entrenamiento muere al calcular `len(seq)` con
    `AttributeError: 'NoneType' object has no attribute 'open'` — pasó de
    verdad: el run con `data/pre_elong_2b_pair` no arrancó nunca.
    """

    def _seq(self, tmp, pair_frac, con_pair=True):
        n_pos, total = 4, 12
        arrays = {"X": np.zeros((total, 128, 1), np.float32),
                  "y": np.zeros((total, 1), np.float32),
                  "n_pos": np.int32(n_pos)}
        if con_pair:
            marcas = np.zeros(total, dtype=np.uint8)
            marcas[1] = 1
            marcas[3] = 1
            arrays["pair"] = marcas
        np.savez(os.path.join(tmp, "0001.npz"), **arrays)
        seq = BinSequence.__new__(BinSequence)     # sin __init__ (no hace falta TF)
        seq.ids = ["0001"]
        seq.preextracted = True
        seq.pre_dir = tmp
        seq.ds_dir = tmp
        seq.max_pts = 10 ** 9
        seq.bg_ratio = 4.0
        seq.pair_frac = pair_frac
        return seq

    def test_lee_pair_sin_cerrar_el_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = self._seq(tmp, 0.5)._points_per_spectrum()
        # n_pos + round(0.5*4) + min(total-n_pos=8, 4*4=16, max_pts-n_pos) = 4+2+8
        self.assertEqual(n, 14)

    def test_pair_frac_cero_no_lo_necesita(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = self._seq(tmp, 0.0)._points_per_spectrum()
        self.assertEqual(n, 12)

    def test_pair_frac_positivo_sin_array_pair_no_inventa_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = self._seq(tmp, 0.5, con_pair=False)._points_per_spectrum()
        self.assertEqual(n, 12)


if __name__ == "__main__":
    unittest.main()