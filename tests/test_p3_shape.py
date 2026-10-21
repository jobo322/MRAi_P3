"""Tests de la métrica de cola (decay) de la P3 por eje.

Motivación
----------
El reporte de `run_train_2d.py` mide aciertos (F1/precisión) pero no cuánto
peso deja la P3 alrededor del pico. Medido en 6 espectros de validación, la P3
de producción conserva 5,9 % a ±3 px en F1 (la S: 0,1 %) y 0,1 % en F2: la
cola F1 es la que produce la cresta que `msep` tiene que colapsar. Esta métrica
la hace visible en cada corrida, sin depender de mirar el PNG.

Definición: para cada pico GT (máximo local en una ventana de 3x3), con centro
normalizado a 1, se promedian las dos muestras simétricas a ±d px en cada eje:

    F1(d) = 0.5 * (|P[r-d, c]| + |P[r+d, c]|) / |P[r, c]|
    F2(d) = 0.5 * (|P[r, c-d]| + |P[r, c+d]|) / |P[r, c]|

con d = 3 y 6. Un pico en el borde (sin las dos muestras) no entra en esa
distancia; una relación F1/F2 > 1 indica cola preferente en F1.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(BASE_DIR, "scripts", "plotting")


def _load():
    if "run_train_2d_shape_test" in sys.modules:
        return sys.modules["run_train_2d_shape_test"]
    sys.path.insert(0, PLOT_DIR)
    spec = importlib.util.spec_from_file_location(
        "run_train_2d_shape_test", os.path.join(PLOT_DIR, "run_train_2d.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_train_2d_shape_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def blob(shape=(41, 61), row=20, col=30, sig_r=1.0, sig_c=1.0, amp=1.0):
    rr = np.arange(shape[0])[:, None]
    cc = np.arange(shape[1])[None, :]
    return amp * np.exp(-((rr - row) ** 2) / (2 * sig_r ** 2)
                        - ((cc - col) ** 2) / (2 * sig_c ** 2))


class TestShapeValues(unittest.TestCase):
    def test_blob_anisotropo_tiene_cola_f1_menor(self):
        mod = _load()
        P = blob(sig_r=1.0, sig_c=4.0)      # estrecho en F1, ancho en F2
        v = mod.shape_values(P, [20], [30])
        self.assertEqual(len(v["f1_at_3"]), 1)
        self.assertEqual(len(v["f2_at_3"]), 1)
        self.assertLess(v["f1_at_3"][0], 0.05)
        self.assertGreater(v["f2_at_3"][0], 0.5)

    def test_blob_isotropo_da_ejes_parecidos(self):
        mod = _load()
        P = blob(sig_r=3.0, sig_c=3.0)
        v = mod.shape_values(P, [20], [30])
        for d in (3, 6):
            self.assertAlmostEqual(float(v[f"f1_at_{d}"][0]),
                                   float(v[f"f2_at_{d}"][0]), places=6)

    def test_pico_en_el_borde_no_entra(self):
        mod = _load()
        P = blob(shape=(10, 10), row=0, col=0, sig_r=1.0, sig_c=1.0)
        v = mod.shape_values(P, [0], [0], offsets=(3, 6))
        self.assertEqual(len(v["f1_at_3"]), 0)
        self.assertEqual(len(v["f2_at_3"]), 0)

    def test_centro_cero_se_ignora(self):
        mod = _load()
        P = np.zeros((21, 21), dtype=np.float32)
        v = mod.shape_values(P, [10], [10])
        self.assertEqual(len(v["f1_at_3"]), 0)

    def test_usa_el_maximo_local_de_la_ventana(self):
        mod = _load()
        P = blob(sig_r=2.0, sig_c=2.0)
        # el GT cae 1 px al lado del máximo real: la ventana 3x3 lo recupera
        v = mod.shape_values(P, [21], [31])
        self.assertEqual(len(v["f1_at_3"]), 1)


class TestShapeReport(unittest.TestCase):
    def test_medianas_y_ratio(self):
        mod = _load()
        v = {
            "f1_at_3": np.array([0.2, 0.4]),
            "f2_at_3": np.array([0.1, 0.2]),
            "f1_at_6": np.array([0.05, 0.15]),
            "f2_at_6": np.array([0.01, 0.03]),
        }
        rep = mod.shape_report(v)
        self.assertEqual(rep["n"], 2)
        self.assertAlmostEqual(rep["at_3px"]["f1"], 0.3)
        self.assertAlmostEqual(rep["at_3px"]["f2"], 0.15)
        self.assertAlmostEqual(rep["at_3px"]["f1_over_f2"], 2.0)
        self.assertAlmostEqual(rep["at_6px"]["f1"], 0.10)
        self.assertAlmostEqual(rep["at_6px"]["f2"], 0.02)
        self.assertAlmostEqual(rep["at_6px"]["f1_over_f2"], 5.0)

    def test_vacio_no_revienta(self):
        mod = _load()
        rep = mod.shape_report({"f1_at_3": np.array([]),
                                "f2_at_3": np.array([]),
                                "f1_at_6": np.array([]),
                                "f2_at_6": np.array([])})
        self.assertEqual(rep["n"], 0)
        self.assertTrue(np.isnan(rep["at_3px"]["f1"]))

    def test_ratio_ignora_denominador_cero(self):
        mod = _load()
        v = {
            "f1_at_3": np.array([0.2, 0.4]),
            "f2_at_3": np.array([0.0, 0.2]),
            "f1_at_6": np.array([0.1, 0.1]),
            "f2_at_6": np.array([0.0, 0.1]),
        }
        rep = mod.shape_report(v)
        self.assertAlmostEqual(rep["at_3px"]["f1_over_f2"], 2.0)


if __name__ == "__main__":
    unittest.main()
