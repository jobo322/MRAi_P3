"""Tests del target alternativo (cresta del pico) — `scripts/ridge_target.py`.

Lo que fijan: la etiqueta de cresta es la componente conexa por encima de
`frac * A` del propio pico, y `components` cuenta las regiones resultantes (que
es el techo del picker «un pico por componente», y donde se paga el solapamiento).
"""

import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from ridge_target import (RIDGE_FRAC, components, count_instances,  # noqa: E402
                          ridge_labels, ridge_owners, ridge_region)


def gaussian(shape=(64, 96), row=32, col=48, sigma_r=3.0, sigma_c=3.0, amp=1.0):
    r = np.arange(shape[0])[:, None]
    c = np.arange(shape[1])[None, :]
    return amp * np.exp(-(((r - row) ** 2) / (2 * sigma_r ** 2) +
                          ((c - col) ** 2) / (2 * sigma_c ** 2)))


class TestRidgeOfPeak(unittest.TestCase):
    def test_area_y_umbral(self):
        S = gaussian()
        lab, (r0, c0) = ridge_region(S, 32, 48)
        area = int(lab.sum())
        # semialtura de una gaussiana isotrópica: r = sigma*sqrt(2 ln 2) ~ 3.53 px
        self.assertGreater(area, 20)
        self.assertLess(area, 70)
        # todos los píxeles están por encima de frac*A
        win = S[r0:r0 + lab.shape[0], c0:c0 + lab.shape[1]]
        self.assertTrue(np.all(win[lab] >= RIDGE_FRAC * S[32, 48] - 1e-12))
        self.assertFalse(np.any(win[~lab][win[~lab] >= RIDGE_FRAC * S[32, 48]]),
                         "no deja fuera ningún píxel conexo por encima del umbral")

    def test_es_conexa(self):
        S = gaussian()
        lab, _ = ridge_region(S, 32, 48)
        self.assertEqual(components(lab), 1)

    def test_no_se_desborda_de_la_ventana(self):
        S = gaussian(row=1, col=1)
        lab, _ = ridge_region(S, 1, 1)
        self.assertGreaterEqual(int(lab.sum()), 1)

    def test_un_pico_debil_junto_a_uno_fuerte_no_hereda_la_cresta(self):
        """El umbral va por `amp` del manifest y la región es exclusiva.

        Con `amp = S[row, col]` (contaminado por el vecino) la región del débil
        se fuga hacia el fuerte; con `amp` explícito y reparto por cercanía, no.
        """
        S = gaussian(row=32, col=48, amp=1.0) + gaussian(row=32, col=40, amp=0.03)
        own = ridge_owners(S, [32, 32], [48, 40], amps=[1.0, 0.03])
        self.assertEqual(count_instances(own), 2, "cada pico conserva su trozo")
        self.assertEqual(int(own[32, 48]), 0)
        self.assertEqual(int(own[32, 40]), 1, "el débil no hereda la cresta del fuerte")


class TestRidgeLabels(unittest.TestCase):
    def test_dos_picos_separados_dan_dos_componentes(self):
        S = gaussian(row=20, col=20) + gaussian(row=20, col=70)
        L = ridge_labels(S, [20, 20], [20, 70])
        self.assertEqual(components(L > 0), 2)
        self.assertEqual(sorted(np.unique(L).tolist()), [0.0, 1.0], "etiqueta 0/1")

    def test_dos_picos_solapados_se_separan_por_cercania(self):
        """Exclusivo: se tocan pero cada uno conserva su componente.

        Sin exclusividad se fusionarían en una, y el picker «un pico por
        componente» perdería uno de los dos sin poder saberlo.
        """
        S = gaussian(row=20, col=44, sigma_c=4.0) + gaussian(row=20, col=50, sigma_c=4.0)
        own = ridge_owners(S, [20, 20], [44, 50])
        self.assertEqual(count_instances(own), 2)
        self.assertEqual(int(own[20, 44]), 0)
        self.assertEqual(int(own[20, 50]), 1, "en la zona de solape manda la cercanía")

    def test_densidad_en_la_banda_de_una_fila_es_uno(self):
        """Propiedad central del cambio: en ±1 fila del ápice la etiqueta es 1.

        Con el marcador de 2 px la densidad en esa banda era ~0.2, que es lo que
        el modelo devolvía y lo que el picker convertía en picos extra.
        """
        S = gaussian(sigma_r=4.0, sigma_c=2.0)
        L = ridge_labels(S, [32], [48])
        banda = L[31:34, 48]
        self.assertTrue(np.all(banda > 0), f"densidad {banda.mean()}")

    def test_amplitud_nula_no_etiqueta(self):
        S = gaussian(row=32, col=48) + gaussian(row=32, col=20, amp=0.0)
        own = ridge_owners(S, [32, 32], [48, 20], amps=[1.0, 0.0])
        self.assertEqual(count_instances(own), 1, "el pico sin amplitud no existe")

    def test_sin_picos_devuelve_ceros(self):
        L = ridge_labels(np.zeros((16, 16)), [], [])
        self.assertEqual(float(L.sum()), 0.0)


class TestComponents(unittest.TestCase):
    def test_cuenta_regiones(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1, 1] = True
        m[5, 5:7] = True
        self.assertEqual(components(m), 2)

    def test_vacia(self):
        self.assertEqual(components(np.zeros((4, 4), dtype=bool)), 0)

    def test_diagonal_no_es_conexa(self):
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = m[1, 1] = True
        self.assertEqual(components(m), 2, "4-vecindad: la diagonal no conecta")


if __name__ == "__main__":
    unittest.main()
