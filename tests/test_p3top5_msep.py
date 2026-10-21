"""Tests for the `msep` knob of `P3toP5`: separación mínima entre picos.

Contexto medido (8 espectros del dataset, `scripts/plotting/run_train_2d.py`):
la etiqueta marca un pico con un marcador de 1/2/4 px, pero el modelo elongado
responde con una cresta de varios píxeles en F1 (P a 1 px en F1 = 0.26 frente a
0.03 en F2). Con el valor histórico `msep=[1,1]` —todos los máximos locales a
1 px— la lista de picos saca 2–4 picos por pico real: 1105 FP de 2505 picos en
8 espectros. Subir `msep` colapsa la cresta en un pico.

Lo que NO debe cambiar: el valor por defecto sigue siendo `[1,1]`, byte a byte
igual que antes, para no alterar el flujo de la carpeta 51 sin decirlo.

El módulo se carga por ruta (como `test_headers.py`) para no disparar el
`__init__.py` del paquete, que importa TensorFlow.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src", "MRAi_P3")

try:
    import nmrglue  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("nmrglue no está instalado")


def _load():
    """Carga `src/MRAi_P3/P3toP5.py` bajo un paquete sintético."""
    import types

    if "MRP_msep" not in sys.modules:
        pkg = types.ModuleType("MRP_msep")
        pkg.__path__ = [SRC_DIR]
        sys.modules["MRP_msep"] = pkg
    spec = importlib.util.spec_from_file_location(
        "MRP_msep.P3toP5", os.path.join(SRC_DIR, "P3toP5.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["MRP_msep.P3toP5"] = mod
    spec.loader.exec_module(mod)
    return mod


P3toP5 = _load()


def _ridge():
    """Cresta mínima: dos máximos a 2 px en F1 (mismo F2) + uno lejano."""
    P = np.zeros((80, 120), dtype=float)
    P[10, 20] = 0.9     # fuerte
    P[12, 20] = 0.6     # a 2 px en F1: el segundo máximo de la MISMA cresta
    P[60, 100] = 0.8    # lejos de todo
    return P


class TestPeakPointsMsep(unittest.TestCase):
    def test_default_es_el_historico_msep_1(self):
        """Sin `msep` el resultado es idéntico a `[1,1]` (no cambiar el legacy)."""
        P = _ridge()
        a = P3toP5.peak_points(P, thres=0.2)
        b = P3toP5.peak_points(P, thres=0.2, msep=[1, 1])
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(len(a), 3, "con msep=1 se ven los dos máximos de la cresta")

    def test_msep_colapsa_la_cresta_y_se_queda_el_fuerte(self):
        P = _ridge()
        loc = P3toP5.peak_points(P, thres=0.2, msep=3)
        self.assertEqual(loc.tolist(), [[10, 20], [60, 100]],
                         "msep=3 colapsa (12,20) en (10,20), el más fuerte")

    def test_msep_no_se_come_picos_bien_separados(self):
        P = np.zeros((80, 120))
        P[10, 20] = 0.5
        P[16, 20] = 0.9      # 6 px: dos picos distintos, no una cresta
        self.assertEqual(len(P3toP5.peak_points(P, thres=0.2, msep=5)), 2)
        # a partir de 6 px sí se consideran el mismo pico, y gana el fuerte
        loc = P3toP5.peak_points(P, thres=0.2, msep=6)
        self.assertEqual(loc.tolist(), [[16, 20]])

    def test_escalar_y_lista_equivalen(self):
        P = _ridge()
        self.assertEqual(P3toP5.peak_points(P, thres=0.2, msep=3).tolist(),
                         P3toP5.peak_points(P, thres=0.2, msep=[3, 3]).tolist())

    def test_longitud_incorrecta_falla_claro(self):
        with self.assertRaises(ValueError):
            P3toP5.peak_points(_ridge(), thres=0.2, msep=[3, 3, 3])

    def test_el_umbral_sigue_mandando(self):
        P = _ridge()
        P[12, 20] = 0.1       # por debajo del umbral: no es candidato
        loc = P3toP5.peak_points(P, thres=0.2, msep=1)
        self.assertNotIn([12, 20], loc.tolist())


class TestNormalizeMsep(unittest.TestCase):
    """La CLI acepta `--msep 3` (todos los ejes) y `--msep 3 1` (por eje).

    `argparse` con `nargs="+"` devuelve SIEMPRE una lista, así que hace falta
    una normalización explícita: `peak_points` espera un escalar o una lista de
    la misma longitud que las dimensiones de P3, y `[3]` no vale para un P3 2D.
    """

    def test_none_pasa_none(self):
        self.assertIsNone(P3toP5.normalize_msep(None))

    def test_escalar_pasa_escalar(self):
        self.assertEqual(P3toP5.normalize_msep(3), 3)

    def test_lista_de_uno_se_colapsa_a_escalar(self):
        self.assertEqual(P3toP5.normalize_msep([3]), 3)
        self.assertEqual(P3toP5.normalize_msep((3,)), 3)

    def test_lista_por_eje_se_mantiene(self):
        self.assertEqual(P3toP5.normalize_msep([3, 1]), [3, 1])
        self.assertEqual(P3toP5.normalize_msep((3, 1)), [3, 1])
        self.assertEqual(P3toP5.normalize_msep(np.array([2, 1])), [2, 1])

    def test_lista_vacia_falla_claro(self):
        with self.assertRaises(ValueError):
            P3toP5.normalize_msep([])

    def test_el_resultado_que_espera_peak_points(self):
        """`[3]` colapsado a 3 funciona; `[3,1]` se aplica por eje."""
        P = _ridge()          # cresta en F1: dos máximos a 2 px en filas
        uno = P3toP5.peak_points(P, thres=0.2,
                                 msep=P3toP5.normalize_msep([3]))
        por_eje = P3toP5.peak_points(P, thres=0.2,
                                     msep=P3toP5.normalize_msep([3, 1]))
        self.assertEqual(uno.tolist(), [[10, 20], [60, 100]])
        # msep=[3,1] solo colapsa en F1 (filas); la cresta está en F1, así que
        # también queda un pico
        self.assertEqual(por_eje.tolist(), [[10, 20], [60, 100]])


class TestMsepCli(unittest.TestCase):
    """Las dos CLIs declaran `--msep` como lista de enteros (sin importarlas:
    ambos scripts importan TensorFlow y este entorno de tests no lo tiene)."""

    SCRIPTS = (
        os.path.join(BASE_DIR, "scripts", "plotting", "run_train_2d.py"),
        os.path.join(BASE_DIR, "scripts", "bruker_2d", "run_51.py"),
    )

    def _kwargs(self, path):
        import ast
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--msep"):
                return {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        return None

    def test_las_dos_clis_aceptan_varios_valores(self):
        for path in self.SCRIPTS:
            with self.subTest(script=os.path.basename(path)):
                kw = self._kwargs(path)
                self.assertIsNotNone(kw, f"falta --msep en {path}")
                self.assertEqual(kw.get("nargs"), "'+'",
                                 "--msep debe aceptar `3 1` (nargs='+')")
                self.assertEqual(kw.get("type"), "int")
                self.assertTrue(kw.get("default", "").startswith("["),
                                "el default debe ser una lista, p.ej. [1]")


if __name__ == "__main__":
    unittest.main()
