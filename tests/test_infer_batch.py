"""Tests de `scripts/plotting/infer_batch.py`: inferencia en lote.

Lo único que se puede probar sin TensorFlow es la selección de ids (y que el
módulo NO arrastre TF al importarse). El resto (predict_P3 + la cadena) ya está
cubierto por los tests de la caché y por `run_train_2d.py`.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(BASE_DIR, "scripts", "plotting")


def _load():
    "Carga el script por ruta, como test_p3top5_msep hace con P3toP5."
    if "infer_batch_test" in sys.modules:
        return sys.modules["infer_batch_test"]
    sys.path.insert(0, PLOT_DIR)
    spec = importlib.util.spec_from_file_location(
        "infer_batch_test", os.path.join(PLOT_DIR, "infer_batch.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infer_batch_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSelectIds(unittest.TestCase):
    def _man(self, train, val):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"train": train, "val": val}, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_primeros_n_del_split(self):
        mod = _load()
        path = self._man(["0001", "0002", "0003", "0004"], ["0000"])
        self.assertEqual(mod.select_ids(path, "train", 3), ["0001", "0002", "0003"])
        self.assertEqual(mod.select_ids(path, "val", 5), ["0000"])

    def test_n_mayor_que_lo_disponible_devuelve_todo(self):
        mod = _load()
        path = self._man(["0001", "0002"], ["0000"])
        self.assertEqual(mod.select_ids(path, "train", 999), ["0001", "0002"])

    def test_ids_siempre_como_texto_de_4_digitos(self):
        """El nombre del .tab se construye con el id: 1 -> '0001'."""
        mod = _load()
        path = self._man([1, "2", 3], [])
        self.assertEqual(mod.select_ids(path, "train", 3), ["0001", "0002", "0003"])

    def test_split_desconocido_falla_con_mensaje(self):
        mod = _load()
        path = self._man(["0001"], ["0000"])
        with self.assertRaises(SystemExit) as ctx:
            mod.select_ids(path, "nope", 5)
        self.assertIn("nope", str(ctx.exception))

    def test_n_cero_no_devuelve_nada(self):
        mod = _load()
        path = self._man(["0001"], ["0000"])
        self.assertEqual(mod.select_ids(path, "train", 0), [])


class TestSinTensorFlow(unittest.TestCase):
    def test_importar_no_arrastra_tf(self):
        _load()
        self.assertNotIn("tensorflow", sys.modules,
                         "infer_batch.py no debe importar TF a nivel de módulo")


class TestWriteP3Map(unittest.TestCase):
    """El visor (`findP3`) busca `<inferencia>/<id>/view2d/p3.bin` y valida
    `shape` en `meta.json`; sin eso el selector «P3 probability» queda muerto
    para todos los espectros del lote."""

    def test_layout_que_espera_el_visor(self):
        mod = _load()
        H, W = 256, 2048
        with tempfile.TemporaryDirectory() as tmp:
            P = np.linspace(-1.0, 1.0, H * W).reshape(H, W).astype(np.float64)
            vdir = mod.write_p3_map(tmp, "0001", P)
            self.assertEqual(os.path.dirname(vdir), os.path.join(tmp, "0001"))
            self.assertEqual(os.path.basename(vdir), "view2d")
            bin = os.path.join(vdir, "p3.bin")
            self.assertTrue(os.path.isfile(bin))
            self.assertEqual(os.path.getsize(bin), H * W * 4)   # float32
            back = np.fromfile(bin, dtype="<f4").reshape(H, W)
            self.assertTrue(np.allclose(back, P.astype(np.float32)))
            meta = json.load(open(os.path.join(vdir, "meta.json")))
            self.assertEqual(meta["shape"], [H, W])
            self.assertEqual(meta["p3"], "p3.bin")

    def test_id_se_normaliza_a_cuatro_digitos(self):
        mod = _load()
        with tempfile.TemporaryDirectory() as tmp:
            vdir = mod.write_p3_map(tmp, 7, np.zeros((256, 2048)))
            self.assertIn(os.path.join(tmp, "0007", "view2d"), vdir)


if __name__ == "__main__":
    unittest.main()
