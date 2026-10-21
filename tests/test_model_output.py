"""Tests for `model_output.single_output`: la clave de salida no es contrato.

Los SavedModels originales (Keras 2) exponen la ultima Dense con su nombre
(`dense_14` en 2D, `dense_9` en 3D). Los re-exportados con Keras 3
(`model.export`, como hace `scripts/train_model.py`) la exponen como
`output_0`. El nombre de la capa es un detalle de serializacion, no parte del
contrato del modelo: hay una unica salida y los consumidores deben tomarla sea
cual sea su clave.

Regresion cubierta: `run_51.py` y `StoP3.Production_2D/3D` indexaban
`predict(...)['dense_14']` y fallaban con `KeyError: 'dense_14'` sobre un
modelo re-exportado con Keras 3.

El modulo se carga por ruta (como en `test_headers.py`) para no disparar el
`__init__.py` del paquete, que importa TensorFlow.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
MODULE_PATH = os.path.join(REPO_ROOT, "src", "MRAi_P3", "model_output.py")

spec = importlib.util.spec_from_file_location("model_output", MODULE_PATH)
model_output = importlib.util.module_from_spec(spec)
sys.modules["model_output"] = model_output
spec.loader.exec_module(model_output)


def _arr(n=3):
    return np.zeros((n, 1), dtype=np.float32)


class TestSingleOutput(unittest.TestCase):
    """La funcion no depende de TensorFlow: solo desenvuelve la salida."""

    def test_legacy_keras2_key(self):
        """SavedModels originales: la salida viene bajo el nombre de la Dense."""
        x = _arr()
        self.assertIs(model_output.single_output({"dense_14": x}), x)

    def test_keras3_export_key(self):
        """Re-export con Keras 3 (`model.export`): la salida es 'output_0'."""
        x = _arr()
        self.assertIs(model_output.single_output({"output_0": x}), x)

    def test_bare_array_passthrough(self):
        """Un modelo sin salida estructurada devuelve el array tal cual."""
        x = _arr()
        self.assertIs(model_output.single_output(x), x)

    def test_ambiguous_output_raises(self):
        """Varias salidas: no se adivina, se avisa con las claves disponibles."""
        with self.assertRaises(ValueError) as cm:
            model_output.single_output({"a": _arr(), "b": _arr()})
        self.assertIn("a", str(cm.exception))
        self.assertIn("b", str(cm.exception))

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            model_output.single_output({})

    def test_matches_legacy_key_lookup(self):
        """Equivalencia con el indexado historico sobre el formato antiguo."""
        x = _arr(5)
        pred = {"dense_14": x}
        np.testing.assert_array_equal(model_output.single_output(pred),
                                      pred["dense_14"])


class TestRealSavedModels(unittest.TestCase):
    """Integracion: ambos formatos cargan y devuelven (n, 1) sin tocar claves."""

    LEGACY = os.path.join(REPO_ROOT, "Trained", "Probability_2D", "US", "DNN_0")
    EXPORT = os.path.join(REPO_ROOT, "Trained", "out_elong_continue", "DNN_0")

    def _load(self, path):
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("tensorflow no instalado")
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(128, 1)),
            tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")])

    def test_keras3_export_is_readable(self):
        if not os.path.isdir(self.EXPORT):
            self.skipTest(f"SavedModel local ausente: {self.EXPORT}")
        model = self._load(self.EXPORT)
        x = np.random.default_rng(0).standard_normal((8, 128, 1)).astype(np.float32)
        out = model_output.single_output(model.predict(x, batch_size=8, verbose=0))
        self.assertEqual(np.asarray(out).shape, (8, 1))

    def test_both_formats_expose_one_output(self):
        if not (os.path.isdir(self.LEGACY) and os.path.isdir(self.EXPORT)):
            self.skipTest("SavedModels locales ausentes")
        x = np.random.default_rng(1).standard_normal((4, 128, 1)).astype(np.float32)
        for path in (self.LEGACY, self.EXPORT):
            with self.subTest(model=path):
                model = self._load(path)
                out = model_output.single_output(
                    model.predict(x, batch_size=4, verbose=0))
                self.assertEqual(np.asarray(out).shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
