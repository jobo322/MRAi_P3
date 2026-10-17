"""Tests del copiado de pesos de `--init` (scripts/weight_roles.py).

El bug original: `--init` emparejaba los pesos base "por forma en orden". Como
las dos ramas y todos los niveles del modelo comparten formas, el orden en que
el SavedModel lista sus variables no es el de creación, y el resultado era
asignar a cada brazo los pesos del OTRO (48 de 62 pesos). Medido sobre el
dataset real: BCE 1,0433 frente a 0,3524 del modelo base, es decir peor que
aleatorio (un modelo sin `--init` da 0,6932).

Lo que se fija aquí:
  * el rol derivado del SavedModel base (tabla explícita, para detectar cualquier
    cambio accidental en la derivación);
  * que `assign_weights` reproduce el base: maxdiff < 1e-6 en ventanas de pico
    reales, y con el base y la copia del mismo orden de magnitud (si no, la
    comparación sería trivial);
  * que un fallo de emparejado lanza ValueError en vez de copiar mal en silencio;
  * que funciona igual con el SavedModel de Keras 2 y con un re-export de Keras 3.
"""
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from weight_roles import _node_name, _port  # noqa: E402  (helpers sin TF)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - el entorno de entrenamiento lo tiene
    tf = None

BASE_US = os.path.join(REPO_ROOT, "Trained", "Probability_2D", "US", "DNN_0")
BASE_EXPORT = os.path.join(REPO_ROOT, "Trained", "out_elong_continue", "DNN_0")
DATASET = os.environ.get(
    "MRAI_DATASET", "/Users/abolanos/git/cheminfo/spectrum-generator/dataset")


def probe_windows():
    """Cruces sintéticos: positivos con pico en las dos ramas y negativos claros.

    Un cross son DOS perfiles: índices 0:64 = brazo F1 (elongado, el ancho),
    64:128 = brazo F2 (estrecho), con el pico centrado en 32 en ambos. Con ruido
    normalizado o con bumps que no respetan esa forma el modelo satura a ~0 y
    comparar no dice nada (es la trampa en la que se cae al escribir este test).
    """
    import numpy as np

    def loren(t, c, hwhm, amp=1.0):
        return amp / (1.0 + ((t - c) / hwhm) ** 2)

    t = np.arange(64.0)
    rows = []
    for h1 in (4.0, 8.0, 12.0):
        for h2 in (1.0, 2.0, 4.0):
            for dc in (0.0, 1.0, 2.0):
                for amp in (1.0, 50.0):
                    rows.append(np.concatenate([loren(t, 32 - dc, h1, amp),
                                                loren(t, 32, h2, amp)]))
    rows.append(np.ones(128))                                        # plano
    rows.append(np.concatenate([loren(t, 32, 8.0), np.ones(64)]))    # solo F1
    rows.append(np.concatenate([np.ones(64), loren(t, 32, 2.0)]))    # solo F2
    rows.append(np.concatenate([loren(t, 2, 8.0), loren(t, 2, 2.0)]))  # en el borde
    rows.append(np.random.default_rng(0).standard_normal(128))       # ruido
    X = np.asarray(rows, dtype=np.float32)[:, :, None]
    X /= np.linalg.norm(X, axis=(1, 2), keepdims=True)
    return X


def real_peak_windows(n_bg=1500, seed=0):
    """Ventanas reales del dataset (positivos + fondo), si está disponible."""
    import json
    ids = json.load(open(os.path.join(REPO_ROOT, "data", "train_2d_elong",
                                      "manifest.json")))["val"]
    sid = ids[0]
    from cross_extract_fast import split_2d_rows
    S = np.fromfile(os.path.join(DATASET, "spectra", f"{sid}.bin"),
                    dtype="<f4").reshape(256, 2048)
    L = np.fromfile(os.path.join(DATASET, "labels", f"{sid}.bin"),
                    dtype=np.uint8).reshape(256, 2048)
    pos = np.where(L.reshape(-1) > 0)[0]
    neg = np.random.default_rng(seed).choice(
        np.where(L.reshape(-1) == 0)[0], n_bg, replace=False)
    X, _ = split_2d_rows(S, np.concatenate([pos, neg]), np.array([64, 64]))
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def load_base(path):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(128, 1)),
        tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")])


def _swap_arms(roles):
    """Roles con F1 y F2 intercambiados: el bug histórico del `--init`."""
    out = []
    for role in roles:
        if "|F1|" in role:
            out.append(role.replace("|F1|", "|F2|"))
        elif "|F2|" in role:
            out.append(role.replace("|F2|", "|F1|"))
        else:
            out.append(role)
    return out


def predict(model, X):
    out = model.predict(X, batch_size=2 ** 14, verbose=0)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    return np.asarray(out).reshape(-1)


class TestGraphHelpers(unittest.TestCase):
    """Parseo de nombres de tensor: no necesita TensorFlow."""

    def test_node_name_and_port(self):
        self.assertEqual(_node_name("model/split:output:1"), "model/split")
        self.assertEqual(_port("model/split:output:1"), 1)
        self.assertEqual(_port("model/split:output:0"), 0)
        self.assertEqual(_port("model/conv"), 0)


@unittest.skipIf(tf is None, "tensorflow no instalado")
class TestDeriveRoles(unittest.TestCase):
    def test_towards_us_base_names(self):
        """Tabla de roles del SavedModel base de Keras 2 (versionado en el repo)."""
        from weight_roles import derive_roles
        roles = derive_roles(BASE_US)
        self.assertEqual(len(roles), 62)
        self.assertEqual(roles["conv|F1|1|t|k"], "conv1d_52/kernel")
        self.assertEqual(roles["conv|F1|1|s|k"], "conv1d_53/kernel")
        self.assertEqual(roles["conv|F2|1|t|k"], "conv1d_65/kernel")
        self.assertEqual(roles["conv|F1|6|s|b"], "conv1d_63/bias")
        self.assertEqual(roles["conv|F2|6|t|k"], "conv1d_75/kernel")
        self.assertEqual(roles["fus|F1|k"], "conv1d_64/kernel")
        self.assertEqual(roles["fus|F2|b"], "conv1d_77/bias")
        self.assertEqual(roles["dense|0|t|k"], "dense_10/kernel")
        self.assertEqual(roles["dense|1|s|b"], "dense_13/bias")
        self.assertEqual(roles["out|k"], "dense_14/kernel")
        self.assertEqual(roles["out|b"], "dense_14/bias")
        # F1 es la 1ª mitad del cross y F2 la 2ª: si esto se cruzase, el brazo
        # elongado recibiría los filtros del otro y el bug volvería.
        self.assertNotEqual(roles["conv|F1|1|t|k"], roles["conv|F2|1|t|k"])

    def test_roles_are_unique(self):
        from weight_roles import derive_roles
        roles = derive_roles(BASE_US)
        self.assertEqual(len(set(roles.keys())), 62)
        self.assertEqual(len(set(roles.values())), 62)

    def test_export_from_keras3_is_supported(self):
        """Generalidad: el mismo análisis vale para un re-export de Keras 3."""
        if not os.path.isdir(BASE_EXPORT):
            self.skipTest(f"SavedModel local ausente: {BASE_EXPORT}")
        from weight_roles import derive_roles
        roles = derive_roles(BASE_EXPORT)
        self.assertEqual(len(roles), 62)
        # build_model: conv1d = F1 L1 Tanh, conv1d_12 = F2 L1 Tanh
        self.assertEqual(roles["conv|F1|1|t|k"], "conv1d/kernel")
        self.assertEqual(roles["conv|F2|1|t|k"], "conv1d_12/kernel")
        self.assertEqual(roles["fus|F1|k"], "conv1d_24/kernel")
        self.assertEqual(roles["out|k"], "dense_14/kernel")

    def test_rejects_a_foreign_graph(self):
        """Un SavedModel que no es P3 2D debe fallar, no adivinar."""
        from weight_roles import derive_roles
        with self.assertRaises(ValueError):
            derive_roles(os.path.join(REPO_ROOT, "Trained", "Probability_3D", "DNN_0"))


@unittest.skipIf(tf is None, "tensorflow no instalado")
class TestAssignWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from train_model import build_model
        # Una sola vez por proceso: el contador global de nombres de Keras 3 hace
        # que la 4ª llamada choque con el nombre explícito 'dense_14'.
        cls.model = build_model()
        cls.X = probe_windows()

    def _copy_from(self, base_path):
        from weight_roles import assign_weights
        base = load_base(base_path)
        reference = predict(base, self.X)
        copied, skipped = assign_weights(self.model, base_path)
        return reference, predict(self.model, self.X), copied, skipped

    def test_weight_roles_cover_every_weight(self):
        self.assertEqual(len(self.model.weight_roles), len(self.model.weights))
        self.assertEqual(len(self.model.weight_roles), 62)
        self.assertEqual(len(set(self.model.weight_roles)), 62)
        self.assertEqual(self.model.weight_roles[0], "conv|F1|1|t|k")
        self.assertEqual(self.model.weight_roles[-1], "out|b")

    def test_probe_is_discriminating(self):
        """El banco de pruebas activa el base y contiene negativos claros."""
        reference = predict(load_base(BASE_US), self.X)
        self.assertGreater(reference.max(), 0.9,
                           "el banco de pruebas no activa el modelo")
        self.assertLess(reference.min(), 0.1,
                        "el banco de pruebas no discrimina")

    def test_probe_detects_a_misassignment(self):
        """Control negativo: con los brazos cruzados el copiado NO reproduce el base.

        Es la garantía de que `test_init_reproduces_the_us_base` no pasa por
        casualidad: el mismo banco de pruebas y el mismo umbral detectan el bug
        histórico (asignar a cada brazo los pesos del otro), que es el único
        emparejado erróneo que conserva las formas.
        """
        from weight_roles import assign_weights
        reference = predict(load_base(BASE_US), self.X)
        original = list(self.model.weight_roles)
        try:
            self.model.weight_roles = _swap_arms(original)
            assign_weights(self.model, BASE_US)
            scrambled = predict(self.model, self.X)
        finally:
            self.model.weight_roles = original
        self.assertGreater(float(np.abs(reference - scrambled).max()), 0.5)

    def test_init_reproduces_the_us_base(self):
        reference, copied_p, copied, skipped = self._copy_from(BASE_US)
        self.assertEqual(copied, 62)
        self.assertEqual(skipped, 0)
        self.assertLess(float(np.abs(reference - copied_p).max()), 1e-6)

    def test_init_reproduces_a_keras3_export(self):
        if not os.path.isdir(BASE_EXPORT):
            self.skipTest(f"SavedModel local ausente: {BASE_EXPORT}")
        reference, copied_p, copied, skipped = self._copy_from(BASE_EXPORT)
        self.assertEqual(copied, 62)
        self.assertLess(float(np.abs(reference - copied_p).max()), 1e-6)

    def test_does_not_silently_skip_unmatched_weights(self):
        """Con roles que el base no tiene, debe lanzar y no dejar pesos a medias."""
        from weight_roles import assign_weights
        original = list(self.model.weight_roles)
        try:
            self.model.weight_roles = ["conv|F9|9|t|k"] * len(original)
            with self.assertRaises(ValueError) as cm:
                assign_weights(self.model, BASE_US)
            self.assertIn("sin peso base equivalente", str(cm.exception))
        finally:
            self.model.weight_roles = original

    def test_requires_a_model_with_roles(self):
        from weight_roles import assign_weights
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=(128, 1)),
                                     tf.keras.layers.Dense(1)])
        with self.assertRaises(ValueError):
            assign_weights(model, BASE_US)


@unittest.skipIf(tf is None, "tensorflow no instalado")
@unittest.skipUnless(os.path.isdir(DATASET), "dataset local ausente")
class TestAssignWeightsOnRealPeaks(unittest.TestCase):
    """La comparación que importa: ventanas reales de pico del dataset."""

    def test_maxdiff_on_real_peak_windows(self):
        from train_model import build_model
        from weight_roles import assign_weights
        X = real_peak_windows()
        base = load_base(BASE_US)
        reference = predict(base, X)
        model = build_model()
        assign_weights(model, BASE_US)
        copied_p = predict(model, X)
        self.assertGreater(reference.max(), 0.5)
        self.assertLess(float(np.abs(reference - copied_p).max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
