"""Tests de pares apretados: peak_pair_flags y row_pair_flags.

Un par apretado es lo que el perfil en cruz NO ve: dos picos a <= `px` el uno
del otro con amplitudes muy distintas (el propio <= `ratio` del otro). La fila
que la red entrenada ve es una cruz (eje F1 + eje F2 por el centro), así que un
vecino en diagonal a 4 px no deja huella en ninguna de las dos ramas; el
manifiesto sí conoce los centros y las amplitudes de todos los picos. El
sobremuestreo (`train_model.py --pair-frac`) ataca exactamente los falsos
negativos enterrados en el flanco de un pico grande.

Reglas que fijan estos tests:

* el par en diagonal se marca (el caso que motivó el módulo);
* los empates de amplitud NO cuentan (ni con ratio 1);
* los límites son INCLUSIVOS: distancia == px marca y ratio == umbral marca;
* n=0 y n=1 no rompen; listas, tuplas, vacíos y dtypes varios se aceptan;
* la propagación píxel->pico marca TODOS los píxeles del pixelSet del pico
  marcado (marcadores de 2 y de 4 px) y ninguno de un pico grande que no
  forma par con nadie; un píxel "pertenece" a un pico marcado si está a
  <= px de su centro;
* idx_flat desordenado da la misma marca por píxel que el ordenado;
* preextract valida --pair-px/--pair-ratio con SystemExit y mensaje claro;
* pair_marks es numpy puro: importarlo no arrastra TensorFlow.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from pair_marks import peak_pair_flags, row_pair_flags  # noqa: E402


class TestPeakPairFlags(unittest.TestCase):
    def test_par_alineado_en_f1(self):
        """Misma columna, filas vecinas: el pequeño queda enterrado."""
        f = peak_pair_flags([10, 13, 40], [20, 20, 40], [10.0, 5.0, 8.0])
        self.assertEqual(f.dtype, np.bool_)
        self.assertEqual(f.tolist(), [False, True, False])

    def test_par_en_diagonal(self):
        """EL caso clave: a (3,3) -> 4,24 px el perfil en cruz no lo ve."""
        f = peak_pair_flags([10, 13], [20, 23], [10.0, 5.0])
        self.assertEqual(f.tolist(), [False, True])

    def test_par_simetrico_no_marcado(self):
        """Dos picos gemelos: ninguno de los dos forma par."""
        f = peak_pair_flags([10, 13], [20, 20], [5.0, 5.0])
        self.assertEqual(f.tolist(), [False, False])

    def test_empate_no_cuenta_ni_con_ratio_1(self):
        """'Otro con amplitud mayor' es estricto: el empate nunca marca."""
        f = peak_pair_flags([10, 13], [20, 20], [5.0, 5.0], ratio=1.0)
        self.assertEqual(f.tolist(), [False, False])

    def test_pico_aislado_no(self):
        self.assertEqual(peak_pair_flags([7], [9], [3.0]).tolist(), [False])

    def test_distancia_exactamente_px_marcada(self):
        """Límite inclusivo: dist == px marca, dist > px no (ratio 1 aísla)."""
        inside = peak_pair_flags([0, 6], [0, 0], [10.0, 5.0],
                                 px=6.0, ratio=1.0)
        self.assertEqual(inside.tolist(), [False, True])
        outside = peak_pair_flags([0, 6.001], [0, 0], [10.0, 5.0],
                                  px=6.0, ratio=1.0)
        self.assertEqual(outside.tolist(), [False, False])

    def test_ratio_exactamente_umbral_marcado(self):
        """Límite inclusivo del ratio, con la distancia de sobra (4 px)."""
        justo = peak_pair_flags([10, 10], [20, 24], [5.0, 10.0],
                                px=6.0, ratio=0.5)
        self.assertEqual(justo.tolist(), [True, False])    # 5/10 == 0.5
        pasado = peak_pair_flags([10, 10], [20, 24], [6.0, 10.0],
                                 px=6.0, ratio=0.5)
        self.assertEqual(pasado.tolist(), [False, False])  # 6/10 = 0.6 > 0.5

    def test_n0_y_n1(self):
        e = peak_pair_flags([], [], [])
        self.assertEqual(len(e), 0)
        self.assertEqual(e.dtype, np.bool_)
        self.assertEqual(peak_pair_flags([], [], [], px=6.0).tolist(), [])

    def test_acepta_listas_tuplas_y_dtypes(self):
        ref = peak_pair_flags([10, 13], [20, 20], [10.0, 5.0])
        a = peak_pair_flags(np.array([10, 13]), np.array([20, 20]),
                            np.array([10.0, 5.0], dtype=np.float32))
        b = peak_pair_flags((10, 13), (20, 20), (10, 5))
        self.assertTrue(np.array_equal(ref, a))
        self.assertTrue(np.array_equal(ref, b))

    def test_longitudes_distintas_fallan_claro(self):
        with self.assertRaises(ValueError):
            peak_pair_flags([1, 2], [3], [1.0, 2.0])


class TestRowPairFlags(unittest.TestCase):
    W = 40

    def _flat(self, pixels, W=None):
        W = self.W if W is None else W
        return np.array([r * W + c for r, c in pixels], dtype=np.int64)

    def test_pixelset_de_2_px_todos_marcados(self):
        """Par alineado; los pixelSet de 2 px de ambos picos salen 1."""
        big_ps = [(10, 20), (10, 21)]
        small_ps = [(12, 20), (12, 21)]
        idx = self._flat(big_ps + small_ps)
        f = row_pair_flags([10, 12], [20, 20], [10.0, 5.0], idx, self.W)
        self.assertEqual(f.dtype, np.uint8)
        self.assertEqual(f.tolist(), [1, 1, 1, 1])

    def test_pixelset_de_4_px_todos_marcados(self):
        """Centro en medio-cuadro: pixelSet de 4 px alrededor de (20.5,30.5)."""
        big_ps = [(20, 30), (20, 31), (21, 30), (21, 31)]
        small_ps = [(23, 32)]                     # 1 px, a ~2.9 px del centro
        idx = self._flat(big_ps + small_ps)
        f = row_pair_flags([20, 23], [30, 32], [10.0, 5.0], idx, self.W)
        self.assertEqual(f.tolist(), [1, 1, 1, 1, 1])

    def test_pico_grande_sin_par_sale_0(self):
        """Un pico grande lejos de todo par marcado: sus píxeles no cuentan."""
        idx = self._flat([(100, 5), (100, 6)])    # pixelSet del pico grande
        f = row_pair_flags([10, 12, 100], [20, 20, 5], [10.0, 5.0, 9.0],
                           idx, self.W)
        self.assertEqual(f.tolist(), [0, 0])

    def test_idx_flat_desordenado(self):
        """El marcaje es por píxel: el orden de idx_flat no cambia nada."""
        idx_sorted = self._flat([(10, 20), (10, 21), (12, 20), (12, 21),
                                 (50, 3)])
        order = [3, 0, 4, 1, 2]
        f_sorted = row_pair_flags([10, 12], [20, 20], [10.0, 5.0],
                                  idx_sorted, self.W)
        f_shuf = row_pair_flags([10, 12], [20, 20], [10.0, 5.0],
                                idx_sorted[order], self.W)
        self.assertTrue(np.array_equal(f_shuf, f_sorted[order]))

    def test_idx_vacio_y_n0(self):
        f = row_pair_flags([], [], [], [], self.W)
        self.assertEqual(len(f), 0)
        self.assertEqual(f.dtype, np.uint8)
        f1 = row_pair_flags([1], [2], [3.0], np.array([], dtype=np.int64),
                            self.W)
        self.assertEqual(len(f1), 0)

    def test_acepta_listas_y_dtypes_varios(self):
        idx = [10 * self.W + 20, 12 * self.W + 20]
        a = row_pair_flags([10, 12], [20, 20], [10.0, 5.0], idx, self.W)
        b = row_pair_flags(np.array([10, 12], dtype=np.int32),
                           np.array([20, 20], dtype=np.int32),
                           np.array([10.0, 5.0], dtype=np.float32),
                           np.array(idx, dtype=np.uint32), self.W)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.tolist(), [1, 1])


class TestValidacionPreextract(unittest.TestCase):
    """--pair-px/--pair-ratio se validan con SystemExit y mensaje claro."""

    def test_pair_px_negativo_falla(self):
        from preextract import _check_pair_args
        with self.assertRaises(SystemExit):
            _check_pair_args(-0.1, 0.5)

    def test_pair_ratio_fuera_de_rango_falla(self):
        from preextract import _check_pair_args
        for r in (0.0, -0.5, 1.0001, 2.0):
            with self.subTest(r=r):
                with self.assertRaises(SystemExit):
                    _check_pair_args(6.0, r)

    def test_valores_validos_pasan(self):
        from preextract import _check_pair_args
        _check_pair_args(0.0, 0.5)      # apagado, ratio da igual
        _check_pair_args(6.0, 0.5)
        _check_pair_args(6.0, 1.0)


class TestPreextractPairFlags(unittest.TestCase):
    """Extremo a extremo en un dataset sintético minúsculo.

    Con --pair-px 0 el npz NO gana claves y el log NO gana columnas; con
    --pair-px > 0 el npz guarda pair (uint8, len M), pair_px y pair_ratio, el
    log gana la columna `pairs`, y las filas del pool marcadas son las que
    predice row_pair_flags (los positivos van primeros, en orden raster).
    """
    H, W = 256, 2048

    @staticmethod
    def _dataset(tmp):
        """ spectra/0000.bin + labels/0000.bin + manifest.jsonl + manifiesto."""
        rng = np.random.default_rng(0)
        S = (rng.standard_normal((256, 2048)) * 0.01).astype("<f4")
        L = np.zeros((256, 2048), dtype=np.uint8)
        # par apretado ((100,1000)-(102,1002): 2,83 px) + un pico aislado
        peaks = [{"row": 100, "col": 1000, "A": 0.5},
                 {"row": 102, "col": 1002, "A": 0.1},
                 {"row": 50, "col": 300, "A": 0.3}]
        for p in peaks:
            # como el dataset real: pixelSet de 2 px y la etiqueta en esos px
            p["pixelSet"] = [[p["row"], p["col"]], [p["row"], p["col"] + 1]]
            for (rr, cc) in p["pixelSet"]:
                L[rr, cc] = 1
        os.makedirs(os.path.join(tmp, "spectra"), exist_ok=True)
        os.makedirs(os.path.join(tmp, "labels"), exist_ok=True)
        S.tofile(os.path.join(tmp, "spectra", "0000.bin"))
        L.tofile(os.path.join(tmp, "labels", "0000.bin"))
        with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
            fh.write('{"id":0,"peaks":[' + ",".join(
                '{"row":%d,"col":%d,"A":%r,"pixelSet":%s}'
                % (p["row"], p["col"], p["A"], json.dumps(p["pixelSet"]))
                for p in peaks) + ']}\n')
        json.dump({"train": ["0000"], "val": []},
                  open(os.path.join(tmp, "manifest.json"), "w"))
        return S

    def _run(self, dataset, out, pair_px):
        import json
        sys.argv = ["preextract.py", "--dataset", dataset, "--ids",
                    os.path.join(dataset, "manifest.json"), "--out", out,
                    "--seed", "0", "--pool-k", "3"]
        if pair_px is not None:
            sys.argv += ["--pair-px", str(pair_px)]
        import preextract
        preextract.main()

    def test_npz_y_log_con_y_sin_pares(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._dataset(tmp)
            base_keys = {"X", "y", "n_pos", "d", "hard_px", "H", "W"}
            # --pair-px 0 (implícito): exactamente el npz y el log de hoy
            out0 = os.path.join(tmp, "out0")
            self._run(tmp, out0, None)
            with np.load(os.path.join(out0, "0000.npz")) as z:
                self.assertEqual(set(z.files), base_keys,
                                 "con --pair-px 0 el npz no gana claves")
            man0 = json.load(open(os.path.join(out0, "manifest.json")))
            self.assertNotIn("pair_px", man0)
            with open(os.path.join(out0, "preextract.log")) as fh:
                rows0 = list(csv.reader(fh))
            self.assertEqual(rows0[0],
                             ["id", "split", "M", "pos", "hard", "band2",
                              "seconds"])
            self.assertEqual(len(rows0[1]), 7)
            # --pair-px 6: claves nuevas, columna nueva, marcas esperadas
            out6 = os.path.join(tmp, "out6")
            self._run(tmp, out6, 6.0)
            with np.load(os.path.join(out6, "0000.npz")) as z:
                self.assertEqual(base_keys <= set(z.files), True)
                self.assertIn("pair", z.files)
                pair = z["pair"]
                self.assertEqual(pair.dtype, np.uint8)
                self.assertEqual(len(pair), int(z["X"].shape[0]))
                self.assertEqual(float(z["pair_px"]), 6.0)
                self.assertEqual(float(z["pair_ratio"]), 0.5)
                n_pos = int(z["n_pos"])
            # positivos en orden raster: 0,1 pico aislado; 2,3 el GRANDE del
            # par; 4,5 el pequeño. Solo se marca el pico PEQUEÑO (el que queda
            # enterrado): el grande no es el problema, y marcarlo sería
            # sobremuestrear positivos fáciles.
            self.assertEqual(pair[:n_pos].tolist(), [0, 0, 0, 0, 1, 1])
            man6 = json.load(open(os.path.join(out6, "manifest.json")))
            self.assertEqual(man6.get("pair_px"), 6.0)
            with open(os.path.join(out6, "preextract.log")) as fh:
                rows6 = list(csv.reader(fh))
            self.assertEqual(rows6[0][:7], rows0[0])
            self.assertEqual(rows6[0][7], "pairs")
            self.assertEqual(len(rows6[1]), 8)
            self.assertEqual(int(rows6[1][7]), int(pair.sum()))

    def test_id_fuera_del_manifest_jsonl_falla_claro(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._dataset(tmp)
            json.dump({"train": ["0007"], "val": []},
                      open(os.path.join(tmp, "manifest.json"), "w"))
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp, os.path.join(tmp, "out"), 6.0)
            self.assertIn("0007", str(ctx.exception))


class TestPairMarksEsNumpyPuro(unittest.TestCase):
    def test_importar_pair_marks_no_trae_tensorflow(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import pair_marks; "
             "assert 'tensorflow' not in sys.modules, "
             "'pair_marks importa TensorFlow'" % SCRIPTS],
            capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


class TestPertenenciaExacta(unittest.TestCase):
    """El flag marca los PÍXELES DE LA ETIQUETA del pico marcado, no un disco.

    Por qué importa (medido en el pool real, id 0001): marcando un disco de
    radio `px` alrededor del centro salían **1112 filas marcadas de 14525**,
    de las cuales solo **49** eran positivas. Como el sobremuestreo elige
    filas marcadas para el bloque de POSITIVOS, con la semántica de disco el
    96 % de lo añadido eran negativos y el diseño no hacía nada.
    """

    W = 100

    def _caso(self):
        # A grande en (10,10); B pequeña en (10,16): dist 6, ratio 0.4 -> B marcada
        rows = np.array([10.0, 10.0])
        cols = np.array([10.0, 16.0])
        amps = np.array([1.0, 0.4])
        pixel_sets = [[(10, 10)], [(10, 16), (10, 17)]]   # B tiene 2 px de etiqueta
        idx = np.array([10 * self.W + 10,    # positivo de A
                        10 * self.W + 16,    # positivo de B
                        10 * self.W + 17,    # segundo px de B
                        10 * self.W + 13,    # FONDO a 3 px del centro de B
                        11 * self.W + 19])   # fondo lejos
        return rows, cols, amps, pixel_sets, idx

    def test_solo_los_pixeles_de_la_etiqueta(self):
        rows, cols, amps, pixel_sets, idx = self._caso()
        got = row_pair_flags(rows, cols, amps, idx, self.W,
                                        px=6.0, ratio=0.5,
                                        pixel_sets=pixel_sets)
        self.assertEqual(got.tolist(), [0, 1, 1, 0, 0])

    def test_el_disco_sigue_disponible_si_no_hay_pixel_sets(self):
        """Sin `pixel_sets` se mantiene la aproximación por radio (documentada)."""
        rows, cols, amps, _, idx = self._caso()
        got = row_pair_flags(rows, cols, amps, idx, self.W,
                                        px=6.0, ratio=0.5)
        # El disco marca TODO: el píxel del pico grande (a 6 px exactos de B,
        # límite inclusivo), el fondo a 3 px y hasta el fondo de otra fila a
        # 3,16 px. Exactamente la patología que se arregla con `pixel_sets`.
        self.assertEqual(got.tolist(), [1, 1, 1, 1, 1])

    def test_pixel_set_de_cuatro_pixeles(self):
        rows = np.array([10.0, 10.0])
        cols = np.array([10.0, 16.0])
        amps = np.array([1.0, 0.4])
        pixel_sets = [[(10, 10)], [(10, 16), (10, 17), (11, 16), (11, 17)]]
        idx = np.array([10 * self.W + 16, 10 * self.W + 17,
                        11 * self.W + 16, 11 * self.W + 17,
                        10 * self.W + 18])
        got = row_pair_flags(rows, cols, amps, idx, self.W,
                                        px=6.0, ratio=0.5,
                                        pixel_sets=pixel_sets)
        self.assertEqual(got.tolist(), [1, 1, 1, 1, 0])

    def test_sin_pares_marcados_nada_se_marca(self):
        rows = np.array([10.0, 10.0])
        cols = np.array([10.0, 16.0])
        amps = np.array([1.0, 0.9])          # ratio 0.9 > 0.5 -> nadie marcado
        pixel_sets = [[(10, 10)], [(10, 16)]]
        idx = np.array([10 * self.W + 10, 10 * self.W + 16])
        got = row_pair_flags(rows, cols, amps, idx, self.W,
                                        px=6.0, ratio=0.5,
                                        pixel_sets=pixel_sets)
        self.assertEqual(got.tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()