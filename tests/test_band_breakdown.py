"""Tests de `scripts/band_breakdown.py`.

La imagen sintética tiene un solo positivo, así que el desglose (d1, d2) del
fondo se puede contar a mano: es lo que fija la semántica que después se usa
para elegir `--hard-aniso` (un error de un píxel en el rectángulo cambia el
reparto del pool sin que nada falle).
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import band_breakdown as bb  # noqa: E402

SHAPE = bb.SHAPE
ROW, COL = 100, 1000
BANDS = ((3, 10, 0, 3), (3, 10, 1, 3))


def write_dataset(tmp, classes=("MIX",), split=None, peak=1, amp=None):
    """Dataset mínimo: un label por id, con un positivo en (ROW, COL).

    Con `amp` escribe también `spectra/<id>.bin` (float32) con ese valor en el
    pico, para poder probar el eje de intensidad.
    """
    os.makedirs(os.path.join(tmp, "labels"))
    if amp is not None:
        os.makedirs(os.path.join(tmp, "spectra"))
    for i, _ in enumerate(classes):
        L = np.zeros(SHAPE, dtype=np.uint8)
        L[ROW, COL] = peak
        L.tofile(os.path.join(tmp, "labels", f"{i:04d}.bin"))
        if amp is not None:
            S = np.zeros(SHAPE, dtype="<f4")
            S[ROW, COL] = amp
            S.tofile(os.path.join(tmp, "spectra", f"{i:04d}.bin"))
    with open(os.path.join(tmp, "manifest.jsonl"), "w") as fh:
        for i, cls in enumerate(classes):
            noise = ',"noise":{"sd":0.001,"snr":1000}' if amp is not None else ""
            fh.write(f'{{"id":{i},"class":"{cls}"{noise},"peaks":[{{"row":1}}]}}\n')
    if split is not None:
        with open(os.path.join(tmp, "dataset.json"), "w") as fh:
            json.dump({"split": split}, fh)
    return tmp


def measure(tmp, sid="0000", bands=BANDS, cls="MIX", noise_sd=None):
    task = (sid, cls, "train", tmp, bands, 30, 10, bb.HARD_PX, noise_sd)
    return bb.measure_one(task)


class TestMeasureOne(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        write_dataset(self.tmp)

    def test_background_and_positives(self):
        row = measure(self.tmp)
        self.assertEqual(row["n_pos"], 1)
        self.assertEqual(row["n_bg"], SHAPE[0] * SHAPE[1] - 1)

    def test_ring_uses_the_rounded_distance(self):
        # `d` se redondea a píxel entero (`np.rint`), así que el anillo d <= 2
        # incluye los offsets de distancia real 2,236 —(1,2) y (2,1)— además de
        # los 12 que ya cumplen d <= 2 exacto: 4 de d2 puro, 2 de d1 puro, 4
        # diagonales a sqrt(2) y 2 a d1 = 2, más 8 de (1,2)/(2,1).
        self.assertEqual(measure(self.tmp)["ring_px"], 12 + 8)

    def test_band_counts_match_the_rectangle(self):
        bands = measure(self.tmp)["bands"]["3-10x0-3"]
        # d1 en 3..10 -> 16 filas; d2 en 0..3 -> 1 + 2 + 2 + 2 = 7 columnas.
        self.assertEqual(bands["total"], 16 * 7)
        self.assertEqual(bands["d2_zero"], 16)
        self.assertEqual(bands["diagonal"], 16 * 6)
        self.assertEqual(sum(bands["per_d2"]), bands["total"])
        self.assertEqual(bands["per_d2"], [16, 32, 32, 32])

    def test_f2lo_one_excludes_the_axis(self):
        bands = measure(self.tmp)["bands"]["3-10x1-3"]
        self.assertEqual(bands["d2_zero"], 0)
        self.assertEqual(bands["total"], 16 * 6)

    def test_histogram_has_no_positive_cell(self):
        row = measure(self.tmp)
        stride = row and 11  # d2_max + 1
        hist = row["hist"]
        self.assertEqual(len(hist), 31 * stride)
        # (d1=0, d2=0) es el positivo: nunca es fondo.
        self.assertEqual(hist[0], 0)
        self.assertEqual(sum(hist), row["n_bg"])


class TestSnr(unittest.TestCase):
    """Eje de intensidad: el brazo del cross contra el ruido del dataset."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Pico de amplitud 0,5 y σ = 0,5 -> snr = 1 dentro de la ventana del brazo.
        write_dataset(self.tmp, amp=0.5)

    def test_snr_buckets_are_edges(self):
        self.assertEqual(bb.snr_buckets([0.0, 1.0, 3.0, 10.0, 100.0, 1000.0, 1e6]),
                         [1, 1, 1, 1, 1, 2])
        self.assertEqual(bb.snr_buckets([]), [0, 0, 0, 0, 0, 0])
        self.assertEqual(bb.snr_labels(bb.SNR_EDGES),
                         ("<1", "1-3", "3-10", "10-100", "100-1000", ">=1000"))

    def test_read_noise_sd(self):
        self.assertAlmostEqual(bb.read_noise_sd(self.tmp), 0.001)

    def test_buckets_cover_each_region(self):
        row = measure(self.tmp, noise_sd=0.5)
        for name, buckets in row["snr"].items():
            with self.subTest(region=name):
                if name == "anillo":
                    expected = row["ring_px"]
                elif name.startswith("band:"):
                    expected = row["bands"][name.split(":")[1]]["total"]
                else:
                    expected = (row["n_bg"] - row["ring_px"]
                                - row["bands"]["3-10x0-3"]["total"])
                self.assertEqual(sum(buckets), expected)

    def test_cross_only_sees_the_peak_on_its_own_row_or_column(self):
        """El cross de un píxel son su fila y su columna: el pico entra al cross
        solo si comparte fila (d1=0) o columna (d2=0) con el píxel.

        Con un único pico, los 16 px de la banda con d2=0 tienen el pico en su
        columna (snr = 0,5/0,5 = 1 -> bucket "1-3") y los 96 diagonales no lo
        tienen en ninguna de las dos (snr = 0 -> bucket "<1").
        """
        row = measure(self.tmp, noise_sd=0.5)
        self.assertEqual(row["snr"]["band:3-10x0-3"], [96, 16, 0, 0, 0, 0])
        self.assertEqual(row["snr"]["band:3-10x1-3"], [96, 0, 0, 0, 0, 0])
        # Anillo: los 8 px sobre la fila o la columna del pico lo ven; los 12 de
        # las diagonales (1,1), (1,2), (2,1) no.
        self.assertEqual(row["snr"]["anillo"], [12, 8, 0, 0, 0, 0])
        # `fácil` también puede ver el pico: 102 px de su fila o columna dentro
        # de la ventana de 64 px quedan fuera del anillo y de la banda.
        self.assertEqual(row["snr"]["facil"][1], 102)


class TestReaders(unittest.TestCase):
    def test_classes_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(tmp, classes=("MIX", "PG", "PL"),
                          split={"train": [0, 1], "val": [2]})
            classes = bb.read_classes(tmp)
            self.assertEqual(classes, {0: "MIX", 1: "PG", 2: "PL"})
            self.assertEqual(bb.split_of(tmp, []), {"0000": "train", "0001": "train",
                                                    "0002": "val"})
            self.assertEqual(bb.label_ids(tmp), ["0000", "0001", "0002"])

    def test_aggregate_sums_bands(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(tmp, classes=("MIX", "PG"))
            rows = [measure(tmp, "0000", cls="MIX"),
                    measure(tmp, "0001", cls="PG")]
            total, by_class, _ = bb.aggregate(rows, 30, 10)
            self.assertEqual(total["n_ids"], 2)
            self.assertEqual(total["n_pos"], 2)
            self.assertEqual(total["bands"]["3-10x0-3"]["total"], 2 * 112)
            self.assertEqual(by_class["PG"]["bands"]["3-10x0-3"]["total"], 112)
            self.assertEqual(sum(total["hist"]), total["n_bg"])

    def test_aggregate_sums_snr(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(tmp, classes=("MIX", "PG"), amp=0.5)
            rows = [measure(tmp, "0000", cls="MIX", noise_sd=0.5),
                    measure(tmp, "0001", cls="PG", noise_sd=0.5)]
            total, by_class, _ = bb.aggregate(rows, 30, 10)
            self.assertEqual(total["snr"]["band:3-10x0-3"], [192, 32, 0, 0, 0, 0])
            self.assertEqual(by_class["PG"]["snr"]["band:3-10x0-3"], [96, 16, 0, 0, 0, 0])
            # Sin `--intensity` no hay clave `snr` y el agregado no la inventa.
            plain = [measure(tmp, "0000", cls="MIX"), measure(tmp, "0001", cls="PG")]
            self.assertEqual(bb.aggregate(plain, 30, 10)[0]["snr"], {})


if __name__ == "__main__":
    unittest.main()
