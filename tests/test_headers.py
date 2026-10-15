"""Tests for the nmrglue fallback header parsing in headers.py."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # tests/
REPO_ROOT = os.path.dirname(BASE_DIR)                   # /Users/abolanos/git/mrai_p3
SRC_DIR = os.path.join(REPO_ROOT, "src", "MRAi_P3")
SPECTRA_DIR = os.path.join(REPO_ROOT, "Spectra", "2D")

# Add repo root to path so relative imports inside the package work
sys.path.insert(0, REPO_ROOT)

# ── Lazy imports that avoid the StoP3 (tensorflow) import ──
# Register a synthetic package so that `from . import headers` inside
# P5toPeaks / PeakstoTab resolves to our MRP.* modules.
_pkg = types.ModuleType("MRP")
_pkg.__path__ = [SRC_DIR]
sys.modules["MRP"] = _pkg

def load(name: str):
    """Load a single module by file path under the synthetic MRP package."""
    spec = importlib.util.spec_from_file_location(
        f"MRP.{name}", os.path.join(SRC_DIR, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"MRP.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_file(path: str):
    """Return the path if it exists, else raise a SkipError."""
    if not os.path.isfile(path):
        raise unittest.SkipTest(f"Missing test file: {path}")
    return path


# ── Spectra ──
AZURIN_FT2 = os.path.join(SPECTRA_DIR, "Azurin.ft2")
AZURIN_P5 = os.path.join(SPECTRA_DIR, "Azurin.P5")
TAU_FT2 = os.path.join(SPECTRA_DIR, "Tau.ft2")
UBIQUITIN_FT2 = os.path.join(SPECTRA_DIR, "Ubiquitin.ft2")
MALT_FT2 = os.path.join(SPECTRA_DIR, "Malt.ft2")


# ── Load the modules we need ──
headers = load("headers")

class TestHeadersFallback(unittest.TestCase):
    """Tests that force the nmrglue fallback path (no showhdr binary)."""

    def setUp(self):
        # Force fallback by monkey-patching subprocess.check_output on the headers module.
        # This makes read_headers take the FileNotFoundError branch without NMRPipe.
        self._orig = headers.subprocess.check_output
        def _raise_file_not_found(*a, **kw):
            raise FileNotFoundError("showhdr not installed (test forced)")
        headers.subprocess.check_output = _raise_file_not_found

    def tearDown(self):
        headers.subprocess.check_output = self._orig

    # ── Azurin tests ──
    def test_azurin_nd_and_nam(self):
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        self.assertEqual(ND, 2)
        self.assertEqual(nam, {1: "N", 2: "HN"})

    def test_azurin_ppm_limits(self):
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        # Authoritative nmrglue ppm_limits() reference values
        self.assertAlmostEqual(pfst[1], 138.012, places=2)  # 15N upper
        self.assertAlmostEqual(plst[1], 98.154, places=2)     # 15N lower
        self.assertAlmostEqual(pfst[2], 10.998, places=2)     # 1H upper
        self.assertAlmostEqual(plst[2], 5.999, places=2)      # 1H lower

    def test_azurin_sizes(self):
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        self.assertEqual(sz[1], 256)
        self.assertEqual(sz[2], 1279)  # REAL FDSIZE, NOT stale FTSIZE=4096

    def test_azurin_sw_positive(self):
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        for dim, value in sw.items():
            self.assertGreater(value, 0, f"sw[{dim}] should be positive, got {value}")

    # ── Tau FT2 ──
    def test_tau_sizes(self):
        tau_path = _ensure_file(TAU_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(tau_path)
        self.assertEqual(ND, 2)
        # Guard the stale FDF2FTSIZE (4096) regression
        self.assertEqual(sz[1], 256)
        self.assertEqual(sz[2], 1233)

    # ── Ubiquitin FT2 ──
    def test_ubiquitin_sizes(self):
        ubi_path = _ensure_file(UBIQUITIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(ubi_path)
        self.assertEqual(ND, 2)
        self.assertGreater(sz[2], 1)

    # ── Malt FT2 ──
    def test_malt_sizes(self):
        malt_path = _ensure_file(MALT_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(malt_path)
        self.assertEqual(ND, 2)
        self.assertEqual(sz[2], 805)  # Guard the stale FDF2FTSIZE regression

    # ── Regression guard: old Hz-as-ppm failure cannot come back ──
    def test_regression_ppm_not_hz(self):
        """pfst[1] must be < 1000 (15N ppm range), not a Hz-like value > 1000."""
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        self.assertLess(pfst[1], 1000,
            f"Regression: pfst[1] = {pfst[1]} exceeds 1000 Hz-as-ppm threshold")
        self.assertGreater(pfst[2], 5,
            f"Regression: pfst[2] = {pfst[2]} below 1H ppm range (expected ~10.998)")

    # ── Regression guard: plst < pfst (upper > lower) ──
    def test_regression_plst_less_than_pfst(self):
        """ppm limits must have plst < pfst (last < first)."""
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        self.assertLess(plst[1], pfst[1],
            f"Regression: plst[1]={plst[1]} >= pfst[1]={pfst[1]} (last >= first)")

    # ── end-to-end: P5→peaks→tab with valid ppm / indices ──
    def test_e2e_p5_to_tab(self):
        """Run P5→peaks→tab and validate ppm, axis labels and point indices.

        Convention: X is the 1H (direct) axis = NMRPipe dim 2, Y is the 15N axis
        = dim 1. P5toPeaks emits coordinates as (F2, F1) in 2D, so .tab must
        calibrate X against dim 2 and Y against dim 1.
        """
        peaksto_peaks = load("P5toPeaks")
        peaksto_tab = load("PeakstoTab")

        azurin_path = _ensure_file(AZURIN_FT2)
        azurin_p5 = _ensure_file(AZURIN_P5)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)
        self.assertEqual(nam, {1: "N", 2: "HN"})

        with tempfile.TemporaryDirectory() as tmpdir:
            peaks_file = os.path.join(tmpdir, "Azurin.peaks")
            peaksto_peaks.main(azurin_p5, spec_path=azurin_path, out=peaks_file)
            self.assertTrue(os.path.isfile(peaks_file), "PeakstoPeaks did not write peaks file")

            tab_file = os.path.join(tmpdir, "Azurin.tab")
            peaksto_tab.main(peaks_file, spec_path=azurin_path, out=tab_file)
            self.assertTrue(os.path.isfile(tab_file), "PeakstoTab did not write tab file")

            with open(tab_file) as f:
                lines = f.readlines()

            # ── .peaks: labels must match the emitted coordinate order ──
            with open(peaks_file) as f:
                pk = f.readlines()
            iname = [l.split() for l in pk if l.startswith("#INAME")]
            self.assertEqual(len(iname), ND)
            # emitted column 1 is F2 (1H), column 2 is F1 (15N)
            self.assertEqual(iname[0][2], "HN")
            self.assertEqual(iname[1][2], "N")

            # ── .tab: DATA lines must declare X=1H, Y=15N ──
            data = [l.split() for l in lines if l.startswith("DATA")]
            self.assertEqual(len(data), 2)
            x_ax, y_ax = data[0], data[1]
            self.assertEqual(x_ax[1], "X_AXIS")
            self.assertEqual(x_ax[2], "HN")            # 1H first
            self.assertEqual(int(x_ax[4]), int(sz[2]) + 2)
            self.assertAlmostEqual(float(x_ax[5].rstrip("ppm")), pfst[2], places=2)
            self.assertEqual(y_ax[1], "Y_AXIS")
            self.assertEqual(y_ax[2], "N")             # 15N second
            self.assertEqual(int(y_ax[4]), int(sz[1]) + 2)
            self.assertAlmostEqual(float(y_ax[5].rstrip("ppm")), pfst[1], places=2)

            # ── .tab: resolve column indices from VARS, not hardcoded ──
            vars_line = None
            for l in lines:
                if l.startswith("VARS   "):
                    vars_line = l
                    break
            self.assertIsNotNone(vars_line, "No VARS line found in tab")

            v_list = vars_line.strip().replace("VARS   ", "").split()
            idx_X1, idx_X3 = v_list.index("X1"), v_list.index("X3")
            idx_Y1, idx_Y3 = v_list.index("Y1"), v_list.index("Y3")
            col_XPPM, col_YPPM = v_list.index("X_PPM"), v_list.index("Y_PPM")

            # Whitespace-tolerant row filter: FORMAT %5d left-pads, so l[:1]
            # is a space for small indices and a naive filter matches nothing.
            rows = [l.split() for l in lines
                    if l.strip() and l.strip()[0].isdigit() and len(l.split()) == len(v_list)]
            self.assertGreater(len(rows), 0, "No data rows parsed from tab")

            for parts in rows:
                x1, x3 = int(parts[idx_X1]), int(parts[idx_X3])
                y1, y3 = int(parts[idx_Y1]), int(parts[idx_Y3])
                # X -> 1H axis (dim 2), Y -> 15N axis (dim 1)
                self.assertGreaterEqual(x1, 1, f"X1={x1} below 1")
                self.assertLessEqual(x3, int(sz[2]), f"X3={x3} exceeds sz[2]={sz[2]}")
                self.assertGreaterEqual(y1, 1, f"Y1={y1} below 1")
                self.assertLessEqual(y3, int(sz[1]), f"Y3={y3} exceeds sz[1]={sz[1]}")
                # X_PPM is a 1H value, Y_PPM is a 15N value
                x_ppm, y_ppm = float(parts[col_XPPM]), float(parts[col_YPPM])
                self.assertGreaterEqual(x_ppm, plst[2] - 0.5, f"X_PPM={x_ppm} below 1H range")
                self.assertLessEqual(x_ppm, pfst[2] + 0.5, f"X_PPM={x_ppm} above 1H range")
                self.assertGreaterEqual(y_ppm, plst[1] - 0.5, f"Y_PPM={y_ppm} below 15N range")
                self.assertLessEqual(y_ppm, pfst[1] + 0.5, f"Y_PPM={y_ppm} above 15N range")

            # ── .peaks/.tab row consistency: first emitted peak ──
            first = rows[0]
            x_ppm, y_ppm = float(first[col_XPPM]), float(first[col_YPPM])
            # X_PPM in .tab is column 2 of the .peaks row -> belongs to dim 2
            self.assertGreaterEqual(x_ppm, plst[2], f"peaks col2 {x_ppm} below plst[2]={plst[2]}")
            self.assertLessEqual(x_ppm, pfst[2], f"peaks col2 {x_ppm} above pfst[2]={pfst[2]}")
            # Y_PPM in .tab is column 3 of the .peaks row -> belongs to dim 1
            self.assertGreaterEqual(y_ppm, plst[1], f"peaks col3 {y_ppm} below plst[1]={plst[1]}")
            self.assertLessEqual(y_ppm, pfst[1], f"peaks col3 {y_ppm} above pfst[1]={pfst[1]}")

            # The same values must be what the .peaks file actually emitted
            pk_rows = [l.split() for l in pk
                       if l.strip() and l.strip()[0].isdigit()]
            self.assertGreater(len(pk_rows), 0, "No peak rows parsed from .peaks")
            self.assertAlmostEqual(float(pk_rows[0][1]), x_ppm, places=3)
            self.assertAlmostEqual(float(pk_rows[0][2]), y_ppm, places=3)

    # ── Run with pytest --collect only (quick smoke test) ──
    def test_headers_module_loads(self):
        """Verify the headers module imports cleanly without tensorflow."""
        self.assertTrue(hasattr(headers, "read_headers"))
        self.assertTrue(callable(headers.read_headers))


class TestHeadersShowhdrRoute(unittest.TestCase):
    """Tests for the `showhdr -verb` primary route (docs follow-up #4).

    These only run when NMRPipe's `showhdr` is reachable on PATH; otherwise
    every test is skipped. The critical invariant is that both routes return
    the *same* canonical six-tuple, because showhdr prints its axis columns in
    display order (X, Y == FDF2, FDF1) whereas the nmrglue fallback keys by
    NMRPipe dimension. A mismatch silently transposes .peaks/.tab output.
    """

    SPECTRA = [AZURIN_FT2, TAU_FT2, UBIQUITIN_FT2, MALT_FT2]

    @staticmethod
    def _showhdr_available():
        return shutil.which("showhdr") is not None

    @staticmethod
    def _forced_fallback(path):
        """Read headers through the nmrglue fallback regardless of PATH."""
        orig = headers.subprocess.check_output

        def _raise(*a, **kw):
            raise FileNotFoundError("showhdr hidden (test forced)")

        headers.subprocess.check_output = _raise
        try:
            return headers.read_headers(path)
        finally:
            headers.subprocess.check_output = orig

    def setUp(self):
        if not self._showhdr_available():
            raise unittest.SkipTest("showhdr not on PATH (NMRPipe not installed)")

    def test_routes_agree_on_every_spectrum(self):
        """showhdr and nmrglue fallback must return the identical tuple."""
        for path in self.SPECTRA:
            if not os.path.isfile(path):
                continue
            with self.subTest(spectrum=os.path.basename(path)):
                primary = headers.read_headers(path)
                fallback = self._forced_fallback(path)

                self.assertEqual(primary[0], fallback[0], "ND differs")
                self.assertEqual(primary[5], fallback[5], "axis names differ")
                # Sizes are integers, so they must match exactly.
                self.assertEqual(primary[1], fallback[1], "sz differs")
                # showhdr prints SW/PPM to 3 decimals; nmrglue keeps full
                # float precision, so allow the print round-off (<=1e-3).
                for idx, label in ((2, "sw"), (3, "pfst"), (4, "plst")):
                    for dim in primary[idx]:
                        self.assertAlmostEqual(
                            primary[idx][dim], fallback[idx][dim], delta=1e-3,
                            msg=f"{label}[{dim}] differs between routes")

    def test_showhdr_keeps_nmrpipe_dim_keying(self):
        """Dim 1 must stay the indirect (15N) axis, dim 2 the direct (1H).

        Guards the transposition: showhdr's first printed column is the X/1H
        axis, which is NMRPipe dim **2**, not dim 1.
        """
        azurin_path = _ensure_file(AZURIN_FT2)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        self.assertEqual(ND, 2)
        self.assertEqual(nam, {1: "N", 2: "HN"})
        self.assertEqual(sz[1], 256)     # 15N indirect
        self.assertEqual(sz[2], 1279)    # 1H direct
        self.assertAlmostEqual(pfst[1], 138.012, places=2)
        self.assertAlmostEqual(pfst[2], 10.998, places=2)

    def test_e2e_p5_to_tab_with_showhdr(self):
        """End-to-end P5→peaks→tab must yield in-range indices via showhdr."""
        peaksto_peaks = load("P5toPeaks")
        peaksto_tab = load("PeakstoTab")

        azurin_path = _ensure_file(AZURIN_FT2)
        azurin_p5 = _ensure_file(AZURIN_P5)
        ND, sz, sw, pfst, plst, nam = headers.read_headers(azurin_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            peaks_file = os.path.join(tmpdir, "Azurin.peaks")
            peaksto_peaks.main(azurin_p5, spec_path=azurin_path, out=peaks_file)
            tab_file = os.path.join(tmpdir, "Azurin.tab")
            peaksto_tab.main(peaks_file, spec_path=azurin_path, out=tab_file)

            with open(tab_file) as f:
                lines = f.readlines()
            with open(peaks_file) as f:
                pk = f.readlines()

            iname = [l.split() for l in pk if l.startswith("#INAME")]
            self.assertEqual(iname[0][2], "HN")
            self.assertEqual(iname[1][2], "N")

            data = [l.split() for l in lines if l.startswith("DATA")]
            self.assertEqual(data[0][2], "HN")
            self.assertEqual(data[1][2], "N")

            vars_line = next((l for l in lines if l.startswith("VARS   ")), None)
            self.assertIsNotNone(vars_line, "No VARS line found in tab")
            v_list = vars_line.strip().replace("VARS   ", "").split()
            idx_X1, idx_Y1 = v_list.index("X1"), v_list.index("Y1")
            idx_X3, idx_Y3 = v_list.index("X3"), v_list.index("Y3")

            rows = [l.split() for l in lines
                    if l.strip() and l.strip()[0].isdigit() and len(l.split()) == len(v_list)]
            self.assertGreater(len(rows), 0, "No data rows parsed from tab")

            for parts in rows:
                x1, x3 = int(parts[idx_X1]), int(parts[idx_X3])
                y1, y3 = int(parts[idx_Y1]), int(parts[idx_Y3])
                self.assertGreaterEqual(x1, 1, f"X1={x1} below 1")
                self.assertLessEqual(x3, int(sz[2]), f"X3={x3} exceeds sz[2]={sz[2]}")
                self.assertGreaterEqual(y1, 1, f"Y1={y1} below 1")
                self.assertLessEqual(y3, int(sz[1]), f"Y3={y3} exceeds sz[1]={sz[1]}")


if __name__ == "__main__":
    unittest.main()