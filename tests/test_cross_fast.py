"""Parity + throughput tests for cross_extract_fast vs cross_extract.

- test_parity_zero_diff: split_2d_fast == split_2d bit-exact (maxdiff 0.0)
  on random spectra incl. edge shapes (wrap padding paths).
- test_speedup: fast version is not slower than the loop version.
"""
import os
import sys
import time
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from cross_extract import split_2d
from cross_extract_fast import split_2d_fast


class TestCrossFast(unittest.TestCase):
    def test_parity_zero_diff(self):
        rng = np.random.default_rng(0)
        for shape in [(256, 512), (256, 2048), (64, 65)]:
            S = rng.standard_normal(shape)
            for cross in [np.array([64, 64])]:
                X1, N1 = split_2d(S, cross)
                X2, N2 = split_2d_fast(S, cross)
                self.assertEqual(X1.shape, X2.shape)
                self.assertEqual(float(np.max(np.abs(X1 - X2))), 0.0)
                self.assertEqual(float(np.max(np.abs(N1 - N2))), 0.0)

    def test_speedup(self):
        rng = np.random.default_rng(1)
        S = rng.standard_normal((256, 2048))
        cross = np.array([64, 64])
        split_2d_fast(S, cross)  # warm-up (import/allocation)
        t0 = time.time()
        split_2d(S, cross)
        t_loop = time.time() - t0
        t0 = time.time()
        split_2d_fast(S, cross)
        t_fast = time.time() - t0
        print(f"\nloop {t_loop:.2f}s fast {t_fast:.2f}s "
              f"speedup {t_loop / t_fast:.1f}x")
        self.assertLessEqual(t_fast, t_loop)


if __name__ == "__main__":
    unittest.main()
