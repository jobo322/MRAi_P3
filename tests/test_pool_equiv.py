"""Pool-mode equivalence tests.

Verifies the invariant that makes the background-pool design safe:

  * pool_k == bg_ratio  -> subsampling the pool yields EXACTLY the same
    set of rows as the legacy fixed sampling (no data change).
  * pool_k > bg_ratio   -> all positives are always present and the
    background subset is a proper subset of the pool, different per epoch
    (variation preserved), while the sample SIZE stays constant.

Pure NumPy: train_model is imported at module level (TF is imported lazily
inside its functions), so no TensorFlow is required here.
"""
import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from preextract import build_pool  # noqa: E402
from train_model import subsample_pool  # noqa: E402


def _fake_spectrum(h=64, w=64, n_pos=10, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((h, w))
    L = np.zeros((h, w), dtype=np.float32)
    flat = rng.choice(h * w, size=n_pos, replace=False)
    L.reshape(-1)[flat] = 1.0
    X = rng.standard_normal((h * w, 128, 1)).astype(np.float32)
    return X, L


def _rows(X, y):
    """Order-independent fingerprint of a sample."""
    return sorted(map(tuple, np.concatenate([X.reshape(len(X), -1), y],
                                            axis=1).tolist()))


class TestPoolEquiv(unittest.TestCase):
    def test_pool_k_equals_bg_ratio_is_legacy_equivalent(self):
        """pool_k == bg_ratio -> same rows as the legacy fixed sample."""
        X, L = _fake_spectrum()
        max_pts = 1 << 20
        k = 4.0
        pool_x, pool_y, n_pos = build_pool(X, L, k, max_pts,
                                           np.random.default_rng(0))
        sub_x, sub_y = subsample_pool(pool_x, pool_y, n_pos, max_pts, k,
                                      np.random.default_rng(1))
        # taking bg_ratio == pool_k consumes the whole background pool
        self.assertEqual(len(sub_x), len(pool_x))
        self.assertEqual(_rows(sub_x, sub_y), _rows(pool_x, pool_y))

    def test_pool_k_gt_bg_ratio_keeps_positives_and_varies(self):
        """pool_k > bg_ratio -> positives intact, size fixed, bg differs."""
        X, L = _fake_spectrum()
        max_pts = 1 << 20
        pool_x, pool_y, n_pos = build_pool(X, L, 10.0, max_pts,
                                           np.random.default_rng(0))
        bg_ratio = 2.0
        s1 = subsample_pool(pool_x, pool_y, n_pos, max_pts, bg_ratio,
                            np.random.default_rng(1))
        s2 = subsample_pool(pool_x, pool_y, n_pos, max_pts, bg_ratio,
                            np.random.default_rng(2))
        exp = n_pos + int(n_pos * bg_ratio)
        self.assertEqual(len(s1[0]), exp)
        self.assertEqual(len(s2[0]), exp)
        # every positive is present in both
        self.assertEqual(int(s1[1].sum()), n_pos)
        self.assertEqual(int(s2[1].sum()), n_pos)
        # background differs between epochs
        self.assertNotEqual(_rows(s1[0], s1[1]), _rows(s2[0], s2[1]))

    def test_sample_size_matches_legacy_formula(self):
        """Same size as the on-demand path for identical bg_ratio/max_pts."""
        X, L = _fake_spectrum()
        n_pos = int(L.sum())
        for max_pts, bg in [(1 << 20, 2.0), (1 << 20, 4.0), (n_pos + 5, 4.0)]:
            pool_x, pool_y, np_ = build_pool(X, L, 10.0, max_pts,
                                             np.random.default_rng(0))
            s_x, _ = subsample_pool(pool_x, pool_y, np_, max_pts, bg,
                                    np.random.default_rng(1))
            expected = n_pos + min(len(pool_x) - n_pos, int(n_pos * bg),
                                   max_pts - n_pos)
            self.assertEqual(len(s_x), expected, (max_pts, bg))


if __name__ == "__main__":
    unittest.main()
