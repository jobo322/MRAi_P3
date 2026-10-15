"""Vectorised cross-section extraction — drop-in replacement for
cross_extract.split_2d / StoP3.split_2D.

Same output contract: X (H*W,128,1) L2-normalised per vector,
Norm (H*W,), wrap padding. The two cross arms are sliding windows
over the padded spectrum, so the per-pixel Python loop
(para_split_2d: 524k iterations/spectrum) becomes two strided
assignments. Speedup ~10-50x; output must satisfy
maxdiff == 0.0 vs the loop version (see tests/test_cross_fast.py).

split_2d_rows(S, idx, Cross_Size) is the sparse counterpart: it returns
only the rows in `idx` and is exactly equivalent to
split_2d(S, Cross_Size)[idx] in both X and Norm, without ever allocating
the full (H*W, 128, 1) matrix. Callers that keep ~1% of the rows (the
training pool) should use it; see tests/test_split_rows.py.

Reference: scripts/cross_extract.py (mirror of
src/MRAi_P3/StoP3.py lines 10-26). Any change in StoP3 must be
mirrored in BOTH cross_extract.py and this module.
"""
import numpy as np


def split_2d_fast(S, Cross_Size):
    S = np.asarray(S)
    c0, c1 = int(Cross_Size[0]), int(Cross_Size[1])
    H, W = S.shape
    p0a, p0b = c0 // 2, int(c0 / 2 - 1 / 2)
    p1a, p1b = c1 // 2, int(c1 / 2 - 1 / 2)
    S_pad = np.pad(S, ((p0a, p0b), (p1a, p1b)), "wrap")
    # F1 arm: for pixel (i,j) rows i..i+c0-1 of column j+c1//2.
    # S_pad[:, p1a:p1a+W] starts at column c1//2, so a sliding window
    # of c0 rows gives col[i,j,k] = S_pad[i+k, j+c1//2]. Exact match.
    col = np.lib.stride_tricks.sliding_window_view(
        S_pad[:, p1a:p1a + W], window_shape=c0, axis=0)
    # F2 arm: for pixel (i,j) cols j..j+c1-1 of row i+c0//2.
    # S_pad[p0a:p0a+H, :] starts at row c0//2, so a sliding window
    # of c1 cols gives row[i,j,k] = S_pad[i+c0//2, j+k]. Exact match.
    row = np.lib.stride_tricks.sliding_window_view(
        S_pad[p0a:p0a + H, :], window_shape=c1, axis=1)
    S_data = np.empty((H * W, c0 + c1, 1), dtype=np.float64)
    S_data[:, :c0, 0] = col.reshape(H * W, c0)
    S_data[:, c0:, 0] = row.reshape(H * W, c1)
    Norm = np.linalg.norm(S_data, axis=(1, 2))
    S_data = S_data / Norm[:, None, None]
    return S_data, Norm


def split_2d_rows(S, idx, Cross_Size):
    """Cross sections for a subset of pixels only.

    Contract: identical to ``split_2d(S, Cross_Size)[idx]`` for both the
    normalised rows X (len(idx), c0+c1, 1) and the row norms Norm
    (len(idx),), bit for bit. Norm is the norm of the UN-normalised row,
    exactly as in split_2d / split_2d_fast.

    `idx` is a flat row-major pixel index (i * W + j); it may be a list or
    an array, any dtype, unsorted, or empty. Out-of-range values raise
    IndexError. Rows whose norm is 0 come out as NaN in X, which callers
    already sanitise with ``X[~np.isfinite(X)] = 0.0``.

    The sliding windows are views, so indexing them with (ii, jj) copies
    only the requested rows. Do NOT reshape them to (H*W, c) first: that
    materialises the whole matrix, which is the cost this avoids.
    """
    S = np.asarray(S).astype(np.float64)
    idx = np.asarray(idx)
    c0, c1 = int(Cross_Size[0]), int(Cross_Size[1])
    H, W = S.shape
    p0a, p0b = c0 // 2, int(c0 / 2 - 1 / 2)
    p1a, p1b = c1 // 2, int(c1 / 2 - 1 / 2)
    S_pad = np.pad(S, ((p0a, p0b), (p1a, p1b)), "wrap")
    # Same two arms as split_2d_fast, but indexed sparsely. The views are
    # already bounded to H x W, so ii/jj need no % H / % W.
    col = np.lib.stride_tricks.sliding_window_view(
        S_pad[:, p1a:p1a + W], window_shape=c0, axis=0)
    row = np.lib.stride_tricks.sliding_window_view(
        S_pad[p0a:p0a + H, :], window_shape=c1, axis=1)
    ii, jj = idx // W, idx % W
    out = np.empty((len(idx), c0 + c1), dtype=np.float64)
    out[:, :c0] = col[ii, jj]
    out[:, c0:] = row[ii, jj]
    norm = np.linalg.norm(out, axis=1)
    out /= norm[:, None]
    return out.reshape(len(idx), c0 + c1, 1), norm
