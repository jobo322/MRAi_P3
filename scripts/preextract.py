"""Pre-extract a per-spectrum background POOL for training.

Runs the expensive split_2d ONCE per spectrum (loop or --fast vectorised
version, bit-identical) and stores a pool: ALL positives (rows 0:n_pos)
plus pool_k * n_pos background pixels chosen with a FIXED --seed.

Training then subsamples that pool every epoch (train_model.subsample_pool
takes all positives + bg_ratio*n_pos background rows with a fresh rng), so
the per-epoch background variation is PRESERVED while split_2d is paid
only once. With pool_k > bg_ratio each epoch sees a different background
subset; with pool_k == bg_ratio the whole pool is consumed (equivalent to
the legacy fixed sampling, see tests/test_pool_equiv.py).

Why: BinSequence._points re-runs split_2d (~1-2 s/spectrum) EVERY epoch
-> 1000 spectra x 210 epochs ~= 60-110 h of CPU just recomputing the
identical X. Pre-extracting removes that entirely.

Output layout (<out>/):
  <id>.npz          X (M,128,1) float32, y (M,1) float32, n_pos, H, W
  manifest.json     {source, seed, pool_k, max_pts, shape, train, val}
  preextract.log    CSV: id, split, M(pool rows), n_pos, seconds

M is the POOL size, not the per-epoch sample: M = n_pos + pool_k*n_pos.
For the jres dataset (n_pos ~= 581) pool_k=10 -> M ~= 6391, ~3.3 MB per
spectrum, ~3.3 GB for 1000 spectra. Sizes measured:

  pool_k   M pool   MB/spectrum   1000 spectra
    2       1743       0.90          0.90 GB
    5       3486       1.80          1.80 GB
   10       6391       3.30          3.30 GB
   20      12201       6.30          6.30 GB
   50      29631      15.29         15.29 GB

Storage is uncompressed npz on purpose: float32 spectra do not
compress (280 vs 300 MB) and compression costs 26x on write and
12x on read (measured: 4.24 vs 0.37 ms/spectrum). --compress is
available only if you must shrink the upload.

Usage:
  python3 scripts/preextract.py --dataset .../dataset \\
      --ids data/train_2d_elong/manifest.json --out data/pre_elong \\
      --seed 0 --pool-k 10 --fast
  # then train (--bg-ratio controls the per-epoch subsampling):
  python3 scripts/train_model.py --data data/pre_elong --preextracted \\
      --bg-ratio 2 --lr 1e-4 ...

Memory: one spectrum -> X complete (524288,128,1) float64 = 512 MB
transient, then only the pool kept (~3.3 MB fp32 with pool_k=10). Runs
spectrum-by-spectrum, peak RAM < 1.5 GB.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from cross_extract import split_2d as split_2d_loop  # noqa: E402

SHAPE = (256, 2048)
CROSS = np.array([64, 64])


def load_pair(ds_dir, sid):
    s_path = os.path.join(ds_dir, "spectra", f"{sid}.bin")
    l_path = os.path.join(ds_dir, "labels", f"{sid}.bin")
    S = np.fromfile(s_path, dtype="<f4").reshape(SHAPE).astype(np.float32)
    L = np.fromfile(l_path, dtype=np.uint8).reshape(SHAPE).astype(np.float32)
    return np.ascontiguousarray(S), np.ascontiguousarray(L)


def build_pool(X, L, pool_k, max_pts=None, rng=None):
    """Pool por espectro: TODOS los positivos (filas 0:n_pos) + pool_k*pos
    píxeles de fondo elegidos una sola vez.

    El submuestreo por época (bg_ratio) lo hace train_model.subsample_pool
    sobre este pool, así el split_2d se paga UNA vez y el fondo sigue
    variando entre épocas mientras pool_k > bg_ratio.

    Devuelve (X_pool, y_pool, n_pos). Las filas [0:n_pos] son positivos,
    [n_pos:] es el pool de fondo.
    """
    y = L.reshape(-1, 1)
    pos = np.where(y[:, 0] > 0.5)[0]
    n_pos = len(pos)
    neg = np.where(y[:, 0] < 0.5)[0]
    n_bg = min(len(neg), int(n_pos * pool_k))
    if max_pts is not None:
        n_bg = min(n_bg, max(0, max_pts - n_pos))
    bg = rng.choice(neg, size=max(0, n_bg), replace=False)
    idx = np.concatenate([pos, bg])
    return X[idx], y[idx], n_pos


def main():
    ap = argparse.ArgumentParser(description="Pre-extract a background pool "
                                             "(positives + pool_k*pos bg).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", required=True,
                    help="manifest.json de Fase 1 con train/val (jres_ingest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="semilla para elegir QUE pixeles de fondo entran al pool")
    ap.add_argument("--pool-k", type=float, default=10.0,
                    help="pool de fondo = pool-k x positivos (10 -> ~3.3 GB/1000; "
                         "debe ser > --bg-ratio de train para que el fondo varíe)")
    ap.add_argument("--max-pts", type=int, default=None,
                    help="tope opcional de puntos por espectro (acota el tamaño)")
    ap.add_argument("--fast", action="store_true",
                    help="use cross_extract_fast (bit-identical, ~6x)")
    ap.add_argument("--compress", action="store_true",
                    help="np.savez_compressed (smaller, 26x slower write)")
    args = ap.parse_args()

    split_2d = split_2d_loop
    if args.fast:
        from cross_extract_fast import split_2d_fast
        split_2d = split_2d_fast

    man = json.load(open(args.ids))
    ids = list(dict.fromkeys(man["train"] + man["val"]))
    split_of = {i: "train" for i in man["train"]}
    split_of.update({i: "val" for i in man["val"]})
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log = open(os.path.join(args.out, "preextract.log"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["id", "split", "M", "pos", "seconds"])
    t_all, done = time.time(), []
    for n, sid in enumerate(ids):
        t0 = time.time()
        S, L = load_pair(args.dataset, sid)
        X, _ = split_2d(S.astype(np.float64), CROSS)
        X = np.asarray(X, dtype=np.float32)
        X[~np.isfinite(X)] = 0.0
        Xs, ys, n_pos = build_pool(X, L, args.pool_k, args.max_pts, rng)
        del X
        saver = np.savez_compressed if args.compress else np.savez
        saver(os.path.join(args.out, f"{sid}.npz"),
              X=np.ascontiguousarray(Xs, dtype=np.float32),
              y=np.ascontiguousarray(ys, dtype=np.float32),
              n_pos=np.int32(n_pos),
              H=np.int32(SHAPE[0]), W=np.int32(SHAPE[1]))
        dt = time.time() - t0
        wr.writerow([sid, split_of[sid], len(Xs), n_pos, f"{dt:.1f}"])
        log.flush()
        done.append(sid)
        if (n + 1) % 25 == 0 or n + 1 == len(ids):
            el = time.time() - t_all
            print(f"[{n + 1}/{len(ids)}] {sid} M={len(Xs)} "
                  f"{dt:.1f}s (elapsed {el / 60:.0f} min, "
                  f"ETA {(el / (n + 1) * (len(ids) - n - 1)) / 60:.0f} min)",
                  flush=True)
    out_man = {"source": os.path.abspath(args.dataset),
               "storage": "pre-extracted background pool (positives + pool_k*pos bg)",
               "seed": args.seed, "pool_k": args.pool_k,
               "max_pts": args.max_pts, "shape": list(SHAPE),
               "train": [i for i in man["train"] if i in done],
               "val": [i for i in man["val"] if i in done]}
    json.dump(out_man, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"Done: {len(out_man['train'])} train / {len(out_man['val'])} val "
          f"-> {args.out} ({(time.time() - t_all) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
