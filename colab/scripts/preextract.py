"""Pre-extract a per-spectrum background POOL for training.

Runs the expensive split_2d ONCE per spectrum and stores a pool: ALL
positives (rows 0:n_pos) plus the background budget (pool_k * n_pos rows)
chosen with a FIXED --seed, plus the distance to the nearest peak of every
background row. Only the pool rows are ever computed (split_2d_rows), so
the 524288-row matrix of the reference implementation is never allocated
(1000 spectra: 219 s -> ~11-16 s, 1.39 GB -> 0.17 GB peak RAM).

--slow-split keeps the reference path (full split_2d + X[idx]) for
verification; --fast is accepted and ignored so the Colab notebook keeps
working. The two are mutually exclusive.

Training then subsamples that pool every epoch (train_model.subsample_pool
takes all positives + bg_ratio*n_pos background rows with a fresh rng), so
the per-epoch background variation is PRESERVED while split_2d is paid
only once. With pool_k > bg_ratio each epoch sees a different background
subset; with pool_k == bg_ratio the whole pool is consumed (equivalent to
the legacy fixed sampling, see tests/test_pool_equiv.py).

The background is NOT sampled uniformly: --hard-frac (0.25 by default)
reserves that share of the pool's background for negatives within --hard-px
px of a peak. Uniform sampling put the zeros inside a peak (the label is a
sub-pixel marker of 1/2/4 px on peaks of ~20-50 px) at 0.48 % of the pool:
per epoch and spectrum, 578 positives against ~11 negatives 1 px away, and
the trained model ended up giving P > 0.5 to 81 % of the pixels right next
to a peak (peak lists ~7.7x too long). Every background row stores its
distance in `d`, so the per-epoch share is tuned in train_model.py
(--hard-frac/--hard-px) without regenerating the pool. --hard-frac 0
reproduces the old uniform pool byte for byte (pinned by
tests/test_split_rows.py).

Why: BinSequence._points re-runs split_2d (~1-2 s/spectrum) EVERY epoch
-> 1000 spectra x 210 epochs ~= 60-110 h of CPU just recomputing the
identical X. Pre-extracting removes that entirely.

Output layout (<out>/):
  <id>.npz          X (M,128,1) float32, y (M,1) float32, n_pos, d (uint8,
                    distancia al pico más cercano de cada fila de fondo),
                    hard_px, H, W
  manifest.json     {source, seed, pool_k, hard_frac, hard_px, max_pts,
                     shape, train, val}
  preextract.log    CSV: id, split, M(pool rows), n_pos, hard, seconds

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
      --seed 0 --pool-k 10
  # then train (--bg-ratio controls the per-epoch subsampling):
  python3 scripts/train_model.py --data data/pre_elong --preextracted \\
      --bg-ratio 2 --lr 1e-4 ...

Memory: one spectrum -> only the pool rows are built (~6400 x 128 fp64
= 6.5 MB), then stored as ~3.3 MB fp32. Peak RAM < 0.2 GB.
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
from hard_negatives import (HARD_FRAC, HARD_PX, choose_background,  # noqa: E402
                            distance_to_positive, pool_sections)

try:
    from cross_extract_fast import split_2d_rows  # noqa: E402
    HAVE_FAST_SPLIT = True
except ImportError:  # pragma: no cover - scripts/ always ships the module
    HAVE_FAST_SPLIT = False

SHAPE = (256, 2048)
CROSS = np.array([64, 64])


def load_pair(ds_dir, sid):
    s_path = os.path.join(ds_dir, "spectra", f"{sid}.bin")
    l_path = os.path.join(ds_dir, "labels", f"{sid}.bin")
    S = np.fromfile(s_path, dtype="<f4").reshape(SHAPE).astype(np.float32)
    L = np.fromfile(l_path, dtype=np.uint8).reshape(SHAPE).astype(np.float32)
    return np.ascontiguousarray(S), np.ascontiguousarray(L)


def plan_pool_idx(L, pool_k, max_pts=None, rng=None, hard_frac=0.0,
                  hard_px=HARD_PX):
    """Rows that form the pool: all positives, then the background budget.

    Layout: [positivos | duros | fáciles]. El fondo se reparte entre negativos a
    <= `hard_px` px de un pico (`hard_frac` del presupuesto) y el resto. Antes
    era uniforme sobre todos los negativos, así que los ceros dentro del pico
    (la etiqueta es un marcador sub-píxel de 1/2/4 px sobre picos de ~20-50 px)
    entraban al 0,48 % del fondo: ~11 negativos a 1 px por cada 578 positivos y
    el modelo acababa dando P > 0,5 al 81 % de los vecinos inmediatos de un
    pico. Ver scripts/hard_negatives.py.

    Pure index planning, independent of X: this is what lets the caller
    compute only the selected cross sections instead of the whole
    (H*W, 128, 1) matrix. Returns (idx, n_pos); rows [0:n_pos] of `idx`
    are the positives and [n_pos:] the background pool.

    Con `hard_frac=0` (por defecto) el sorteo es EXACTAMENTE el histórico: una
    sola llamada a `rng.choice` sobre todos los negativos, en el mismo orden, así
    que el pool guardado para una semilla no cambia (tests/test_split_rows.py).
    """
    y = L.reshape(-1, 1)
    pos = np.where(y[:, 0] > 0.5)[0]
    n_pos = len(pos)
    neg = np.where(y[:, 0] < 0.5)[0]
    n_bg = min(len(neg), int(n_pos * pool_k))
    if max_pts is not None:
        n_bg = min(n_bg, max(0, max_pts - n_pos))
    n_hard, _ = pool_sections(n_bg, hard_frac)
    if hard_frac > 0 and n_hard > 0:
        d_neg = distance_to_positive(L).reshape(-1)[neg]
        bg = neg[choose_background(neg, d_neg, n_bg, hard_frac, hard_px, rng)]
    else:
        bg = rng.choice(neg, size=max(0, n_bg), replace=False)
    return np.concatenate([pos, bg]), n_pos


def build_pool(X, L, pool_k, max_pts=None, rng=None, hard_frac=0.0,
               hard_px=HARD_PX):
    """Pool por espectro: TODOS los positivos (filas 0:n_pos) + el fondo.

    El submuestreo por época (bg_ratio) lo hace train_model.subsample_pool
    sobre este pool, así el split_2d se paga UNA vez y el fondo sigue
    variando entre épocas mientras pool_k > bg_ratio.

    Devuelve (X_pool, y_pool, n_pos). Las filas [0:n_pos] son positivos,
    [n_pos:] es el pool de fondo.

    Wrapper fino sobre plan_pool_idx: se mantiene porque es la fuente única
    de verdad de la semántica del pool (tests/test_pool_equiv.py).
    """
    idx, n_pos = plan_pool_idx(L, pool_k, max_pts, rng,
                               hard_frac=hard_frac, hard_px=hard_px)
    return X[idx], L.reshape(-1, 1)[idx], n_pos


def main():
    ap = argparse.ArgumentParser(description="Pre-extract a background pool "
                                             "(positives + ring + bg).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", required=True,
                    help="manifest.json de Fase 1 con train/val (jres_ingest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="semilla para elegir QUE pixeles de fondo entran al pool")
    ap.add_argument("--hard-frac", type=float, default=HARD_FRAC,
                    help=f"fracción del fondo del pool reservada al anillo del "
                         f"pico (d <= --hard-px); {HARD_FRAC} por defecto, 0 = "
                         f"fondo uniforme como antes")
    ap.add_argument("--hard-px", type=int, default=HARD_PX,
                    help=f"radio en px del anillo considerado 'negativo duro' "
                         f"({HARD_PX})")
    ap.add_argument("--pool-k", type=float, default=10.0,
                    help="pool de fondo = pool-k x positivos (10 -> ~3.3 GB/1000; "
                         "debe ser > --bg-ratio de train para que el fondo varíe)")
    ap.add_argument("--max-pts", type=int, default=None,
                    help="tope opcional de puntos por espectro (acota el tamaño)")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--slow-split", action="store_true",
                       help="referencia: split_2d completo (matriz de 524288 "
                            "filas) + X[idx]; lento, solo para verificar")
    split.add_argument("--fast", action="store_true",
                       help="[no-op] antes elegía cross_extract_fast; el split "
                            "por filas ya es el camino por defecto")
    ap.add_argument("--compress", action="store_true",
                    help="np.savez_compressed (smaller, 26x slower write)")
    args = ap.parse_args()

    slow_split = args.slow_split or not HAVE_FAST_SPLIT
    if args.slow_split and not HAVE_FAST_SPLIT:
        print("aviso: cross_extract_fast no disponible", flush=True)
    elif not HAVE_FAST_SPLIT:
        print("aviso: cross_extract_fast no disponible -> split completo",
              flush=True)

    man = json.load(open(args.ids))
    ids = list(dict.fromkeys(man["train"] + man["val"]))
    split_of = {i: "train" for i in man["train"]}
    split_of.update({i: "val" for i in man["val"]})
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log = open(os.path.join(args.out, "preextract.log"), "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["id", "split", "M", "pos", "hard", "seconds"])
    t_all, done = time.time(), []
    for n, sid in enumerate(ids):
        t0 = time.time()
        S, L = load_pair(args.dataset, sid)
        # Plan the pool BEFORE splitting: split_2d does not touch the rng, so
        # the draw order (and therefore the pool) is unchanged.
        idx, n_pos = plan_pool_idx(L, args.pool_k, args.max_pts, rng,
                                   hard_frac=args.hard_frac, hard_px=args.hard_px)
        d_pool = distance_to_positive(L).reshape(-1)[idx]
        n_hard, _ = pool_sections(len(idx) - n_pos, args.hard_frac)
        if slow_split:
            # Reference path: full matrix, then keep the pool rows.
            X, _ = split_2d_loop(S.astype(np.float64), CROSS)
            Xs = np.asarray(X, dtype=np.float32)[idx]
        else:
            # Only the pool rows are ever built (524288 -> ~6400 rows).
            Xs, _ = split_2d_rows(S, idx, CROSS)
            Xs = np.asarray(Xs, dtype=np.float32)
        Xs[~np.isfinite(Xs)] = 0.0
        ys = np.ascontiguousarray(L.reshape(-1, 1)[idx], dtype=np.float32)
        saver = np.savez_compressed if args.compress else np.savez
        saver(os.path.join(args.out, f"{sid}.npz"),
              X=np.ascontiguousarray(Xs, dtype=np.float32),
              y=ys,
              n_pos=np.int32(n_pos),
              # distancia al positivo más cercano de cada fila de fondo: es lo que
              # permite minar el anillo por época sin volver a generar el pool
              d=np.ascontiguousarray(d_pool[n_pos:], dtype=np.uint8),
              hard_px=np.int32(args.hard_px),
              H=np.int32(SHAPE[0]), W=np.int32(SHAPE[1]))
        dt = time.time() - t0
        wr.writerow([sid, split_of[sid], len(Xs), n_pos, n_hard, f"{dt:.1f}"])
        log.flush()
        done.append(sid)
        if (n + 1) % 25 == 0 or n + 1 == len(ids):
            el = time.time() - t_all
            print(f"[{n + 1}/{len(ids)}] {sid} M={len(Xs)} "
                  f"{dt:.1f}s (elapsed {el / 60:.0f} min, "
                  f"ETA {(el / (n + 1) * (len(ids) - n - 1)) / 60:.0f} min)",
                  flush=True)
    out_man = {"source": os.path.abspath(args.dataset),
               "storage": "pre-extracted background pool (positives + ring + bg)",
               "seed": args.seed, "pool_k": args.pool_k,
               "hard_frac": args.hard_frac, "hard_px": args.hard_px,
               "max_pts": args.max_pts, "shape": list(SHAPE),
               "train": [i for i in man["train"] if i in done],
               "val": [i for i in man["val"] if i in done]}
    json.dump(out_man, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"Done: {len(out_man['train'])} train / {len(out_man['val'])} val "
          f"-> {args.out} ({(time.time() - t_all) / 3600:.2f} h)")
    print(f"fondo: {args.hard_frac * 100:.0f} % del pool reservado al anillo "
          f"(d <= {args.hard_px} px) · el submuestreo por época lo elige con "
          f"train_model.py --hard-frac (0 = fondo uniforme)")


if __name__ == "__main__":
    main()
