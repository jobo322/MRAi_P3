"""Pre-extract fixed (X, y) training pairs per spectrum (one seed).

Runs the expensive split_2d ONCE per spectrum (loop or --fast
vectorised version, bit-identical) plus the BinSequence sampling
(positives + bg_ratio background, max_pts cap) with a FIXED seed,
and stores per-spectrum .npz files + a manifest. Training then
reads contiguous batches (no split, no sampling per epoch).

Why fixed: BinSequence._points re-runs split_2d (~1-2 s/spectrum)
EVERY epoch -> 1000 spectra x 210 epochs ~= 60-110 h of CPU just
recomputing the identical X. The per-epoch rng sampling variation
is frozen by --seed; compensate with global shuffle each epoch
(train_model --shuffle-buffer, cheap) — see COLAB.md.

Output layout (<out>/):
  <id>.npz          X (M,128,1) float32, y (M,1) float32, H, W
  manifest.json    {source, seed, bg_ratio, max_pts, shape, train, val}
  preextract.log   CSV: id, split, M(points), pos, seconds

M is NOT max_pts: sampling takes ALL positives (~580) plus
bg_ratio*positives of background, so M ~= pos*(1+bg_ratio) ~= 2900
for the jres dataset. 1000 spectra -> ~1.5 GB on disk.

Storage is uncompressed npz on purpose: float32 spectra do not
compress (280 vs 300 MB) and compression costs 26x on write and
12x on read (measured: 4.24 vs 0.37 ms/spectrum). --compress is
available only if you must shrink the upload.

Usage:
  python3 scripts/preextract.py --dataset .../dataset \\
      --ids data/train_2d_elong/manifest.json --out /tmp/pre \\
      --seed 0 --fast
  # then train:
  python3 scripts/train_model.py --data /tmp/pre --preextracted --lr 1e-4 ...

Memory: one spectrum -> X complete (524288,128,1) float64 = 512 MB
transient, then only the M sample kept (~1.5 MB fp32). Runs
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


def sample_points(X, L, max_pts, bg_ratio, rng):
    y = L.reshape(-1, 1)
    pos = np.where(y[:, 0] > 0.5)[0]
    n_bg = min(len(y) - len(pos), int(len(pos) * bg_ratio),
               max_pts - len(pos))
    neg = np.where(y[:, 0] < 0.5)[0]
    take_neg = rng.choice(neg, size=max(0, n_bg), replace=False)
    idx = np.concatenate([pos, take_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser(description="Pre-extract fixed (X,y)/spectrum.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ids", required=True,
                    help="manifest.json de Fase 1 con train/val (jres_ingest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pts", type=int, default=65536)
    ap.add_argument("--bg-ratio", type=float, default=4.0)
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
        Xs, ys = sample_points(X, L, args.max_pts, args.bg_ratio, rng)
        del X
        saver = np.savez_compressed if args.compress else np.savez
        saver(os.path.join(args.out, f"{sid}.npz"),
              X=np.ascontiguousarray(Xs, dtype=np.float32),
              y=np.ascontiguousarray(ys, dtype=np.float32),
              H=np.int32(SHAPE[0]), W=np.int32(SHAPE[1]))
        dt = time.time() - t0
        wr.writerow([sid, split_of[sid], len(Xs), int(ys.sum()), f"{dt:.1f}"])
        log.flush()
        done.append(sid)
        if (n + 1) % 25 == 0 or n + 1 == len(ids):
            el = time.time() - t_all
            print(f"[{n + 1}/{len(ids)}] {sid} M={len(Xs)} "
                  f"{dt:.1f}s (elapsed {el / 60:.0f} min, "
                  f"ETA {(el / (n + 1) * (len(ids) - n - 1)) / 60:.0f} min)",
                  flush=True)
    out_man = {"source": os.path.abspath(args.dataset),
               "storage": "pre-extracted fixed (X,y)/spectrum, one seed",
               "seed": args.seed, "bg_ratio": args.bg_ratio,
               "max_pts": args.max_pts, "shape": list(SHAPE),
               "train": [i for i in man["train"] if i in done],
               "val": [i for i in man["val"] if i in done]}
    json.dump(out_man, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"Done: {len(out_man['train'])} train / {len(out_man['val'])} val "
          f"-> {args.out} ({(time.time() - t_all) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
