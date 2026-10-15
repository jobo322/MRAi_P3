"""Prepare synthetic 2D-US spectra for MR-Ai retraining.

Consumes the producer JSON/NPZ layout described in training_data_contract.py and
produces per-spectrum (X, y) pairs via src/MRAi_P3/StoP3.split_2D WITHOUT modification,
guaranteeing train<->inference compatibility (StoP3.Production_2D / P3_2D).

Usage:
  python scripts/prepare_training_data.py --input data/synth_2d --out data/train_2d --val-frac 0.2
  python scripts/prepare_training_data.py --input data/synth_2d --validate-only
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from cross_extract import split_2d as split_2D  # noqa: E402  (NumPy mirror of StoP3.split_2D)



def load_spectrum(path):
    """Load one spectrum. Returns (S, L or None, meta). S is (N_F1, N_F2) float32."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        S = np.asarray(z["S"], dtype=np.float32)
        L = np.asarray(z["L"], dtype=np.float32) if "L" in z else None
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return np.ascontiguousarray(S), L, meta
    with open(path) as f:
        obj = json.load(f)
    S = np.asarray(obj["data"], dtype=np.float32)
    L = np.asarray(obj["labels"], dtype=np.float32) if "labels" in obj else None
    return np.ascontiguousarray(S), L, obj.get("meta", {})


def validate(S, L, meta, name, min_positives=None):
    """Acceptance checks per training_data_contract.py. Returns error list."""
    errs = []
    H, W = S.shape
    if not (H < W):
        errs.append(f"{name}: orientation {S.shape}, expected (F1_short, F2_long)")
    if not np.all(np.isfinite(S)):
        errs.append(f"{name}: non-finite values in S")
    if L is not None:
        if L.shape != S.shape:
            errs.append(f"{name}: L.shape={L.shape} != S.shape={S.shape}")
        if not set(np.unique(L).tolist()) <= {0.0, 1.0}:
            errs.append(f"{name}: L values not in {{0,1}}")
        npos = int(L.sum())
        # scale acceptance band with matrix size: ~256 peaks on 256x512 reference patch
        expected = 256 * (H * W) / (256 * 512)
        lo = min_positives if min_positives is not None else 0.5 * expected
        if not (lo <= npos <= 2.0 * expected + 8):
            errs.append(f"{name}: positives={npos}, expected ~{expected:.0f} (256-peak ref scaled)")
    return errs



def prepare_one(S, L):
    """Run split_2D (repo code, unmodified). Sanitises Norm==0 NaNs."""
    S_data, Norm = split_2D(S, np.array([64, 64]))
    X = np.asarray(S_data, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0  # flat background 0/0 guard
    y = np.asarray(L.reshape(-1, 1), dtype=np.float32) if L is not None else None
    return X, y, np.asarray(Norm, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Prepare synthetic 2D-US spectra.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-positives", type=int, default=None,
                    help="Override acceptance floor (debug probes only)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.input) if f.endswith((".json", ".npz")))
    if not files:
        sys.exit(f"No spectra in {args.input}")
    print(f"Found {len(files)} spectra")
    rng = np.random.default_rng(args.seed)
    stats, all_errs = [], []
    for f in files:
        S, L, meta = load_spectrum(os.path.join(args.input, f))
        errs = validate(S, L, meta, f, args.min_positives)
        all_errs.extend(errs)
        stats.append((f, S.shape, int(L.sum()) if L is not None else -1, errs))
    print(f"Validated: {len(files)} files, {sum(1 for s in stats if s[3])} with errors")
    for f, sh, npos, errs in stats:
        if errs:
            print("  FAIL", f, sh, errs)
    if args.validate_only or all_errs:
        if all_errs and not args.validate_only:
            sys.exit(f"Aborting: {len(all_errs)} acceptance errors")
        return
    os.makedirs(args.out, exist_ok=True)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    val_set = set(idx[:max(1, int(len(files) * args.val_frac))].tolist())
    manifest = {"train": [], "val": []}
    for k, f in enumerate(files):
        S, L, _ = load_spectrum(os.path.join(args.input, f))
        X, y, Norm = prepare_one(S, L)
        H, W = S.shape
        out_p = os.path.join(args.out, os.path.splitext(f)[0] + ".npz")
        kw = dict(X=X, Norm=Norm, H=H, W=W)
        if y is not None:
            kw["y"] = y
        np.savez_compressed(out_p, **kw)
        split = "val" if k in val_set else "train"
        manifest[split].append(os.path.basename(out_p))
        print(f"  [{split}] {f} X={X.shape} pos={int(y.sum()) if y is not None else -1}")
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"Done: {len(manifest['train'])} train / {len(manifest['val'])} val -> {args.out}")


if __name__ == "__main__":
    main()
