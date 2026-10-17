#!/usr/bin/env python
"""Run the folder-51 StoP3 chain on a TRAINING spectrum.

Mirror of `scripts/bruker_2d/run_51.py`: same inference, same P3 tail, same
peak-list chain (`P3toP5` -> `P5toPeaks` -> `PeakstoTab`). The only difference
is the input: one of the synthetic J-resolved spectra the model was trained on
(`dataset/spectra/<id>.bin`) instead of a Bruker `pdata/1` directory. The point
is to *look* at what the model does on data it has seen.

Steps (identical to folder 51):

  1. Repackage the raw `.bin` spectrum as a Pipe `.ft2`, because the peak-list
     chain reads NMRPipe headers.
  2. Blocked `Production_2D` with `--model`, candidate mask
     `|S| >= spe_the * std(S)` (the blocked variant is required: stock
     `StoP3.Production_2D` allocates the full (N,128,1) float64 context block,
     ~537 MB here).
  3. P3_2D noise/sign tail.
  4. `P3toP5` (threshold) -> `P5toPeaks` -> `PeakstoTab`.
  5. Artifacts for the zero-dependency web viewer, plus a matplotlib PNG, plus
     a match report against the ground truth in `dataset/manifest.jsonl`.

Why a synthetic `.ft2` and why the calibration is exact
-------------------------------------------------------
The training data has no physical ppm calibration (the grid is in points), so
the Pipe header is written with `car = N/2 - 1`, `sw = N`, `obs = 1`, which
gives

    ppm = (N - 1) - index

That is *exactly* what `P3toP5` writes (`FDF2ORIG/FDF2OBS + (FDSIZE - idx - 1)
* (SW/OBS)/FDSIZE`) and what `nmrglue.pipe.make_uc(...).ppm()` reads — the two
agree to 0.0 — so a detected peak reported at ppm `p` maps back to pixel
`index = (N - 1) - round(p)` with no ambiguity. `PeakstoTab.point_from_ppm` is
used for that inverse (it is the codebase's own), shifting its 1-based result.

Usage
-----
    tflow/bin/python scripts/plotting/run_train_2d.py --id 1
    tflow/bin/python scripts/plotting/run_train_2d.py --id 7 \\
        --model Trained/out_elong_continue/DNN_0
    # then look at it:  node scripts/plotting/view2d.mjs --out /tmp/train_view2d

Outputs in `--out` (default /tmp/train_p3/<id>): `<id>.ft2`, `<id>.P3`,
`<id>.P5`, `<id>_P5.npy`, `<id>.peaks`, `<id>.tab`, `<id>.png` and
`view2d/` (meta.json + *.bin) unless `--no-view2d`.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_DATASET = "/Users/abolanos/git/cheminfo/spectrum-generator/dataset"
# First existing wins. The elongated models are the point of this script; the
# 2D-US base is the fallback so it still runs on a fresh checkout.
MODEL_CANDIDATES = (
    # "Trained/Probability_2D_Elongated/DNN_0_local_serving/DNN_0",
    # "Trained/Probability_2D_Elongated/DNN_0_local_savedmodel",
    # "Trained/Probability_2D_Elongated/DNN_0_savedmodel",
    # "Trained/out_elong_continue/DNN_0",
    # "Trained/out_elong_savedmodel",
    "Trained/Probability_2D/US/DNN_0",
)
SPEC_TYPE = "JRES"


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None,
                    help="training spectrum id (default: first train id)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--model", default=None,
                    help="SavedModel dir (default: first of %s that exists)"
                         % ", ".join(MODEL_CANDIDATES))
    ap.add_argument("--out", default=None, help="output dir (default /tmp/train_p3/<id>)")
    ap.add_argument("--spe-the", type=float, default=0.5,
                    help="candidate mask threshold, |S| >= the * std(S) (0.5)")
    ap.add_argument("--pl-the", type=float, default=0.2,
                    help="P3toP5 probability threshold (0.2)")
    ap.add_argument("--spec-type", default=SPEC_TYPE)
    ap.add_argument("--row-block", type=int, default=8,
                    help="rows per inference block (8; memory knob)")
    ap.add_argument("--match-px", type=float, default=2.0,
                    help="a detection counts as a true positive if it lands "
                         "within this many pixels of a ground-truth peak (2)")
    ap.add_argument("--no-view2d", action="store_true",
                    help="skip the web-viewer artifact set")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG")
    ap.add_argument("--plot-only", action="store_true",
                    help="reuse <out>/<id>.P3 and <out>/<id>.tab and only redraw "
                         "the PNG (the inference is the expensive part)")
    return ap.parse_args()


def pick_model(explicit):
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(REPO, explicit)
        if not os.path.isfile(os.path.join(p, "saved_model.pb")):
            sys.exit(f"--model is not a SavedModel dir (no saved_model.pb): {p}")
        return p
    for c in MODEL_CANDIDATES:
        p = os.path.join(REPO, c)
        if os.path.isfile(os.path.join(p, "saved_model.pb")):
            return p
    sys.exit("no SavedModel found. Tried:\n  " + "\n  ".join(
        os.path.join(REPO, c) for c in MODEL_CANDIDATES))


def _bin_path(dataset, kind, sid):
    """Resolve `spectra/0001.bin` (ids are zero-padded to 4) or `spectra/1.bin`."""
    d = os.path.join(dataset, kind)
    for name in (f"{int(sid):04d}.bin", f"{int(sid)}.bin"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    sys.exit(f"no {kind} file for id {sid} in {d}")


def load_spectrum(dataset, sid):
    S = np.fromfile(_bin_path(dataset, "spectra", sid),
                    dtype="<f4").reshape(256, 2048).astype(np.float32)
    L = np.fromfile(_bin_path(dataset, "labels", sid),
                    dtype=np.uint8).reshape(256, 2048)
    return S, L


def load_ground_truth(dataset, sid):
    """Peaks of this spectrum from manifest.jsonl: (rows=F1, cols=F2, class, r).

    The manifest is strict JSONL: one *compact* object per line (`{"id":0,...`,
    no space after the colon), so the record is cheap to locate by prefix and
    parsing stops at the first match instead of reading all 1000.
    """
    path = os.path.join(dataset, "manifest.jsonl")
    want = int(sid)
    prefix = '{"id":%d,' % want
    with open(path) as fh:
        for line in fh:
            if not line.startswith(prefix):
                continue
            rec = json.loads(line)
            peaks = rec.get("peaks", [])
            # manifest keys: col -> F2/x, row -> F1/y
            cols = np.array([p["col"] for p in peaks], dtype=float)
            rows = np.array([p["row"] for p in peaks], dtype=float)
            return rows, cols, rec.get("class", "?"), rec.get("r")
    sys.exit(f"id {sid} not found in {path}")


def first_train_id(dataset):
    d = json.load(open(os.path.join(dataset, "dataset.json")))
    return int(d["split"]["train"][0])


# --------------------------------------------------------------------------- #
# Step 1 - raw .bin -> Pipe .ft2 with an exact ppm <-> pixel calibration
# --------------------------------------------------------------------------- #
def write_pipe(ft2_path, S):
    """Write S (F1, F2) as a Pipe file where ppm == (N-1) - index.

    `ng.pipe.create_dic` maps `udic[i]` -> `FDF{i+1}`, so key 0 is FDF1 (the
    indirect J axis, 256 points) and key 1 is FDF2 (the direct 1H axis, 2048).
    `car = N/2 - 1` puts the upper ppm limit at N-1, which is what makes the
    round-trip exact.
    """
    import nmrglue as ng

    def dim(n, label):
        return {"size": n, "label": label, "sw": float(n), "obs": 1.0,
                "car": n / 2.0 - 1.0, "encoding": "direct",
                "time": False, "complex": False, "freq": True}

    udic = {"ndim": 2, 0: dim(S.shape[0], "J"), 1: dim(S.shape[1], "1H")}
    os.makedirs(os.path.dirname(os.path.abspath(ft2_path)), exist_ok=True)
    ng.pipe.write(ft2_path, ng.pipe.create_dic(udic), S.astype(np.float32),
                  overwrite=True)
    return ft2_path


# --------------------------------------------------------------------------- #
# Step 2 - blocked Production_2D (same maths as run_51.py / StoP3.Production_2D)
# --------------------------------------------------------------------------- #
def predict_P3(S, model, spe_the=0.5, row_block=8, verbose=True):
    from MRAi_P3.model_output import single_output

    H, W = S.shape
    Sd = S.astype(np.float64)
    RI = np.zeros_like(Sd, dtype=bool)
    RI[np.abs(Sd) >= spe_the * np.std(Sd)] = True
    if verbose:
        print(f"candidates: {RI.sum():,} / {Sd.size:,} "
              f"({RI.mean() * 100:.2f}%)  spe_the={spe_the}", flush=True)

    pred = np.zeros((Sd.size, 1), dtype=np.float64)
    Sp = np.pad(Sd, ((32, 31), (32, 31)), "wrap")
    flat = RI.flatten()
    t00 = time.time()
    for r0 in range(0, H, row_block):
        r1 = min(r0 + row_block, H)
        # 64-column window above/below + 64-row window left/right -> (n, 128)
        col = np.stack([Sp[r0 + k:r1 + k, 32:32 + W] for k in range(64)], axis=2)
        row = np.stack([Sp[r0 + 32:r1 + 32, k:k + W] for k in range(64)], axis=2)
        blk = np.concatenate([col, row], axis=2).reshape((r1 - r0) * W, 128)
        n = np.linalg.norm(blk, axis=1, keepdims=True)
        n[n == 0] = 1.0
        blk = blk / n
        m = flat[r0 * W:r1 * W]
        if m.any():
            out = single_output(model.predict(blk[m], batch_size=2 ** 16,
                                              verbose=0))
            # (n, 1), same shape as the masked view -- do not flatten here
            pred[r0 * W:r1 * W][m] = np.asarray(out).reshape(-1, 1)
            del out
        del col, row, blk, n, m
        if verbose:
            print(f"  rows {r0:>4}-{r1:<4} {time.time() - t00:6.1f}s", flush=True)
    P = pred.reshape(S.shape)
    del RI, Sp, flat, pred, Sd
    gc.collect()
    return P


def p3_tail(P, S):
    """P3_2D tail from run_51.py: deterministic noise + the sign of S."""
    np.random.seed(0)
    noise = np.random.normal(size=S.shape)
    noise *= 0.001 / np.std(noise)
    P = (P + noise) * np.sign(S)
    return P


# --------------------------------------------------------------------------- #
# Step 3 - peak-list chain
# --------------------------------------------------------------------------- #
def run_chain(P3_path, ft2_path, out_dir, name, pl_the, spec_type):
    import nmrglue as ng
    import MRAi_P3 as P3

    dic, P = ng.pipe.read(P3_path)
    P5_path = os.path.join(out_dir, f"{name}.P5")
    P3.P3toP5.main(P3_path, ft2_path, "P", pl_the, "Pipe", out=P5_path)
    peaks_path = os.path.join(out_dir, f"{name}.peaks")
    P3.P5toPeaks.main(P5_path, ft2_path, spec_type, out=peaks_path)
    tab_path = os.path.join(out_dir, f"{name}.tab")
    P3.PeakstoTab.main(peaks_path, ft2_path, out=tab_path)
    return P5_path, peaks_path, tab_path


def read_tab(tab_path, ft2_path):
    """Detected peaks as 0-based (row, col) pixels, via the codebase's inverse.

    `PeakstoTab.point_from_ppm` returns a 1-based point index; shift it. The
    calibration written in `write_pipe` makes this exact.
    """
    from MRAi_P3.headers import read_headers
    from MRAi_P3.PeakstoTab import point_from_ppm

    ND, sz, sw, pfst, plst, nam = read_headers(ft2_path)
    vars_line, rows = None, []
    for line in open(tab_path):
        s = line.strip()
        if s.startswith("VARS"):
            vars_line = s.split()[1:]
            continue
        if vars_line and s and s[0].isdigit() and len(s.split()) == len(vars_line):
            rows.append(s.split())
    if not vars_line:
        return np.zeros((0, 2)), []
    ix, iy = vars_line.index("X_PPM"), vars_line.index("Y_PPM")
    pix, meta = [], []
    for r in rows:
        x_ppm, y_ppm = float(r[ix]), float(r[iy])
        col = int(round(point_from_ppm(x_ppm, pfst[2], plst[2], sz[2]))) - 1
        row = int(round(point_from_ppm(y_ppm, pfst[1], plst[1], sz[1]))) - 1
        pix.append((row, col))
        meta.append({"x_ppm": x_ppm, "y_ppm": y_ppm,
                     "height": float(r[vars_line.index("HEIGHT")]),
                     "type": r[vars_line.index("TYPE")]})
    return np.array(pix, dtype=int).reshape(-1, 2), meta


# --------------------------------------------------------------------------- #
# Step 4 - outputs
# --------------------------------------------------------------------------- #
def dump_view2d(out_dir, S, P, gt_rows, gt_cols, det, meta, labels=("J", "1H")):
    """Artifact set consumed by scripts/plotting/view2d.mjs (see its README)."""
    vdir = os.path.join(out_dir, "view2d")
    os.makedirs(vdir, exist_ok=True)

    H, W = S.shape
    # ppm scales: ppm = (N-1) - index, matching write_pipe
    f2ppm = (W - 1) - np.arange(W, dtype=np.float32)
    f1hz = (H - 1) - np.arange(H, dtype=np.float32)      # obs = 1 -> ppm == Hz

    S.astype("<f4").tofile(os.path.join(vdir, "spectrum.bin"))
    P.astype("<f4").tofile(os.path.join(vdir, "p3.bin"))
    f2ppm.tofile(os.path.join(vdir, "f2ppm.bin"))
    f1hz.tofile(os.path.join(vdir, "f1hz.bin"))

    # Ground truth and detections both live in the same pixel space; the viewer
    # wants (x = F2 ppm, y = F1 Hz).
    gt_x = (W - 1) - gt_cols
    gt_y = (H - 1) - gt_rows
    det_x = (W - 1) - det[:, 1] if len(det) else np.array([])
    det_y = (H - 1) - det[:, 0] if len(det) else np.array([])

    m = {
        "shape": [H, W], "obs": 1.0, "labels": [labels[0], labels[1]],
        "spectrum": "spectrum.bin", "p3": "p3.bin",
        "f2ppm": "f2ppm.bin", "f1hz": "f1hz.bin",
        "peaks": {"x": [float(v) for v in det_x], "y": [float(v) for v in det_y],
                  "rows": [{"index": i + 1, "xPpm": float(det_x[i]),
                            "yHz": float(det_y[i]), "height": meta[i]["height"],
                            "vol": 0.0, "type": meta[i]["type"]}
                           for i in range(len(det_x))]},
        "truth": {"x": [float(v) for v in gt_x], "y": [float(v) for v in gt_y]},
        "source": {"ft2": "synthetic", "peakSource": "P3 chain",
                   "model": os.path.basename(out_dir)},
    }
    with open(os.path.join(vdir, "meta.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    return vdir


def plot(out_dir, S, P, gt_rows, gt_cols, det, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = S.shape
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5))

    Sn = S / np.max(np.abs(S))
    levels = 0.02 * 1.6 ** np.arange(14)

    # P is a probability, so the colour scale is fixed 0..1 rather than
    # relative to this map: two runs are then directly comparable, and a
    # saturated colour really means "P near 1".
    a = ax[0]
    a.contour(Sn, levels, colors="black", linewidths=0.5)
    a.set_title("spectrum")

    a = ax[1]
    im = a.imshow(P, cmap="viridis", origin="upper", aspect="auto",
                  vmin=0.0, vmax=1.0)
    a.set_title("P3 probability")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="P(peak)")

    for a in ax[:2]:
        a.scatter(gt_cols, gt_rows, s=42, facecolors="none", edgecolors="red",
                  linewidths=1.0, label=f"ground truth ({len(gt_cols)})")
        if len(det):
            a.scatter(det[:, 1], det[:, 0], s=34, marker="x", c="cyan",
                      linewidths=1.0, label=f"detected ({len(det)})")
        a.set_xlabel("F2 index"); a.set_ylabel("F1 index")
        a.legend(loc="upper right", fontsize=8)
        a.set_xlim(0, W); a.set_ylim(H, 0)

    # Zoom on the region with most ground truth around it, to actually see the
    # per-peak behaviour (overlaps, shoulders, false positives next to a peak).
    c0 = int(gt_cols.mean()) if len(gt_cols) else W // 2
    r0 = int(gt_rows.mean()) if len(gt_rows) else H // 2
    hw, hh = 260, 70
    x0, x1 = max(0, c0 - hw), min(W, c0 + hw)
    y0, y1 = max(0, r0 - hh), min(H, r0 + hh)
    a = ax[2]
    im = a.imshow(P[y0:y1, x0:x1], cmap="viridis", origin="upper", aspect="auto",
                  vmin=0.0, vmax=1.0, alpha=0.55)
    a.contour(Sn[y0:y1, x0:x1], levels, colors="black", linewidths=0.5)
    sel_gt = (gt_cols >= x0) & (gt_cols < x1) & (gt_rows >= y0) & (gt_rows < y1)
    a.scatter(gt_cols[sel_gt] - x0, gt_rows[sel_gt] - y0, s=46, facecolors="none",
              edgecolors="red", linewidths=1.1, label="ground truth")
    if len(det):
        sel_d = (det[:, 1] >= x0) & (det[:, 1] < x1) & (det[:, 0] >= y0) & (det[:, 0] < y1)
        a.scatter(det[sel_d, 1] - x0, det[sel_d, 0] - y0, s=36, marker="x",
                  c="cyan", linewidths=1.0, label="detected")
    a.set_title(f"zoom x[{x0}:{x1}] y[{y0}:{y1}]")
    a.set_xlabel("F2 index (local)"); a.set_ylabel("F1 index (local)")
    a.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="P(peak)")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, "spectrum_P3_peaks.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return png


def match_report(gt_rows, gt_cols, det, match_px):
    """Greedy nearest-neighbour matching between detections and ground truth."""
    n_gt, n_det = len(gt_rows), len(det)
    gt = np.stack([gt_rows, gt_cols], axis=1).astype(float) if n_gt else np.zeros((0, 2))
    matched_gt, matched_det, dists = set(), set(), []
    if n_gt and n_det:
        d = np.linalg.norm(gt[:, None, :] - det[None, :, :].astype(float), axis=2)
        order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
        for gi, di in order:
            if d[gi, di] > match_px:
                break
            if gi in matched_gt or di in matched_det:
                continue
            matched_gt.add(int(gi)); matched_det.add(int(di)); dists.append(d[gi, di])
    return {
        "n_gt": n_gt, "n_detected": n_det,
        "true_positive": len(matched_gt),
        "false_positive": n_det - len(matched_det),
        "false_negative": n_gt - len(matched_gt),
        "recall": len(matched_gt) / n_gt if n_gt else float("nan"),
        "precision": len(matched_det) / n_det if n_det else float("nan"),
        "median_dist_px": float(np.median(dists)) if dists else float("nan"),
        "match_px": match_px,
    }


def main():
    args = parse_args()
    import tensorflow as tf

    sid = args.id if args.id is not None else first_train_id(args.dataset)
    model_path = pick_model(args.model)
    out_dir = args.out or f"/tmp/train_p3/{sid}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"spectrum : id={sid}  ({args.dataset})")
    print(f"model    : {os.path.relpath(model_path, REPO)}")
    print(f"output   : {out_dir}")

    S, L = load_spectrum(args.dataset, sid)
    gt_rows, gt_cols, cls, r = load_ground_truth(args.dataset, sid)
    print(f"           S {S.shape} std={S.std():.4g}  label={L.sum()} px  "
          f"class={cls}  r={r if r is None else round(float(r), 3)}  "
          f"ground-truth peaks={len(gt_cols)}")

    ft2 = os.path.join(out_dir, f"{sid}.ft2")
    P3_path = os.path.join(out_dir, f"{sid}.P3")
    tab = os.path.join(out_dir, f"{sid}.tab")

    if args.plot_only:
        import nmrglue as ng
        for p in (ft2, P3_path, tab):
            if not os.path.isfile(p):
                sys.exit(f"--plot-only needs {p} -- run once without it first")
        P = ng.pipe.read(P3_path)[1].astype(np.float64)
        det, det_meta = read_tab(tab, ft2)
        print(f"reusing {P3_path} + {tab}  ({len(det)} peaks)")
    else:
        write_pipe(ft2, S)
        print(f"\n[1/5] pipe     -> {ft2}")

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(128, 1)),
            tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")])
        print("[2/5] inference")
        P = predict_P3(S, model, args.spe_the, args.row_block)
        del model
        gc.collect()
        P = p3_tail(P, S)
        print(f"      P3 range [{P.min():.4f}, {P.max():.4f}]")

        import nmrglue as ng
        dic = ng.pipe.read(ft2)[0]
        ng.pipe.write(P3_path, dic, np.float32(P), overwrite=True)
        print(f"[3/5] P3       -> {P3_path}")

        print("[4/5] peak chain")
        run_chain(P3_path, ft2, out_dir, str(sid), args.pl_the, args.spec_type)
        det, det_meta = read_tab(tab, ft2)
        print(f"      detected {len(det)} peaks  -> {tab}")

    rep = match_report(gt_rows, gt_cols, det, args.match_px)
    print(f"\n[5/5] match vs ground truth (tolerance {args.match_px:.0f} px)")
    print(f"      ground truth : {rep['n_gt']}")
    print(f"      detected     : {rep['n_detected']}"
          f"  ({rep['n_detected'] / max(rep['n_gt'], 1):.2f}x the truth)")
    print(f"      TP {rep['true_positive']}   FP {rep['false_positive']}   "
          f"FN {rep['false_negative']}")
    print(f"      recall {rep['recall']:.3f}   precision {rep['precision']:.3f}   "
          f"median dist {rep['median_dist_px']:.2f} px")

    # How sure is the model of the zeros right next to a peak? This is the
    # metric that motivated --hard-frac: uniform background sampled them at
    # 0.48 %, and the model then gave P > 0.5 to 81 % of them.
    #
    # Reported twice, because it depends on the candidate mask: P is 0 wherever
    # |S| < spe_the * std(S), so a tighter mask hides ring pixels that the model
    # would otherwise judge. `over candidates` is the mask-independent figure.
    near = (L > 0)
    try:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(near, iterations=2) & ~near
        if ring.any():
            mask = np.abs(S) >= args.spe_the * S.std()
            hot = np.abs(P)[ring] > 0.5      # one entry per ring pixel
            in_mask = mask[ring]             # same length as hot
            n_cand = int(in_mask.sum())
            frac_all = float(hot.mean())
            frac_cand = float(hot[in_mask].mean()) if n_cand else float("nan")
            print(f"      P > 0.5 on the 2-px ring around a label: "
                  f"{frac_all * 100:.1f}% of {int(ring.sum())} px  "
                  f"({frac_cand * 100:.1f}% over the {n_cand} candidates)")
    except ImportError:
        pass

    if not args.no_view2d:
        vdir = dump_view2d(out_dir, S, P, gt_rows, gt_cols, det, det_meta)
        print(f"\nviewer artifacts -> {vdir}")
        print(f"  node scripts/plotting/view2d.mjs --out {vdir}")
    if not args.no_plot:
        png = plot(out_dir, S, P, gt_rows, gt_cols, det,
                   f"training spectrum id={sid}  class={cls}  "
                   f"model={os.path.basename(model_path)}  "
                   f"GT {rep['n_gt']} / det {rep['n_detected']}")
        print(f"png -> {png}")

    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump({"id": sid, "class": cls, "r": r,
                   "model": os.path.relpath(model_path, REPO),
                   "spe_the": args.spe_the, "pl_the": args.pl_the,
                   "match": rep}, fh, indent=1)
    print("ALL DONE")


if __name__ == "__main__":
    main()
