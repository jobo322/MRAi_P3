"""Full Bruker 2D chain: 2rr -> Pipe .ft2 -> blocked P3_2D -> P3 -> P5 -> .peaks -> .tab.

Worked example for Folder 51 (J-resolved, jresgpprqf, 600 MHz), reusable for
other Bruker 2D data sets by overriding the paths / thresholds below or via
the CLI flags.

Memory-safe: vectorised row blocks (parity vs StoP3.split_2D proven maxdiff
0.0, vs Production_2D proven maxdiff ~1e-7). Stock StoP3.Production_2D
allocates the full (N,128,1) float64 context block (~8 GiB for 512x16384),
so the blocked variant is required on 16 GB machines.

Steps:
  1. Bruker pdata/1/2rr -> Pipe .ft2 (nmrglue read_pdata + create_dic).
     `showhdr` rejects raw Bruker 2rr, so this repackaging is required for
     the downstream header parsing. Verified byte-exact (maxdiff 0.0).
  2. Blocked Production_2D with Trained/Probability_2D/US/DNN_<axis>,
     candidate mask |S| >= SPE_THE * std(S), P3_2D noise/sign tail.
  3. P3toP5 (PL_THE) -> P5toPeaks (SPEC_TYPE) -> PeakstoTab.

Usage:
  tflow/bin/python scripts/bruker_2d/run_51.py
  tflow/bin/python scripts/bruker_2d/run_51.py --pdata /path/to/51/pdata/1 \\
      --outdir /tmp/my_run --name myexp --pl-the 0.2

Outputs in OUTDIR: <name>.ft2, <name>.P3, <name>.P5, <name>_P5.npy,
<name>.peaks, <name>.tab
"""
import argparse
import sys, os, gc, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'src'))
import numpy as np, nmrglue as ng, tensorflow as tf
import MRAi_P3 as P3

# Defaults reproduce the Folder-51 run exactly.
BRUKER_PDATA = '/Users/abolanos/Downloads/jresolved_spectrum/51/pdata/1'
FT2 = '/tmp/p51/51.ft2'
OUT = '/tmp/p51_run'
NAME = '51'
D2D = os.path.join(REPO, 'Trained/Probability_2D/US')
SPE_THE = 0.5
PL_THE = 0.2
SPEC_TYPE = 'JRES'
ROW_BLOCK = 8
AXIS = 0

def parse_args():
    ap = argparse.ArgumentParser(description='Bruker 2D -> P3 -> peak list chain.')
    ap.add_argument('--pdata', default=BRUKER_PDATA, help='Bruker pdata dir holding 2rr')
    ap.add_argument('--ft2', default=FT2, help='Pipe .ft2 path (written if missing)')
    ap.add_argument('--outdir', default=OUT, help='Output directory')
    ap.add_argument('--name', default=NAME, help='Output basename')
    ap.add_argument('--model-dir', default=D2D, help='Trained/Probability_2D dir')
    ap.add_argument('--spe-the', type=float, default=SPE_THE)
    ap.add_argument('--pl-the', type=float, default=PL_THE)
    ap.add_argument('--spec-type', default=SPEC_TYPE)
    ap.add_argument('--row-block', type=int, default=ROW_BLOCK)
    ap.add_argument('--axis', type=int, default=AXIS)
    ap.add_argument('--skip-convert', action='store_true',
                    help='Reuse existing --ft2 instead of converting from --pdata')
    return ap.parse_args()


def bruker_to_pipe(pdata_dir, ft2_path):
    """Repackage Bruker 2rr as Pipe .ft2 (byte-exact data matrix)."""
    b_dic, data = ng.bruker.read_pdata(pdata_dir)
    udic = ng.bruker.guess_udic(b_dic, data)
    for i in range(udic['ndim']):
        udic[i]['time'] = False
        udic[i]['complex'] = False
    p_dic = ng.pipe.create_dic(udic)
    ng.pipe.write(ft2_path, p_dic, data.astype(np.float32), overwrite=True)
    return ft2_path


def main():
    args = parse_args()
    FT2 = args.ft2
    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    t00 = time.time()

    if not args.skip_convert or not os.path.isfile(FT2):
        os.makedirs(os.path.dirname(os.path.abspath(FT2)), exist_ok=True)
        bruker_to_pipe(args.pdata, FT2)
        print(f'converted {args.pdata}/2rr -> {FT2}', flush=True)

    dic, Sraw = ng.pipe.read(FT2)
    S = Sraw.astype(np.float64)
    print(f'S {S.shape} std={S.std():.4g} max={S.max():.4g}', flush=True)

    # ---- blocked Production_2D (== StoP3.Production_2D) ----
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(shape=(128, 1)),
        tf.keras.layers.TFSMLayer(args.model_dir + '/DNN_' + str(args.axis), call_endpoint='serving_default')])
    RI = np.zeros_like(S, dtype=bool)
    RI[np.abs(S) >= args.spe_the * np.std(S)] = True
    print(f'RI candidates: {RI.sum():,} / {S.size:,} ({RI.mean()*100:.2f}%)', flush=True)
    pred = np.zeros((S.size, 1))
    Sp = np.pad(S, ((32, 31), (32, 31)), 'wrap')
    H, W = S.shape
    flat = RI.flatten()
    for r0 in range(0, H, args.row_block):
        t0 = time.time()
        r1 = min(r0 + args.row_block, H)
        col = np.stack([Sp[r0+k:r1+k, 32:32+W] for k in range(64)], axis=2)
        row = np.stack([Sp[r0+32:r1+32, k:k+W] for k in range(64)], axis=2)
        blk = np.concatenate([col, row], axis=2).reshape((r1-r0)*W, 128)
        n = np.linalg.norm(blk, axis=1, keepdims=True); n[n == 0] = 1.0
        blk = blk / n
        m = flat[r0*W:r1*W]
        if m.any():
            out = model.predict(blk[m], batch_size=2**16, verbose=0)['dense_14']
            pred[r0*W:r1*W][m] = out
            del out
        del col, row, blk, n, m
        print(f'rows {r0}-{r1} done in {time.time()-t0:.1f}s', flush=True)
    P = pred.reshape(S.shape)
    del model, RI, Sp, flat, pred
    gc.collect()

    # ---- P3_2D tail: deterministic noise, sign of S ----
    np.random.seed(0)
    noise = np.random.normal(size=S.shape); noise *= 0.001 / np.std(noise)
    P += noise; P *= np.sign(S)
    del noise, S
    gc.collect()
    print(f'P3 range [{P.min():.4f},{P.max():.4f}]  ({time.time()-t00:.0f}s total)', flush=True)

    P3_PATH = os.path.join(OUT, f'{args.name}.P3')
    ng.pipe.write(P3_PATH, dic, np.float32(P), overwrite=True)
    print('wrote', P3_PATH, flush=True)

    # ---- peak list chain ----
    P5_PATH = os.path.join(OUT, f'{args.name}.P5')
    P3.P3toP5.main(P3_PATH, FT2, 'P', args.pl_the, 'Pipe', out=P5_PATH)
    print('wrote', P5_PATH, flush=True)
    PEAKS_PATH = os.path.join(OUT, f'{args.name}.peaks')
    P3.P5toPeaks.main(P5_PATH, FT2, args.spec_type, out=PEAKS_PATH)
    TAB_PATH = os.path.join(OUT, f'{args.name}.tab')
    P3.PeakstoTab.main(PEAKS_PATH, FT2, out=TAB_PATH)
    print('wrote', PEAKS_PATH, TAB_PATH, flush=True)
    rows = [l for l in open(TAB_PATH) if l.strip() and l.strip()[0].isdigit()]
    print(f'{len(rows)} peaks in tab', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
