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

--hard-band2/--hard-frac2 add an isotropic middle band (hard_px < d <=
hard_band2). --hard-aniso F1LO F1HI F2LO F2HI replaces it with a rectangle
per axis (F1 = rows, F2 = columns) using d1/d2, the row/column distance to
the nearest positive; the npz then also stores d1/d2 so train_model.py
--hard-aniso can resample the band each epoch. The ring keeps priority and
a row never lands in two bands.

Why: BinSequence._points re-runs split_2d (~1-2 s/spectrum) EVERY epoch
-> 1000 spectra x 210 epochs ~= 60-110 h of CPU just recomputing the
identical X. Pre-extracting removes that entirely.

Pares apretados (--pair-px > 0): entre los falsos negativos que quedan hay
picos enterrados en el flanco de uno mayor (a <= 6 px), y el perfil en cruz no
ve a un vecino en diagonal, así que el par no se puede derivar de X. Se marca
desde <dataset>/manifest.jsonl (centros + amplitudes de TODOS los picos), una
sola pasada para todo el pool, y viaja en el npz como un byte por fila (`pair`)
para que train_model.py --pair-frac sobremuestre esas filas cada época. Ver
scripts/pair_marks.py y docs/ring_and_picking_metrics.md §11.

Output layout (<out>/):
  <id>.npz          X (M,128,1) float32, y (M,1) float32, n_pos, d (uint8,
                    distancia al pico más cercano de cada fila de fondo),
                    d1/d2 (uint8, solo con --hard-aniso: distancia por eje),
                    hard_px, H, W
  manifest.json     {source, seed, pool_k, hard_frac, hard_px, hard_aniso,
                     max_pts, shape, train, val}
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
from hard_negatives import (HARD_FRAC, HARD_PX, aniso_band_mask,  # noqa: E402
                            axis_distance_to_positive, band_sections,
                            choose_background, distance_to_positive,
                            pool_sections)
from pair_marks import row_pair_flags  # noqa: E402

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
                  hard_px=HARD_PX, band2_px=None, band2_frac=0.0,
                  aniso=None):
    """Rows that form the pool: all positives, then the background budget.

    Layout: [positivos | anillo | banda2 | fáciles]. El fondo se reparte entre
    negativos a <= `hard_px` px de un pico (`hard_frac`), negativos en
    `hard_px < d <= band2_px` (`hard_frac2`) y el resto uniforme. Antes era
    uniforme sobre todos los negativos, así que los ceros dentro del pico
    (la etiqueta es un marcador sub-píxel de 1/2/4 px sobre picos de ~20-50 px)
    entraban al 0,48 % del fondo: ~11 negativos a 1 px por cada 578 positivos y
    el modelo acababa dando P > 0,5 al 81 % de los vecinos inmediatos de un
    pico. Ver scripts/hard_negatives.py.

    Dos bandas: medido, el anillo (d<=2) entraba al 54 % pero la banda 3-6 px
    —donde caen los hombros que se vuelven falsos positivos— al 0,5 %, porque el
    fondo uniforme la diluye frente a las ~500 000 filas a d>6. `band2_frac`
    reparte el presupuesto explícitamente.

    Pure index planning, independent of X: this is what lets the caller
    compute only the selected cross sections instead of the whole
    (H*W, 128, 1) matrix. Returns (idx, n_pos); rows [0:n_pos] of `idx`
    are the positives and [n_pos:] the background pool.

    Con `hard_frac=0` y `band2_frac=0` (por defecto) el sorteo es EXACTAMENTE el
    histórico: una sola llamada a `rng.choice` sobre todos los negativos, en el
    mismo orden, así que el pool guardado para una semilla no cambia
    (tests/test_split_rows.py).
    """
    if aniso is not None and band2_frac <= 0:
        raise ValueError(
            "--hard-aniso necesita --hard-frac2 > 0: es la fracción del "
            "presupuesto de fondo que se reserva a la banda anisótropa")
    if aniso is not None and band2_px is not None:
        raise ValueError("--hard-aniso y --hard-band2 son excluyentes")
    y = L.reshape(-1, 1)
    pos = np.where(y[:, 0] > 0.5)[0]
    n_pos = len(pos)
    neg = np.where(y[:, 0] < 0.5)[0]
    n_bg = min(len(neg), int(n_pos * pool_k))
    if max_pts is not None:
        n_bg = min(n_bg, max(0, max_pts - n_pos))
    n_hard, n_band2, _ = band_sections(n_bg, hard_frac, band2_frac)
    if (hard_frac > 0 and n_hard > 0) or (band2_frac > 0 and n_band2 > 0):
        band2_mask = None
        if aniso is not None:
            d_full, d1_full, d2_full = axis_distance_to_positive(L)
            d_neg = d_full.reshape(-1)[neg]
            band2_mask = aniso_band_mask(d1_full.reshape(-1)[neg],
                                         d2_full.reshape(-1)[neg], *aniso)
        else:
            d_neg = distance_to_positive(L).reshape(-1)[neg]
        bg = neg[choose_background(neg, d_neg, n_bg, hard_frac, hard_px, rng,
                                   band2_px=band2_px,
                                   band2_frac=band2_frac,
                                   band2_mask=band2_mask)]
    else:
        bg = rng.choice(neg, size=max(0, n_bg), replace=False)
    return np.concatenate([pos, bg]), n_pos


def build_pool(X, L, pool_k, max_pts=None, rng=None, hard_frac=0.0,
               hard_px=HARD_PX, band2_px=None, band2_frac=0.0, aniso=None):
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
                               hard_frac=hard_frac, hard_px=hard_px,
                               band2_px=band2_px, band2_frac=band2_frac,
                               aniso=aniso)
    return X[idx], L.reshape(-1, 1)[idx], n_pos


def _check_pair_args(pair_px, pair_ratio):
    """Validación de las flags de pares apretados: SystemExit y mensaje claro."""
    if pair_px < 0:
        raise SystemExit(f"--pair-px ({pair_px}) no puede ser negativo "
                         f"(0 = apagado)")
    if not (0.0 < pair_ratio <= 1.0):
        raise SystemExit(f"--pair-ratio ({pair_ratio}) debe estar en (0, 1]: "
                         f"es amps[propio]/amps[otro] del pico enterrado")


def load_peak_manifest(ds_dir, ids):
    """(rows, cols, amps, pixel_sets) por id de <dataset>/manifest.jsonl, una pasada.

    El manifiesto es JSONL compacto (`{"id":0,...`, sin espacio tras los dos
    puntos), así que el id se localiza por prefijo sin parsear la línea entera;
    solo se parsean los ids que pide el pool. Una sola pasada reutilizable
    sirve para todos los espectros: del manifiesto salen las amplitudes, la
    única diferencia entre picos que el perfil en cruz no puede ver.
    """
    path = os.path.join(ds_dir, "manifest.jsonl")
    want = {int(sid) for sid in ids}
    peaks = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith('{"id":'):
                continue
            try:
                sid = int(line[6:line.index(",")])
            except ValueError:
                continue
            if sid not in want or sid in peaks:
                continue
            rec = json.loads(line)
            pk = rec.get("peaks", [])
            peaks[sid] = (
                np.array([p["row"] for p in pk], dtype=np.float64),
                np.array([p["col"] for p in pk], dtype=np.float64),
                np.array([p["A"] for p in pk], dtype=np.float64),
                # píxeles de la etiqueta de cada pico: el marcado exacto de
                # los pares apretados (ver row_pair_flags)
                # si un manifiesto mínimo no trae pixelSet, basta el centro
                [p.get("pixelSet") or [[p["row"], p["col"]]] for p in pk])
    faltan = [sid for sid in ids if int(sid) not in peaks]
    if faltan:
        raise SystemExit(
            f"ERROR: {len(faltan)}/{len(ids)} ids de la lista no están en "
            f"{path} (p.ej. {faltan[:3]}): sin sus picos no se pueden marcar "
            f"los pares apretados.")
    return peaks


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
    ap.add_argument("--hard-band2", type=int, default=0,
                    help="radio exterior de la SEGUNDA banda de fondo, en px "
                         "(0 = sin banda 2). Medido: la banda 3-6 px entraba al "
                         "pool al 0,5 %% y es donde caen los hombros que se "
                         "vuelven falsos positivos; con --hard-band2 6 el "
                         "presupuesto la cubre de verdad")
    ap.add_argument("--hard-frac2", type=float, default=0.0,
                    help="fracción del fondo reservada a la banda 2 "
                         "(--hard-px < d <= --hard-band2, o la banda de "
                         "--hard-aniso); 0 = apagada, y "
                         "hard-frac + hard-frac2 <= 1")
    ap.add_argument("--hard-aniso", type=int, nargs=4, default=None,
                    metavar=("F1LO", "F1HI", "F2LO", "F2HI"),
                    help="banda de fondo anisótropa en px del pico, por eje: "
                         "F1 (filas) en [F1LO, F1HI] y F2 (columnas) en "
                         "[F2LO, F2HI]; p.ej. 3 10 1 3. Sustituye a "
                         "--hard-band2 (excluyentes) y usa --hard-frac2 como "
                         "fracción; guarda d1/d2 en el npz para que "
                         "train_model.py --hard-aniso pueda remuestrear")
    ap.add_argument("--pair-px", type=float, default=0.0,
                    help="radio en px para marcar PARES APRETADOS "
                         "(0 = apagado): un pico se marca si otro con amplitud "
                         "mayor está a <= --pair-px px y su relación de "
                         "amplitudes <= --pair-ratio. Necesita "
                         "<dataset>/manifest.jsonl; guarda `pair` en el npz "
                         "para train_model.py --pair-frac")
    ap.add_argument("--pair-ratio", type=float, default=0.5,
                    help="amps[propio]/amps[otro] <= --pair-ratio para marcar "
                         "el par (0.5); solo se usa con --pair-px > 0")
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

    if args.hard_frac2 > 0 and not (args.hard_band2 or args.hard_aniso):
        ap.error("--hard-frac2 necesita --hard-band2 o --hard-aniso (la banda "
                 "que se lleva esa fracción)")
    if args.hard_band2 and args.hard_band2 <= args.hard_px:
        ap.error(f"--hard-band2 ({args.hard_band2}) debe ser mayor que "
                 f"--hard-px ({args.hard_px})")
    if args.hard_aniso:
        if args.hard_band2:
            ap.error("--hard-aniso y --hard-band2 son excluyentes")
        if args.hard_frac2 <= 0:
            ap.error("--hard-aniso necesita --hard-frac2 > 0 (la fracción del "
                     "fondo que se reserva a la banda anisótropa)")
        f1lo, f1hi, f2lo, f2hi = args.hard_aniso
        if f1lo < 0 or f2lo < 0 or f1lo > f1hi or f2lo > f2hi:
            ap.error(f"--hard-aniso {args.hard_aniso}: se espera "
                     f"0 <= F1LO <= F1HI y 0 <= F2LO <= F2HI")
    if args.hard_frac + args.hard_frac2 > 1.0 + 1e-9:
        ap.error(f"--hard-frac ({args.hard_frac}) + --hard-frac2 "
                 f"({args.hard_frac2}) no pueden sumar más de 1")
    _check_pair_args(args.pair_px, args.pair_ratio)

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
    # la columna `pairs` (filas del pool marcadas como par apretado) solo
    # aparece con el marcaje encendido: el log de hoy no cambia
    wr.writerow(["id", "split", "M", "pos", "hard", "band2", "seconds"]
                + (["pairs"] if args.pair_px > 0 else []))
    peak_man = (load_peak_manifest(args.dataset, ids)
                if args.pair_px > 0 else None)
    t_all, done = time.time(), []
    for n, sid in enumerate(ids):
        t0 = time.time()
        S, L = load_pair(args.dataset, sid)
        # Plan the pool BEFORE splitting: split_2d does not touch the rng, so
        # the draw order (and therefore the pool) is unchanged.
        idx, n_pos = plan_pool_idx(L, args.pool_k, args.max_pts, rng,
                                   hard_frac=args.hard_frac, hard_px=args.hard_px,
                                   band2_px=args.hard_band2 or None,
                                   band2_frac=args.hard_frac2,
                                   aniso=args.hard_aniso)
        if args.hard_aniso:
            d_full, d1_full, d2_full = axis_distance_to_positive(L)
            d_pool = d_full.reshape(-1)[idx]
            d1_pool = d1_full.reshape(-1)[idx]
            d2_pool = d2_full.reshape(-1)[idx]
        else:
            d_pool = distance_to_positive(L).reshape(-1)[idx]
            d1_pool = d2_pool = None
        n_hard, n_band2, _ = band_sections(len(idx) - n_pos, args.hard_frac,
                                           args.hard_frac2)
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
        extra = {}
        if peak_man is not None:
            rows_pk, cols_pk, amps_pk, pix_pk = peak_man[int(sid)]
            pair_flags = row_pair_flags(rows_pk, cols_pk, amps_pk, idx,
                                        SHAPE[1], px=args.pair_px,
                                        ratio=args.pair_ratio,
                                        pixel_sets=pix_pk)
            extra = {"pair": np.ascontiguousarray(pair_flags, dtype=np.uint8),
                     "pair_px": np.float64(args.pair_px),
                     "pair_ratio": np.float64(args.pair_ratio)}
        saver = np.savez_compressed if args.compress else np.savez
        saver(os.path.join(args.out, f"{sid}.npz"),
              X=np.ascontiguousarray(Xs, dtype=np.float32),
              y=ys,
              n_pos=np.int32(n_pos),
              # distancia al positivo más cercano de cada fila de fondo: es lo que
              # permite minar el anillo por época sin volver a generar el pool
              d=np.ascontiguousarray(d_pool[n_pos:], dtype=np.uint8),
              hard_px=np.int32(args.hard_px),
              H=np.int32(SHAPE[0]), W=np.int32(SHAPE[1]),
              **({"d1": np.ascontiguousarray(d1_pool[n_pos:], dtype=np.uint8),
                  "d2": np.ascontiguousarray(d2_pool[n_pos:], dtype=np.uint8)}
                 if args.hard_aniso else {}),
              **extra)
        dt = time.time() - t0
        wr.writerow([sid, split_of[sid], len(Xs), n_pos, n_hard, n_band2,
                     f"{dt:.1f}"]
                    + ([int(pair_flags.sum())] if peak_man is not None else []))
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
               "hard_band2": args.hard_band2, "hard_frac2": args.hard_frac2,
               "hard_aniso": args.hard_aniso,
               "max_pts": args.max_pts, "shape": list(SHAPE),
               "train": [i for i in man["train"] if i in done],
               "val": [i for i in man["val"] if i in done]}
    if args.pair_px > 0:
        out_man["pair_px"] = args.pair_px
        out_man["pair_ratio"] = args.pair_ratio
    json.dump(out_man, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"Done: {len(out_man['train'])} train / {len(out_man['val'])} val "
          f"-> {args.out} ({(time.time() - t_all) / 3600:.2f} h)")
    print(f"fondo: {args.hard_frac * 100:.0f} % del pool reservado al anillo "
          f"(d <= {args.hard_px} px)" +
          (f" y {args.hard_frac2 * 100:.0f} % a la banda "
           + (f"anisótropa (F1 {args.hard_aniso[0]}-{args.hard_aniso[1]} px, "
              f"F2 {args.hard_aniso[2]}-{args.hard_aniso[3]} px)"
              if args.hard_aniso else
              f"({args.hard_px} < d <= {args.hard_band2} px)")
           if args.hard_frac2 > 0 else "") +
          f" · el submuestreo por época lo elige con train_model.py --hard-frac"
          f" (0 = fondo uniforme)")


if __name__ == "__main__":
    main()
