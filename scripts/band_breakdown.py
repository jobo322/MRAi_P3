"""Desglose (d1, d2) de la etiqueta sobre el dataset completo.

Por qué
-------
`aniso_band_mask` define la banda de negativos duros como un rectángulo en
(d1, d2) = (|Δfila|, |Δcolumna|) al positivo más cercano, así que elegir
`--hard-aniso F1LO F1HI F2LO F2HI` es elegir cuánta masa de la etiqueta entra al
entrenamiento. El pool guarda solo una muestra de ese fondo (`pool_k * n_pos`
filas, con `hard_frac2` reservado a la banda): sirve para proporciones —medido,
20,2 % en d2=0 en el pool frente a 20,0 % en la imagen— pero no para contar
píxeles, porque guarda ~27 % de la banda. Este script cuenta sobre los labels
completos (`<dataset>/labels/<id>.bin`), sin pool, para los 1000 espectros, y
desglosa por clase (PG = puro gaussiano, PL = puro lorentziano, MIX) y por split.

Qué reporta
-----------
  * `hist`: histograma 2D de (d1, d2) del fondo, recortado a `--d1-max` /
    `--d2-max` (el último bin acumula la cola).
  * `bands`: por cada rectángulo pedido con `--band F1LO F1HI F2LO F2HI`:
    píxeles de fondo por espectro, % del fondo, desglose por d2, y cuántos caen
    en el eje F1 puro (d2=0) frente a la diagonal (d1>0 y d2>0).
  * `by_class` y `by_split`: los mismos totales agrupados.
  * `ring_px`: fondo con distancia euclídea `d <= --hard-px`. Es lo que acota
    `hard_frac`: si el anillo no alcanza, `choose_background` rellena con fondo
    fácil y la fracción efectiva queda por debajo de la pedida.

El fondo se cuenta igual que el pool: solo píxeles con `label == 0` (los
positivos son la etiqueta sub-píxel de 1/2/4 px y no entran a ninguna banda).

Fondo de etiqueta vs. intensidad
--------------------------------
`fondo` acá es una noción de ETIQUETA, no de ruido: son los 523 712 px de
`label == 0` (el 99,9 % de la imagen). Incluye tres cosas muy distintas:

  * ruido solo, lejos de todo pico;
  * las colas y hombros de los picos, con intensidad por encima del ruido;
  * el INTERIOR de los picos: la etiqueta es un marcador de 1/2/4 px en el
    centro, así que el resto del pico también es `label == 0`.

`fácil` (en `choose_background`) tampoco es sinónimo de ruido: es el fondo
fuera del anillo y de la banda, un criterio de DISTANCIA al positivo más
cercano. Puede ser ruido o la cola lejana de un pico fuerte.

Con `--intensity` se agrega el eje de intensidad, que es ortogonal: para cada
píxel de fondo se toma la ventana de 64 px de su brazo de cross (lo que ve el
modelo) y se mide `snr = max(brazo) / σ`, con `σ` el ruido gaussiano del dataset
(SD = A_min = 0,001, uniforme para los 1000 espectros). `snr < 1` es ruido puro;
`snr >= 3` tiene un bulto discernible.

Ojo con la geometría: el cross de un píxel son SU fila y SU columna, así que un
pico entra al cross del píxel solo si comparte fila (d1 = 0) o columna (d2 = 0)
con él. Un negativo en diagonal (d1 > 0 y d2 > 0) tiene un cross que NO contiene
a ese pico: lo que ve son las colas y los picos que crucen su fila o su columna.
Eso hace que `d2 = 0` y la diagonal no sean el mismo tipo de negativo, por más
que estén en el mismo rectángulo. Distancia e intensidad tampoco coinciden: cerca
de un pico débil (A ≈ 0,002) la cola está por debajo del ruido, y lejos de un
pico fuerte (A ≈ 1) puede seguir por encima.

Y el modelo NO ve la amplitud absoluta: `cross_extract.split_2d` y
`split_2d_rows` normalizan cada cross por su norma L2 y la primera capa es
`softsign(10 * x)`, así que la entrada es invariante a escala. `snr` mide si el
dato tiene un bulto discernible —lo que hace que la FORMA del cross sea
informativa—, no lo que el modelo recibe como número.

Uso
---
    python scripts/band_breakdown.py --out /tmp/band_breakdown.json
    python scripts/band_breakdown.py --intensity --out /tmp/band_breakdown_snr.json

Por defecto recorre los 1000 labels, lee la clase de `<dataset>/manifest.jsonl`,
usa todos los núcleos y escribe el JSON. `--ids-file data/train_2d_elong/manifest.json`
restringe a un split del repo.
"""
import argparse
import bisect
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_negatives import aniso_band_mask, axis_distance_to_positive  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = "/Users/abolanos/git/cheminfo/spectrum-generator/dataset"
SHAPE = (256, 2048)
HARD_PX = 2
# Ventana de cada brazo del cross (CROSS = [64, 64] en preextract.py) y ruido
# gaussiano del dataset (dataset.json, conventions.noise: SD = A_min = 0.001).
ARM = 64
NOISE_SD = 0.001
SNR_EDGES = (1.0, 3.0, 10.0, 100.0, 1000.0)
# Bandas por defecto: la del pool f2lo0 actual, la del runbook, la ablación
# `d2 <= 1` y el anillo (para contraste).
DEFAULT_BANDS = ((3, 10, 0, 3), (3, 10, 1, 3), (3, 10, 0, 1), (1, 2, 0, 3))
CLASS_RE = re.compile(r'^\{"id":(\d+),"class":"([A-Z]+)"')
NOISE_RE = re.compile(r'"noise":\{[^}]*?"sd":([0-9.eE+-]+)')


def read_noise_sd(dataset):
    """σ del ruido gaussiano añadido al render, de `<dataset>/manifest.jsonl`.

    `dataset.json` lo fija en `SD = A_min` = 0,001 para los 1000 espectros; se lee
    del manifiesto por si alguna vez cambia por espectro.
    """
    path = os.path.join(dataset, "manifest.jsonl")
    vals = []
    if os.path.isfile(path):
        with open(path) as fh:
            for line in fh:
                m = NOISE_RE.search(line)
                if m:
                    vals.append(float(m.group(1)))
    return float(np.median(vals)) if vals else NOISE_SD


def snr_labels(edges):
    """Etiquetas de los buckets de SNR: `<1`, `1-3`, ..., `>=1000`."""
    labels, lo = [], 0.0
    for hi in edges:
        labels.append(f"<{hi:g}" if lo == 0.0 else f"{lo:g}-{hi:g}")
        lo = hi
    labels.append(f">={lo:g}")
    return tuple(labels)


def band_key(band):
    """Clave de una banda en los agregados: `band:3-10x0-3`."""
    f1lo, f1hi, f2lo, f2hi = band
    return f"band:{f1lo}-{f1hi}x{f2lo}-{f2hi}"


def snr_buckets(values, edges=None):
    """Cuenta `values` en los buckets de SNR definidos por `edges`."""
    v = np.asarray(values, dtype=np.float64)
    edges = SNR_EDGES if edges is None else tuple(edges)
    return [int(np.count_nonzero((v >= lo) & (v < hi)))
            for lo, hi in zip((0.0,) + edges, edges + (np.inf,))]


def read_classes(dataset):
    """{sid: 'PG'|'PL'|'MIX'} de `<dataset>/manifest.jsonl`, sin parsear el JSON entero.

    El manifiesto es JSONL con `{"id":N,"class":"X"` al principio de cada línea
    y ~100 kB por línea de picos; el prefijo basta para la clase.
    """
    path = os.path.join(dataset, "manifest.jsonl")
    classes = {}
    if not os.path.isfile(path):
        return classes
    with open(path) as fh:
        for line in fh:
            m = CLASS_RE.match(line)
            if m:
                classes[int(m.group(1))] = m.group(2)
    return classes


def label_ids(dataset):
    """Ids (str de 4 dígitos) con label en `<dataset>/labels/`."""
    d = os.path.join(dataset, "labels")
    ids = [f[:-4] for f in os.listdir(d) if f.endswith(".bin")]
    return sorted(ids)


def split_of(dataset, ids):
    """{sid: 'train'|'val'} leído de `<dataset>/dataset.json` (o vacío)."""
    path = os.path.join(dataset, "dataset.json")
    out = {}
    try:
        with open(path) as fh:
            sp = (json.load(fh) or {}).get("split") or {}
    except (OSError, ValueError):
        return out
    for name in ("train", "val"):
        for sid in sp.get(name, []) or []:
            out[str(sid).zfill(4)] = name
    return out


def measure_one(task):
    """Mide un espectro: histograma (d1, d2) y conteos por banda, solo del fondo.

    Con `noise_sd` (no None) agrega `snr`: el máximo del brazo de 64 px del cross
    dividido por σ, por región del pool (anillo, cada banda y `fácil`).
    """
    sid, cls, split, dataset, bands, d1_max, d2_max, hard_px, noise_sd = task
    L = np.fromfile(os.path.join(dataset, "labels", f"{sid}.bin"),
                    dtype=np.uint8).reshape(SHAPE)
    d, d1, d2 = axis_distance_to_positive(L)
    bg = (L == 0).reshape(-1)
    d1b = d1.reshape(-1)[bg].astype(np.int64)
    d2b = d2.reshape(-1)[bg].astype(np.int64)
    db = d.reshape(-1)[bg].astype(np.int64)

    hist = np.bincount(np.minimum(d1b, d1_max) * (d2_max + 1) +
                       np.minimum(d2b, d2_max),
                       minlength=(d1_max + 1) * (d2_max + 1))
    band_counts = {}
    for (f1lo, f1hi, f2lo, f2hi) in bands:
        m = aniso_band_mask(d1b, d2b, f1lo, f1hi, f2lo, f2hi)
        band_counts[f"{f1lo}-{f1hi}x{f2lo}-{f2hi}"] = {
            "total": int(m.sum()),
            "per_d2": [int((m & (d2b == v)).sum())
                       for v in range(f2lo, f2hi + 1)],
            "d2_zero": int((m & (d2b == 0)).sum()),
            "diagonal": int((m & (d1b > 0) & (d2b > 0)).sum()),
        }
    out = {
        "id": sid, "class": cls, "split": split,
        "n_pos": int((L > 0).sum()), "n_bg": int(bg.sum()),
        "ring_px": int((db <= hard_px).sum()),
        "hist": hist.tolist(),
        "bands": band_counts,
    }
    if noise_sd:
        S = np.fromfile(os.path.join(dataset, "spectra", f"{sid}.bin"),
                        dtype="<f4").reshape(SHAPE)
        # Los dos brazos del cross, con el mismo padding wrap del modelo.
        arm = np.maximum(maximum_filter1d(S, ARM, axis=0, mode="wrap"),
                         maximum_filter1d(S, ARM, axis=1, mode="wrap"))
        snr = (arm.reshape(-1)[bg] / float(noise_sd)).astype(np.float64)
        regions = {"anillo": db <= hard_px}
        for (f1lo, f1hi, f2lo, f2hi) in bands:
            regions[band_key((f1lo, f1hi, f2lo, f2hi))] = aniso_band_mask(
                d1b, d2b, f1lo, f1hi, f2lo, f2hi)
        # `fácil` es lo que el pool deja fuera del anillo y de la PRIMERA banda.
        easy = ~(regions["anillo"] | regions[band_key(bands[0])])
        regions["facil"] = easy
        out["snr"] = {name: snr_buckets(snr[mask]) for name, mask in regions.items()}
    return out


def _sum_snr(dst, src):
    for name, buckets in src.items():
        cur = dst.setdefault(name, [0] * len(buckets))
        for i, n in enumerate(buckets):
            cur[i] += n


def _sum_bands(dst, src):
    for key, val in src.items():
        cur = dst.setdefault(key, {"total": 0, "per_d2": [0] * len(val["per_d2"]),
                                   "d2_zero": 0, "diagonal": 0})
        cur["total"] += val["total"]
        cur["d2_zero"] += val["d2_zero"]
        cur["diagonal"] += val["diagonal"]
        for i, n in enumerate(val["per_d2"]):
            cur["per_d2"][i] += n


def aggregate(rows, d1_max, d2_max):
    """Totales del dataset y agrupados por clase y por split."""
    total = {"n_ids": 0, "n_pos": 0, "n_bg": 0, "ring_px": 0,
             "hist": [0] * ((d1_max + 1) * (d2_max + 1)), "bands": {}, "snr": {}}
    by_class, by_split = {}, {}
    for row in rows:
        for acc in (total, by_class.setdefault(row["class"], {
                        "n_ids": 0, "n_pos": 0, "n_bg": 0, "ring_px": 0,
                        "hist": [0] * len(total["hist"]), "bands": {}, "snr": {}}),
                    by_split.setdefault(row["split"], {
                        "n_ids": 0, "n_pos": 0, "n_bg": 0, "ring_px": 0,
                        "hist": [0] * len(total["hist"]), "bands": {}, "snr": {}})):
            acc["n_ids"] += 1
            acc["n_pos"] += row["n_pos"]
            acc["n_bg"] += row["n_bg"]
            acc["ring_px"] += row["ring_px"]
            for i, n in enumerate(row["hist"]):
                acc["hist"][i] += n
            _sum_bands(acc["bands"], row["bands"])
            if "snr" in row:
                _sum_snr(acc["snr"], row["snr"])
    return total, by_class, by_split


def _print_hist(acc, d1_max, d2_max, d2_show=6, d1_show=12):
    n_bg = acc["n_bg"] or 1
    print(f"  histograma (d1 x d2), % del fondo de {acc['n_ids']} espectros:")
    cols = list(range(d2_show + 1)) + [d2_max]
    head = "".join(f"{('d2>=%d' % d2_max) if c == d2_max else ('d2=%d' % c):>9}"
                   for c in cols)
    print(f"  {'':>6}{head}")
    for r in list(range(d1_show + 1)) + [d1_max]:
        cells = []
        for c in cols:
            v = acc["hist"][r * (d2_max + 1) + c]
            cells.append(f"{100.0 * v / n_bg:>8.3f}%")
        label = f"d1>={d1_max}" if r == d1_max else f"d1={r}"
        print(f"  {label:>6}" + "".join(cells))
    print("  (el último bin de cada eje acumula la cola)")


def _print_bands(acc, bands):
    n_bg = acc["n_bg"] or 1
    n_ids = acc["n_ids"] or 1
    print(f"  bandas (fondo: {acc['n_bg'] / n_ids:,.0f} px/espectro, "
          f"positivos {acc['n_pos'] / n_ids:,.0f}/esp, "
          f"anillo d<={HARD_PX} {acc['ring_px'] / n_ids:,.0f}/esp):")
    for (f1lo, f1hi, f2lo, f2hi) in bands:
        b = acc["bands"][f"{f1lo}-{f1hi}x{f2lo}-{f2hi}"]
        tot = b["total"]
        print(f"    {f1lo}-{f1hi} x {f2lo}-{f2hi}: {tot / n_ids:,.0f} px/esp "
              f"({100.0 * tot / n_bg:.3f}% del fondo)")
        parts = ", ".join(f"d2={f2lo + i} {100.0 * n / tot:.1f}%"
                          for i, n in enumerate(b["per_d2"]) if tot)
        print(f"      {parts}")
        print(f"      eje F1 puro (d2=0) {b['d2_zero'] / n_ids:,.0f}/esp "
              f"({100.0 * b['d2_zero'] / tot:.1f}% de la banda) | "
              f"diagonal {b['diagonal'] / n_ids:,.0f}/esp "
              f"({100.0 * b['diagonal'] / tot:.1f}%)")


def _print_snr(acc, bands, noise_sd):
    if not acc.get("snr"):
        return
    n_ids = acc["n_ids"] or 1
    print(f"  intensidad (σ = {noise_sd:g}, brazo de {ARM} px; snr = max(brazo)/σ):")
    labels = snr_labels(SNR_EDGES)
    print(f"    {'región':<22}{'px/esp':>10}" +
          "".join(f"{'snr' + lab:>9}" for lab in labels))
    names = ["anillo"] + [band_key(b) for b in bands] + ["facil"]
    for name in names:
        buckets = acc["snr"].get(name)
        if not buckets:
            continue
        n = sum(buckets) or 1
        cells = "".join(f"{100.0 * b / n:>8.1f}%" for b in buckets)
        print(f"    {name:<22}{n / n_ids:>10,.0f}{cells}")


def _snr_above(buckets, threshold):
    """Suma de los buckets cuyo borde inferior es >= `threshold`."""
    start = bisect.bisect_left(list(SNR_EDGES), threshold) + 1
    return sum(buckets[start:])


def _print_snr_by_class(by_class, bands):
    """Línea de snr de la primera banda por clase: px/esp y % con snr >= 3."""
    key = band_key(bands[0])
    parts = []
    for cls in sorted(by_class):
        buckets = by_class[cls]["snr"].get(key)
        if not buckets:
            continue
        n = sum(buckets) or 1
        parts.append(f"{cls} {n / max(by_class[cls]['n_ids'], 1):,.0f}/esp "
                     f"({100.0 * _snr_above(buckets, 3) / n:.1f}% snr>=3)")
    if parts:
        print(f"    {key} por clase: " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--ids-file", default=None,
                    help="manifest.json del repo (train/val) para restringir los ids")
    ap.add_argument("--band", type=int, nargs=4, action="append", default=None,
                    metavar=("F1LO", "F1HI", "F2LO", "F2HI"),
                    help="rectángulo a medir; repetible (default: las 4 bandas de referencia)")
    ap.add_argument("--d1-max", type=int, default=30)
    ap.add_argument("--d2-max", type=int, default=10)
    ap.add_argument("--hard-px", type=int, default=HARD_PX)
    ap.add_argument("--intensity", action="store_true",
                    help="agrega el eje de intensidad (snr del brazo del cross): lee "
                         "también spectra/<id>.bin, ~3x más lento")
    ap.add_argument("--noise-sd", type=float, default=None,
                    help=f"σ del ruido para snr (default: el del manifiesto, {NOISE_SD:g})")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default="/tmp/band_breakdown.json")
    args = ap.parse_args()

    bands = tuple(tuple(b) for b in (args.band or DEFAULT_BANDS))
    for f1lo, f1hi, f2lo, f2hi in bands:
        if min(f1lo, f1hi, f2lo, f2hi) < 0 or f1lo > f1hi or f2lo > f2hi:
            raise SystemExit(f"banda inválida: {f1lo} {f1hi} {f2lo} {f2hi}")

    ids = label_ids(args.dataset)
    if args.ids_file:
        with open(args.ids_file) as fh:
            manifest = json.load(fh)
        wanted = set(manifest.get("train", [])) | set(manifest.get("val", []))
        ids = [i for i in ids if i in wanted]
    if not ids:
        raise SystemExit(f"no hay labels en {args.dataset}/labels")

    classes = read_classes(args.dataset)
    splits = split_of(args.dataset, ids)
    noise_sd = None
    if args.intensity:
        noise_sd = args.noise_sd if args.noise_sd is not None else read_noise_sd(args.dataset)
        if noise_sd <= 0:
            raise SystemExit(f"σ inválido: {noise_sd}")
    tasks = [(sid, classes.get(int(sid), "?"), splits.get(sid, "?"), args.dataset,
              bands, args.d1_max, args.d2_max, args.hard_px, noise_sd) for sid in ids]

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(measure_one, tasks, chunksize=4))
    else:
        rows = [measure_one(t) for t in tasks]

    total, by_class, by_split = aggregate(rows, args.d1_max, args.d2_max)
    cls_count = {}
    for row in rows:
        cls_count[row["class"]] = cls_count.get(row["class"], 0) + 1

    print(f"ids: {len(ids)} | clases: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(cls_count.items()))} | "
          f"split: " + ", ".join(f"{k} {v['n_ids']}" for k, v in sorted(by_split.items())))
    _print_hist(total, args.d1_max, args.d2_max)
    _print_bands(total, bands)
    if noise_sd:
        _print_snr(total, bands, noise_sd)
        _print_snr_by_class(by_class, bands)
    print("  por clase (misma banda, en px/espectro):")
    for cls in sorted(by_class):
        acc = by_class[cls]
        cells = ", ".join(
            f"{f1lo}-{f1hi}x{f2lo}-{f2hi} "
            f"{acc['bands'][f'{f1lo}-{f1hi}x{f2lo}-{f2hi}']['total'] / max(acc['n_ids'], 1):,.0f}"
            for (f1lo, f1hi, f2lo, f2hi) in bands)
        print(f"    {cls}: fondo {acc['n_bg'] / max(acc['n_ids'], 1):,.0f}/esp | {cells}")

    payload = {
        "dataset": args.dataset,
        "ids": len(ids),
        "bands": [list(b) for b in bands],
        "d1_max": args.d1_max, "d2_max": args.d2_max, "hard_px": args.hard_px,
        "noise_sd": noise_sd,
        "total": total, "by_class": by_class, "by_split": by_split,
        "per_id": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
