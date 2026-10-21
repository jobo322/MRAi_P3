#!/usr/bin/env python3
"""Inferencia EN LOTE sobre el dataset: un solo modelo cargado para N espectros.

Hace exactamente la misma cadena que `scripts/plotting/run_train_2d.py`
(`predict_P3` -> `p3_tail` -> `P3toP5` -> `P5toPeaks` -> `PeakstoTab`), pero
cargando el modelo UNA vez: 100 espectros tardan ~2,5 min en vez de ~12.

Por cada id deja `<id>.tab` en `--out` — el formato que consume
`view2d.mjs --dataset ... --inference <out>` — y borra los intermedios
(`ft2`, `P3`, `P5`, `peaks`) salvo con `--keep`. Al final escribe
`<out>/batch_report.json` con la configuración, las métricas agregadas, el
recuento por espectro, la cola de la P3 por eje (`p3_shape`: decay medianas a
3/6 px y relación F1/F2) y los que fallaron.

Uso (desde la raíz del repo, con el entorno de TensorFlow):

    ~/.pyenv/versions/tflow/bin/python scripts/plotting/infer_batch.py \\
        --n 100 --split train --model Trained/out_pair_savedmodel \\
        --out /tmp/inf100 --spe-the 0.2 --pl-the 0.6 --msep 3 1

Ojo con qué se mide: `--split train` son espectros VISTOS en entrenamiento (las
métricas salen optimistas); para juzgar el modelo usa `--split val`. Medido:
los mismos 200 de val dan F1 0,844 (`out_elong_2b`) frente a 0,832
(`out_pair`), mientras que en los 8 de la tabla corta la diferencia se invierte
(ver `docs/ring_and_picking_metrics.md` §13).

Y para mirarlo:

    node scripts/plotting/view2d.mjs --dataset <dataset> --inference <out> \\
        --port 8787
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "scripts", "plotting"))
import run_train_2d as R  # noqa: E402  (numpy puro a nivel de módulo)

SPLIT_MANIFEST = os.path.join(REPO, "data", "train_2d_elong", "manifest.json")


def select_ids(manifest_path, split, n):
    """Los primeros `n` ids de `split`, como texto de 4 dígitos.

    Los ids se usan para construir `<id>.bin` y `<id>.tab`, así que se
    normalizan a 4 dígitos (`1` -> `0001`): el manifiesto los trae así, pero un
    JSON escrito a mano puede traerlos como enteros.
    """
    with open(manifest_path) as fh:
        man = json.load(fh)
    if split not in man:
        raise SystemExit(
            f"split '{split}' no está en {manifest_path} "
            f"(hay: {', '.join(k for k in man if isinstance(man[k], list))})")
    return [str(v).zfill(4) for v in man[split][: max(0, int(n))]]


def write_p3_map(out_dir, sid, P):
    """Escribe `<out>/<id>/view2d/{p3.bin,meta.json}` y devuelve ese dir.

    Es el layout que `view2d.mjs` busca con `findP3()` para habilitar el
    selector «P3 probability» del visor de dataset: `<inferencia>/<id>/view2d/
    p3.bin` (float32 LE, filas=F1, columnas=F2) mas un `meta.json` cuyo `shape`
    se valida contra `dataset.json` (256 x 2048). Sin esto el selector queda
    deshabilitado para todos los espectros del lote.
    """
    sid = str(sid).zfill(4)
    vdir = os.path.join(out_dir, sid, "view2d")
    os.makedirs(vdir, exist_ok=True)
    P = np.asarray(P, dtype=np.float32)
    P.astype("<f4").tofile(os.path.join(vdir, "p3.bin"))
    with open(os.path.join(vdir, "meta.json"), "w") as fh:
        json.dump({"shape": [int(P.shape[0]), int(P.shape[1])],
                   "p3": "p3.bin", "source": "infer_batch.py"}, fh, indent=1)
    return vdir


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100,
                    help="cuántos espectros procesar (100)")
    ap.add_argument("--split", default="train", choices=("train", "val"),
                    help="train = vistos en entrenamiento (optimista); "
                         "val = para juzgar el modelo (train)")
    ap.add_argument("--model", default=None,
                    help="SavedModel dir (default: el primero que exista de "
                         "MODEL_CANDIDATES, igual que run_train_2d.py)")
    ap.add_argument("--out", required=True, help="directorio de salida")
    ap.add_argument("--dataset", default=R.DEFAULT_DATASET)
    ap.add_argument("--ids-file", default=SPLIT_MANIFEST,
                    help=f"manifiesto con train/val ({SPLIT_MANIFEST})")
    ap.add_argument("--spe-the", type=float, default=0.2,
                    help="máscara de candidatos, |S| >= the * std(S) (0.2)")
    ap.add_argument("--pl-the", type=float, default=0.6,
                    help="umbral de P3toP5 (0.6: medido, el mejor en 200 ids)")
    ap.add_argument("--msep", type=int, nargs="+", default=[3, 1],
                    help="separación mínima por eje: `3 1` = colapsa la cresta "
                         "en F1 y respeta los pares en F2")
    ap.add_argument("--spec-type", default=R.SPEC_TYPE)
    ap.add_argument("--match-px", type=float, default=2.0,
                    help="tolerancia del emparejado contra la etiqueta (2 px)")
    ap.add_argument("--no-p3-maps", action="store_true",
                    help="no escribe <out>/<id>/view2d/p3.bin (2 MB/espectro): "
                         "sin él el visor no puede mostrar la capa P3")
    ap.add_argument("--keep", action="store_true",
                    help="no borra ft2/P3/P5/peaks de cada espectro")
    ap.add_argument("--fail-fast", action="store_true",
                    help="aborta al primer espectro que falle (por defecto sigue)")
    args = ap.parse_args()

    import nmrglue as ng
    import tensorflow as tf
    from MRAi_P3.P3toP5 import normalize_msep

    model_path = R.pick_model(args.model)
    ids = select_ids(args.ids_file, args.split, args.n)
    msep = normalize_msep(args.msep)
    os.makedirs(args.out, exist_ok=True)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(128, 1)),
        tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")])
    print(f"{len(ids)} espectros de '{args.split}' · modelo "
          f"{os.path.relpath(model_path, REPO)}")
    print(f"cadena: spe_the {args.spe_the} · pl_the {args.pl_the} · msep {msep}",
          flush=True)

    agg = dict(n_detected=0, true_positive=0, n_gt=0)
    shape_acc = {}
    per_id, failed = {}, []
    t0 = time.time()
    for k, sid in enumerate(ids, 1):
        try:
            S, _ = R.load_spectrum(args.dataset, sid)
            gr, gc, _, _ = R.load_ground_truth(args.dataset, sid)
            ft2 = os.path.join(args.out, f"{sid}.ft2")
            R.write_pipe(ft2, S)
            P = R.p3_tail(R.predict_P3(S, model, args.spe_the, verbose=False), S)
            p3 = os.path.join(args.out, f"{sid}.P3")
            dic = ng.pipe.read(ft2)[0]
            ng.pipe.write(p3, dic, np.float32(P), overwrite=True)
            if not args.no_p3_maps:
                write_p3_map(args.out, sid, P)
            R.run_chain(p3, ft2, args.out, sid, args.pl_the, args.spec_type, msep)
            det, _meta = R.read_tab(os.path.join(args.out, f"{sid}.tab"), ft2)
            rep = R.match_report(gr, gc, det, args.match_px)
            shape = R.shape_values(P, gr, gc)
            for key, vals in shape.items():
                shape_acc.setdefault(key, []).extend(vals.tolist())
            for key in agg:
                agg[key] += rep[key]
            per_id[sid] = {k2: rep[k2] for k2 in
                           ("n_gt", "n_detected", "true_positive",
                            "false_positive", "false_negative",
                            "precision", "recall")}
            per_id[sid]["p3_shape"] = R.shape_report(shape)
            if not args.keep:
                for ext in (".ft2", ".P3", ".P5", "_P5.npy", ".peaks"):
                    p = os.path.join(args.out, f"{sid}{ext}")
                    if os.path.isfile(p):
                        os.remove(p)
        except Exception as exc:                      # noqa: BLE001
            failed.append({"id": sid, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise
            print(f"  FALLO {sid}: {type(exc).__name__}: {exc}", flush=True)
        if k % 10 == 0 or k == len(ids):
            n = max(1, k - len(failed))
            P_ = agg["true_positive"] / max(1, agg["n_detected"])
            Rc = agg["true_positive"] / max(1, agg["n_gt"])
            f1 = 2 * P_ * Rc / max(1e-12, P_ + Rc)
            print(f"  {k}/{len(ids)}  {time.time()-t0:5.0f}s  det "
                  f"{agg['n_detected']} tp {agg['true_positive']}  "
                  f"P {P_:.3f} R {Rc:.3f} F1 {f1:.3f}", flush=True)

    P_ = agg["true_positive"] / max(1, agg["n_detected"])
    Rc = agg["true_positive"] / max(1, agg["n_gt"])
    F1 = 2 * P_ * Rc / max(1e-12, P_ + Rc)
    p3_shape = R.shape_report({k: np.asarray(v, dtype=float)
                               for k, v in shape_acc.items()})
    report = {
        "split": args.split, "n_requested": args.n, "n_done": len(per_id),
        "model": os.path.relpath(model_path, REPO), "dataset": args.dataset,
        "chain": {"spe_the": args.spe_the, "pl_the": args.pl_the,
                  "msep": msep, "spec_type": args.spec_type,
                  "match_px": args.match_px},
        "aggregate": {"n_gt": agg["n_gt"], "n_detected": agg["n_detected"],
                      "true_positive": agg["true_positive"],
                      "det_over_gt": agg["n_detected"] / max(1, agg["n_gt"]),
                      "precision": P_, "recall": Rc, "f1": F1},
        "p3_shape": p3_shape,
        "p3_maps": not args.no_p3_maps,
        "seconds": round(time.time() - t0, 1),
        "failed": failed, "per_id": per_id,
        "caveat": ("--split train son espectros vistos en entrenamiento: las "
                   "métricas son optimistas. Para juzgar el modelo usa "
                   "--split val."),
    }
    with open(os.path.join(args.out, "batch_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    print(f"\nTOTAL {len(per_id)} espectros ({agg['n_gt']} picos reales): "
          f"picos {agg['n_detected']} ({agg['n_detected']/max(1, agg['n_gt']):.2f}x)"
          f"  precision {P_:.3f}  recall {Rc:.3f}  F1 {F1:.3f}"
          f"  en {time.time()-t0:.0f}s")
    if p3_shape["n"]:
        s3 = p3_shape["at_3px"]
        print(f"P3 shape: F1/F2 @3px {s3['f1_over_f2']:.2f} "
              f"(F1 {s3['f1']:.3f}, F2 {s3['f2']:.3f})  "
              f"@6px {p3_shape['at_6px']['f1_over_f2']:.2f}")
    if failed:
        print(f"fallaron {len(failed)}: {[f['id'] for f in failed][:5]}"
              f"{' ...' if len(failed) > 5 else ''}")
    print(f"\nreport -> {os.path.join(args.out, 'batch_report.json')}")
    print("para mirarlo:\n"
          f"  node scripts/plotting/view2d.mjs --dataset {args.dataset} "
          f"--inference {args.out} --port 8787")


if __name__ == "__main__":
    main()
