"""Fase 1 — Ingesta del dataset jres-peak-picking-2D (.bin + manifest.jsonl).

Lee spectra/<id>.bin (float32 LE, shape (256,2048) = (F1,F2)) y labels/<id>.bin
(uint8, misma forma), valida contra manifest.jsonl + dataset.json y escribe un
manifiesto de entrenamiento SIN pre-expandir X (1000 X completos ~= 268 GB).

El entrenamiento (Fase 2) extrae batches bajo demanda con cross_extract.split_2d
vectorizado sobre los .bin originales via memmap.

Uso:
  python3 scripts/jres_ingest.py --dataset /Users/abolanos/git/cheminfo/spectrum-generator/dataset \
      --out /Users/abolanos/git/mrai_p3/data/train_2d_elong --validate-only
  python3 scripts/jres_ingest.py --dataset .../dataset --out .../train_2d_elong
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from cross_extract import split_2d  # noqa: E402

SHAPE = (256, 2048)  # (F1=y corta, F2=x larga)


def load_pair(ds_dir, sid):
    s_path = os.path.join(ds_dir, "spectra", f"{sid}.bin")
    l_path = os.path.join(ds_dir, "labels", f"{sid}.bin")
    S = np.fromfile(s_path, dtype="<f4").reshape(SHAPE).astype(np.float32)
    L = np.fromfile(l_path, dtype=np.uint8).reshape(SHAPE).astype(np.float32)
    return np.ascontiguousarray(S), np.ascontiguousarray(L)


def main():
    ap = argparse.ArgumentParser(description="Fase 1: ingesta dataset jres .bin")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--probe", type=int, default=3,
                    help="n. espectros con extraccion X completa de verificacion")
    args = ap.parse_args()

    ds = args.dataset
    cfg = json.load(open(os.path.join(ds, "dataset.json")))
    st = cfg["storage"]
    assert st["spectra"]["shape"] == [256, 2048] and st["spectra"]["dtype"] == "float32"
    assert st["labels"]["shape"] == [256, 2048] and st["labels"]["dtype"] == "uint8"
    print(f"dataset kind={cfg['kind']} v={cfg['version']} class-mix={cfg['classes']}")

    man_lines = open(os.path.join(ds, "manifest.jsonl")).read().strip().split("\n")
    n = len(man_lines)
    print(f"manifest: {n} espectros")
    split = cfg["split"]
    print(f"split dataset.json: {len(split['train'])} train / {len(split['val'])} val")

    errs, tot_pos, bands = [], 0, {"solid": 0, "gray": 0, "detect": 0}
    for li, ln in enumerate(man_lines):
        e = json.loads(ln)
        sid = f"{e['id']:04d}"
        S, L = load_pair(ds, sid)
        H, W = S.shape
        if (H, W) != SHAPE:
            errs.append(f"{sid}: shape {S.shape}")
        if not np.isfinite(S).all():
            errs.append(f"{sid}: non-finite S")
        if set(np.unique(L).tolist()) - {0.0, 1.0}:
            errs.append(f"{sid}: L no binaria")
        # pixelSet manifest vs bits .bin
        ps = set(tuple(px) for p in e["peaks"] for px in p["pixelSet"])
        lab = set(map(tuple, np.argwhere(L == 1).tolist()))
        if ps != lab:
            errs.append(f"{sid}: pixelSet({len(ps)}) != labels({len(lab)})")
        tot_pos += int(L.sum())
        sd = e["noise"]["sd"]
        for p in e["peaks"]:
            snr = p["A"] / sd
            bands["solid" if snr > 10 else ("gray" if snr >= 3 else "detect")] += 1
        if errs and len(errs) > 10:
            break
    tot = sum(bands.values())
    print(f"picos totales: {tot} bandas: " +
          ", ".join(f"{k} {v} ({v/tot*100:.1f}%)" for k, v in bands.items()))
    print(f"positivos en labels: {tot_pos} (incluye pixeles dobles sub-pixel)")
    if errs:
        print(f"ERRORES ({len(errs)}):")
        [print("  FAIL", x) for x in errs[:20]]
        sys.exit(1)
    print("validacion binaria OK: 0 errores")

    # sonda de extraccion X en los primeros --probe espectros
    for li in range(min(args.probe, n)):
        e = json.loads(man_lines[li])
        sid = f"{e['id']:04d}"
        S, _ = load_pair(ds, sid)
        X, Norm = split_2d(S.astype(np.float64), np.array([64, 64]))
        X = np.asarray(X, dtype=np.float32)
        X[~np.isfinite(X)] = 0.0
        bi, bj = np.unravel_index(int(np.argmax(S)), S.shape)
        v = X[bi * SHAPE[1] + bj, :, 0]
        w1 = int((v[:64] >= 0.5 * v[:64].max()).sum())
        w2 = int((v[64:] >= 0.5 * v[64:].max()).sum())
        flag = "OK" if (w1 >= w2 and np.isfinite(X).all()) else "SUSPECT"
        print(f"  probe {sid}: X={X.shape} FWHM=({w1},{w2}) finite={bool(np.isfinite(X).all())} {flag}")

    if args.validate_only:
        return
    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "source": os.path.abspath(ds),
        "storage": "referenced .bin via memmap (no pre-expandidos)",
        "shape": list(SHAPE),
        "train": [f"{i:04d}" for i in split["train"]],
        "val": [f"{i:04d}" for i in split["val"]],
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"Done: {len(manifest['train'])} train / {len(manifest['val'])} val -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
