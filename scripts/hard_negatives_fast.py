#!/usr/bin/env python3
"""Alternativa rápida de `hard_negatives.choose_background`, SIN tocar el original.

Implementación paralela para poder medir y entrenar mientras el repo está en
uso: no modifica `scripts/hard_negatives.py`, `scripts/train_model.py` ni
`scripts/preextract.py`. Todo lo que hace falta para adoptarla después es
cambiar un import.

El único cambio de fondo es el complemento ordenado:

    rest = np.setdiff1d(np.arange(len(candidates)), reserved)

`np.arange` ya viene ordenado y `reserved` son posiciones únicas, así que
`setdiff1d` paga dos `uniquify+sort` para devolver lo que una máscara booleana
da en dos pasadas O(n). En un pool de 14 525 filas son ~3 ms por espectro, y
el bucle de datos de una época lo llama 800 veces.

Uso:
    python scripts/hard_negatives_fast.py --check       # equivalencia (sin TF)
    python scripts/hard_negatives_fast.py --check-e2e   # equivalencia por subsample_pool
    python scripts/hard_negatives_fast.py --bench       # tiempos función a función
    python scripts/hard_negatives_fast.py --train -- <args de train_model.py>

La equivalencia no es "parecida": `--check` compara los índices y el estado del
`rng` resultante contra el original en miles de configuraciones, y `--check-e2e`
compara el lote completo que devuelve `subsample_pool` (X e y) bit a bit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hard_negatives import band_sections  # noqa: E402
from hard_negatives import choose_background as _reference  # noqa: E402


def _sorted_complement(n, excluded):
    """Posiciones 0..n-1 que NO están en `excluded`, en orden ascendente.

    Equivalente exacto a `np.setdiff1d(np.arange(n), excluded)` pero sin
    ordenar: `np.arange` ya sale ordenado, `excluded` son posiciones únicas y
    `np.flatnonzero` devuelve los índices en orden creciente.
    """
    mask = np.ones(n, dtype=bool)
    if len(excluded):
        mask[np.asarray(excluded, dtype=np.intp)] = False
    return np.flatnonzero(mask)


def choose_background(candidates, d_candidates, n_bg, hard_frac, hard_px,
                      rng, band2_px=None, band2_frac=0.0, band2_mask=None):
    """Copia de `hard_negatives.choose_background` con `rest` por máscara.

    Misma firma, mismo consumo de `rng`, misma salida: la única diferencia es
    cómo se calcula el complemento de las filas reservadas.
    """
    candidates = np.asarray(candidates)
    d_candidates = np.asarray(d_candidates)
    n_bg = min(int(n_bg), len(candidates))
    if n_bg <= 0:
        return np.array([], dtype=np.int64)
    if band2_mask is not None and band2_px is not None:
        raise ValueError("band2_px y band2_mask son excluyentes")
    if band2_mask is not None:
        if band2_frac <= 0:
            raise ValueError("band2_mask necesita band2_frac > 0")
        band2_mask = np.asarray(band2_mask)
        if band2_mask.shape != candidates.shape:
            raise ValueError(
                f"band2_mask {band2_mask.shape} != candidates {candidates.shape}")
    hard = np.where(d_candidates <= hard_px)[0]
    use_band2 = band2_frac > 0 and (band2_px is not None or band2_mask is not None)
    if use_band2:
        if band2_mask is not None:
            band2 = np.where(band2_mask & (d_candidates > hard_px))[0]
        else:
            band2 = np.where((d_candidates > hard_px) &
                             (d_candidates <= band2_px))[0]
    else:
        band2 = np.array([], dtype=np.int64)

    if not use_band2 and (hard_frac <= 0 or len(hard) == 0):
        # Camino histórico intacto: un único sorteo uniforme sobre todo el fondo.
        return rng.choice(len(candidates), size=n_bg, replace=False)

    n_hard_req, n_band2_req, _ = band_sections(n_bg, hard_frac, band2_frac)
    n_hard = min(len(hard), n_hard_req)
    take_hard = (rng.choice(hard, size=n_hard, replace=False) if n_hard
                 else np.array([], dtype=np.int64))
    n_band2 = min(len(band2), n_band2_req)
    take_band2 = (rng.choice(band2, size=n_band2, replace=False) if n_band2
                  else np.array([], dtype=np.int64))

    # El resto absorbe lo que falte en las bandas, así el tamaño nunca cambia.
    reserved = np.concatenate([hard, band2]) if use_band2 else hard
    rest = _sorted_complement(len(candidates), reserved)
    n_easy = n_bg - len(take_hard) - len(take_band2)
    if n_easy > len(rest):
        # Rama fría (solo si el anillo agota el fondo): `setdiff1d` cuesta poco
        # aquí y se deja idéntico al original, duplicados incluidos.
        taken = np.concatenate([take_hard, take_band2])
        remaining = np.setdiff1d(reserved, taken)
        extra = rng.choice(remaining, size=n_easy - len(rest), replace=False)
        take_easy = np.concatenate([rest, extra])
    elif n_easy:
        take_easy = rng.choice(rest, size=n_easy, replace=False)
    else:
        take_easy = np.array([], dtype=np.int64)
    return np.concatenate([take_hard, take_band2, take_easy]).astype(np.int64)


choose_background_fast = choose_background


def install(modules=("train_model", "preextract")):
    """Parchea `choose_background` en los módulos ya importados.

    `subsample_pool` y `plan_pool_idx` resuelven el global del módulo en cada
    llamada, así que basta con reemplazar el atributo. Devuelve los módulos
    parcheados de verdad (los que estaban en `sys.modules`).
    """
    import hard_negatives as hn
    hn.choose_background = choose_background
    patched = []
    for name in modules:
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "choose_background"):
            mod.choose_background = choose_background
            patched.append(name)
    return patched


# --------------------------------------------------------------------------
# verificación / medida
# --------------------------------------------------------------------------

def _pool_fixture(seed=0, n=14525, n_pos=581):
    """Pool sintético con la geometría real (d, d1, d2 y anillo no vacío)."""
    rng = np.random.default_rng(seed)
    d = np.concatenate([rng.integers(0, 3, size=n_pos + 120),
                        rng.integers(3, 40, size=n - n_pos - 120)])
    d1 = np.abs(d + rng.integers(-2, 3, size=n)).astype(np.int32)
    d2 = np.abs(d // 3 + rng.integers(-1, 2, size=n)).astype(np.int32)
    return d, d1, d2


def _cases():
    """Configuraciones de `choose_background` a comparar."""
    d, d1, d2 = _pool_fixture()
    from hard_negatives import aniso_band_mask
    for n in (1, 7, 300, len(d)):
        for hard_frac in (0.0, 0.25, 0.55, 0.9, 1.0):
            for band2_frac in (0.0, 0.25):
                if band2_frac:
                    mask = aniso_band_mask(d1[:n], d2[:n], 3, 10, 0, 3)
                    if hard_frac:
                        yield (n, hard_frac, band2_frac, None, mask)
                    else:
                        # banda anisótropa sin anillo: camino distinto
                        yield (n, hard_frac, band2_frac, None, mask)
                else:
                    yield (n, hard_frac, 0.0, 40, None)
    # banda isotrópica
    for hard_frac in (0.0, 0.2, 0.55, 0.9):
        for band2_px in (5, 20, 60):
            yield (len(d), hard_frac, 0.3, band2_px, None)


def cmd_check(verbose=True):
    d, d1, d2 = _pool_fixture()
    n_pos = 581
    bad = 0
    total = 0
    for n, hard_frac, band2_frac, band2_px, mask in _cases():
        if hard_frac + band2_frac > 1.0:
            # `band_sections` lo rechaza: se comprueba aparte que ambas
            # implementaciones fallen igual.
            continue
        cand = np.arange(n_pos, n_pos + n)
        dd = d[:n]
        for n_bg in sorted({1, 37, 581, 1452, 2324, 4000}):
            for seed in (0, 1, 7):
                a_rng = np.random.default_rng(seed)
                b_rng = np.random.default_rng(seed)
                a = _reference(cand, dd, n_bg, hard_frac, 2, a_rng,
                               band2_px=band2_px, band2_frac=band2_frac,
                               band2_mask=mask)
                b = choose_background(cand, dd, n_bg, hard_frac, 2, b_rng,
                                      band2_px=band2_px, band2_frac=band2_frac,
                                      band2_mask=mask)
                total += 1
                same = (np.array_equal(a, b) and a.dtype == b.dtype
                        and a_rng.bit_generator.state == b_rng.bit_generator.state)
                if not same:
                    bad += 1
                    if bad <= 5:
                        print(f"  DIFIERE: n={n} n_bg={n_bg} hard_frac={hard_frac} "
                              f"band2_frac={band2_frac} band2_px={band2_px} "
                              f"mask={mask is not None} seed={seed}")
                        print(f"    ref={a[:12]} len={len(a)}")
                        print(f"    new={b[:12]} len={len(b)}")
    if verbose:
        estado = "OK" if bad == 0 else f"{bad} DIFERENCIAS"
        print(f"check función: {total - bad}/{total} idénticos -> {estado}")

    # Paridad de errores: una combinación inválida debe fallar igual en las dos.
    cand = np.arange(n_pos, n_pos + 100)
    dd = d[:100]
    err = []
    for fn in (_reference, choose_background):
        try:
            fn(cand, dd, 40, 0.9, 2, np.random.default_rng(0),
               band2_frac=0.25, band2_px=40)
            err.append(None)
        except ValueError as e:
            err.append(str(e))
    err_ok = err[0] == err[1] and err[0] is not None
    if verbose:
        print(f"check errores: {'idénticos' if err_ok else 'DIFIEREN'}"
              + ("" if err_ok else f" -> {err}"))
    return bad == 0 and err_ok


def cmd_check_e2e():
    """Compara el lote completo de `subsample_pool` con y sin el parche."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import train_model as tm
    from hard_negatives import aniso_band_mask

    d, d1, d2 = _pool_fixture(seed=0)
    rng0 = np.random.default_rng(123)
    X = rng0.random((len(d), 128, 1), dtype=np.float32)
    n_pos = 581
    y = np.zeros(len(d), dtype=np.float32)
    y[:n_pos] = 1.0
    band2 = aniso_band_mask(d1, d2, 3, 10, 0, 3)

    kw = dict(d=d, hard_frac=0.55, hard_px=2, band2_frac=0.25,
              d1=d1, d2=d2, aniso=(3, 10, 0, 3))

    orig = tm.choose_background
    a = tm.subsample_pool(X, y, n_pos, 65536, 4.0, np.random.default_rng(5), **kw)
    tm.choose_background = choose_background
    b = tm.subsample_pool(X, y, n_pos, 65536, 4.0, np.random.default_rng(5), **kw)
    tm.choose_background = orig

    ok = all(np.array_equal(x, z) for x, z in zip(a, b))
    print(f"check e2e subsample_pool: X={a[0].shape} y={a[1].shape} "
          f"-> {'IDÉNTICO' if ok else 'DIFIERE'}")
    if not ok:
        for i, (x, z) in enumerate(zip(a, b)):
            if not np.array_equal(x, z):
                print(f"  bloque {i}: {x.shape} vs {z.shape} "
                      f"distintos={int((x != z).sum()) if x.shape == z.shape else 'forma'}")
    assert band2.shape == (len(d),)
    return ok


def cmd_bench(iters=200, n=14525, n_pos=581, n_bg=2324):
    d, d1, d2 = _pool_fixture()
    from hard_negatives import aniso_band_mask
    mask = aniso_band_mask(d1, d2, 3, 10, 0, 3)
    cand = np.arange(n_pos, n_pos + n)
    dd = d[:n]
    mask = mask[:n]

    def run(fn):
        rng = np.random.default_rng(0)
        t0 = os.times()
        import time
        w0 = time.perf_counter()
        for _ in range(iters):
            fn(cand, dd, n_bg, 0.55, 2, rng, band2_frac=0.25, band2_mask=mask)
        w = (time.perf_counter() - w0) / iters * 1000
        return w

    run(_reference)   # calentamiento
    run(choose_background)
    a = run(_reference)
    b = run(choose_background)
    print(f"n={n} n_bg={n_bg} hard_frac=0.55 band2=0.25 (anisótropa), {iters} iters")
    print(f"  original : {a:7.3f} ms/llamada")
    print(f"  máscara  : {b:7.3f} ms/llamada   ({a / b:.2f}x)")
    print(f"  por época (800 espectros): {a * 0.8:5.2f} s -> {b * 0.8:5.2f} s"
          f"   ahorro {a * 0.8 - b * 0.8:.2f} s")
    return a, b


def cmd_train(args):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import train_model as tm
    patched = install()
    print(f"[hard_negatives_fast] parcheado en: {', '.join(patched) or 'ninguno'}")
    sys.argv = ["train_model.py"] + list(args)
    tm.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--train" in argv:
        i = argv.index("--train")
        rest = argv[i + 1:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        cmd_train(rest)
        return 0
    if "--check-e2e" in argv:
        return 0 if cmd_check_e2e() else 1
    if "--bench" in argv:
        cmd_bench()
        return 0
    if "--check" in argv:
        return 0 if cmd_check() else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
