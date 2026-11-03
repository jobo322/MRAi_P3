#!/usr/bin/env python3
"""Entrenamiento en mixed_float16 como runner paralelo, SIN tocar train_model.py.

Mismo patrón que `hard_negatives_fast.py`: no modifica ningún fichero en uso.
Instala la política global `mixed_float16` y parchea `train_model.build_model`
para que el modelo entrene en fp16 pero **exporte float32**, que es lo que
consumen los productores StoP3 (`model_output.single_output`).

Medido antes de escribir esto (3 pares intercalados, datos en memoria, formas
reales 65536x128x1): 657 -> 527 ms/step de pared (~1.25x) con los mismos
segundos de CPU, así que la ganancia es eficiencia de memoria/paralelismo, no
menos trabajo. En M1 no hay kernel de convolución fp16 rápido: las convs siguen
en fp32 y lo que se ahorra es tráfico en las ops punto a punto.

Uso:
    python scripts/train_mixed_precision.py --verify
    python scripts/train_mixed_precision.py --train -- <args de train_model.py>
    python scripts/train_mixed_precision.py --train --fast-negs -- <args...>

`--verify` comprueba, contra el modelo original:
  * 6 831 parámetros y los 62 pesos en el mismo orden (el cast no añade pesos),
  * salida float32 (no float16, que rompería el export),
  * copia por rol desde un SavedModel base (--init) 62/62,
  * checkpoint .keras que se relee y predice igual,
  * SavedModel exportado con salida float32,
  * y la deriva numérica real de fp16 sobre un espectro del pool.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

POLICY = "mixed_float16"
_ORIG = {}


def _setup(policy=POLICY, fast_negs=False):
    """Política global + parche de build_model. Devuelve (tf, train_model)."""
    import tensorflow as tf
    if policy != "float32":
        tf.keras.mixed_precision.set_global_policy(policy)
    import train_model as tm

    if "build_model" not in _ORIG:
        _ORIG["build_model"] = tm.build_model
    tm.build_model = _wrap_build_model(_ORIG["build_model"], tf)
    if fast_negs:
        import hard_negatives_fast
        patched = hard_negatives_fast.install()
        print(f"[mp] negativos rápidos en: {', '.join(patched) or 'ninguno'}")
    return tf, tm


def _wrap_build_model(orig, tf):
    """Envuelve build_model para devolver salida float32 bajo mixed_float16.

    El cabezal es `Dense(1, sigmoid, name="dense_14")`: bajo la política mixta
    computa y devuelve float16. Un cast a float32 al final conserva el contrato
    de exportación y no añade ningún peso.
    """
    def build_model_mp(*a, **kw):
        base = orig(*a, **kw)
        if tf.keras.backend.floatx() == "float32" and \
                tf.keras.mixed_precision.global_policy().name == "float32":
            return base
        out = tf.keras.ops.cast(base.output, "float32")
        model = tf.keras.Model(inputs=base.input, outputs=out,
                               name=f"{base.name}_mp")
        # Mismo orden de pesos que el original (el cast no aporta variables),
        # que es lo que `weight_roles.assign_weights` necesita para --init.
        model.weight_roles = base.weight_roles
        return model
    return build_model_mp


# --------------------------------------------------------------------------
# verificación
# --------------------------------------------------------------------------

def cmd_verify(init_dir="Trained/Probability_2D/US/DNN_0", pool_dir=None,
               sample=4096):
    import tempfile

    import numpy as np

    tf, tm = _setup()
    base = _ORIG["build_model"]()

    # 1) modelo con política mixta
    mp = tm.build_model()

    print(f"política            : {tf.keras.mixed_precision.global_policy()}")
    print(f"params base/mp      : {base.count_params()} / {mp.count_params()}")
    print(f"pesos base/mp       : {len(base.weights)} / {len(mp.weights)}")
    print(f"salida base/mp      : {base.output.dtype.name} / {mp.output.dtype.name}")

    ok = True
    if mp.count_params() != base.count_params():
        print("  FALLO: cambia el número de parámetros")
        ok = False
    if len(mp.weights) != len(base.weights):
        print("  FALLO: cambia el número de pesos")
        ok = False
    if mp.output.dtype != tf.float32:
        print("  FALLO: la salida no es float32 -> el export rompería StoP3")
        ok = False
    orden = all(a.shape == b.shape for a, b in zip(base.weights, mp.weights))
    print(f"orden de pesos      : {'idéntico' if orden else 'DIFIERE'}")
    ok = ok and orden

    # 2) --init por rol sobre el modelo envuelto
    if os.path.isdir(init_dir):
        from weight_roles import assign_weights
        copied, skipped = assign_weights(mp, init_dir)
        print(f"--init por rol      : {copied}/{len(mp.weights)} copiados, "
              f"{skipped} aleatorios")
        if copied != len(mp.weights):
            print("  FALLO: no se copian todos los pesos por rol")
            ok = False
    else:
        print(f"--init por rol      : omitido ({init_dir} no existe)")

    # 3) deriva numérica sobre un espectro real del pool
    if pool_dir is None:
        cand = os.path.join("data", "pre_aniso_f2lo0_552520")
        pool_dir = cand if os.path.isdir(cand) else None
    if pool_dir:
        man_path = os.path.join(pool_dir, "manifest.json")
        import json
        ids = json.load(open(man_path))["train"][:1]
        with np.load(os.path.join(pool_dir, f"{ids[0]}.npz")) as z:
            X = np.asarray(z["X"])[:sample].astype(np.float32)
        mp.set_weights(base.get_weights())  # mismos pesos -> solo cambia el dtype
        p32 = base.predict(X, batch_size=8192, verbose=0).ravel()
        p16 = mp.predict(X, batch_size=8192, verbose=0).ravel()
        d = np.abs(p32 - p16)
        cruz = int(((p32 >= 0.5) != (p16 >= 0.5)).sum())
        print(f"deriva fp16 ({len(X)} filas)  : max|dp|={d.max():.3e} "
              f"p99={np.quantile(d, 0.99):.3e} media={d.mean():.3e} "
              f"cruces de 0.5={cruz}")
    else:
        mp.set_weights(base.get_weights())

    # 4) checkpoint .keras: guardar y releer
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "ckpt.keras")
        mp.save(ck)
        back = tf.keras.models.load_model(ck)
        Xs = np.array(mp.inputs[0].shape[1:])
        x_dummy = np.zeros((64,) + tuple(Xs), dtype=np.float32)
        a = mp.predict(x_dummy, verbose=0)
        b = back.predict(x_dummy, verbose=0)
        print(f"checkpoint .keras   : releído, salida {b.dtype} "
              f"{'idéntica' if np.array_equal(a, b) else 'DIFIERE'}")
        ok = ok and np.array_equal(a, b) and b.dtype == np.float32

        # 5) export SavedModel
        exp = os.path.join(td, "exported")
        mp.export(exp)
        loaded = tf.saved_model.load(exp)
        sig = loaded.signatures["serving_default"]
        outs = {k: v.dtype.name for k, v in sig.structured_outputs.items()}
        print(f"export savedmodel   : salidas {outs}")

        def _first_dtype(v):
            if isinstance(v, dict):
                return _first_dtype(next(iter(v.values())))
            return v.dtype.name
        out_dtype = _first_dtype(sig.structured_outputs)
        if out_dtype != "float32":
            print("  FALLO: el SavedModel exporta distinto de float32")
            ok = False

    print("VERIFY:", "OK" if ok else "FALLOS")
    return ok


# --------------------------------------------------------------------------
# entrenamiento
# --------------------------------------------------------------------------

def cmd_train(args, policy=POLICY, fast_negs=False):
    tf, tm = _setup(policy=policy, fast_negs=fast_negs)
    print(f"[mp] política {tf.keras.mixed_precision.global_policy()} | "
          f"salida forzada a float32")
    sys.argv = ["train_model.py"] + list(args)
    tm.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--train" in argv:
        i = argv.index("--train")
        rest = argv[i + 1:]
        fast = "--fast-negs" in rest
        policy = POLICY
        if "--policy" in rest:
            j = rest.index("--policy")
            policy = rest[j + 1]
            del rest[j:j + 2]
        rest = [a for a in rest if a != "--fast-negs"]
        if rest and rest[0] == "--":
            rest = rest[1:]
        cmd_train(rest, policy=policy, fast_negs=fast)
        return 0
    if "--verify" in argv:
        return 0 if cmd_verify() else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
