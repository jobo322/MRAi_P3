"""Fase 2 — Re-entrenamiento / fine-tune del modelo 2D-US para picos elongados en F1.

Arquitectura: WNN_2D del material suplementario (docs/sciadv.ady7995_sm.pdf p.2):
  Input([128,1]) -> softsign(10*x) -> split[64,64]
  -> 2 ramas x 6 bloques Conv1D(10,k=2,s=2,tanh) x Conv1D(10,k=2,s=2,sigmoid)
  -> fusion Conv1D(2*filter,k=1,linear) por rama (=20, como en pesos guardados)
  -> concat + squeeze -> Dense(20,tanh)xDense(20,sig) -> Dense(10,tanh)xDense(10,sig)
  -> Dense(1,sigmoid) [= dense_14 en SavedModels originales]
Loss: BinaryCrossentropy(from_logits=False). Optimizer: Adam.

Datos: manifest.json de Fase 1 (800 train / 200 val), .bin originales via memmap,
extraccion de crosses bajo demanda con cross_extract.split_2d (vectorizado).
NO pre-expande X (1000 X completos ~= 268 GB en float32).

Uso:
  # fine-tune (recomendado primero):
  python3 scripts/train_model.py --data data/train_2d_elong --init Trained/Probability_2D/US/DNN_0 --lr 1e-4
  # desde cero:
  python3 scripts/train_model.py --data data/train_2d_elong --lr 1e-3
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from cross_extract import split_2d as split_2d_loop
try:  # vectorised, bit-identical (tests/test_cross_fast.py: maxdiff == 0.0)
    from cross_extract_fast import split_2d_fast, split_2d_rows
    HAVE_FAST_SPLIT = True
except ImportError:  # older checkout without the module
    split_2d_rows = None
    HAVE_FAST_SPLIT = False

SHAPE = (256, 2048)
CROSS = np.array([64, 64])


def build_model(filter_n=10):
    import tensorflow as tf

    def WNN(layer, filt):
        def inner(x):
            for _ in range(layer):
                t = tf.keras.layers.Conv1D(filt, kernel_size=2, strides=2,
                                           activation="tanh")(x)
                s = tf.keras.layers.Conv1D(filt, kernel_size=2, strides=2,
                                           activation="sigmoid")(x)
                x = t * s
            return x
        return inner

    def DNN(filt):
        def inner(x):
            t = tf.keras.layers.Dense(2 * filt, activation="tanh")(x)
            s = tf.keras.layers.Dense(2 * filt, activation="sigmoid")(x)
            x = t * s
            t = tf.keras.layers.Dense(filt, activation="tanh")(x)
            s = tf.keras.layers.Dense(filt, activation="sigmoid")(x)
            x = t * s
            return x
        return inner

    size = [64, 64]
    nlev = [int(np.log2(s)) for s in size]  # [6, 6]
    inp = tf.keras.layers.Input(shape=[size[0] + size[1], 1])
    # Keras 3: las ops TF puras no aceptan KerasTensor; envolver en Lambda.
    import keras.ops as K
    x = tf.keras.layers.Lambda(lambda t: K.softsign(10 * t),
                                 output_shape=(size[0] + size[1], 1))(inp)
    # Slicing con forma estatica explicita (K.split pierde la longitud en Keras 3).
    x_0 = tf.keras.layers.Lambda(lambda t: t[:, :size[0], :],
                                 output_shape=(size[0], 1))(x)
    x_1 = tf.keras.layers.Lambda(lambda t: t[:, size[0]:, :],
                                 output_shape=(size[1], 1))(x)
    x_0 = WNN(nlev[0], filter_n)(x_0)
    x_1 = WNN(nlev[1], filter_n)(x_1)
    x_0 = tf.keras.layers.Conv1D(2 * filter_n, kernel_size=1)(x_0)
    x_1 = tf.keras.layers.Conv1D(2 * filter_n, kernel_size=1)(x_1)
    out = tf.keras.layers.Lambda(lambda ts: K.squeeze(K.concatenate(ts, axis=-1), axis=1))([x_0, x_1])
    out = DNN(filter_n)(out)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_14")(out)
    return tf.keras.Model(inputs=inp, outputs=out)


def subsample_pool(X, y, n_pos, max_pts, bg_ratio, rng):
    """Muestreo por época desde un pool pre-extraído.

    Toma TODOS los positivos (filas 0:n_pos) + `bg_ratio * n_pos` filas de
    fondo elegidas al azar del pool (filas n_pos:). Como el pool tiene
    `pool_k * n_pos` filas de fondo, con pool_k > bg_ratio cada época ve un
    subconjunto distinto -> se conserva la aumentación de fondo sin volver a
    pagar split_2d. Si pool_k == bg_ratio se toman todas las filas del pool
    (equivalente al muestreo fijo; ver tests/test_pool_equiv.py).
    """
    n_bg = min(len(X) - n_pos, int(n_pos * bg_ratio), max_pts - n_pos)
    bg = rng.choice(len(X) - n_pos, size=max(0, n_bg), replace=False) + n_pos
    idx = np.concatenate([np.arange(n_pos), bg])
    rng.shuffle(idx)
    return X[idx], y[idx]


class BinSequence:
    """Generador de batches: memmap .bin + split_2d bajo demanda.

    Por epoca recorre n espectros; de cada uno muestrea hasta max_pts puntos
    (todos los positivos + submuestra de fondo hasta completar) para acotar RAM.
    """

    def __init__(self, ds_dir, ids, batch=65536, max_pts=65536, seed=0,
                 shuffle=True, bg_ratio=4.0, fast_split=True):
        import tensorflow as tf
        self.ds_dir = ds_dir
        self.ids = list(ids)
        self.batch = batch
        self.max_pts = max_pts
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.bg_ratio = bg_ratio
        self.tf = tf
        # Por defecto solo se calculan las filas muestreadas (split_2d_rows,
        # bit-identico, ~60x menos trabajo); --slow-split usa la matriz
        # completa como referencia.
        self.fast_split = bool(fast_split and HAVE_FAST_SPLIT)
        self.split_2d = split_2d_fast if self.fast_split else split_2d_loop

    def _points_per_spectrum(self):
        """Puntos reales por espectro: positivos + bg_ratio fondo (cap max_pts).

        max_pts NO es el valor efectivo: con ~580 positivos y bg_ratio=4
        el tope es 580*(1+4) ~= 2900, no 65536. Leerlo del primer
        espectro evita declarar steps/época inflados (bug: se declaraban
        800 y el generador producía 35 -> la época terminaba a los 35).
        """
        if not self.ids:
            return 1
        sid = self.ids[0]
        if getattr(self, "preextracted", False):
            base = getattr(self, "pre_dir", None) or self.ds_dir
            with np.load(os.path.join(base, f"{sid}.npz")) as z:
                total = int(z["X"].shape[0])
                n_pos = int(z["n_pos"]) if "n_pos" in z else None
            if n_pos is None:  # formato antiguo: la muestra es el fichero
                return max(1, total)
            # pool: lo que se entrena por época es el submuestreo, no el pool
            n_bg = min(total - n_pos, int(n_pos * self.bg_ratio),
                       self.max_pts - n_pos)
            return max(1, n_pos + n_bg)
        L = np.memmap(os.path.join(self.ds_dir, "labels", f"{sid}.bin"),
                      dtype=np.uint8, mode="r", shape=SHAPE).astype(np.float32)
        npos = int((L.reshape(-1) > 0.5).sum())
        n_bg = min(L.size - npos, int(npos * self.bg_ratio),
                   self.max_pts - npos)
        return max(1, npos + n_bg)

    def __len__(self):
        return max(1, (len(self.ids) * self._points_per_spectrum()) // self.batch)

    def _load(self, sid):
        S = np.memmap(os.path.join(self.ds_dir, "spectra", f"{sid}.bin"),
                      dtype="<f4", mode="r", shape=SHAPE).astype(np.float32)
        L = np.memmap(os.path.join(self.ds_dir, "labels", f"{sid}.bin"),
                      dtype=np.uint8, mode="r", shape=SHAPE).astype(np.float32)
        return np.ascontiguousarray(S), np.ascontiguousarray(L)

    def _points(self, S, L):
        """Muestrea los puntos del espectro (todos los positivos + fondo).

        El plan de indices se calcula ANTES del split y el rng se consume en
        el mismo orden que antes (choice -> shuffle), asi que la muestra no
        cambia. Con fast_split solo se construyen las filas de `idx`
        (split_2d_rows); la ruta lenta mantiene split_2d completo + X[idx].
        """
        y = L.reshape(-1, 1)
        pos = np.where(y[:, 0] > 0.5)[0]
        n_bg = min(len(y) - len(pos), int(len(pos) * self.bg_ratio),
                   self.max_pts - len(pos))
        neg = np.where(y[:, 0] < 0.5)[0]
        take_neg = self.rng.choice(neg, size=max(0, n_bg), replace=False)
        idx = np.concatenate([pos, take_neg])
        self.rng.shuffle(idx)
        if self.fast_split:
            X, _ = split_2d_rows(S, idx, CROSS)
            X = np.asarray(X, dtype=np.float32)
        else:
            X, _ = self.split_2d(S.astype(np.float64), CROSS)
            X = np.asarray(X, dtype=np.float32)[idx]
        X[~np.isfinite(X)] = 0.0
        return X, y[idx]

    def _load_pre(self, sid):
        base = getattr(self, "pre_dir", None) or self.ds_dir
        with np.load(os.path.join(base, f"{sid}.npz")) as z:
            X, y = z["X"], z["y"]
            n_pos = int(z["n_pos"]) if "n_pos" in z else None
        if n_pos is None:  # formato antiguo: muestra fija, ya es final
            return np.ascontiguousarray(X), np.ascontiguousarray(y)
        Xs, ys = subsample_pool(X, y, n_pos, self.max_pts, self.bg_ratio,
                                self.rng)
        return np.ascontiguousarray(Xs), np.ascontiguousarray(ys)

    def __iter__(self):
        order = self.rng.permutation(len(self.ids)) if self.shuffle else np.arange(len(self.ids))
        buf_x, buf_y, have = [], [], 0
        for k in order:
            if getattr(self, "preextracted", False):
                X, y = self._load_pre(self.ids[k])
            else:
                S, L = self._load(self.ids[k])
                X, y = self._points(S, L)
            buf_x.append(X)
            buf_y.append(y)
            have += len(X)
            while have >= self.batch:
                # Un solo concatenate por buffer: antes se concatenaba dos
                # veces cada uno (Xc y rest_x salian de arrays distintos).
                all_x = np.concatenate(buf_x)
                all_y = np.concatenate(buf_y)
                Xc, rest_x = all_x[:self.batch], all_x[self.batch:]
                yc, rest_y = all_y[:self.batch], all_y[self.batch:]
                buf_x = [rest_x] if len(rest_x) else []
                buf_y = [rest_y] if len(rest_y) else []
                have = len(rest_x)
                yield Xc, yc

    def to_dataset(self):
        """Dataset infinito: `steps_per_epoch` gobierna la duracion de epoca.

        `.repeat()` NO es opcional. Keras 3 (epoch_iterator.py) solo recrea el
        iterador cuando `_current_iterator is None`, y el bloque que lo resetea
        al final de una epoca exige `self._num_batches`, que es None para un
        `from_generator` (cardinalidad desconocida). Sin `.repeat()`, un
        generador que se agota justo en el limite de epoca deja un iterador
        muerto: la epoca siguiente hace 0 batches, todas las metricas salen
        0.0 y solo aparece un UserWarning (paso en el run del 15-oct-2026:
        105 de 210 epocas sin entrenar). Con `.repeat()` el dataset es
        infinito y el agotamiento no puede ocurrir; cada vuelta de `repeat()`
        llama otra vez a `gen()`, asi que `__iter__` rebaraja con el rng y el
        fondo sigue variando entre epocas.
        """
        def gen():
            yield from self.__iter__()
        return self.tf.data.Dataset.from_generator(
            gen, output_signature=(
                self.tf.TensorSpec(shape=(None, 128, 1), dtype=self.tf.float32),
                self.tf.TensorSpec(shape=(None, 1), dtype=self.tf.float32))
        ).repeat().prefetch(2)


class _Syncer:
    """Copia local -> destino cada N epocas, sin depender de TensorFlow.

    Escribir en Drive cuesta entre 0,3 y 20 s por epoca (es una API de red
    montada como sistema de ficheros), y esa varianza domina el tiempo de
    epoca cuando el computo son ~2 s: el entrenamiento oscilaba entre 1 s y
    20 s por epoca con el step constante. Aqui el entrenamiento escribe en
    un directorio local (--scratch) y esto lo copia al destino cada `every`
    epocas, asi que el I/O variable sale del bucle de epocas sin perder la
    red de seguridad frente al reciclaje de la VM de Colab.

    La copia es atomica (fichero temporal + os.replace) para que un corte a
    mitad de escritura no deje un checkpoint truncado en Drive.
    """

    def __init__(self, src, dst, every=5,
                 files=("ckpt.keras", "training.log")):
        self.src = src
        self.dst = dst
        self.every = max(1, int(every))
        self.files = tuple(files)

    def due(self, epoch):
        """Toca sincronizar al terminar esta epoca? (epoch es 0-based)."""
        return (int(epoch) + 1) % self.every == 0

    def maybe_sync(self, epoch):
        if self.due(epoch):
            self.sync()

    def sync(self):
        os.makedirs(self.dst, exist_ok=True)
        for name in self.files:
            src = os.path.join(self.src, name)
            if not os.path.isfile(src):
                continue  # p.ej. aun no hay ckpt con save_best_only
            dst = os.path.join(self.dst, name)
            tmp = dst + ".tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def make_drive_sync(tf, src, dst, every=5,
                    files=("ckpt.keras", "training.log")):
    """Callback de Keras que envuelve `_Syncer`.

    Se construye con una fabrica porque este modulo importa TensorFlow de
    forma perezosa (tests/ lo importa sin TF instalado).
    """
    syncer = _Syncer(src, dst, every, files)

    class DriveSync(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            syncer.maybe_sync(epoch)

        def on_train_end(self, logs=None):
            syncer.sync()

    return DriveSync()


class _StepGuard:
    """Cuenta batches por epoca y detecta epocas que no entrenan nada.

    Red de seguridad del fallo del 15-oct-2026: cuando el generador se agotaba
    en el limite de epoca, Keras arrancaba la siguiente con un iterador muerto
    y la epoca hacia 0 batches. Todas las metricas de train salian 0.0 y el
    unico aviso era un UserWarning, asi que se perdio la mitad del computo
    (105 de 210 epocas) sin que se notara. `to_dataset()` lo arregla con
    `.repeat()`; esto convierte cualquier recaida en un error inmediato.
    """

    def __init__(self):
        self.seen = 0
        self.epoch = None

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self.seen = 0

    def on_train_batch_end(self, batch):
        self.seen += 1

    def check(self):
        """Mensaje de error si la epoca no entreno nada, o None si esta bien."""
        if self.seen == 0:
            return (
                f"epoca {self.epoch}: 0 batches de entrenamiento. El generador "
                "se agoto en el limite de epoca y Keras no recreo el iterador "
                "(keras/src/trainers/epoch_iterator.py). Revisa "
                "BinSequence.to_dataset(): necesita .repeat() antes de "
                ".prefetch() para que steps_per_epoch gobierne la epoca."
            )
        return None


def make_step_guard(tf):
    """Callback de Keras que aborta si una epoca no entrena nada."""
    guard = _StepGuard()

    class StepGuard(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            guard.on_epoch_begin(epoch)

        def on_train_batch_end(self, batch, logs=None):
            guard.on_train_batch_end(batch)

        def on_epoch_end(self, epoch, logs=None):
            problem = guard.check()
            if problem:
                raise RuntimeError(problem)

    return StepGuard()


def main():
    ap = argparse.ArgumentParser(description="Fase 2: entrenamiento 2D-US elongado")
    ap.add_argument("--data", required=True, help="dir con manifest.json de Fase 1")
    ap.add_argument("--out", default="Trained/Probability_2D_Elongated/DNN_0")
    ap.add_argument("--init", default=None, help="SavedModel base para fine-tune")
    ap.add_argument("--resume", default=None,
                    help="ckpt .keras para continuar entrenamiento (pesos exactos)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=210)
    ap.add_argument("--max-pts", type=int, default=65536)
    ap.add_argument("--bg-ratio", type=float, default=4.0,
                    help="fondo por cada positivo (4=paper-like, 2=mas agresivo)")
    ap.add_argument("--n-train", type=int, default=None,
                    help="usa solo los primeros N espectros de train (pruebas rapidas)")
    ap.add_argument("--n-val", type=int, default=None,
                    help="usa solo los primeros N espectros de val (pruebas rapidas)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preextracted", action="store_true",
                    help="leer el pool pre-extraido de preextract.py "
                         "(<id>.npz) en vez de .bin + split_2d bajo demanda; "
                         "el fondo se submuestrea por epoca con --bg-ratio")
    ap.add_argument("--slow-split", action="store_true",
                    help="usar el split_2d con bucle Python (referencia "
                         "bit-idéntica); por defecto se usa el vectorizado 6x")
    ap.add_argument("--scratch", default=None, metavar="DIR",
                    help="escribir checkpoints y log aqui (local) y copiarlos "
                         "a --out cada --sync-every epocas; evita el I/O "
                         "variable de Drive en el bucle de entrenamiento")
    ap.add_argument("--sync-every", type=int, default=5,
                    help="cada cuantas epocas copiar --scratch a --out (5)")
    args = ap.parse_args()

    import tensorflow as tf
    print("TF", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))

    man = json.load(open(os.path.join(args.data, "manifest.json")))
    ds_dir = man["source"]
    print(f"train {len(man['train'])} / val {len(man['val'])} desde {ds_dir}")

    lr = args.lr if args.lr else (1e-4 if (args.init or args.resume) else 1e-3)
    model = build_model()
    if args.resume:
        print(f"continuar desde {args.resume}")
        model.load_weights(args.resume)
        print("pesos restaurados exactos (incluye optimizer si se reanuda con compile previo)")
    elif args.init:
        print(f"fine-tune desde {args.init}")
        base = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(128, 1)),
            tf.keras.layers.TFSMLayer(args.init, call_endpoint="serving_default")])
        # Los nombres difieren (conv1d vs conv1d_52...): copiar por FORMA en orden.
        # Los pesos base vienen intercalados kernel/bias; emparejar secuencialmente
        # cada peso del modelo nuevo con el siguiente peso base de igual forma.
        pool = {}
        for w in base.weights:
            pool.setdefault(tuple(w.shape), []).append(w)
        copied, skipped = 0, 0
        for w in model.weights:
            cands = pool.get(tuple(w.shape), [])
            if cands:
                w.assign(cands.pop(0).numpy())
                copied += 1
            else:
                skipped += 1
        print(f"pesos copiados: {copied}, aleatorios: {skipped}")
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.AUC(name="auc")])
    model.summary()

    train_ids = man["train"][:args.n_train] if args.n_train else man["train"]
    val_ids = man["val"][:args.n_val] if args.n_val else man["val"]
    print(f"subset: {len(train_ids)} train / {len(val_ids)} val")
    train_seq = BinSequence(ds_dir, train_ids, batch=args.batch,
                            max_pts=args.max_pts, seed=args.seed, shuffle=True,
                            bg_ratio=args.bg_ratio,
                            fast_split=not args.slow_split)
    val_seq = BinSequence(ds_dir, val_ids, batch=args.batch,
                          max_pts=args.max_pts, seed=args.seed + 1, shuffle=False,
                          bg_ratio=args.bg_ratio,
                          fast_split=not args.slow_split)
    if not args.preextracted:
        used = ("por filas (split_2d_rows)" if train_seq.fast_split
                else "completo (--slow-split)")
        print(f"split_2d: {used}"
              + ("" if HAVE_FAST_SPLIT else "  [cross_extract_fast no disponible]"))
    if args.preextracted:
        train_seq.preextracted = True
        val_seq.preextracted = True
        train_seq.pre_dir = args.data
        val_seq.pre_dir = args.data
        print(f"modo pre-extraído: {man.get('storage', '')} "
              f"(seed={man.get('seed')}, pool_k={man.get('pool_k')}, "
              f"bg_ratio/época={args.bg_ratio})")
    os.makedirs(args.out, exist_ok=True)
    # --scratch: el entrenamiento escribe en local y se sincroniza a --out.
    ckpt_dir = args.scratch or args.out
    if args.scratch:
        os.makedirs(ckpt_dir, exist_ok=True)
    cbs = [
        # El guard va primero: si una epoca no entrena nada, aborta antes de
        # que los demas callbacks escriban checkpoint o log.
        make_step_guard(tf),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(ckpt_dir, "ckpt.keras"), monitor="val_loss",
            save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(ckpt_dir, "training.log")),
    ]
    if args.scratch:
        # Despues de ModelCheckpoint/CSVLogger para que los ficheros existan.
        cbs.append(make_drive_sync(tf, ckpt_dir, args.out,
                                   every=args.sync_every))
        print(f"checkpoints en {ckpt_dir}; sincronizados a {args.out} "
              f"cada {max(1, args.sync_every)} epocas")
    t0 = time.time()
    n_val_steps = max(1, len(val_seq))
    hist = model.fit(train_seq.to_dataset(), validation_data=val_seq.to_dataset(),
                     epochs=args.epochs, callbacks=cbs,
                     steps_per_epoch=len(train_seq), validation_steps=n_val_steps)
    print(f"entrenamiento: {(time.time()-t0)/3600:.2f} h")
    if "val_loss" in hist.history:
        h = np.array([hist.history["loss"], hist.history["val_loss"]]).T
    else:
        h = np.array([hist.history["loss"], hist.history["loss"]]).T
    np.save(os.path.join(args.out, "history.npy"), h)
    with open(os.path.join(args.out, "history.txt"), "w") as f:
        for tr, va in h:
            f.write(f"{tr:.6e} {va:.6e}\n")
    model.export(args.out + "_savedmodel")
    print(f"OK -> {args.out} (+ {args.out}_savedmodel), history.npy {h.shape}")
    # verificacion estilo inferencia StoP3
    xb = np.zeros((1024, 128, 1), dtype=np.float32)
    print("predict smoke:", model.predict(xb, batch_size=1024).shape)


if __name__ == "__main__":
    main()
