"""Tests for the Drive-sync callback and BinSequence batching.

Two independent regressions, both about what happens *around* the training
step rather than inside it:

- `_Syncer` / `make_drive_sync`: the checkpoint and the CSV log used to be
  written to Drive on every epoch. Drive is a network filesystem mounted
  through FUSE, so each write costs anywhere from 0.3 s to 20 s, and that
  variance showed up as epoch durations oscillating between 1 s and 20 s
  while the training step itself stayed constant. The run now writes to a
  local scratch directory and copies to the destination every `every`
  epochs, so the variable I/O leaves the epoch loop while the crash-safety
  net is preserved.

- `BinSequence.__iter__`: the batching loop used to concatenate its buffer
  four times per yielded batch (twice for X, twice for y). It now
  concatenates once. The yielded stream must be unchanged: batches are
  still consecutive slices of the per-spectrum sample stream, and the
  trailing partial batch is still dropped.
"""
import os
import sys
import tempfile
import types
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from cross_extract import split_2d  # noqa: E402
from cross_extract_fast import split_2d_fast  # noqa: E402
from train_model import BinSequence, _Syncer, make_drive_sync  # noqa: E402

CKPT = "ckpt.keras"
LOG = "training.log"


def _write(src, name, payload):
    with open(os.path.join(src, name), "wb") as f:
        f.write(payload)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


class TestSyncer(unittest.TestCase):
    """Copy logic and cadence, with no TensorFlow involved."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.src = os.path.join(self._tmp.name, "scratch")
        self.dst = os.path.join(self._tmp.name, "drive")
        os.makedirs(self.src)
        os.makedirs(self.dst)

    def test_nothing_copied_before_the_first_due_epoch(self):
        _write(self.src, CKPT, b"a")
        _write(self.src, LOG, b"b")
        syncer = _Syncer(self.src, self.dst, every=3)
        syncer.maybe_sync(0)
        syncer.maybe_sync(1)
        self.assertFalse(os.path.exists(os.path.join(self.dst, CKPT)))
        self.assertFalse(os.path.exists(os.path.join(self.dst, LOG)))

    def test_copies_on_cadence(self):
        _write(self.src, CKPT, b"a")
        syncer = _Syncer(self.src, self.dst, every=3)
        syncer.maybe_sync(2)  # epoch index 2 == third epoch
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_cadence_is_by_epoch_index_not_by_call_count(self):
        """Repeated calls for the same epoch must not trigger a copy."""
        _write(self.src, CKPT, b"a")
        syncer = _Syncer(self.src, self.dst, every=5)
        for _ in range(5):
            syncer.maybe_sync(0)  # five calls, still the 1st epoch
        self.assertFalse(os.path.exists(os.path.join(self.dst, CKPT)))
        syncer.maybe_sync(4)  # the 5th epoch
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_every_one_syncs_every_epoch(self):
        _write(self.src, CKPT, b"a")
        syncer = _Syncer(self.src, self.dst, every=1)
        syncer.maybe_sync(0)
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_sync_is_unconditional_and_picks_up_new_content(self):
        _write(self.src, CKPT, b"first")
        syncer = _Syncer(self.src, self.dst, every=100)
        syncer.sync()
        self.assertEqual(_read(os.path.join(self.dst, CKPT)), b"first")
        _write(self.src, CKPT, b"second")
        syncer.sync()
        self.assertEqual(_read(os.path.join(self.dst, CKPT)), b"second")

    def test_copy_is_byte_identical(self):
        payload = bytes(range(256)) * 400  # 100 KB, like a real ckpt
        _write(self.src, CKPT, payload)
        _write(self.src, LOG, b"epoch,loss\n1,0.5\n")
        _Syncer(self.src, self.dst, every=1).sync()
        self.assertEqual(_read(os.path.join(self.dst, CKPT)), payload)
        self.assertEqual(_read(os.path.join(self.dst, LOG)), b"epoch,loss\n1,0.5\n")

    def test_missing_source_files_are_skipped(self):
        """No checkpoint yet (save_best_only) must not raise."""
        _write(self.src, LOG, b"only the log")
        _Syncer(self.src, self.dst, every=1).sync()
        self.assertTrue(os.path.exists(os.path.join(self.dst, LOG)))
        self.assertFalse(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_no_temp_file_left_behind(self):
        """The copy is atomic: write to .tmp then replace."""
        _write(self.src, CKPT, b"payload")
        _Syncer(self.src, self.dst, every=1).sync()
        self.assertEqual(sorted(os.listdir(self.dst)), [CKPT])

    def test_creates_the_destination_if_absent(self):
        _write(self.src, CKPT, b"payload")
        dst = os.path.join(self._tmp.name, "does", "not", "exist")
        _Syncer(self.src, dst, every=1).sync()
        self.assertTrue(os.path.exists(os.path.join(dst, CKPT)))

    def test_every_is_clamped_to_at_least_one(self):
        _write(self.src, CKPT, b"payload")
        syncer = _Syncer(self.src, self.dst, every=0)
        syncer.maybe_sync(0)
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_custom_file_list(self):
        _write(self.src, "other.bin", b"x")
        _write(self.src, CKPT, b"y")
        _Syncer(self.src, self.dst, every=1, files=("other.bin",)).sync()
        self.assertEqual(sorted(os.listdir(self.dst)), ["other.bin"])


class _FakeCallback:
    """Stand-in for tf.keras.callbacks.Callback (train_model must not need TF)."""

    def __init__(self):
        self.model = None


def _fake_tf():
    return types.SimpleNamespace(
        keras=types.SimpleNamespace(
            callbacks=types.SimpleNamespace(Callback=_FakeCallback)))


class TestMakeDriveSync(unittest.TestCase):
    """The Keras adapter: cadence on on_epoch_end, unconditional at the end."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.src = os.path.join(self._tmp.name, "scratch")
        self.dst = os.path.join(self._tmp.name, "drive")
        os.makedirs(self.src)

    def _cb(self, every):
        _write(self.src, CKPT, b"payload")
        return make_drive_sync(_fake_tf(), self.src, self.dst, every=every)

    def test_is_a_callback_subclass(self):
        self.assertIsInstance(self._cb(3), _FakeCallback)

    def test_on_epoch_end_respects_the_cadence(self):
        cb = self._cb(every=3)
        cb.on_epoch_end(0)
        cb.on_epoch_end(1)
        self.assertFalse(os.path.exists(os.path.join(self.dst, CKPT)))
        cb.on_epoch_end(2)
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_on_train_end_always_syncs(self):
        cb = self._cb(every=1000)
        cb.on_train_end()
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))

    def test_on_epoch_end_accepts_logs_argument(self):
        cb = self._cb(every=1)
        cb.on_epoch_end(0, logs={"loss": 1.0})  # Keras passes logs
        self.assertTrue(os.path.exists(os.path.join(self.dst, CKPT)))


def _seq(batch, seed=0, bg_ratio=4.0, max_pts=65536, fast_split=True,
         points_per_spectrum=50):
    """BinSequence without __init__ (which would import TensorFlow)."""
    seq = object.__new__(BinSequence)
    seq.rng = np.random.default_rng(seed)
    seq.bg_ratio = bg_ratio
    seq.max_pts = max_pts
    seq.fast_split = fast_split
    seq.split_2d = split_2d_fast if fast_split else split_2d
    seq.batch = batch
    seq.shuffle = False
    seq.preextracted = False
    seq.ids = []
    # list(seq) consults __len__ through the length hint, and __len__ calls
    # _points_per_spectrum(), which would read the dataset from disk. Stub it
    # so the batching tests need no dataset.
    seq._points_per_spectrum = lambda: points_per_spectrum
    return seq


def _fake_load(n_pos=10, shape=(64, 64)):
    """Deterministic spectrum with exactly n_pos positives."""
    def load(sid):
        rng = np.random.default_rng(1000 + int(sid))
        S = rng.standard_normal(shape).astype(np.float32)
        L = np.zeros(shape, dtype=np.float32)
        L.reshape(-1)[rng.choice(L.size, size=n_pos, replace=False)] = 1.0
        return S, L
    return load


class TestIterBatching(unittest.TestCase):
    """__iter__ must yield consecutive slices of the sample stream.

    This is the invariant the single-concatenate refactor has to preserve:
    the batches partition a prefix of the per-spectrum samples, and the
    trailing partial batch is dropped.
    """

    N_SPECTRA = 8
    N_POS = 10
    # _points returns n_pos + bg_ratio*n_pos = 10 + 40 = 50 rows per spectrum
    POINTS = N_POS * (1 + 4)
    BATCH = 120

    def _instrumented(self, batch=BATCH, fast_split=True):
        seq = _seq(batch, fast_split=fast_split,
                   points_per_spectrum=self.POINTS)
        seq.ids = list(range(self.N_SPECTRA))
        seq._load = _fake_load(self.N_POS)
        seen = []
        orig = seq._points

        def spy(S, L):
            X, y = orig(S, L)
            seen.append((X, y))
            return X, y

        seq._points = spy
        return seq, seen

    def test_points_per_spectrum_is_what_the_test_assumes(self):
        seq, _ = self._instrumented()
        seq.ids = seq.ids[:1]
        X, _ = seq._points(*seq._load(0))
        self.assertEqual(len(X), self.POINTS)

    def test_batches_are_a_prefix_of_the_sample_stream(self):
        seq, seen = self._instrumented()
        batches = list(seq)
        stream_x = np.concatenate([x for x, _ in seen])
        stream_y = np.concatenate([y for _, y in seen])
        used = (len(stream_x) // seq.batch) * seq.batch
        self.assertEqual(used, 3 * self.BATCH)  # 400 rows -> 3 full batches
        got_x = np.concatenate([x for x, _ in batches])
        got_y = np.concatenate([y for _, y in batches])
        self.assertEqual(len(got_x), used)
        self.assertTrue(np.array_equal(got_x, stream_x[:used]))
        self.assertTrue(np.array_equal(got_y, stream_y[:used]))

    def test_every_batch_has_exactly_batch_rows(self):
        seq, _ = self._instrumented()
        for X, y in seq:
            self.assertEqual(X.shape, (self.BATCH, 128, 1))
            self.assertEqual(y.shape, (self.BATCH, 1))

    def test_trailing_partial_batch_is_dropped(self):
        seq, _ = self._instrumented()
        batches = list(seq)
        total = len(batches) * self.BATCH
        self.assertLess(total, self.N_SPECTRA * self.POINTS)
        self.assertEqual(total, 360)  # 400 - 40

    def test_same_stream_on_the_slow_split_path(self):
        fast, seen_f = self._instrumented(fast_split=True)
        slow, seen_s = self._instrumented(fast_split=False)
        bf, bs = list(fast), list(slow)
        self.assertEqual(len(bf), len(bs))
        for (xf, _), (xs, _) in zip(bf, bs):
            self.assertTrue(np.array_equal(xf, xs))

    def test_is_deterministic_for_a_given_seed(self):
        a, _ = self._instrumented()
        b, _ = self._instrumented()
        for (xa, ya), (xb, yb) in zip(list(a), list(b)):
            self.assertTrue(np.array_equal(xa, xb))
            self.assertTrue(np.array_equal(ya, yb))

    def test_no_state_leaks_between_epochs(self):
        """A second pass starts from an empty buffer.

        The rng advances, so the background sample legitimately differs
        between epochs (that is the augmentation). What must not happen is
        a leftover buffer: the second pass has to be a clean prefix of its
        own sample stream.
        """
        seq, seen = self._instrumented()
        first = list(seq)
        seen.clear()
        second = list(seq)
        self.assertEqual(len(first), len(second))
        self.assertEqual([len(x) for x, _ in first],
                         [len(x) for x, _ in second])
        stream_x = np.concatenate([x for x, _ in seen])
        stream_y = np.concatenate([y for _, y in seen])
        got_x = np.concatenate([x for x, _ in second])
        got_y = np.concatenate([y for _, y in second])
        used = (len(stream_x) // seq.batch) * seq.batch
        self.assertEqual(len(got_x), used)
        self.assertTrue(np.array_equal(got_x, stream_x[:used]))
        self.assertTrue(np.array_equal(got_y, stream_y[:used]))

    def test_shuffle_permutes_spectrum_order_but_not_the_stream_shape(self):
        seq, _ = self._instrumented()
        seq.shuffle = True
        seq.ids = list(range(self.N_SPECTRA))
        batches = list(seq)
        self.assertEqual(len(batches), 3)
        self.assertEqual(sum(len(x) for x, _ in batches), 360)

    def test_no_spectra_yields_nothing(self):
        seq, _ = self._instrumented()
        seq.ids = []
        self.assertEqual(list(seq), [])


if __name__ == "__main__":
    unittest.main()
