"""Tests for the epoch-boundary fixes in train_model.py.

The 15-oct Colab run logged 210 epochs of which 105 did no training at all:
every odd epoch had acc = auc = loss = 0.0 and val_loss moved only by
resampling noise. The cause is in Keras 3, not in our generator:

    keras/src/trainers/epoch_iterator.py
    if self._current_iterator is None or self.steps_per_epoch is None:
        self._current_iterator = iter(self._get_iterator())

`_current_iterator` is only recreated when it is None. With `steps_per_epoch`
set and an unknown dataset cardinality (`from_generator` gives no cardinality),
the post-loop reset block (`if self._num_batches and ...`) never fires, so a
generator that is exhausted exactly at the epoch boundary leaves a dead
iterator behind. The next epoch then runs zero batches, emits
`_interrupted_warning()`, and only then sets `_current_iterator = None` --
which is why it alternates instead of dying for good.

Two fixes, tested here:

- `to_dataset()` must `.repeat()` so the dataset is infinite and
  `steps_per_epoch` alone governs the epoch length. This is what the Keras
  warning itself recommends.
- `_StepGuard` / `make_step_guard` turn a silent empty epoch into a loud
  failure, so an equivalent regression can never cost half a run unnoticed.

Both are testable without TensorFlow: the guard logic is TF-free, and the
dataset transformation is checked through a recording fake.
"""
import os
import sys
import types
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from cross_extract_fast import split_2d_fast  # noqa: E402
from train_model import (BinSequence, _StepGuard,  # noqa: E402
                         make_step_guard)


class _RecordingDataset:
    """Records the order of the dataset transformations applied to it."""

    def __init__(self):
        self.calls = []
        self.gen = None
        self.output_signature = None

    def repeat(self):
        self.calls.append("repeat")
        return self

    def prefetch(self, n):
        self.calls.append(("prefetch", n))
        return self


class _FakeCallback:
    """Stand-in for tf.keras.callbacks.Callback."""

    def __init__(self):
        self.model = None


def _fake_tf():
    rec = _RecordingDataset()

    def from_generator(gen, output_signature=None):
        rec.gen = gen
        rec.output_signature = output_signature
        return rec

    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            Dataset=types.SimpleNamespace(from_generator=from_generator)),
        TensorSpec=lambda shape, dtype=None: ("TensorSpec", shape, dtype),
        float32="float32",
        keras=types.SimpleNamespace(
            callbacks=types.SimpleNamespace(Callback=_FakeCallback)),
    ), rec


def _seq(tf=None):
    """BinSequence without __init__ (which would import TensorFlow)."""
    seq = object.__new__(BinSequence)
    seq.rng = np.random.default_rng(0)
    seq.bg_ratio = 4.0
    seq.max_pts = 65536
    seq.fast_split = True
    seq.split_2d = split_2d_fast
    seq.batch = 65536
    seq.shuffle = True
    seq.preextracted = False
    seq.ids = []
    seq.tf = tf
    return seq


class TestToDatasetRepeats(unittest.TestCase):
    """The dataset must be infinite so steps_per_epoch governs the epoch."""

    def test_repeat_is_applied(self):
        tf, rec = _fake_tf()
        _seq(tf).to_dataset()
        self.assertIn("repeat", rec.calls)

    def test_repeat_comes_before_prefetch(self):
        """repeat() first: prefetch must buffer an infinite stream, and the
        epoch boundary must not be a generator exhaustion point."""
        tf, rec = _fake_tf()
        _seq(tf).to_dataset()
        self.assertEqual(rec.calls, ["repeat", ("prefetch", 2)])

    def test_output_signature_is_preserved(self):
        tf, rec = _fake_tf()
        _seq(tf).to_dataset()
        shapes = [s[1] for s in rec.output_signature]
        self.assertEqual(shapes, [(None, 128, 1), (None, 1)])

    def test_prefetch_buffer_is_two(self):
        tf, rec = _fake_tf()
        _seq(tf).to_dataset()
        self.assertIn(("prefetch", 2), rec.calls)


class TestStepGuard(unittest.TestCase):
    """Counting logic, with no TensorFlow involved."""

    def test_starts_at_zero_and_counts_batches(self):
        g = _StepGuard()
        g.on_epoch_begin(0)
        self.assertEqual(g.seen, 0)
        for _ in range(7):
            g.on_train_batch_end(0)
        self.assertEqual(g.seen, 7)
        self.assertIsNone(g.check())

    def test_an_epoch_with_no_batches_is_reported(self):
        g = _StepGuard()
        g.on_epoch_begin(3)
        problem = g.check()
        self.assertIsNotNone(problem)
        self.assertIn("3", problem)
        self.assertIn("repeat", problem)

    def test_the_counter_resets_each_epoch(self):
        g = _StepGuard()
        g.on_epoch_begin(0)
        g.on_train_batch_end(0)
        self.assertIsNone(g.check())
        g.on_epoch_begin(1)  # a fresh epoch
        self.assertEqual(g.seen, 0)
        self.assertIsNotNone(g.check())

    def test_a_later_epoch_is_still_checked(self):
        """The 15-oct failure started at epoch 1, not 0."""
        g = _StepGuard()
        g.on_epoch_begin(0)
        for _ in range(35):
            g.on_train_batch_end(0)
        self.assertIsNone(g.check())
        g.on_epoch_begin(1)
        self.assertIsNotNone(g.check())


class TestMakeStepGuard(unittest.TestCase):
    """The Keras adapter."""

    def test_is_a_callback_subclass(self):
        self.assertIsInstance(make_step_guard(_fake_tf()[0]), _FakeCallback)

    def test_does_not_raise_when_the_epoch_trained(self):
        cb = make_step_guard(_fake_tf()[0])
        cb.on_epoch_begin(0)
        for _ in range(35):
            cb.on_train_batch_end(0)
        cb.on_epoch_end(0)  # must not raise

    def test_raises_on_an_empty_epoch(self):
        cb = make_step_guard(_fake_tf()[0])
        cb.on_epoch_begin(1)
        with self.assertRaises(RuntimeError) as ctx:
            cb.on_epoch_end(1)
        self.assertIn("0 batches", str(ctx.exception))

    def test_accepts_the_logs_argument(self):
        cb = make_step_guard(_fake_tf()[0])
        cb.on_epoch_begin(0)
        cb.on_train_batch_end(0, logs={"loss": 1.0})
        cb.on_epoch_end(0, logs={"loss": 1.0})


if __name__ == "__main__":
    unittest.main()
