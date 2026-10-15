"""Colab GPU sanity check — run as cell 0 before anything else.

Verifies standard TF (CUDA build) sees an NVIDIA GPU.
On a correct Colab GPU runtime prints e.g.:
  TF 2.1x.y GPUs: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
On CPU-only runtime prints GPUs: [] -> switch runtime to GPU first.
"""
import tensorflow as tf

print("TF", tf.__version__)
print("devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))
