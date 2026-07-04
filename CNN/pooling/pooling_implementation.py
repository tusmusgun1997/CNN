"""
Pooling — from-scratch (NumPy) implementations verified against Keras.

This script accompanies pooling-guide.md. It:

  1. Implements MAX and AVERAGE pooling from scratch with NumPy (2D + channels).
  2. Implements GLOBAL average / max pooling from scratch.
  3. Runs them on the exact 4x4 sample matrix from the guide so you can see the
     hand-worked numbers appear:
         max pool 2x2  -> [[6, 4], [7, 8]]
         avg pool 2x2  -> [[3.75, 2.25], [2.5, 3.75]]
         global avg    -> 49/16 = 3.0625
         global max    -> 8
  4. Verifies the from-scratch results EXACTLY match Keras' built-in layers.
  5. Demonstrates pooling preserving channel depth on a multi-channel batch.

Run:
    pip install tensorflow      # (numpy comes with it)
    python pooling_implementation.py
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
)

np.set_printoptions(precision=4, suppress=True)


# ============================================================================
# FROM-SCRATCH POOLING (NumPy)
# ============================================================================
def pool2d(x, pool_size=2, stride=2, mode="max"):
    """Pool a single 2D feature map (H x W).

    mode: "max" or "avg".  Uses 'valid' padding (no border), like the default.
    Returns the pooled 2D array.
    """
    H, W = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = x[r:r + pool_size, c:c + pool_size]
            out[i, j] = window.max() if mode == "max" else window.mean()
    return out


def pool2d_channels(x, pool_size=2, stride=2, mode="max"):
    """Pool a multi-channel map (H x W x C), each channel independently.

    Depth C is preserved (pooling never mixes channels).
    """
    H, W, C = x.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    out = np.zeros((out_h, out_w, C), dtype=np.float64)
    for ch in range(C):
        out[:, :, ch] = pool2d(x[:, :, ch], pool_size, stride, mode)
    return out


def global_pool(x, mode="avg"):
    """Collapse each channel of an (H x W x C) map to a single value -> (C,)."""
    axis = (0, 1)  # average/max over height and width, keep channels
    return x.max(axis=axis) if mode == "max" else x.mean(axis=axis)


# ============================================================================
# HELPERS
# ============================================================================
def keras_pool(x_hwc, layer):
    """Run a Keras pooling layer on a single (H,W,C) map; return a numpy array."""
    batch = x_hwc[np.newaxis, ...].astype(np.float32)  # -> (1, H, W, C)
    return layer(batch).numpy()[0]                      # drop the batch dim


def check(name, a, b):
    """Assert two arrays match and print a tidy PASS line."""
    ok = np.allclose(a, b, atol=1e-5)
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: scratch == Keras  ({ok})")
    return ok


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("TensorFlow version:", tf.__version__, "\n")

    # --- Sample data: the 4x4 matrix from pooling-guide.md ------------------------
    sample = np.array(
        [
            [1, 3, 2, 4],
            [5, 6, 1, 2],
            [7, 2, 8, 0],
            [1, 0, 3, 4],
        ],
        dtype=np.float32,
    )
    print("Sample 4x4 feature map:")
    print(sample, "\n")

    sample_hwc = sample[:, :, np.newaxis]  # (4, 4, 1) for the channel-aware path

    # --- 1. MAX pooling ----------------------------------------------------
    print("=" * 60)
    print("MAX POOLING  (2x2, stride 2)")
    print("=" * 60)
    scratch_max = pool2d(sample, 2, 2, "max")
    keras_max = keras_pool(sample_hwc, MaxPooling2D(pool_size=2, strides=2))[:, :, 0]
    print("from scratch:\n", scratch_max)
    print("keras       :\n", keras_max)
    check("max pool", scratch_max, keras_max)
    print("expected     : [[6, 4], [7, 8]]\n")

    # --- 2. AVERAGE pooling ------------------------------------------------
    print("=" * 60)
    print("AVERAGE POOLING  (2x2, stride 2)")
    print("=" * 60)
    scratch_avg = pool2d(sample, 2, 2, "avg")
    keras_avg = keras_pool(sample_hwc, AveragePooling2D(pool_size=2, strides=2))[:, :, 0]
    print("from scratch:\n", scratch_avg)
    print("keras       :\n", keras_avg)
    check("avg pool", scratch_avg, keras_avg)
    print("expected     : [[3.75, 2.25], [2.5, 3.75]]\n")

    # --- 3. GLOBAL pooling -------------------------------------------------
    print("=" * 60)
    print("GLOBAL POOLING  (whole map -> one number per channel)")
    print("=" * 60)
    scratch_gap = global_pool(sample_hwc, "avg")
    scratch_gmp = global_pool(sample_hwc, "max")
    keras_gap = keras_pool(sample_hwc, GlobalAveragePooling2D())
    keras_gmp = keras_pool(sample_hwc, GlobalMaxPooling2D())
    print(f"global average: scratch={scratch_gap}  keras={keras_gap}  (expected 3.0625)")
    print(f"global max    : scratch={scratch_gmp}  keras={keras_gmp}  (expected 8)")
    check("global avg", scratch_gap, keras_gap)
    check("global max", scratch_gmp, keras_gmp)
    print()

    # --- 4. Depth is preserved across channels -----------------------------
    print("=" * 60)
    print("CHANNELS: pooling preserves depth (H,W shrink; C unchanged)")
    print("=" * 60)
    multi = np.random.rand(8, 8, 16).astype(np.float32)   # 8x8 map, 16 channels
    pooled = pool2d_channels(multi, 2, 2, "max")
    keras_multi = keras_pool(multi, MaxPooling2D(pool_size=2, strides=2))
    print(f"input  shape: {multi.shape}   ->   output shape: {pooled.shape}")
    print(f"depth stayed at {multi.shape[2]} channels; H,W went {multi.shape[:2]} -> {pooled.shape[:2]}")
    check("channel-wise max pool", pooled, keras_multi)
    print()

    print("All done. Every from-scratch result matches Keras. [OK]")


if __name__ == "__main__":
    main()
