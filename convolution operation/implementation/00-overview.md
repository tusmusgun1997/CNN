# Implementation — Padding & Strides in Keras

This folder turns the theory into runnable code. The script
[`conv_padding_strides_keras.py`](conv_padding_strides_keras.py) implements
**both padding and strides** with Keras `Conv2D`, using the exact **6×6
vertical-edge** sample image from the markdown lessons — so you can watch the
hand-worked numbers appear for real.

## What the script does

1. **Builds sample data** — the same 6×6 vertical-edge image (left bright `10`,
   right dark `0`).
2. **Loads a hand-made filter** — the `[1 0 -1; 1 0 -1; 1 0 -1]` vertical-edge
   detector, injected as the `Conv2D` kernel with `set_weights()`.
3. **Runs all four padding × stride combinations** and prints each output's
   shape and values:

   | padding | strides | output |
   |---------|---------|--------|
   | `valid` | 1 | 4 × 4 |
   | `same`  | 1 | 6 × 6 |
   | `valid` | 2 | 2 × 2 |
   | `same`  | 2 | 3 × 3 |

4. **Bonus model** — a small `Sequential` CNN on a random image batch showing
   how `padding` and `strides` are used normally, with `model.summary()`.

## How to run

```bash
# 1. Install TensorFlow (includes Keras)
pip install tensorflow

# 2. Run the script
python conv_padding_strides_keras.py
```

> Python 3.9–3.12 recommended. First run downloads/initializes TensorFlow, so it
> may take a few seconds. No GPU required — everything here is tiny.

## Expected output (abridged)

```
Sample 6x6 input image (vertical edge, left=10 bright, right=0 dark):
     10    10    10     0     0     0
     ...

  padding=valid  strides=1  ->  output shape (1, 4, 4, 1)  (spatial (4, 4))
      output values:
          0.0   30.0   30.0    0.0
          0.0   30.0   30.0    0.0
          0.0   30.0   30.0    0.0
          0.0   30.0   30.0    0.0

  padding=same   strides=1  ->  output shape (1, 6, 6, 1)  (spatial (6, 6))
      ...

  padding=valid  strides=2  ->  output shape (1, 2, 2, 1)  (spatial (2, 2))
      output values:
          0.0   30.0
          0.0   30.0

  padding=same   strides=2  ->  output shape (1, 3, 3, 1)  (spatial (3, 3))
      ...
```

The `valid/stride1 → 4×4` block reproduces
[convolution operation/01](../convolution%20operation/01-convolution-6x6-with-3x3.md),
and the four shapes match the table in
[strides/02](../strides/02-strides-worked-examples.md). Theory ✅ verified by
code.

## Key API points illustrated

| Concept | Keras code |
|---------|------------|
| Padding mode | `Conv2D(..., padding="valid")` or `padding="same"` |
| Stride | `Conv2D(..., strides=2)` (or `strides=(2, 2)`) |
| Number of filters → output depth | `Conv2D(filters=32, ...)` |
| Input shape (H, W, C) | `Input(shape=(28, 28, 3))` |
| Injecting a fixed kernel | `layer.build(...)` then `layer.set_weights([kernel, bias])` |
| Kernel tensor shape | `(kernel_h, kernel_w, in_channels, out_channels)` |

## Where this fits

```
convolution operation/
├── convolution operation/   ← the math, by hand
├── padding/                 ← padding theory
├── strides/                 ← strides theory
└── implementation/          ← YOU ARE HERE: run it in Keras
```

← Back to the [convolution operation index](../00-overview.md).
