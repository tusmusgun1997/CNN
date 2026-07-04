"""
Convolution in Keras — demonstrating PADDING and STRIDES on sample data.

This script makes the hand-worked examples from the `convolution operation`
folder real. It:

  1. Builds a 6x6 "vertical edge" sample image (matches the markdown examples).
  2. Loads a hand-made vertical-edge detector as the Conv2D kernel.
  3. Runs the convolution with every combination of:
         padding = "valid" | "same"
         strides = 1 | 2
     ...and prints the OUTPUT SHAPE and OUTPUT VALUES for each.
  4. Confirms the numbers match the by-hand results:
         valid, stride 1 -> 4x4
         same,  stride 1 -> 6x6
         valid, stride 2 -> 2x2
         same,  stride 2 -> 3x3
  5. Bonus: a small Sequential model on a random image batch, showing how
     padding/strides are used in a normal CNN (via model.summary()).

Run:
    pip install tensorflow
    python conv_padding_strides_keras.py
"""

import numpy as np

# Keras ships inside TensorFlow. This import works for TF 2.x.
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D

# Make results reproducible and printouts clean.
np.set_printoptions(precision=1, suppress=True)
tf.random.set_seed(42)
np.random.seed(42)


# ----------------------------------------------------------------------------
# 1. SAMPLE DATA — a 6x6 image with a clear vertical edge (left bright, right dark)
# ----------------------------------------------------------------------------
def make_sample_image():
    """Return a 6x6 vertical-edge image shaped for Keras: (batch, H, W, channels)."""
    image = np.array(
        [
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    # Keras Conv2D expects a 4D tensor: (batch_size, height, width, channels).
    return image.reshape(1, 6, 6, 1)


# ----------------------------------------------------------------------------
# 2. THE FILTER — a 3x3 vertical-edge detector:  [ 1 0 -1 ; 1 0 -1 ; 1 0 -1 ]
# ----------------------------------------------------------------------------
def make_vertical_edge_kernel():
    """Return (kernel, bias) shaped the way Keras Conv2D expects."""
    kernel_2d = np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        dtype=np.float32,
    )
    # Keras kernel shape = (kernel_h, kernel_w, in_channels, out_channels).
    kernel = kernel_2d.reshape(3, 3, 1, 1)
    bias = np.zeros(shape=(1,), dtype=np.float32)  # one bias per output filter
    return kernel, bias


# ----------------------------------------------------------------------------
# 3. RUN ONE CONVOLUTION with a given padding + stride, and print the result.
# ----------------------------------------------------------------------------
def run_conv(x, kernel, bias, padding, strides):
    """Apply a single Conv2D layer with fixed weights and report the output."""
    layer = Conv2D(
        filters=1,            # one output feature map
        kernel_size=3,        # 3x3 filter
        strides=strides,      # <-- the STRIDE knob
        padding=padding,      # <-- the PADDING knob: "valid" or "same"
        use_bias=True,
    )
    # Build the layer so it has weight tensors, then inject our hand-made filter.
    layer.build(input_shape=(None, 6, 6, 1))
    layer.set_weights([kernel, bias])

    output = layer(x)                 # forward pass
    output_2d = output.numpy()[0, :, :, 0]  # drop batch and channel dims for display

    print(f"  padding={padding:<5}  strides={strides}  ->  "
          f"output shape {tuple(output.shape)}  (spatial {output_2d.shape})")
    print("      output values:")
    for row in output_2d:
        print("        " + "  ".join(f"{v:5.1f}" for v in row))
    print()
    return output_2d


# ----------------------------------------------------------------------------
# 4. BONUS — a normal CNN stub using padding + strides on a random batch.
# ----------------------------------------------------------------------------
def bonus_model_demo():
    """Show how padding/strides appear in a typical multi-filter CNN."""
    print("=" * 68)
    print("BONUS: a small CNN on a random 8-image batch of 28x28 RGB images")
    print("=" * 68)

    # Sample data: 8 random color images (like a mini training batch).
    x_batch = np.random.rand(8, 28, 28, 3).astype(np.float32)

    model = Sequential(
        [
            Input(shape=(28, 28, 3)),
            # 'same' padding keeps size; 16 learnable filters.
            Conv2D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
            # stride 2 downsamples 28 -> 14 while detecting features.
            Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu"),
            # 'valid' padding + stride 2 shrinks further: 14 -> 6.
            Conv2D(64, kernel_size=3, strides=2, padding="valid", activation="relu"),
        ]
    )

    model.summary()
    out = model(x_batch)
    print(f"\nInput batch shape : {x_batch.shape}")
    print(f"Output batch shape: {tuple(out.shape)}")
    print("Notice how H and W shrink (28 -> 14 -> 6) while depth grows (3 -> 16 -> 32 -> 64).\n")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("TensorFlow version:", tf.__version__, "\n")

    x = make_sample_image()
    kernel, bias = make_vertical_edge_kernel()

    print("Sample 6x6 input image (vertical edge, left=10 bright, right=0 dark):")
    for row in x[0, :, :, 0]:
        print("   " + "  ".join(f"{v:4.0f}" for v in row))
    print()

    print("=" * 68)
    print("Convolving with the vertical-edge filter for every padding/stride combo")
    print("=" * 68 + "\n")

    # These four calls reproduce the table from strides/02-strides-worked-examples.md
    run_conv(x, kernel, bias, padding="valid", strides=1)  # expect 4x4
    run_conv(x, kernel, bias, padding="same",  strides=1)  # expect 6x6
    run_conv(x, kernel, bias, padding="valid", strides=2)  # expect 2x2
    run_conv(x, kernel, bias, padding="same",  strides=2)  # expect 3x3

    print("Expected shapes (from the hand-worked math):")
    print("   valid/stride1 -> 4x4 | same/stride1 -> 6x6 "
          "| valid/stride2 -> 2x2 | same/stride2 -> 3x3\n")

    bonus_model_demo()


if __name__ == "__main__":
    main()
