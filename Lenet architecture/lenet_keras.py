"""
LeNet-5 in Keras — the classic 1998 CNN by Yann LeCun et al.

Architecture (for a 32x32 grayscale input):

  Input 32x32x1
    -> C1: Conv  6 @ 5x5  (tanh)   -> 28x28x6      # feature extraction
    -> S2: AvgPool 2x2  stride 2   -> 14x14x6      # subsampling
    -> C3: Conv 16 @ 5x5  (tanh)   -> 10x10x16
    -> S4: AvgPool 2x2  stride 2   ->  5x5x16
    -> C5: Conv 120 @ 5x5 (tanh)   ->  1x1x120     # collapses to a vector
    -> Flatten                     ->  120
    -> F6: Dense 84  (tanh)        ->  84          # classification
    -> Output: Dense 10 (softmax)  ->  10

Notes
-----
* The original 1998 paper used scaled tanh activations and a Gaussian-RBF
  output. Here we use tanh + softmax, the common modern teaching version.
* Original S2/S4 were trainable subsampling; we use plain AveragePooling2D.
* Total parameters come out to 61,706 — matching the classic figure.

Run
---
    pip install tensorflow
    python lenet_keras.py            # build model, print summary, dummy forward pass
    python lenet_keras.py --train    # also train on MNIST (downloads the data)
"""

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input


def build_lenet(input_shape=(32, 32, 1), num_classes=10, activation="tanh"):
    """Build and compile the classic LeNet-5 model."""
    model = Sequential(
        name="LeNet-5",
        layers=[
            Input(shape=input_shape),
            # --- Feature extraction ------------------------------------------
            Conv2D(6, kernel_size=5, strides=1, activation=activation, name="C1_conv"),      # 28x28x6
            AveragePooling2D(pool_size=2, strides=2, name="S2_pool"),                          # 14x14x6
            Conv2D(16, kernel_size=5, strides=1, activation=activation, name="C3_conv"),      # 10x10x16
            AveragePooling2D(pool_size=2, strides=2, name="S4_pool"),                          # 5x5x16
            Conv2D(120, kernel_size=5, strides=1, activation=activation, name="C5_conv"),     # 1x1x120
            # --- Classification ----------------------------------------------
            Flatten(name="flatten"),                                                            # 120
            Dense(84, activation=activation, name="F6_dense"),                                  # 84
            Dense(num_classes, activation="softmax", name="output"),                            # 10
        ],
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_on_mnist(model, epochs=3, batch_size=128):
    """Train LeNet on MNIST. MNIST is 28x28, so we pad it to 32x32."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def prep(x):
        x = x.astype("float32") / 255.0                     # normalize to [0, 1]
        x = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="constant")  # 28x28 -> 32x32
        return x[..., np.newaxis]                            # add channel dim

    x_train, x_test = prep(x_train), prep(x_test)

    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
    )
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {acc:.4f}  (loss {loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 in Keras")
    parser.add_argument("--train", action="store_true", help="train on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print("TensorFlow version:", tf.__version__, "\n")

    model = build_lenet()
    model.summary()

    # A dummy forward pass so the script does something even without internet.
    dummy = np.random.rand(4, 32, 32, 1).astype("float32")
    preds = model(dummy)
    print(f"\nDummy batch {dummy.shape}  ->  output {tuple(preds.shape)}  (10 class probs)")
    print(f"Total parameters: {model.count_params():,}")

    if args.train:
        print("\nTraining on MNIST (downloads ~11 MB the first time)...")
        train_on_mnist(model, epochs=args.epochs)
    else:
        print("\nTip: run  'python lenet_keras.py --train'  to train on MNIST.")


if __name__ == "__main__":
    main()
