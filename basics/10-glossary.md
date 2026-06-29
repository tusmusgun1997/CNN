# 10. Glossary

A quick-reference for every term used in this course. Skim it now, return to it
whenever a word trips you up.

## Core concepts

- **CNN (Convolutional Neural Network)** — A neural network specialized for
  grid-like data (especially images) that learns visual patterns automatically.
- **Tensor** — A multi-dimensional array of numbers. Images flow through a CNN
  as tensors shaped `Height × Width × Channels`.
- **Pixel** — A single point in an image, stored as a brightness value
  (0–255) or a set of RGB values.
- **Channel** — One color plane of an image. RGB color = 3 channels; grayscale
  = 1 channel.
- **Feature** — A pattern the network detects (an edge, a texture, an eye).
- **Feature map / activation map** — The grid output of one filter; shows
  *where* a feature appears in the input.

## The operations

- **Convolution** — Sliding a small filter across the input, computing a
  weighted sum at each position. The defining operation of a CNN.
- **Filter / kernel** — A small grid of learnable weights that detects one
  specific pattern.
- **Stride** — How many pixels the filter jumps between positions. Larger
  stride → smaller output.
- **Padding** — Adding a border (usually of zeros) so the filter can cover edge
  pixels and the output keeps its size.
- **Kernel size** — The dimensions of a filter (e.g. 3×3, 5×5).
- **Activation function** — A non-linear function applied after convolution so
  that stacking layers adds real power.
- **ReLU (Rectified Linear Unit)** — The standard activation: `max(0, x)`.
  Keeps positives, zeros out negatives.
- **Softmax** — Final-layer function that converts raw scores into
  probabilities summing to 1.
- **Pooling** — Downsampling a feature map to shrink it and add position
  tolerance.
- **Max pooling** — Pooling that keeps the maximum value in each window.
- **Average / global average pooling** — Pooling that keeps the mean; global
  pooling collapses a whole map to one number.
- **Flatten** — Unrolling a 3D feature tensor into a 1D vector for the dense
  layers.
- **Fully connected / dense layer** — A layer where every input connects to
  every neuron; combines features into a decision.

## Brain / biology terms

- **Visual cortex** — The brain region that processes sight; the inspiration
  for CNNs.
- **Simple cells** — Neurons that detect edges at a specific location and
  orientation → analogous to **convolution filters**.
- **Complex cells** — Neurons that detect a feature regardless of exact
  position → analogous to **pooling**.
- **Receptive field** — The region of the input that a given neuron responds
  to. Small in early layers, large in deep layers.
- **Hierarchy** — Processing in stages of increasing abstraction: edges →
  shapes → objects. Shared by brains and CNNs.
- **Translation invariance** — Recognizing an object regardless of where it is
  in the image.

## Training terms

- **Weights / parameters** — The numbers the network learns (filter values,
  dense-layer connections).
- **Forward pass** — Running an input through the network to get an output.
- **Loss function** — A number measuring how wrong the network's prediction is.
- **Cross-entropy loss** — The standard loss for classification.
- **Backpropagation** — Working backward to compute how each weight affected the
  error (assigning "blame").
- **Gradient** — The direction and amount a weight should change to reduce loss.
- **Gradient descent** — The optimization method that steps weights downhill to
  minimize loss.
- **Optimizer** — The algorithm performing the updates (e.g. **SGD**, **Adam**).
- **Learning rate** — The step size for weight updates; too big = unstable, too
  small = slow.
- **Batch** — A small group of samples processed before one weight update.
- **Epoch** — One full pass through the entire training dataset.
- **Overfitting** — Memorizing training data instead of learning general
  patterns; fails on new data.
- **Dropout** — Randomly disabling neurons during training to prevent
  overfitting.
- **Data augmentation** — Creating variety by flipping/rotating/cropping
  training images.
- **Validation / test set** — Held-out data used to honestly measure
  performance.
- **Transfer learning** — Reusing a pretrained network and fine-tuning it on a
  new task.

## Architectures (names you'll hear)

- **LeNet-5 (1998)** — The pioneering CNN, used for handwritten digits.
- **AlexNet (2012)** — Sparked the deep learning revolution by winning ImageNet.
- **VGGNet (2014)** — Deep, simple stacks of 3×3 convolutions.
- **Inception / GoogLeNet (2014)** — Parallel filters of different sizes.
- **ResNet (2015)** — Introduced **skip connections**, enabling very deep
  networks (100+ layers).

---

← Back to the [course index](README.md)
