# 7. Fully Connected Layers & Classification

By now the CNN has extracted rich features (edges → shapes → object parts). But
a pile of feature maps isn't an answer. The final stage **converts features
into a decision** — "this is a cat." That's the job of the fully connected
layers and the output.

## Two halves of a CNN

It helps to see a CNN as two distinct phases:

```
┌──────────────────────────────┐   ┌───────────────────────────┐
│   FEATURE EXTRACTION          │   │   CLASSIFICATION          │
│   (conv + ReLU + pooling)     │ → │   (flatten + dense layers │
│   "What patterns are here?"   │   │    + softmax)             │
│                               │   │   "So what is it?"        │
└──────────────────────────────┘   └───────────────────────────┘
```

Files 4–6 covered the left half. This file covers the right half.

## Step 1: Flatten

The convolutional part outputs a 3D tensor, e.g. `7 × 7 × 512`. A fully
connected layer expects a flat **1D list of numbers**. So we **flatten** —
unroll the tensor into one long vector:

```
7 × 7 × 512  →  flatten  →  a vector of 25,088 numbers
```

Nothing is lost or computed here; the same numbers are just laid out in a line.

> Modern networks often replace flattening with **global average pooling**
> (file 6), which produces a much shorter vector (e.g. 512 numbers) and reduces
> overfitting. Both feed the same kind of layer next.

## Step 2: Fully connected (dense) layers

A **fully connected layer** (also called **dense**) connects *every* input
number to *every* neuron — the classic neural network from before CNNs. Its job
is to look at the high-level features all at once and reason about their
combination:

> "There are pointy-ear features AND whisker features AND fur texture →
> strongly suggests **cat**."

Each neuron computes a weighted sum of all inputs, adds a bias, and applies an
activation (usually ReLU). Stacking one or two of these mixes the features into
class evidence.

```
Features ──► [dense neurons] ──► [dense neurons] ──► class scores
 (vector)        (e.g. 256)          (e.g. 128)        (e.g. 10)
```

## Step 3: The output layer + Softmax

The final dense layer has **one neuron per class**. For a 3-class problem (cat,
dog, bird) it outputs three raw scores called **logits**, e.g.:

```
cat: 2.0    dog: 1.0    bird: 0.1
```

These aren't probabilities yet. **Softmax** converts them into a clean
probability distribution that sums to 1:

```
cat: 0.66    dog: 0.24    bird: 0.10     (sums to 1.00)
```

Now the network can say: **"66% confident this is a cat."** The class with the
highest probability is the prediction.

### What Softmax does, intuitively

Softmax exponentiates each score (making big scores dominate) and then
normalizes so everything adds to 100%. It turns "raw opinions" into "calibrated
confidence."

| Output type | Final activation |
|-------------|------------------|
| Pick one of many classes (cat/dog/bird) | **Softmax** |
| Yes/no, single label | **Sigmoid** |
| Multiple independent labels at once | **Sigmoid per class** |

## Why FC layers are "expensive"

Because every input connects to every neuron, dense layers hold the **majority
of a CNN's parameters**. A flatten of 25,088 into 256 neurons is already
~6.4 million weights. This is why modern designs lean on global average pooling
to keep them small.

## The brain parallel

This stage is like the **higher association areas** of your brain (the IT
cortex and beyond) where assembled visual features are integrated and finally
**named**. The early visual cortex says "edges and shapes here"; the later
regions say "that's a cat" — exactly the division of labor between a CNN's
convolutional front end and its dense classifier back end.

---

## Key takeaways

- A CNN = **feature extraction** (conv/pool) + **classification** (dense layers).
- **Flatten** turns the 3D feature tensor into a 1D vector (or use global
  average pooling).
- **Fully connected / dense** layers combine features into class evidence.
- The **output layer** has one neuron per class; **Softmax** turns scores into
  probabilities that sum to 1.
- Dense layers hold most of the network's parameters.
- This mirrors the brain's high-level "naming" regions.
- Next: see [the whole pipeline assembled](08-cnn-pipeline-high-level.md).
