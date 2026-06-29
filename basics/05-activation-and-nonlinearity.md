# 5. Activation & Non-Linearity

After every convolution, a CNN applies an **activation function**. It's a tiny
step but absolutely essential. Here's why.

## The problem: stacking linear things is pointless

Convolution is a **linear** operation (just multiplies and adds). And here's a
mathematical fact:

> Stacking linear operations gives you... another linear operation.

So if you stacked 50 convolution layers with *nothing* in between, the whole
50-layer network would be mathematically equivalent to a **single** layer. All
that depth would be wasted. The network could only learn straight-line
relationships and would fail at anything complex (which is everything
interesting).

## The fix: add a non-linearity

An **activation function** is a simple non-linear function applied to every
number in a feature map. It "bends" the signal so that stacking layers actually
adds power. This is what lets CNNs learn curved, complex decision boundaries —
the difference between "draw one straight line" and "draw any shape you want."

## ReLU: the workhorse

The most common activation in CNNs is **ReLU** (Rectified Linear Unit). It is
delightfully simple:

```
ReLU(x) = max(0, x)
```

In words: **if the number is negative, make it zero; otherwise keep it.**

```
   output
     │        ╱
     │      ╱
     │    ╱
─────┼──╱──────── input
     │ (flat at 0 for negatives)
```

### Example

```
Before ReLU:   -3   5   -1   8   0   -7
After ReLU:     0   5    0   8   0    0
```

Why this helps:

- **Cheap.** Just a comparison with zero — extremely fast.
- **Avoids "vanishing gradients."** Older functions (sigmoid, tanh) squash
  signals into tiny ranges, making deep networks hard to train. ReLU keeps
  positive signals fully intact.
- **Sparse activations.** Many outputs become exactly zero, so only the
  relevant detectors "speak up" — efficient and often more robust.

## The brain parallel — a neuron "firing"

This is one of the most beautiful biological echoes in deep learning. A real
neuron collects incoming signals and **only fires** if the total exceeds a
**threshold**. Below threshold: silence. Above it: it sends a spike.

ReLU is a cartoon of exactly that:

```
Below threshold (negative)  →  output 0      ("neuron stays quiet")
Above threshold (positive)  →  output passes  ("neuron fires")
```

When we say a neuron "activates" or a feature map "lights up," we mean its
ReLU output is positive — the detector found its pattern and is passing the
signal forward.

## Other activation functions (good to know)

You'll meet these later; ReLU is your default for the hidden layers of a CNN.

| Function | Formula / idea | When used |
|----------|----------------|-----------|
| **ReLU** | `max(0, x)` | Default for conv/hidden layers |
| **Leaky ReLU** | tiny slope for negatives | Fixes "dead neurons" that get stuck at 0 |
| **Sigmoid** | squashes to 0–1 | Binary output ("yes/no") |
| **Softmax** | turns scores into probabilities | **Final layer** for multi-class output (file 7) |
| **Tanh** | squashes to −1 to 1 | Older networks, some RNNs |

> **Rule of thumb:** ReLU in the middle of the network, **Softmax** at the very
> end when you need class probabilities.

---

## Key takeaways

- Convolution is **linear**; without help, depth is wasted.
- An **activation function** adds **non-linearity**, unlocking the power of deep
  stacks.
- **ReLU** (`max(0, x)`) is the standard: cheap, effective, train-friendly.
- It mirrors a biological neuron's **fire/don't-fire threshold**.
- **Softmax** is the special activation used at the output for probabilities.
- Next: shrinking the data with [pooling](06-pooling-and-downsampling.md).
