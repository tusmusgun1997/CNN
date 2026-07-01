# Pooling — The Complete Guide

Pooling is the CNN operation that **shrinks feature maps while keeping their
essence**. It sits right after convolution + activation and is one of the three
pillars of a classic CNN (convolution → activation → **pooling**). This document
covers *everything* about pooling: what it is, why it exists, every variant, the
math, backpropagation, criticisms, modern alternatives, and the API.

> Companion code: [`pooling_implementation.py`](pooling_implementation.py)
> implements max/average/global pooling **from scratch in NumPy** and verifies it
> against **Keras** on sample data.

---

## Table of contents

1. [What is pooling?](#1-what-is-pooling)
2. [Why do we pool? (the motivation)](#2-why-do-we-pool)
3. [Where pooling sits in a CNN](#3-where-pooling-sits-in-a-cnn)
4. [Max pooling — worked example](#4-max-pooling--worked-example)
5. [Average pooling — worked example](#5-average-pooling--worked-example)
6. [Max vs. average — when to use which](#6-max-vs-average--when-to-use-which)
7. [Global pooling](#7-global-pooling)
8. [Other pooling variants](#8-other-pooling-variants)
9. [Pooling parameters & output-size formula](#9-pooling-parameters--output-size-formula)
10. [Pooling across channels](#10-pooling-across-channels)
11. [Backpropagation through pooling](#11-backpropagation-through-pooling)
12. [Pooling & translation invariance](#12-pooling--translation-invariance)
13. [Pooling vs. strided convolution](#13-pooling-vs-strided-convolution)
14. [Criticisms & downsides](#14-criticisms--downsides)
15. [The brain parallel](#15-the-brain-parallel)
16. [Practical defaults & cheatsheet](#16-practical-defaults--cheatsheet)
17. [API reference (Keras & PyTorch)](#17-api-reference-keras--pytorch)
18. [Key takeaways](#18-key-takeaways)

---

## 1. What is pooling?

**Pooling** (a.k.a. **subsampling** or **downsampling**) slides a small window
over a feature map and replaces each window with a **single summary value** —
usually its **maximum** or its **average**. The result is a smaller feature map
that keeps the important information and discards fine positional detail.

```
Feature map (4×4)          Pool window (2×2), stride 2         Output (2×2)
┌───────────────┐
│ 1  3 │ 2  4  │                 ┌─────┐                       ┌───────┐
│ 5  6 │ 1  2  │   summarize     │ max │  or  │ avg │          │ 6   4 │
├──────┼───────┤   each 2×2  →   └─────┘       └─────┘   →     │ 7   8 │
│ 7  2 │ 8  0  │   window                                      └───────┘
│ 1  0 │ 3  4  │
└───────────────┘
```

Unlike convolution, **pooling has no learnable weights**. It's a fixed rule
(take the max, take the mean), so it adds *zero parameters* to the network.

## 2. Why do we pool?

Pooling delivers four benefits:

1. **Dimensionality reduction.** A 2×2 pool with stride 2 cuts height and width
   in half, dropping ~75% of the numbers. Fewer numbers → less computation, less
   memory, faster training and inference.
2. **Translation invariance.** By summarizing a neighborhood, pooling makes the
   network care *that* a feature exists, not its exact pixel. A cat shifted a few
   pixels produces (nearly) the same pooled output. (See §12.)
3. **Larger receptive field.** After pooling, each deeper neuron effectively
   "sees" a bigger chunk of the original image, helping detect larger structures.
4. **Overfitting control.** Fewer parameters downstream and a compressed
   representation act as a mild regularizer.

## 3. Where pooling sits in a CNN

```
INPUT ─► [ CONV ─► ReLU ─► POOL ] ─► [ CONV ─► ReLU ─► POOL ] ─► ... ─► FLATTEN ─► DENSE ─► OUTPUT
                          ▲                              ▲
                     pooling here                  and here — after each
                                                   conv block, shrinking as we go
```

A typical pattern: **one pooling layer after each convolution block**, steadily
reducing spatial size while convolutions grow the depth. (See
[basics/08](../basics/08-cnn-pipeline-high-level.md) for the full pipeline.)

## 4. Max pooling — worked example

**Max pooling** keeps the **largest** value in each window. Meaning: "was this
feature strongly present anywhere in this patch? Keep the strongest response."

Using a 2×2 window with stride 2 on our 4×4 map:

```
Input (4×4)              Windows                        Max pool output (2×2)
 1  3  2  4          TL[1,3,5,6]=6   TR[2,4,1,2]=4
 5  6  1  2      →                                  →        6   4
 7  2  8  0          BL[7,2,1,0]=7   BR[8,0,3,4]=8            7   8
 1  0  3  4
```

- Top-left window `{1,3,5,6}` → max = **6**
- Top-right window `{2,4,1,2}` → max = **4**
- Bottom-left window `{7,2,1,0}` → max = **7**
- Bottom-right window `{8,0,3,4}` → max = **8**

The 16 numbers became 4, and the **strongest activations survived**. This is by
far the most common pooling in classic CNNs (AlexNet, VGG).

## 5. Average pooling — worked example

**Average pooling** keeps the **mean** of each window — a smoother, gentler
summary that considers every value, not just the peak.

```
Input (4×4)                Average pool output (2×2)
 1  3  2  4          TL (1+3+5+6)/4 = 3.75     TR (2+4+1+2)/4 = 2.25
 5  6  1  2      →
 7  2  8  0          BL (7+2+1+0)/4 = 2.50     BR (8+0+3+4)/4 = 3.75
 1  0  3  4

                     ┌──────────────┐
                     │ 3.75    2.25 │
                     │ 2.50    3.75 │
                     └──────────────┘
```

## 6. Max vs. average — when to use which

| | **Max pooling** | **Average pooling** |
|---|-----------------|---------------------|
| Keeps | the strongest activation | the overall level |
| Good at | detecting *presence* of a sharp feature (edges, textures) | preserving background/context, smooth signals |
| Behavior | sparse, high-contrast | smooth, blurred |
| Sensitivity to noise | can latch onto a single bright outlier | averages noise out |
| Classic use | hidden layers of most CNNs | final "global" layer, some smooth tasks |

**Rule of thumb:** use **max pooling** inside the network for feature detection;
use **global average pooling** at the end before classification (see §7). Max
pooling won historically because "did the feature fire strongly?" is usually the
question that matters.

## 7. Global pooling

**Global pooling** is an extreme pool: the window is the **entire feature map**,
so each map collapses to a **single number**.

```
One feature map (H × W)  ──global average──►  a single number (its mean)
      7 × 7 × 512         ──────────────────►  1 × 1 × 512  (512 numbers total)
```

- **Global Average Pooling (GAP)** — average of the whole map. Extremely popular
  in modern architectures (ResNet, Inception) as a replacement for large fully
  connected layers, because it:
  - has **zero parameters** (huge reduction vs. a flatten→dense),
  - strongly **resists overfitting**,
  - accepts **variable input sizes** (any H×W collapses to one number).
- **Global Max Pooling (GMP)** — the single largest activation in the whole map.

For our 4×4 example: **GAP = 49/16 ≈ 3.06**, **GMP = 8**.

## 8. Other pooling variants

You'll mostly use max/average/global, but these exist:

| Variant | What it does | Notes |
|---------|--------------|-------|
| **Min pooling** | keeps the minimum | rare; useful on inverted/dark-feature tasks |
| **Sum pooling** | sums the window | average pooling without the divide |
| **L2 / RMS pooling** | root of mean of squares | emphasizes energy, smoother than max |
| **Overlapping pooling** | window > stride, so windows overlap (e.g. 3×3 pool, stride 2) | used in AlexNet; slightly reduces overfitting |
| **Fractional / stochastic pooling** | randomized region sizes / random selection | regularization research techniques |
| **Adaptive pooling** | you specify the *output* size; it computes the window | PyTorch's `AdaptiveAvgPool2d`, handy for fixed-size heads |
| **Mixed / gated pooling** | learnable blend of max and average | learns the best mix per layer |

## 9. Pooling parameters & output-size formula

Pooling has the same geometric knobs as convolution — **except no weights**:

| Parameter | Meaning | Common value |
|-----------|---------|--------------|
| **Pool size (F)** | window dimensions | 2×2 (sometimes 3×3) |
| **Stride (S)** | step between windows | usually = pool size (2) |
| **Padding (P)** | border added | usually 0 (`valid`); default in Keras |

The **output size** uses the same formula as convolution:

```
            ⌊ N + 2P − F ⌋
  Output =  ⌊ ─────────── ⌋ + 1
            ⌊      S      ⌋
```

Example — 4×4 map, F=2, S=2, P=0:  `(4 − 2)/2 + 1 = 2` → **2×2**. ✅

| Input | Pool F | Stride S | Output |
|-------|--------|----------|--------|
| 4×4 | 2 | 2 | 2×2 |
| 6×6 | 2 | 2 | 3×3 |
| 7×7 | 2 | 2 | 3×3 (floor drops a row/col) |
| 8×8 | 2 | 2 | 4×4 |
| 28×28 | 2 | 2 | 14×14 |

> **Default convention:** if you say "2×2 max pooling," people assume **stride
> 2** and **no padding**. Keras' `MaxPooling2D` defaults `strides = pool_size`.

## 10. Pooling across channels

Pooling is applied to **each channel independently** — it never mixes channels.
If the input is `H × W × C`, pooling shrinks `H` and `W` but **leaves depth `C`
unchanged**:

```
Input  4 × 4 × 32   ──2×2 max pool──►   Output  2 × 2 × 32
        (depth preserved; each of the 32 maps pooled on its own)
```

Contrast with convolution, which *sums across* channels and changes depth to the
number of filters. **Pooling preserves depth; convolution transforms it.**

## 11. Backpropagation through pooling

Pooling has no weights to learn, but gradients must still flow **backward**
through it during training. How they route depends on the pooling type:

- **Max pooling — gradient routing.** During the forward pass, remember *which*
  input was the max ("the argmax"). In the backward pass, the full gradient goes
  **only to that winning position**; every other cell in the window gets **0**.
  Think of it as: "only the neuron that won gets the credit/blame."

  ```
  Forward:  window {1,3,5,6} → max = 6 (position bottom-left)
  Backward: gradient g → routed entirely to that position; others get 0
            ┌──────┐          ┌──────┐
            │ 1  3 │  forward  │ 0  0 │  backward
            │ 5  6 │  ──────►  │ 0  g │
            └──────┘           └──────┘
  ```

- **Average pooling — gradient spreading.** The gradient is **divided equally**
  among all cells in the window (each gets `g / (F×F)`), since each contributed
  equally to the average.

This is why max pooling produces **sparse** gradients and average pooling
produces **smooth, distributed** ones.

## 12. Pooling & translation invariance

This is pooling's signature benefit. Because a max (or average) over a
neighborhood doesn't change when the peak shifts by one pixel, the pooled output
is **stable under small translations**:

```
Feature at (1,1):  window {0,0,0,9} → max 9
Feature at (1,2):  window {0,0,9,0} → max 9     ← same pooled output!
```

The network gains a useful *blindness to exact position* — it recognizes "there's
an eye in this region" without caring about the precise pixel. This is the
engineering echo of the visual cortex's **complex cells** (see §15).

⚠️ Caveat: pooling gives **small, local** invariance only. It does **not** make a
CNN invariant to large shifts, rotation, or scaling — those need data
augmentation or specialized architectures.

## 13. Pooling vs. strided convolution

A modern alternative to pooling is to **downsample inside the convolution** using
`stride = 2` (see [strides](../convolution%20operation/strides/00-overview.md)).

| | **Pooling** | **Strided convolution** |
|---|-------------|-------------------------|
| Learnable? | ❌ fixed rule | ✅ learns how to downsample |
| Parameters | none | has weights |
| Interpretability | simple, predictable | flexible, learned |
| Trend | classic (LeNet→VGG) | common in modern nets (ResNet variants, many all-conv nets) |

Some influential work ("Striving for Simplicity: The All Convolutional Net")
showed you can **replace pooling entirely** with strided convolutions and match
accuracy. Both remain valid; many architectures still use max pooling for its
simplicity and strong feature-detection behavior.

## 14. Criticisms & downsides

Pooling isn't free of controversy:

- **Loss of spatial information.** Throwing away exact positions hurts tasks that
  need precise localization (segmentation, keypoint detection). Such models often
  reduce or avoid pooling, or use architectures that recover resolution.
- **Geoffrey Hinton's critique.** Hinton famously called max pooling "a big
  mistake" because it discards the *precise spatial relationships* between parts.
  This motivated **Capsule Networks**, which try to preserve pose information.
- **Too aggressive.** Repeated pooling can destroy fine detail needed for
  fine-grained recognition.
- **Not learnable.** A fixed rule can't adapt to the data the way a strided conv
  can — one reason for the modern shift toward strided convolutions.

These are trade-offs, not deal-breakers. For plain image classification, pooling
still works very well.

## 15. The brain parallel

Pooling is the engineering version of the visual cortex's **complex cells**
(Hubel & Wiesel — see [basics/02](../basics/02-cnn-and-the-human-brain.md)).
Complex cells respond to a feature (like an oriented edge) **anywhere within a
region**, ignoring its exact location — precisely what max pooling does by taking
the strongest response over a neighborhood. Both trade *positional precision* for
*robust detection*, and both build the position tolerance that lets you recognize
an object wherever it appears in your field of view.

## 16. Practical defaults & cheatsheet

If you're just building a CNN and want sensible choices:

```
• Inside the network:   2×2 MAX pooling, stride 2   (halves H and W)
• After conv blocks:    one pool per block
• Before the classifier: GLOBAL AVERAGE pooling      (replaces flatten→dense)
• Padding:              'valid' (none) is the norm for pooling
• Modern alternative:   stride-2 convolutions instead of pooling
```

Quick decision guide:

- Detecting sharp features in hidden layers → **max pooling**.
- Collapsing to a feature vector for classification → **global average pooling**.
- Need precise localization (segmentation) → **minimize pooling** / use strided
  or dilated convs.
- Want a fully learnable downsampler → **strided convolution**.

## 17. API reference (Keras & PyTorch)

**Keras (TensorFlow):**

```python
from tensorflow.keras.layers import (
    MaxPooling2D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
)

MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")  # strides defaults to pool_size
AveragePooling2D(pool_size=(2, 2), strides=None, padding="valid")
GlobalAveragePooling2D()   # H×W×C -> C
GlobalMaxPooling2D()       # H×W×C -> C
```

**PyTorch:**

```python
import torch.nn as nn

nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
nn.AdaptiveAvgPool2d(output_size=1)   # global average pooling
nn.AdaptiveMaxPool2d(output_size=1)   # global max pooling
```

See [`pooling_implementation.py`](pooling_implementation.py) for these in action,
plus from-scratch NumPy versions that reproduce the exact numbers in this doc.

## 18. Key takeaways

- **Pooling** downsamples feature maps by summarizing windows into one value; it
  has **no learnable weights**.
- **Max pooling** keeps the strongest activation (default in hidden layers);
  **average pooling** keeps the mean (smoother).
- **Global average pooling** collapses each map to one number — a
  parameter-free, overfitting-resistant replacement for dense layers.
- Output size follows `⌊(N + 2P − F)/S⌋ + 1`; a **2×2 stride-2** pool **halves**
  H and W and **preserves depth**.
- Pooling is applied **per channel**; it gives **local translation invariance**
  (the CNN echo of the brain's **complex cells**).
- **Backprop:** max pooling routes the gradient to the winner; average pooling
  spreads it evenly.
- Modern nets increasingly use **strided convolutions** instead; pooling has
  known **downsides** (lost spatial info — Hinton's critique).
- Companion code: [`pooling_implementation.py`](pooling_implementation.py).
