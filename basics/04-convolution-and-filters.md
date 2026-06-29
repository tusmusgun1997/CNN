# 4. Convolution & Filters

This is the heart of a CNN — the operation it's named after. Take your time
here; once convolution clicks, everything else is easy.

## The idea in one picture

Imagine sliding a small magnifying glass across a photo, one patch at a time,
checking each patch for a specific pattern ("is there a vertical edge here?").
That sliding-and-checking is **convolution**. The magnifying glass is a
**filter** (also called a **kernel**).

```
        Image (5×5)                 Filter (3×3)
   ┌──┬──┬──┬──┬──┐
   │  │  │  │  │  │              ┌──┬──┬──┐
   ├──┼──┼──┼──┼──┤              │ 1│ 0│-1│
   │ ▓▓▓▓▓▓ │  │  │   slide →    │ 1│ 0│-1│     →  produces a number
   │ ▓ win ▓│  │  │   the        │ 1│ 0│-1│        for each position
   │ ▓▓▓▓▓▓ │  │  │   window
   ├──┼──┼──┼──┼──┤              └──┴──┴──┘
   └──┴──┴──┴──┴──┘
```

## What a filter actually is

A **filter** is a small grid of numbers (weights), typically **3×3** or **5×5**.
These numbers define *what pattern the filter responds to*.

For example, this 3×3 filter detects **vertical edges**:

```
 1  0  -1
 1  0  -1
 1  0  -1
```

It gives a big response where the left side is bright and the right side is
dark (a vertical edge), and near zero on a flat, uniform region.

> **The magic:** In a CNN, *you don't design these filters by hand.* The
> network **learns** the best filter values automatically during training
> (file 9). Early in training they're random; by the end they've become edge
> detectors, color-blob detectors, texture detectors, and more.

## The convolution operation, step by step

At each position, you:

1. **Overlay** the filter on a patch of the image.
2. **Multiply** each filter number by the pixel under it.
3. **Sum** all those products into a single number.
4. **Write** that number into the output grid.
5. **Slide** the filter over and repeat.

### A worked example

Patch of image and a filter:

```
Image patch        Filter
┌─────────┐      ┌──────────┐
│ 1  1  1 │      │ 1  0  -1 │
│ 1  1  1 │  ×   │ 1  0  -1 │
│ 1  1  1 │      │ 1  0  -1 │
└─────────┘      └──────────┘
```

Multiply element-by-element and sum:

```
(1×1)+(1×0)+(1×-1)
+(1×1)+(1×0)+(1×-1)
+(1×1)+(1×0)+(1×-1)
= 0
```

Result is **0** — a flat region has no vertical edge, so the edge detector
stays quiet. If the right column were dark (0s instead of 1s), the result would
be a large positive number — "edge found here!"

## Feature maps

When a filter has slid across the *entire* image, the grid of numbers it
produced is called a **feature map** (or *activation map*). It's a map showing
**where in the image that particular feature appears**.

One filter → one feature map. A convolution layer has **many filters** (say 32
or 64), so it produces **many feature maps stacked together**:

```
Input image (H × W × 3)
        │
   [ 32 filters ]
        │
        ▼
Output (H × W × 32)   ← 32 feature maps, one per filter
```

This is exactly why depth grows as you go deeper (from file 3): each layer adds
more filters, each looking for a different pattern.

## Key knobs (hyperparameters)

You'll see these terms constantly. They control *how* the filter slides:

| Term | What it means | Effect |
|------|---------------|--------|
| **Kernel size** | Filter dimensions (e.g. 3×3) | Bigger = sees more context per step |
| **Stride** | How many pixels the filter jumps each move | Bigger stride = smaller output, faster |
| **Padding** | Adding a border of zeros around the image | Keeps output the same size; protects edges |
| **Number of filters** | How many patterns this layer looks for | More = richer, but heavier |

### Padding, visually

Without padding, the filter can't center on edge pixels, so the output shrinks
and border information is lost. Adding a ring of zeros ("same padding") fixes
this:

```
Original          Padded with zeros
┌───────┐        ┌───────────┐
│ image │   →    │ 0 0 0 0 0 │
└───────┘        │ 0 image 0 │
                 │ 0 0 0 0 0 │
                 └───────────┘
```

## Why shared weights are a superpower

The same filter is used at **every** position of the image. This means:

- **Fewer parameters.** A 3×3 filter has just 9 numbers (plus a bias), no matter
  how big the image is. Compare that to millions in a fully connected layer.
- **Translation invariance.** A learned "eye detector" works whether the eye is
  top-left or bottom-right. Move the cat — the detector still fires. This is the
  CNN version of the brain's position-tolerant **complex cells** (file 2).

## The brain parallel

A single filter scanning for one pattern across the whole image is exactly like
a population of identical **simple cells** (file 2), each watching its own spot
for the same oriented edge. The CNN just reuses *one* set of weights instead of
duplicating them — an engineering shortcut for the same biological idea.

---

## Key takeaways

- **Convolution** = slide a small **filter** over the image, computing a
  weighted sum at each spot.
- A filter detects **one pattern** (edge, color, texture); its values are
  **learned, not hand-coded**.
- The output of one filter is a **feature map** showing *where* that pattern
  appears.
- One conv layer uses **many filters** → many feature maps → growing depth.
- **Stride, padding, and kernel size** control how the filter scans.
- **Shared weights** give efficiency and **translation invariance**.
- Next: making the network non-linear with [activations](05-activation-and-nonlinearity.md).
