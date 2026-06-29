# 3. How CNNs "See" Images

Before we can talk about convolution, you need to know what an image *is* to a
computer. Spoiler: it's just numbers.

## An image is a grid of numbers

A grayscale image is a **2D grid (matrix)** of numbers. Each number is a
**pixel** representing brightness, usually from **0 (black)** to **255 (white)**.

A tiny 5×5 grayscale image of a diagonal line might look like:

```
  0    0    0    0  255
  0    0    0  255    0
  0    0  255    0    0
  0  255    0    0    0
255    0    0    0    0
```

To you that's a line. To a computer it's just 25 numbers. **The CNN's entire
job is to find meaning in patterns of numbers like these.**

## Color images have channels

A color image isn't one grid — it's **three grids stacked together**, one each
for **Red, Green, and Blue (RGB)**. Each is called a **channel**.

```
        Red channel        Green channel       Blue channel
       ┌───────────┐      ┌───────────┐      ┌───────────┐
       │ 255 200 ..│      │  0  50  ..│      │  0  10  ..│
       │ ...       │  +   │ ...       │  +   │ ...       │
       └───────────┘      └───────────┘      └───────────┘
```

So a color image is a **3D block of numbers** with three dimensions:

```
Height  ×  Width  ×  Channels
 200    ×   200   ×     3
```

This 3D block is called a **tensor**. Throughout a CNN, data always flows as
tensors — they just change shape as they pass through layers.

## The "shape" of data

Whenever you work with CNNs you'll constantly talk about **shape** — the
dimensions of the tensor. For example:

| Stage | Shape (H × W × C) | Meaning |
|-------|-------------------|---------|
| Input photo | 224 × 224 × 3 | Color image |
| After early conv layer | 224 × 224 × 32 | 32 feature maps |
| After pooling | 112 × 112 × 32 | Smaller, same depth |
| Deep in the network | 7 × 7 × 512 | Tiny but very "deep" |

Notice the pattern: as data flows through a CNN, the **height and width
shrink** while the **depth (channels) grows**. The network trades *spatial
detail* for *semantic richness* — it stops caring "where exactly" and starts
caring "what is it."

## Why values get normalized

Raw pixels run 0–255, but CNNs train better when numbers are small and centered
around zero. So almost always the first step is **normalization**, e.g.:

```
normalized_pixel = pixel / 255          → range 0 to 1
```

or scaling to a mean of 0 and standard deviation of 1. This is a small but
important practical detail — it helps the network learn faster and more stably.

## The brain parallel

Your retina also converts light into signals — not numbers, but electrical
impulses whose *rate* encodes brightness and color (via red/green/blue-ish
cone cells, conveniently!). In both cases, the raw scene becomes a **signal
that downstream processing can work on**. The RGB channels of a CNN are a
rough echo of the three cone types in your eye.

## A note on grayscale vs. color

Many tutorials use grayscale (1 channel) to keep things simple — for example
the famous **MNIST** handwritten-digit dataset is 28 × 28 × 1. Everything in
this course works the same way; color just means "do it three times, once per
channel, and add the results." Don't let channels intimidate you.

---

## Key takeaways

- To a computer, an image is just a **grid of numbers** (pixels, 0–255).
- Grayscale = 1 channel; color = 3 channels (**RGB**).
- Data flows through a CNN as a **tensor**: Height × Width × Channels.
- As you go deeper, **spatial size shrinks** and **depth grows**.
- Pixels are usually **normalized** before training.
- Next: the operation that scans these grids — [convolution](04-convolution-and-filters.md).
