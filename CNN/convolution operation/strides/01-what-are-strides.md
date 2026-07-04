# 1. What Are Strides?

## The idea

So far, every time the filter moved, it moved **one pixel**. That "one pixel per
step" is a **stride of 1**. The **stride (S)** is simply *how many pixels the
filter jumps each move*.

```
Stride 1:  ▓░░░░░  →  ░▓░░░░  →  ░░▓░░░  →  ░░░▓░░   (visit every position)
Stride 2:  ▓░░░░░  →  ░░▓░░░  →  ░░░░▓░           (skip every other position)
```

A larger stride means **fewer placements**, so the **output is smaller** and the
convolution is **cheaper** to compute.

## Why use a stride bigger than 1?

1. **Downsampling.** Stride 2 roughly **halves** the width and height. It's an
   alternative to pooling ([basics/06](../../basics/06-pooling-and-downsampling.md))
   for shrinking feature maps — and a *learnable* one, since the filter weights
   are trained.
2. **Speed & memory.** Fewer output positions = less computation and smaller
   tensors. Big early strides are common in fast networks.
3. **Bigger effective receptive field.** By covering ground faster, later layers
   "see" more of the original image sooner.

The trade-off: a bigger stride **skips information** and loses spatial detail.
Stride 1 keeps everything; stride 2+ is a deliberate compression.

## The formula (stride's role)

Stride sits in the denominator of the master output-size formula:

```
            ⌊ N + 2P − F ⌋
  Output =  ⌊ ─────────── ⌋ + 1
            ⌊      S      ⌋
```

The bigger the `S`, the smaller the output.

| N | F | P | S | Output | Effect |
|---|---|---|---|--------|--------|
| 6 | 3 | 0 | 1 | (6−3)/1 + 1 = **4** | full size |
| 6 | 3 | 0 | 2 | (6−3)/2 + 1 = 2.5 → ⌊⌋ = **2** | roughly halved |
| 7 | 3 | 0 | 2 | (7−3)/2 + 1 = **3** | roughly halved |
| 6 | 3 | 1 | 2 | (6+2−3)/2 + 1 = 2.5 → **2** | padded, then halved |

> **The floor matters.** When `(N + 2P − F)` doesn't divide evenly by `S`, we
> round **down** (⌊ ⌋). That means some rightmost/bottommost pixels may be
> **ignored** because the filter would hang off the edge. Padding can rescue
> those pixels if you need them.

## Stride in each dimension

You can set different strides for height and width (e.g. `strides=(2, 1)`), but
in practice they're almost always **equal** (`strides=2` means 2 in both). Apply
the formula independently per dimension.

## Stride vs. pooling — two ways to shrink

| | Strided convolution | Pooling |
|---|---------------------|---------|
| Has learnable weights? | ✅ Yes | ❌ No (fixed rule) |
| What it does | Detects a feature **and** downsamples | Only downsamples |
| Trend | Favored in many modern nets | Classic, still common |

Both achieve downsampling; strided conv folds it into the convolution itself.

---

## Key takeaways

- **Stride (S)** = how many pixels the filter jumps per step.
- Stride **1** = every position (full size); stride **2** = skip one (≈ half
  size).
- Larger stride → **smaller, cheaper output**, but **skips detail**.
- It appears as the **denominator** in `⌊(N + 2P − F)/S⌋ + 1`, and results are
  **floored** (edge pixels may be dropped).
- It's a **learnable alternative to pooling** for downsampling.
- Next: [see it worked out by hand](02-strides-worked-examples.md).
