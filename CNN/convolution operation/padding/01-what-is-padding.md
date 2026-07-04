# 1. What Is Padding?

## The two problems padding solves

### Problem 1: the output keeps shrinking

From the [convolution folder](../convolution%20operation/01-convolution-6x6-with-3x3.md),
a 6×6 image with a 3×3 filter gives a **4×4** output. Stack several conv layers
and the image melts away:

```
6×6  →  4×4  →  2×2  →  ...  → nothing left!
```

Each 3×3 convolution chops 2 off each dimension. In a deep network you'd run out
of pixels long before you run out of layers. We need a way to **preserve the
spatial size**.

### Problem 2: edge pixels are treated unfairly

Look at how many times each pixel gets "visited" by a 3×3 filter sliding over a
6×6 image:

```
A corner pixel:   used by  1  filter placement
An edge pixel:    used by ~3  placements
A center pixel:   used by  9  placements
```

```
   1 ··· corner pixel
   ╱
  ┌───────────────┐
  │ 1  2  3  3  2 1│   ← the numbers show roughly how often
  │ 2  4  6  6  4 2│     each pixel is covered by the filter
  │ 3  6  9  9  6 3│
  │ 3  6  9  9  6 3│
  │ 2  4  6  6  4 2│
  │ 1  2  3  3  2 1│
  └───────────────┘
```

The center is sampled **9×** more than the corners. So information at the edges
of the image gets **washed out** — bad if something important sits near a border.

## The fix: add a border

**Padding** adds extra rows and columns around the input before convolving.
Almost always these added cells are **zeros** ("zero padding"):

```
Original 6×6                 Padded with 1 ring of zeros → 8×8
┌───────────────┐            ┌─────────────────────┐
│ 10 10 10 0 0 0│            │ 0  0  0  0  0  0  0 0│
│ 10 10 10 0 0 0│            │ 0 10 10 10  0  0  0 0│
│ 10 10 10 0 0 0│            │ 0 10 10 10  0  0  0 0│
│ 10 10 10 0 0 0│    →       │ 0 10 10 10  0  0  0 0│
│ 10 10 10 0 0 0│            │ 0 10 10 10  0  0  0 0│
│ 10 10 10 0 0 0│            │ 0 10 10 10  0  0  0 0│
└───────────────┘            │ 0 10 10 10  0  0  0 0│
                             │ 0  0  0  0  0  0  0 0│
                             └─────────────────────┘
```

Now the 3×3 filter can center itself **on the original corner pixels**, because
they have neighbors (the zeros) to sit on. Two wins at once:

1. **Output stays larger.** An 8×8 padded input with a 3×3 filter → `(8−3)+1 = 6`
   → a **6×6** output. Same size as the original input!
2. **Edge pixels get sampled more often**, so border information is preserved.

## Why zeros?

Zeros are the neutral choice: they add no brightness and bias the result the
least. Other border strategies exist (you'll meet them occasionally):

| Padding type | Border filled with | When used |
|--------------|--------------------|-----------|
| **Zero padding** | `0` | The default, by far most common |
| **Reflect** | mirror of edge pixels | Avoids dark halos in image processing |
| **Replicate / edge** | copy of the nearest edge pixel | Smoother borders |
| **Constant** | a chosen constant value | Special cases |

For standard CNNs, assume **zero padding** unless told otherwise.

## The cost

Padding is nearly free, but note:

- It adds a few computations (the extra border cells).
- Zero-padded borders introduce a tiny artificial "edge" of their own (bright
  pixel next to a zero), but networks learn to ignore it.

The benefit — controllable output size and fair edge treatment — is well worth
it.

---

## Key takeaways

- Without padding, convolution **shrinks** the image and **under-samples edges**.
- **Padding** adds a border (usually **zeros**) around the input before
  convolving.
- It lets the output **keep its size** and gives **edge pixels fair coverage**.
- A 6×6 input padded by 1 → 8×8 → after a 3×3 filter → back to **6×6**.
- Next: the two named modes — [valid vs. same padding](02-valid-vs-same-padding.md).
