# 2. Valid vs. Same Padding

In practice you rarely set the padding number by hand. Instead you pick a
**mode**. The two you'll use 99% of the time are **valid** and **same**. These
are the exact keywords used in TensorFlow/Keras and the same concepts in
PyTorch.

## "Valid" padding = NO padding

"Valid" means **P = 0** — no border is added. The filter only sits on **valid**
(real) positions fully inside the image. This is exactly what we did in the
convolution folder.

```
Input 6×6, filter 3×3, VALID padding
   →  output = (6 − 3) + 1 = 4   →   4 × 4   (shrinks!)
```

- ✅ No made-up border values.
- ❌ Output is **smaller** than the input.
- ❌ Edge pixels are under-sampled.

Use it when you *want* the size to drop, or when borders don't matter.

## "Same" padding = keep the size the SAME

"Same" automatically adds **just enough** zero padding so the output has the
**same** height and width as the input (when stride = 1).

```
Input 6×6, filter 3×3, SAME padding
   pad P = 1 on each side  →  padded 8×8
   →  output = (8 − 3) + 1 = 6   →   6 × 6   (preserved!)
```

- ✅ Output size **matches** input — easy to stack many layers.
- ✅ Edge pixels get fairer coverage.
- ⚠️ Introduces a thin border of zeros (a minor, learnable artifact).

The padding needed for "same" with stride 1 is:

```
P = (F − 1) / 2
```

| Filter F | "Same" padding P |
|----------|------------------|
| 3×3 | 1 |
| 5×5 | 2 |
| 7×7 | 3 |

(This is why odd-sized filters like 3×3 and 5×5 are popular — `(F−1)/2` comes out
to a whole number, so padding is symmetric.)

## Side-by-side comparison

```
                    VALID                         SAME
              (no padding, P=0)            (pad to preserve size)

Input         6 × 6                        6 × 6
Padding       none                         1-pixel zero border → 8 × 8
Filter        3 × 3                         3 × 3
Output        4 × 4  ← shrinks             6 × 6  ← unchanged
Edges         under-sampled                fairly sampled
```

## How to choose

| Goal | Use |
|------|-----|
| Keep spatial size across many layers | **Same** |
| Build very deep networks without size collapse | **Same** |
| Deliberately downsample / reduce size | **Valid** (often with stride > 1) |
| Final layers where you want a small output | **Valid** |

A very common modern pattern: use **same** padding for the convolutions (so size
is controlled and predictable) and let **pooling** or **strided convolutions**
do the deliberate downsizing. That cleanly separates "detect features" (conv,
same) from "shrink" (pool/stride).

## A note on stride + same

When **stride > 1**, "same" no longer means identical size — it means the output
is `ceil(N / S)`. For example, 6×6 input, 3×3 filter, stride 2, "same" →
`ceil(6/2) = 3` → a 3×3 output. With stride 1 (the usual case) "same" truly
preserves the size. The next file gives the exact formulas.

---

## Key takeaways

- **Valid** padding = **no padding**; output **shrinks** to `(N − F) + 1`.
- **Same** padding = add **`(F−1)/2`** zeros per side (stride 1) so output
  **size is preserved**.
- **Odd** filter sizes (3, 5, 7) make "same" padding symmetric — a reason they're
  preferred.
- Common recipe: **same** padding for conv layers, **stride/pooling** for
  downsizing.
- Next: the [exact formulas and worked examples](03-padding-formulas-and-examples.md).
