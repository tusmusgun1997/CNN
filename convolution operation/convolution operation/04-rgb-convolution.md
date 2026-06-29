# 4. Convolution on RGB (Color) Images

Everything so far used a single grayscale grid. Real photos have **three color
channels** (Red, Green, Blue). This file shows the one key change: the filter
gains **depth**.

## The change in one sentence

> For a color image, a "3×3 filter" is really a **3×3×3** filter — it has a
> separate 3×3 grid for Red, Green, and Blue — and it still produces **one**
> 2D output by summing across all three.

## Picture the data

A grayscale 6×6 image is a flat sheet. An RGB 6×6 image is **three stacked
sheets** — a 6×6×3 volume:

```
        Grayscale                       RGB (6 × 6 × 3)
      ┌──────────┐                  ┌──────────┐
      │  6 × 6   │                  │ R: 6×6   │┐
      └──────────┘                  └──────────┘│┐
                                     └──────────┘│   (3 channels deep)
                                      └──────────┘
```

## The filter must match the input depth

This is the rule to burn into memory:

> **A filter's depth always equals the input's depth.**

So over an RGB image (depth 3), a single 3×3 filter is actually **3×3×3 = 27
weights** (plus 1 bias). It has one 3×3 slice per channel:

```
Filter for an RGB input  (3 × 3 × 3)
   W_R (for Red)     W_G (for Green)    W_B (for Blue)
   ┌──────────┐      ┌──────────┐       ┌──────────┐
   │ . . .    │      │ . . .    │       │ . . .    │
   │ . . .    │      │ . . .    │       │ . . .    │
   │ . . .    │      │ . . .    │       │ . . .    │
   └──────────┘      └──────────┘       └──────────┘
```

## How one output number is computed

You convolve **each channel with its matching filter slice**, then **add the
three results together** (plus the bias) into a **single** number:

```
output = (R_patch ⊙ W_R)  +  (G_patch ⊙ W_G)  +  (B_patch ⊙ W_B)  +  bias
         └─ 9 products ─┘     └─ 9 products ─┘     └─ 9 products ─┘
                          all 27 products summed → ONE value
```

Note: the three channels **collapse into one**. The output of a single filter on
a color image is still a **2D feature map**, not three. Color goes *in*; a single
combined response comes *out*.

## A clean numeric example

Take a 3×3 patch where each channel is uniform, and simple all-ones filters:

```
R patch (all 2)   G patch (all 1)   B patch (all 3)
 2 2 2             1 1 1             3 3 3
 2 2 2             1 1 1             3 3 3
 2 2 2             1 1 1             3 3 3

W_R (all 1)       W_G (all 1)       W_B (all 1)        bias = 0
```

Per-channel contributions:

```
R: 2 × 9 cells = 18
G: 1 × 9 cells =  9
B: 3 × 9 cells = 27
                 ───
output = 18 + 9 + 27 + 0 = 54
```

One number, **54**, summarizing all three channels at that location. Slide the
filter to fill the whole feature map exactly as before.

## Output shape for RGB

The spatial formula is unchanged — only depth bookkeeping differs:

```
Input:  6 × 6 × 3
Filter: 3 × 3 × 3   (depth 3 to match input)
        ↓  stride 1, no padding
Output: 4 × 4 × 1   (per filter)
```

With **K** filters you get **K** feature maps stacked:

```
Input  6 × 6 × 3   ──[ K filters, each 3×3×3 ]──►   Output  4 × 4 × K
```

For example, 10 filters on an RGB image → `4 × 4 × 10`.

## Parameters for an RGB conv layer

Using the formula from [03-formulas.md](03-formulas.md), `K × (F·F·C + 1)`:

```
One 3×3 filter on RGB:   1 × (3·3·3 + 1) = 28 parameters
32 such filters:        32 × (27 + 1)    = 896 parameters
```

The `C` (channel) factor is why a color filter has 3× the weights of a grayscale
one of the same size.

## Going deeper: channels keep growing

After the first layer, the "channels" are no longer R/G/B — they're the **feature
maps** from the previous layer. A second conv layer with input depth 32 uses
filters of depth 32:

```
Layer 1 input:   6 × 6 ×  3   →  output 4 × 4 × 32   (filters are 3×3×3)
Layer 2 input:   4 × 4 × 32   →  output 2 × 2 × 64   (filters are 3×3×32)
```

The rule **"filter depth = input depth"** holds at every layer. This is why CNNs
get *deeper* (more channels) as they go — each layer's filters span all the
feature maps beneath them.

## The brain parallel

Recall from [basics/03](../../basics/03-how-cnns-see-images.md) that your eye has
three cone types (roughly red/green/blue-sensitive). A neuron deeper in the
visual system **combines** signals from all three cone types into a single
response — just like an RGB filter sums R, G, and B into one number. Color
information is *integrated*, not kept separate.

---

## Key takeaways

- An RGB image is a **6 × 6 × 3** volume (three stacked channels).
- A filter's **depth equals the input's depth**, so a "3×3" filter on RGB is
  really **3×3×3 = 27 weights** (+1 bias).
- Convolve each channel with its slice, then **sum across channels** → **one**
  output value. Color collapses into a single feature map.
- Output shape: `6×6×3 → 4×4×K` for K filters; **output depth = number of
  filters**.
- The rule **"filter depth = input depth"** holds at every layer, which is why
  channel counts grow with depth.
- Next folder: [padding](../padding/README.md) — controlling output size and
  protecting the edges.
