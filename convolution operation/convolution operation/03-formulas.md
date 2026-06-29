# 3. Formulas

Now we formalize what we did by hand. These formulas let you predict the output
size and parameter count of any convolution without drawing a single grid.

## Notation

| Symbol | Meaning |
|--------|---------|
| **N** | Input size (width or height), e.g. 6 |
| **F** | Filter (kernel) size, e.g. 3 |
| **P** | Padding (border of zeros added on each side), e.g. 0 |
| **S** | Stride (step size of the slide), e.g. 1 |
| **K** | Number of filters in the layer |
| **C** | Number of input channels (1 for grayscale, 3 for RGB) |

## 1. Output size formula (the important one)

For one spatial dimension:

```
            ⌊ N + 2P − F ⌋
  Output =  ⌊ ─────────── ⌋ + 1
            ⌊      S      ⌋
```

(⌊ ⌋ means "floor" — round down.)

Apply it to our running example — **N=6, F=3, P=0, S=1**:

```
Output = (6 + 0 − 3) / 1 + 1 = 3 + 1 = 4      →   4 × 4  ✅
```

This matches the 4×4 we computed by hand in file 01.

### Try a few more

| N | F | P | S | Output | Note |
|---|---|---|---|--------|------|
| 6 | 3 | 0 | 1 | **4** | our example |
| 6 | 3 | 1 | 1 | **6** | "same" padding keeps size (see padding folder) |
| 6 | 3 | 0 | 2 | **2** | stride 2 halves it: (6−3)/2+1 = 2.5 → floor 2 |
| 7 | 3 | 0 | 2 | **3** | (7−3)/2+1 = 3 |
| 5 | 5 | 0 | 1 | **1** | a 5×5 filter on a 5×5 image → single number |

> For a 2D image you apply the formula **twice** — once for height, once for
> width. They're independent, so a 6×8 image with a 3×3 filter gives `4 × 6`.

## 2. Stride: how the step size shrinks output

**Stride** is how many pixels the filter jumps each move.

```
Stride 1:  ▓░░░░  →  ░▓░░░  →  ░░▓░░   (every position)
Stride 2:  ▓░░░░  →  ░░▓░░  →  ░░░░▓   (skip every other)
```

Bigger stride → fewer placements → **smaller output** and **less computation**,
at the cost of spatial detail. Stride is one way CNNs downsample (an alternative
to pooling).

## 3. Number of parameters in a conv layer

Each filter has a weight per cell **plus one bias**. For `K` filters over `C`
input channels with an `F×F` kernel:

```
Parameters = K × (F × F × C + 1)
                  └─────────┘  └┘
                   weights    bias
```

Examples:

| Layer | F | C | K | Parameters |
|-------|---|---|---|------------|
| Grayscale, 1 filter | 3 | 1 | 1 | 3·3·1 + 1 = **10** |
| Grayscale, 32 filters | 3 | 1 | 32 | 32·(9+1) = **320** |
| RGB, 32 filters | 3 | 3 | 32 | 32·(27+1) = **896** |
| RGB, 64 filters, 5×5 | 5 | 3 | 64 | 64·(75+1) = **4,864** |

Compare this to a fully connected layer on the same 6×6 image (36 inputs → even
a modest 100 neurons = 3,700 weights). Convolution's **weight sharing** is what
keeps these counts tiny — the same 9 numbers are reused at every position.

## 4. Output volume shape (the full picture)

A conv layer transforms an input **volume** into an output **volume**:

```
Input:   N × N × C
                 │  apply K filters, each F×F×C
                 ▼
Output:  W × W × K        where  W = ⌊(N + 2P − F)/S⌋ + 1
```

Key facts to memorize:

- **Output depth = number of filters (K)** — *not* the input channels.
- **Each filter spans the full input depth C** (more on this in the RGB file).
- **Output width/height** come from the size formula above.

For our 6×6 example with one filter: `6×6×1 → 4×4×1`. With 4 directional filters:
`6×6×1 → 4×4×4`.

## 5. The convolution sum, written formally

For completeness, the value at output position `(i, j)` for filter `k` is:

```
              C-1  F-1  F-1
  O[i,j,k] =  ∑    ∑    ∑   Input[i·S + m,  j·S + n,  c] × W[m, n, c, k]   +  b[k]
              c=0  m=0  n=0
```

In words: *for this output spot, run over every channel `c` and every cell
`(m,n)` of the filter, multiply input by weight, add them all up, then add the
filter's bias `b[k]`.* That's exactly the by-hand procedure from file 01,
written compactly.

> **Convolution vs. cross-correlation:** strictly, mathematical convolution
> flips the filter before sliding. Deep learning libraries skip the flip (it's
> learned anyway) and use **cross-correlation**, but everyone calls it
> "convolution." The formula above is the cross-correlation used in practice.

---

## Key takeaways

- **Output size** = `⌊(N + 2P − F)/S⌋ + 1`, applied per dimension.
- **Stride** shrinks the output; **padding** can grow it back (next folder).
- **Parameters** = `K × (F·F·C + 1)` — small thanks to weight sharing.
- **Output depth = K** (number of filters); each filter spans all **C** input
  channels.
- Next: what "spanning all channels" means for [RGB images](04-rgb-convolution.md).
