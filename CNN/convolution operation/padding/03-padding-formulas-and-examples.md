# 3. Padding Formulas & Examples

This file collects every padding formula and walks through worked numbers so you
can compute output sizes confidently.

## The master output-size formula (repeated)

```
            ⌊ N + 2P − F ⌋
  Output =  ⌊ ─────────── ⌋ + 1
            ⌊      S      ⌋
```

| Symbol | Meaning |
|--------|---------|
| N | input size (per dimension) |
| F | filter size |
| P | padding **per side** |
| S | stride |

Note: `2P` because padding is added to **both** sides (left+right, or top+bottom).

## Formula 1 — output size given padding

Plug in your P directly.

**Example A: 6×6, 3×3, P=0, S=1 (valid)**

```
(6 + 0 − 3)/1 + 1 = 4      →  4 × 4
```

**Example B: 6×6, 3×3, P=1, S=1 (same)**

```
(6 + 2 − 3)/1 + 1 = 6      →  6 × 6  ✅ size preserved
```

**Example C: 6×6, 5×5, P=2, S=1 (same for a 5×5 filter)**

```
(6 + 4 − 5)/1 + 1 = 6      →  6 × 6  ✅
```

## Formula 2 — padding needed for "same" (stride 1)

To make output = input when S = 1, solve the formula for P:

```
P = (F − 1) / 2
```

| Filter F | P for "same" | Padded size of a 6×6 | Output |
|----------|--------------|----------------------|--------|
| 3 | 1 | 8 × 8 | 6 × 6 |
| 5 | 2 | 10 × 10 | 6 × 6 |
| 7 | 3 | 12 × 12 | 6 × 6 |

## Formula 3 — "same" output with stride > 1

When stride > 1, "same" means:

```
Output = ⌈ N / S ⌉          (⌈ ⌉ = ceiling, round up)
```

And the total padding required is:

```
P_total = max( (Output − 1)·S + F − N ,  0 )
```

split as evenly as possible between the two sides (if odd, frameworks usually
add the extra pixel on the right/bottom).

**Example D: 6×6, 3×3, S=2, "same"**

```
Output  = ⌈6 / 2⌉ = 3
P_total = (3−1)·2 + 3 − 6 = 4 + 3 − 6 = 1   → 1 pixel total
                                              (0 left, 1 right)
Check:  (6 + 1 − 3)/2 + 1 = 4/2 + 1 = 3      →  3 × 3  ✅
```

## A handy reference table

For a **6×6** input (one of the most common teaching sizes):

| Mode | F | P | S | Output | Size effect |
|------|---|---|---|--------|-------------|
| valid | 3 | 0 | 1 | 4 × 4 | −2 |
| same | 3 | 1 | 1 | 6 × 6 | unchanged |
| valid | 5 | 0 | 1 | 2 × 2 | −4 |
| same | 5 | 2 | 1 | 6 × 6 | unchanged |
| valid | 3 | 0 | 2 | 2 × 2 | halved-ish |
| same | 3 | 1 | 2 | 3 × 3 | ⌈6/2⌉ |

## Worked example: "same" padding on our edge image

Take the 6×6 vertical-edge image and apply the vertical-edge filter with **same
padding (P=1)**. After padding, the input is 8×8 with a zero border:

```
Padded input (8×8)                          Output (6×6, same size as original)
0  0  0  0  0  0  0  0                       -10  0  30  30  0  0
0 10 10 10  0  0  0  0                       -10  0  30  30  0  0
0 10 10 10  0  0  0  0          →            -10  0  30  30  0  0
0 10 10 10  0  0  0  0                        ...
0 10 10 10  0  0  0  0
... (etc)
```

Two things to notice versus the valid result (`0 30 30 0`, a 4×4):

1. The output is **6×6**, matching the input — that's the point of "same."
2. New nonzero values appear at the **left border** (e.g. `−10`). These come from
   the filter straddling the zero padding next to the bright pixels — a small
   **border artifact**. It's harmless; the network learns to discount it.

## Quick mental checklist

When you see a conv layer, ask:

```
1. What is N (input size)?
2. What is F (filter size)?
3. valid or same?  →  sets P
4. What is the stride S?
5. Plug into  ⌊(N + 2P − F)/S⌋ + 1
```

Do this once per dimension and you'll always know the output shape.

---

## Key takeaways

- Master formula: `Output = ⌊(N + 2P − F)/S⌋ + 1` (apply per dimension).
- "Same" padding (stride 1): **`P = (F − 1)/2`** preserves size.
- "Same" with stride > 1: output = **`⌈N/S⌉`**, with
  `P_total = max((out−1)·S + F − N, 0)`.
- Zero-padded borders create a small, **harmless artifact** the network learns to
  ignore.
- Run the 5-step checklist on any conv layer to predict its output shape.
- ← Back to the [convolution operation index](../00-overview.md).
