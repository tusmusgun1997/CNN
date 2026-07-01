# 2. Strides — Worked Examples

Let's watch stride change the output using the same **6×6 vertical-edge image**
and **vertical-edge filter** from the convolution folder.

```
Input (6×6)                     Filter (3×3)
10 10 10  0  0  0                 1  0  -1
10 10 10  0  0  0                 1  0  -1
10 10 10  0  0  0                 1  0  -1
10 10 10  0  0  0
10 10 10  0  0  0
10 10 10  0  0  0
```

## Stride 1 (the baseline)

The filter's top-left corner visits columns **0, 1, 2, 3** and rows **0, 1, 2,
3** → 4 × 4 = 16 positions → a **4×4** output (this is the file-01 result):

```
Output, stride 1  (4 × 4)
 0  30  30   0
 0  30  30   0
 0  30  30   0
 0  30  30   0
```

## Stride 2 (skip every other position)

Now the corner only lands on columns **0, 2** and rows **0, 2** (starting at 4
would overflow: 4 + 3 = 7 > 6). That's 2 × 2 = 4 positions → a **2×2** output.

```
Output size = ⌊(6 − 3)/2⌋ + 1 = ⌊1.5⌋ + 1 = 1 + 1 = 2   →   2 × 2
```

### Computing each of the 4 cells

Column start 0 → window cols 0,1,2 (left=10, right=10) → **0**
Column start 2 → window cols 2,3,4 (left=10, right=0) → 10 per row → **30**

Because the image is vertically uniform, both row positions (start 0 and start
2) give the same values:

```
Output, stride 2  (2 × 2)
 0  30
 0  30
```

### Which positions were visited vs. skipped

```
Corner-start positions the filter lands on (■ = used, · = skipped):

        col0 col1 col2 col3 col4 col5
 row0    ■    ·    ■    ·    ·    ·
 row1    ·    ·    ·    ·    ·    ·
 row2    ■    ·    ■    ·    ·    ·
 row3    ·    ·    ·    ·    ·    ·
 row4    ·    ·    ·    ·    ·    ·
 row5    ·    ·    ·    ·    ·    ·
```

Only 4 placements (vs. 16 at stride 1). Notice the corner starting at column 4
is skipped, and the floor in the formula quietly dropped it — some rightmost
data is not covered.

## Side-by-side

```
              STRIDE 1                 STRIDE 2
Positions     16 (4×4 grid)            4 (2×2 grid)
Output        4 × 4                    2 × 2
Detail        full                     downsampled ~½
Compute       more                     ~4× less
```

## Adding padding to the mix

Combine "same" padding (P=1) with stride 2 on the 6×6 image. From the padding
folder, "same" with stride > 1 gives `⌈N/S⌉`:

```
Output = ⌈6 / 2⌉ = 3    →    3 × 3
```

Check with the formula (padded input is 8×8 → but total pad for "same" is
computed to land on 3): `(6 + 1 − 3)/2 + 1 = 4/2 + 1 = 3`. ✅

So on a single 6×6 image you can get:

| Padding | Stride | Output |
|---------|--------|--------|
| valid | 1 | 4 × 4 |
| same | 1 | 6 × 6 |
| valid | 2 | 2 × 2 |
| same | 2 | 3 × 3 |

These are exactly the four cases the [Keras implementation](../implementation/00-overview.md)
runs and prints, so you can verify these hand calculations against real output.

---

## Key takeaways

- **Stride 1** on 6×6 with a 3×3 filter → **4×4**; **stride 2** → **2×2**.
- Stride 2 lands on **half** the positions and produces a **downsampled** map.
- The **floor** in the formula can **drop** edge positions that would overflow.
- Padding + stride combine predictably: `valid` shrinks more, `same` gives
  `⌈N/S⌉`.
- Verify all of this yourself in the [implementation](../implementation/00-overview.md).
