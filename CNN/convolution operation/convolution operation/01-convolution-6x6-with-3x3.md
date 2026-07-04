# 1. Convolution: a 6×6 Matrix with a 3×3 Filter

This is the canonical teaching example. Master this one page and you understand
convolution.

## The setup

We have:

- An **input** image: a **6×6** matrix (36 numbers).
- A **filter (kernel)**: a **3×3** matrix (9 numbers).
- **Stride = 1** (move one step at a time), **no padding**.

The result will be a **4×4** output. (Why 4×4? See the formula below and in
[03-formulas.md](03-formulas.md).)

### Our input: an image with a vertical edge

The left half is bright (`10`), the right half is dark (`0`). There's a clear
**vertical edge** down the middle, between column 2 and column 3.

```
Input  (6 × 6)
        c0  c1  c2  c3  c4  c5
      ┌────────────────────────┐
 r0   │ 10  10  10   0   0   0 │
 r1   │ 10  10  10   0   0   0 │
 r2   │ 10  10  10   0   0   0 │
 r3   │ 10  10  10   0   0   0 │
 r4   │ 10  10  10   0   0   0 │
 r5   │ 10  10  10   0   0   0 │
      └────────────────────────┘
```

### Our filter: a vertical-edge detector

```
Filter  (3 × 3)
   ┌──────────┐
   │  1  0  -1│
   │  1  0  -1│
   │  1  0  -1│
   └──────────┘
```

Intuition: this filter computes **(left column) − (right column)** of whatever
3×3 patch it sits on. Where the left is brighter than the right, the output is
large and positive — that's exactly what a "vertical edge going bright→dark"
looks like.

## Why the output is 4×4

The 3×3 filter must sit **fully inside** the 6×6 image. Its top-left corner can
start at columns 0, 1, 2, 3 (4 positions) and rows 0, 1, 2, 3 (4 positions). So
there are 4 × 4 = **16 valid placements** → a 4×4 output.

```
Output size = (6 − 3) + 1 = 4    →    4 × 4
```

## Step-by-step: computing ONE output cell

Let's compute the output at **position (0, 1)** — filter's top-left corner at
row 0, column 1.

**1. Grab the patch** the filter covers (rows 0–2, columns 1–3):

```
Patch              Filter
┌──────────┐      ┌──────────┐
│ 10 10  0 │      │  1  0  -1│
│ 10 10  0 │  ⊙   │  1  0  -1│
│ 10 10  0 │      │  1  0  -1│
└──────────┘      └──────────┘
```

**2. Multiply each overlapping pair** (element-wise, the ⊙ symbol):

```
(10×1) + (10×0) + (0×-1)  = 10
(10×1) + (10×0) + (0×-1)  = 10
(10×1) + (10×0) + (0×-1)  = 10
```

**3. Add them all up:**

```
10 + 10 + 10 = 30
```

So `output[0][1] = 30`. A big positive number — the filter found a vertical edge
in this patch. ✅

## Computing the whole 4×4 output

Repeat that process for all 16 positions. Because this image is identical in
every row, every row of the output comes out the same:

| Filter columns over | Left col | Right col | Result per row | Row total |
|---------------------|----------|-----------|----------------|-----------|
| cols 0,1,2 | 10 | 10 | 0 | **0** |
| cols 1,2,3 | 10 | 0 | 10 | **30** |
| cols 2,3,4 | 10 | 0 | 10 | **30** |
| cols 3,4,5 | 0 | 0 | 0 | **0** |

Final output:

```
Output  (4 × 4)
   ┌─────────────────┐
   │  0   30  30   0 │
   │  0   30  30   0 │
   │  0   30  30   0 │
   │  0   30  30   0 │
   └─────────────────┘
```

## How to read this result

```
Input (edge between c2|c3)        Output (edge highlighted)
10 10 10 │ 0 0 0                   0  30  30  0
                                       ▲▲▲▲▲
                          the high values sit right where the
                          bright→dark transition happens
```

The flat regions (all bright or all dark) produced **0** — no edge there. The
**transition zone** produced **30** — "edge detected here!" That single feature
map is the network saying *where* in the image this particular pattern lives.

## The sliding-window animation (in text)

```
Step 1: cols 0-2     Step 2: cols 1-3     Step 3: cols 2-4    Step 4: cols 3-5
┌──────┐             ┌──────┐              ┌──────┐            ┌──────┐
│▓▓▓│..│ → 0         │.▓▓▓│.│ → 30         │..▓▓▓│ → 30        │...▓▓▓│ → 0
└──────┘             └──────┘              └──────┘            └──────┘
```

…then drop down a row and repeat, until all 16 cells are filled.

---

## Key takeaways

- Convolution = **overlay filter → multiply → sum → one output number → slide**.
- A **6×6** input with a **3×3** filter (stride 1, no padding) → a **4×4**
  output, because `(6 − 3) + 1 = 4`.
- Big output values mark **where the filter's pattern appears**.
- Next: use [different filters](02-directional-edge-filters.md) to detect left,
  right, top, and bottom edges.
