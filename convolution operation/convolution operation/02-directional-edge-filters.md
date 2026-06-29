# 2. Directional Edge Filters (Left, Right, Top, Bottom)

The convolution mechanics never change — only the **numbers inside the filter**
change. Different numbers detect different patterns. Here we build four classic
edge detectors that respond to edges facing **left, right, top, and bottom**.

> Reminder from file 01: the filter computes a weighted sum of the patch under
> it. Positive weights "look for brightness here," negative weights "look for
> darkness here." An edge is just *brightness on one side, darkness on the
> other.*

## The four directional filters

All are 3×3. Each one fires when brightness sits on its "preferred" side.

### Vertical edges (left vs. right)

**Left-edge / vertical detector** — fires when the **left** side is brighter:

```
   1   0  -1
   1   0  -1
   1   0  -1
   (left −1·right)  →  positive when left is bright
```

**Right-edge detector** — the mirror image, fires when the **right** is brighter:

```
  -1   0   1
  -1   0   1
  -1   0   1
   (right − left)  →  positive when right is bright
```

### Horizontal edges (top vs. bottom)

**Top-edge detector** — fires when the **top** row is brighter:

```
   1   1   1
   0   0   0
  -1  -1  -1
   (top − bottom)  →  positive when top is bright
```

**Bottom-edge detector** — fires when the **bottom** row is brighter:

```
  -1  -1  -1
   0   0   0
   1   1   1
   (bottom − top)  →  positive when bottom is bright
```

Notice the symmetry: **right = −left**, and **bottom = −top**. Flipping a
filter's sign flips which direction of edge it responds to.

> These specific patterns are the **Prewitt** filters. The closely related
> **Sobel** filters just weight the middle row/column by 2 (e.g. `1 2 1` instead
> of `1 1 1`) for a smoother response. Same idea, slightly different numbers.

## Worked example 1 — vertical edge image

Reusing the vertical-edge image from file 01 (left bright `10`, right dark `0`):

```
Input                          Left-edge filter result (4×4)
10 10 10 │ 0 0 0               0  30  30  0
10 10 10 │ 0 0 0               0  30  30  0
10 10 10 │ 0 0 0      →        0  30  30  0
10 10 10 │ 0 0 0               0  30  30  0
10 10 10 │ 0 0 0
10 10 10 │ 0 0 0
```

- **Left-edge filter:** strong **+30** at the transition (left is brighter). ✅
- **Right-edge filter:** the exact negative, **−30** at the transition. The
  *magnitude* is the same; the sign tells you the edge's direction.
- **Top/bottom filters:** all **0** here — there's no horizontal edge in this
  image, so the horizontal detectors stay silent.

This is the key lesson: **each filter only lights up for the kind of edge it was
designed for.**

## Worked example 2 — horizontal edge image

Now an image with a horizontal edge (top bright, bottom dark):

```
Input  (6×6)
10 10 10 10 10 10
10 10 10 10 10 10
10 10 10 10 10 10
──────────────────  ← horizontal edge here (between r2 and r3)
 0  0  0  0  0  0
 0  0  0  0  0  0
 0  0  0  0  0  0
```

Apply the **top-edge filter** (`top − bottom`):

| Filter rows over | Top row | Bottom row | Result per col | Output row |
|------------------|---------|------------|----------------|------------|
| rows 0,1,2 | 10 | 10 | 0 | **0** |
| rows 1,2,3 | 10 | 0 | 10 | **30** |
| rows 2,3,4 | 10 | 0 | 10 | **30** |
| rows 3,4,5 | 0 | 0 | 0 | **0** |

```
Top-edge filter result (4×4)
   0   0   0   0
  30  30  30  30
  30  30  30  30
   0   0   0   0
```

The strong values land on the **horizontal transition**. Meanwhile the **left/
right (vertical) filters output all 0** on this image — no vertical edge to find.

## Putting it together: a filter bank

In a real CNN, a single convolution layer applies **many filters at once** — a
"filter bank." Each produces its own feature map:

```
                    ┌─► Left-edge   filter ─► feature map 1
   Input image  ────┤─► Right-edge  filter ─► feature map 2
   (6 × 6)          ├─► Top-edge    filter ─► feature map 3
                    └─► Bottom-edge filter ─► feature map 4
                                              (stacked into a 4×4×4 output)
```

So 4 filters on a 6×6 image give a **4 × 4 × 4** output volume — four 4×4 feature
maps stacked. The network now knows, at every location, whether there's a left,
right, top, or bottom edge.

> **Crucial point:** in `basics`, we said filters are *learned*. These hand-made
> edge detectors are just to build intuition. During training, a CNN **discovers
> filters like these on its own** — plus hundreds of others for textures,
> colors, and curves you'd never think to design by hand.

---

## Key takeaways

- The convolution math is identical; **only the filter values change** the
  pattern detected.
- **Left/right** filters detect **vertical** edges; **top/bottom** filters detect
  **horizontal** edges.
- Flipping a filter's **sign** flips the edge direction it responds to
  (right = −left, bottom = −top).
- These are the **Prewitt/Sobel** filters; real CNNs *learn* their own.
- Many filters together form a **filter bank**, producing a stack of feature
  maps.
- Next: the [formulas](03-formulas.md) governing output size and parameters.
