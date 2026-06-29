# The Convolution Operation

This sub-folder is the hands-on core. We take a **6×6 input matrix**, slide a
**3×3 filter** over it, and compute the **4×4 output** — by hand, cell by cell.
Then we swap in different filters and extend everything to color (RGB).

| File | What it teaches |
|------|-----------------|
| [01 — Convolution: 6×6 with 3×3](01-convolution-6x6-with-3x3.md) | The mechanics, one full worked example |
| [02 — Directional edge filters](02-directional-edge-filters.md) | Left, right, top, bottom edge detectors |
| [03 — Formulas](03-formulas.md) | Output size, stride, parameter counts |
| [04 — RGB convolution](04-rgb-convolution.md) | How depth/channels change the operation |

**The one rule of convolution:**

> Place the filter on a patch → multiply each overlapping pair → add them all up
> → that single number is one output pixel. Slide and repeat.
