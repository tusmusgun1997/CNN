# Strides

**Stride** is how far the filter jumps between positions as it slides across the
image. It's the main knob (besides pooling) for **downsampling** — deliberately
shrinking the output. This sub-folder explains it with the same worked-by-hand
style as the rest of this course.

| File | What it teaches |
|------|-----------------|
| [01 — What are strides?](01-what-are-strides.md) | The concept, why we use them, the formula |
| [02 — Strides worked examples](02-strides-worked-examples.md) | Stride 1 vs 2 on our 6×6 image, by hand |

**The one-line summary:**

> Stride = the step size of the sliding filter. Stride 1 visits every position;
> stride 2 skips every other one, halving the output size.

See also the [implementation folder](../implementation/00-overview.md), which runs
both **padding** and **strides** in real Keras code.
