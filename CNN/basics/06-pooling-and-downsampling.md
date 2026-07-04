# 6. Pooling & Downsampling

After convolution and activation, CNNs usually **pool**. Pooling shrinks the
feature maps while keeping the important information. It's the network's way of
zooming out.

## Why shrink at all?

Feature maps can be large, and most of the fine detail is redundant. Pooling
gives you three big wins:

1. **Less computation.** Smaller maps = faster, cheaper, less memory.
2. **Position tolerance.** It no longer matters if a feature is at pixel (10,10)
   or (11,11) — pooling smooths over tiny shifts. The network cares *that* a
   feature exists, not its exact pixel.
3. **Bigger receptive fields.** After pooling, each later neuron effectively
   "sees" a larger part of the original image, helping it detect bigger
   structures.

## Max pooling (the common one)

**Max pooling** slides a small window (usually 2×2) over the feature map and
keeps only the **largest** value in each window. "Was this feature strongly
present anywhere in this patch? Keep the strongest signal."

```
Input feature map (4×4)        Max pool 2×2, stride 2

┌─────┬─────┐
│ 1 3 │ 2 1 │                    ┌─────┬─────┐
│ 4 2 │ 0 5 │       →            │  4  │  5  │
├─────┼─────┤                    ├─────┼─────┤
│ 7 6 │ 1 2 │                    │  8  │  4  │
│ 8 3 │ 4 0 │                    └─────┴─────┘
└─────┴─────┘
```

How each output was computed:

- Top-left window `[1,3,4,2]` → max = **4**
- Top-right window `[2,1,0,5]` → max = **5**
- Bottom-left window `[7,6,8,3]` → max = **8**
- Bottom-right window `[1,2,4,0]` → max = **4**

The 4×4 map became a 2×2 map — **75% fewer numbers**, but the strongest
features survived.

## Average pooling

**Average pooling** keeps the *average* of each window instead of the max. It's
smoother and less aggressive. Max pooling is more common in classic CNNs
because "did the feature appear strongly?" is usually what matters.

```
Window [1,3,4,2]  →  max = 4   |   average = 2.5
```

## Global average pooling (modern favorite)

Instead of pooling a little, **global average pooling** collapses an *entire*
feature map down to a **single number** (its average). A `7 × 7 × 512` tensor
becomes just `1 × 1 × 512` — one value per feature. Modern architectures often
use this right before the final classification step instead of large fully
connected layers, because it has **no parameters** and resists overfitting.

## Pooling vs. strided convolution

A modern alternative to pooling is using a **stride** in the convolution itself
(file 4) to skip positions and shrink the output. Many newer architectures
prefer this because it *learns* how to downsample rather than using a fixed
rule. Both approaches achieve the same goal: smaller spatial size.

## The brain parallel

Pooling is the engineering version of the brain's **complex cells** (file 2),
which respond to a feature *anywhere in a region*, ignoring its exact location.
By taking the max over a neighborhood, the CNN gains the same useful blindness
to small shifts — this is a key source of **translation invariance**, the
ability to recognize an object no matter where it sits in the frame.

## A note: pooling has no learnable weights

Unlike convolution, a pooling layer has **nothing to learn** — it's a fixed
rule (take the max, or the average). That makes it simple and fast. The
*intelligence* lives in the convolution filters; pooling just tidies up.

---

## Key takeaways

- **Pooling** shrinks feature maps, cutting compute and adding position
  tolerance.
- **Max pooling** keeps the strongest value in each window (most common).
- **Average / global average pooling** keep means; global pooling is a popular,
  parameter-free modern choice.
- Pooling has **no learnable weights** — it's a fixed operation.
- It mirrors the brain's position-tolerant **complex cells**.
- Next: turning features into a decision with [fully connected layers](07-fully-connected-and-classification.md).
