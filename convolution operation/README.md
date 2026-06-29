# Convolution Operation — Deep Dive

This folder zooms into the single most important operation in a CNN: the
**convolution** itself. If the [`basics`](../basics/README.md) folder gave you
the big picture, this folder works out the actual arithmetic with concrete
numbers.

## Structure

```
convolution operation/
├── convolution operation/      ← the operation, step by step
│   ├── 01-convolution-6x6-with-3x3.md   (a full worked example)
│   ├── 02-directional-edge-filters.md   (left / right / top / bottom filters)
│   ├── 03-formulas.md                   (output size, parameters, the math)
│   └── 04-rgb-convolution.md            (how it changes for color images)
│
└── padding/                    ← controlling output size & edges
    ├── 01-what-is-padding.md
    ├── 02-valid-vs-same-padding.md
    └── 03-padding-formulas-and-examples.md
```

## Suggested reading order

1. **convolution operation/01** — see a 6×6 image convolved with a 3×3 filter
   into a 4×4 output, every cell computed by hand.
2. **convolution operation/02** — the same machinery with different filters to
   detect edges on the left, right, top, and bottom.
3. **convolution operation/03** — the formulas behind output size and parameter
   counts.
4. **convolution operation/04** — what changes when the image has 3 color
   channels (RGB).
5. **padding/** — why and how we add borders to control the output.

> Prerequisite: skim [basics/04-convolution-and-filters.md](../basics/04-convolution-and-filters.md)
> first if the words "filter," "stride," and "feature map" are new to you.
