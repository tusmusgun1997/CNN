# 1. What Is a CNN?

## The plain-English definition

A **Convolutional Neural Network (CNN)** is a type of deep learning model
designed to understand **visual data** — images and video. It learns to
recognize patterns directly from raw pixels, without a human telling it what
to look for.

Give a CNN a million photos labeled "cat" or "dog," and it will *teach itself*
the difference: the shape of ears, the texture of fur, the geometry of a snout.
Nobody writes rules like "if pointy ears then cat." The network discovers those
rules on its own.

## Why not a regular neural network?

A standard (fully connected) neural network treats every pixel as an
independent input. For a small 200×200 color image that's:

```
200 × 200 × 3 (color channels) = 120,000 inputs
```

If the first layer has just 1,000 neurons, that's **120 million connections**
in the first layer alone. This is:

- **Too expensive** to compute and store.
- **Wasteful**, because it ignores a basic truth about images: *nearby pixels
  are related*. A pixel in the top-left corner has nothing to do with one in
  the bottom-right when you're detecting an edge.
- **Fragile**, because if the cat moves 10 pixels to the right, every input
  changes and the network has to relearn everything.

CNNs fix all three problems with one clever idea: **convolution** (covered in
file 4).

## The core insight

CNNs are built on three powerful ideas:

1. **Local connections** — A neuron only looks at a small patch of the image
   at a time (like looking through a small window), not the whole thing.
2. **Shared weights** — The *same* pattern detector slides across the entire
   image. If a detector learns to find an edge, it can find that edge
   *anywhere*, not just where it first saw one.
3. **Hierarchy** — Early layers find simple things (edges, colors). Later
   layers combine those into complex things (shapes, objects). This mirrors how
   your own brain works (see file 2).

## What CNNs are used for

| Domain | Example task |
|--------|-------------|
| Image classification | "Is this a cat or a dog?" |
| Object detection | "Where are the cars in this street photo?" |
| Face recognition | Unlocking your phone with your face |
| Medical imaging | Spotting tumors in X-rays or MRIs |
| Self-driving cars | Reading road signs and detecting pedestrians |
| Style & generation | Filters, deepfakes, image upscaling |

> **Note:** While CNNs were born for images, the same ideas now apply to audio
> (spectrograms), text, and even game-playing. But vision is where they shine
> and where intuition is easiest to build.

## A mental model

Think of a CNN as an **assembly line of inspectors**:

```
Raw image  →  [Edge inspectors]  →  [Shape inspectors]  →  [Object inspectors]  →  Decision
   🖼️              ╱│╲                    ╱│╲                    ╱│╲              "Cat: 94%"
```

Each station looks at the output of the previous one and reports something
slightly more abstract. By the end of the line, the network has gone from
"here are some pixels" to "this is a cat."

---

## Key takeaways

- A CNN is a neural network specialized for images.
- It **learns patterns from data** instead of being hand-programmed.
- It is efficient because of **local connections** and **shared weights**.
- It builds understanding **hierarchically**: simple → complex.
- Next: see [how this mirrors the human brain](02-cnn-and-the-human-brain.md).
