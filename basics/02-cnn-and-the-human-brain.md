# 2. CNNs and the Human Brain

CNNs are not *just loosely* inspired by biology — their core design comes
directly from discoveries about how the brain processes sight. This file
explains that connection, because understanding the brain makes the math
intuitive later.

## The discovery that started it all

In the late 1950s and 1960s, neuroscientists **David Hubel and Torsten Wiesel**
recorded electrical signals from neurons in a cat's **visual cortex** (the part
of the brain that processes vision). They won a Nobel Prize for what they found:

- Some neurons fired **only** when a line at a *specific angle* appeared in a
  *specific spot* of the cat's vision. These are called **simple cells**.
- Other neurons fired for that same line *anywhere* in a region — they didn't
  care about exact position. These are **complex cells**.
- Vision is processed in a **hierarchy**: simple features first, combined into
  complex ones deeper in.

Every one of these findings maps directly onto a part of a CNN. 👇

## The mapping: brain ↔ CNN

| Human visual system | CNN equivalent | What it does |
|---------------------|----------------|--------------|
| Eye / retina | Input image (pixels) | Captures raw light/color |
| **Simple cells** (detect edges at a spot) | **Convolution filters** | Detect local patterns like edges |
| **Complex cells** (position-tolerant) | **Pooling layers** | Recognize features regardless of exact location |
| Hierarchy V1 → V2 → V4 → IT cortex | **Stacked layers** | Build complex features from simple ones |
| A neuron "firing" past a threshold | **Activation function (ReLU)** | Decide whether a signal is strong enough to pass on |
| Recognition / naming the object | **Fully connected layer + output** | Make the final decision |

## The visual hierarchy in your brain

When you look at your friend's face, your brain does *not* see "a face" all at
once. It processes it in stages, each region adding abstraction:

```
Light → Retina → V1 (edges, orientations)
                  → V2 (corners, textures)
                     → V4 (shapes, color patches)
                        → IT cortex (whole objects, faces)
                           → "That's Maya!"
```

A CNN does the **exact same thing**, layer by layer:

```
Pixels → Conv layer 1 (edges)
          → Conv layer 2 (corners, curves)
             → Conv layer 3 (eyes, noses, textures)
                → Deep layers (whole faces)
                   → Output ("That's Maya!")
```

This is the single most important analogy in this whole course. **Simple
features get combined into complex ones, level by level.**

## "Receptive fields" — a shared concept

In neuroscience, a neuron's **receptive field** is the small region of your
vision that it responds to. A neuron near the start of the visual system has a
*tiny* receptive field (it sees a dot); a neuron deeper in has a *huge* one (it
responds to a whole face).

CNNs use the exact same term and the exact same idea:

- Early-layer neurons have **small receptive fields** → they see tiny patches.
- Deep-layer neurons have **large receptive fields** → they "see" most of the
  image and can respond to whole objects.

## Where the analogy ends (be honest!)

The brain inspired CNNs, but they are **not** the same:

- **Learning is different.** Brains learn from few examples; CNNs usually need
  thousands or millions. The brain doesn't use backpropagation (file 9) the way
  CNNs do.
- **Brains are recurrent and bidirectional.** Real vision has feedback loops
  (your expectations change what you see). A basic CNN flows one direction.
- **Energy & robustness.** Your brain runs on ~20 watts and handles novelty,
  occlusion, and weird lighting far better than most CNNs.
- **No real "understanding."** A CNN detects statistical patterns; it does not
  *know* what a cat is the way you do.

> **Takeaway phrasing:** CNNs are *bio-inspired*, not *bio-identical*. The
> inspiration is real and deep, but a CNN is a simplified, mathematical
> cartoon of the visual cortex — not a brain.

---

## Key takeaways

- Hubel & Wiesel found **simple cells** (detect local edges) and **complex
  cells** (position-tolerant) in the visual cortex.
- These map onto **convolution filters** and **pooling layers**.
- Both brain and CNN build understanding in a **hierarchy**: edges → shapes →
  objects.
- The shared concept of a **receptive field** grows from tiny to large as you
  go deeper.
- The analogy is powerful but imperfect — CNNs are a simplified model.
- Next: [how a CNN actually represents an image](03-how-cnns-see-images.md).
