# 8. The CNN Pipeline (High Level)

Now we zoom out and assemble every piece from files 3–7 into one complete
picture. This is the "what high-level things happen in a CNN" overview.

## The full journey of an image

```
   🖼️  INPUT IMAGE  (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────────────────────┐
│  CONVOLUTION  → detect local patterns        │  "edges, colors"
│  ACTIVATION (ReLU) → keep strong signals     │
│  POOLING → shrink, gain position tolerance   │
└─────────────────────────────────────────────┘
        │   (repeat this block several times,
        ▼    features get more complex each time)
┌─────────────────────────────────────────────┐
│  CONVOLUTION + ReLU + POOLING                │  "corners, textures"
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  CONVOLUTION + ReLU + POOLING                │  "eyes, wheels, fur"
└─────────────────────────────────────────────┘
        │
        ▼
   FLATTEN / GLOBAL POOLING  → 1D feature vector
        │
        ▼
┌─────────────────────────────────────────────┐
│  FULLY CONNECTED LAYERS  → combine features  │  "this looks like a cat"
└─────────────────────────────────────────────┘
        │
        ▼
   SOFTMAX  → probabilities
        │
        ▼
   🎯  OUTPUT:  cat 0.94 | dog 0.05 | bird 0.01
```

## The high-level operations, summarized

These are the "high-level things we do" inside any CNN:

| Operation | Purpose | Brain analogy |
|-----------|---------|---------------|
| **Convolution** | Detect local patterns with sliding filters | Simple cells (edge detectors) |
| **Activation (ReLU)** | Add non-linearity; pass strong signals | Neuron firing past a threshold |
| **Pooling** | Downsample; tolerate position shifts | Complex cells (position-tolerant) |
| **Stacking layers** | Build complex features from simple ones | Visual hierarchy V1→V2→V4→IT |
| **Flatten / global pool** | Convert feature maps to a vector | — |
| **Fully connected** | Combine features into class evidence | Higher association cortex |
| **Softmax** | Produce confidence probabilities | Recognition / naming |

## The feature hierarchy (the big idea, again)

The single most important pattern to remember:

```
Layer depth   What it detects        Receptive field
───────────────────────────────────────────────────
Early    →    edges, colors          tiny  (a few pixels)
Middle   →    corners, textures      medium
Deep     →    object parts (eyes)    large
Deepest  →    whole objects (faces)  most of the image
```

Simple features combine into complex ones, layer by layer — exactly like the
brain's visual hierarchy from file 2. **If you remember only one thing from
this course, remember this.**

## How dimensions evolve

Watch the tensor shape change through a typical small CNN:

```
Input         224 × 224 ×   3
Conv+Pool     112 × 112 ×  32
Conv+Pool      56 ×  56 ×  64
Conv+Pool      28 ×  28 × 128
Conv+Pool      14 ×  14 × 256
Conv+Pool       7 ×   7 × 512
Flatten             25,088
Dense                  256
Output (classes)        10
```

The pattern: **spatial size shrinks, depth grows, then collapse to a decision.**
The network funnels from "lots of pixels, little meaning" to "few numbers,
rich meaning."

## Famous CNN architectures (for context)

You'll hear these names — they're all variations of the pipeline above:

| Architecture | Year | Claim to fame |
|--------------|------|---------------|
| **LeNet-5** | 1998 | The original CNN; read handwritten digits |
| **AlexNet** | 2012 | Won ImageNet, ignited the deep learning boom |
| **VGGNet** | 2014 | Showed that simple, deep stacks of 3×3 convs work great |
| **GoogLeNet/Inception** | 2014 | Multiple filter sizes in parallel |
| **ResNet** | 2015 | "Skip connections" enabled networks 100+ layers deep |

You don't need these yet — just know they're all built from the same
convolution / activation / pooling / dense building blocks.

---

## Key takeaways

- A CNN pipeline = repeated **(conv → ReLU → pool)** blocks, then
  **flatten → dense → softmax**.
- The defining behavior is a **feature hierarchy**: simple → complex, layer by
  layer.
- **Spatial size shrinks while depth grows**, funneling pixels into a decision.
- Famous architectures (LeNet, AlexNet, VGG, ResNet) are all variations on
  these same blocks.
- Next: how the network *learns* its filters — [training](09-training-a-cnn.md).
