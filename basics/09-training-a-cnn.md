# 9. Training a CNN

We've said filters are "learned, not hand-coded." This file explains *how*. You
don't need heavy math — just the intuition of the learning loop.

## What "learning" means here

A CNN has millions of numbers (the filter values and dense-layer weights). At
the start, they're **random**, so the network's guesses are garbage. **Training**
is the process of nudging all those numbers, little by little, until the guesses
become accurate.

The whole thing is a feedback loop:

```
   Guess  →  Measure how wrong  →  Adjust  →  Guess again  →  ...
```

## The ingredients

You need:

1. **Data** — many labeled examples (e.g. 50,000 images each tagged "cat,"
   "dog," etc.). The labels are the "right answers."
2. **A loss function** — a number measuring how wrong the network is.
3. **An optimizer** — the rule for adjusting weights to reduce the loss.

## Step 1: Forward pass

Feed an image through the whole pipeline (file 8). The network outputs
probabilities, e.g. `cat 0.30 | dog 0.60 | bird 0.10`. If the true label was
"cat," that's a bad guess — it favored dog.

## Step 2: Compute the loss

The **loss function** turns "how wrong" into a single number. For
classification, the standard choice is **cross-entropy loss**: it's small when
the network puts high probability on the correct class, and large when it
confidently picks the wrong one.

```
Confident & correct  →  low loss   😀
Unsure               →  medium loss 😐
Confident & wrong    →  high loss  😱
```

The goal of training is simply: **make the loss as small as possible.**

## Step 3: Backpropagation

This is the clever core. **Backpropagation** works *backward* through the
network and asks, for every single weight: *"Did you push the answer toward
right or wrong, and by how much?"* It computes a **gradient** — the direction
each weight should move to reduce the loss.

Think of it as **assigning blame**: every filter value gets told "you
contributed this much to the error."

```
Loss  ←─────── propagate the error backward ───────
                                                    │
   Output ← Dense ← Pool ← Conv ← Pool ← Conv ← Input
   (each layer learns how it should change)
```

## Step 4: Gradient descent (the update)

The **optimizer** uses those gradients to nudge each weight a small step in the
direction that lowers the loss. The classic method is **gradient descent**.

The "small step" size is the **learning rate** — the most important dial in
training:

- Too **large** → the network overshoots and never settles (unstable).
- Too **small** → training crawls and takes forever.

Popular optimizers like **Adam** and **SGD** are smart variations on this idea.

### The hill analogy

Imagine you're blindfolded on a foggy hill and want to reach the bottom (lowest
loss). You feel the slope under your feet (the gradient) and take a step
downhill. Repeat thousands of times and you reach a valley. That's gradient
descent.

```
   loss
    │＼
    │ ＼   ● you start here (random weights)
    │  ＼ ╱
    │   ╲╱  ← step downhill, again and again
    │   ▼  ← low loss = trained network
    └──────────── weights
```

## Step 5: Repeat — epochs and batches

You don't do this once. You repeat over the whole dataset many times:

| Term | Meaning |
|------|---------|
| **Batch** | A small group of images processed together before one update |
| **Iteration** | One weight update (one batch) |
| **Epoch** | One full pass through the *entire* training dataset |

Training might run for dozens of epochs. With each pass, the random filters
gradually morph into meaningful edge, texture, and object detectors. **Nobody
programmed those filters — gradient descent discovered them.**

## The overfitting trap

A network can "cheat" by **memorizing** the training images instead of learning
general patterns. Then it aces training data but fails on new images. This is
**overfitting**. Common defenses:

| Technique | Idea |
|-----------|------|
| **More data** | Harder to memorize a huge, varied set |
| **Data augmentation** | Randomly flip/rotate/crop images to add variety |
| **Dropout** | Randomly switch off neurons during training so the network can't rely on any single one |
| **Regularization** | Penalize overly large weights |
| **Validation set** | Held-out images to honestly check generalization |

> **Golden rule:** Always judge a model on data it has **never seen** during
> training (the validation/test set), never on the training data itself.

## Transfer learning (a practical shortcut)

Training from scratch needs huge data and compute. In practice, people often
take a CNN already trained on millions of images (like ResNet on ImageNet) and
**fine-tune** it on their own smaller dataset. The early filters (edges,
textures) are universal and transfer beautifully — you only retrain the later,
task-specific layers. This is called **transfer learning** and it's how most
real-world CNN projects start.

## The brain parallel (and its limits)

Both brains and CNNs improve through feedback and repetition. But:

- The brain does **not** use backpropagation in the literal CNN sense; how
  biological learning works is still an open research question.
- Brains learn from **few** examples and generalize from one shot ("that's a
  zebra" after seeing one). CNNs typically need thousands.
- Brains learn continuously without forgetting old skills; CNNs can suffer
  "catastrophic forgetting."

So training is the place where CNNs and brains **diverge most** — the
architecture is brain-*inspired*, but the learning algorithm is pure
mathematics.

---

## Key takeaways

- Training = a loop: **forward pass → loss → backprop → update**, repeated.
- **Loss** measures wrongness (usually **cross-entropy** for classification).
- **Backpropagation** assigns blame to each weight; **gradient descent** nudges
  them downhill.
- The **learning rate** controls step size — a critical setting.
- Repeat over **batches** and **epochs**; filters self-organize into detectors.
- Watch for **overfitting**; defend with augmentation, dropout, and a
  validation set.
- **Transfer learning** reuses a pretrained network as a shortcut.
- This is where CNNs diverge most from real biological learning.
- Finish with the [glossary](10-glossary.md) for quick reference.
