# 1. One-to-One

**One input → one output.** A single fixed-size item goes in, a single result
comes out, and there is **no sequence anywhere** — which means there is no
recurrence either.

## The diagram

```
        y            e.g.  image → "cat"
        ▲                  features → price
     ┌─────┐
     │ net │        no loop, no time steps,
     └─────┘        no memory needed
        ▲
        x
```

## What it really is

One-to-one is the **degenerate case**: a plain feed-forward network (an MLP, or a
CNN for images). It appears in the RNN taxonomy only as the **baseline** — the
starting point the other three types generalize away from:

```
Oₜ = f(xₜ·Wᵢ + Oₜ₋₁·Wₕ)      with only t = 1 and O₀ = 0
   = f(x₁·Wᵢ)                 ← the recurrent term vanishes; it's just a dense layer!
```

With a single time step, the memory term `Oₜ₋₁·Wₕ` contributes nothing. The
"recurrent" network collapses into an ordinary layer.

## Use cases

| Task | Input | Output |
|------|-------|--------|
| Image classification | one image | one label ("cat") |
| House-price prediction | one feature vector | one number |
| Tabular classification | one row of features | one class |
| Digit recognition (MNIST) | one 28×28 image | one of 10 digits |

All of these are covered by the [CNN](../../CNN/basics/00-overview.md) and
plain-ANN material — no RNN required.

## When one-to-one is the WRONG choice

The moment your input or output has **order and variable length** — words,
audio frames, sensor readings over time — a fixed one-shot mapping breaks
(same arguments as [CNN vs ANN](../../CNN/CNN%20vs%20ANN/cnn-vs-ann.md): wrong
inductive bias). That's the cue to move to the other three types.

## Key takeaways

- One-to-one = **no sequence in, no sequence out** → effectively a feed-forward
  network; the recurrence term disappears.
- It's in the taxonomy as the **reference point**, not as a real RNN use case.
- Sequence in input? → [many-to-one](03-many-to-one.md) /
  [many-to-many](04-many-to-many.md). Sequence in output? →
  [one-to-many](02-one-to-many.md).
