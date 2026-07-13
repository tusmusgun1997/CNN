# 1. Why LSTM? The Problem

Before the solution, feel the problem. LSTM exists to fix **one specific
weakness** of the vanilla RNN: it can't hold on to information over long
sequences.

## The long-term dependency problem

Language often needs context from **far back**:

```
"I grew up in France … [50 words about other things] … so I speak fluent ___"
```

To fill the blank with **French**, the network must remember "France" from ~50
words ago. A vanilla RNN struggles: by the time it reaches the blank, the memory
of "France" has been **overwritten** many times.

> New to *predicting a word* (vs. classifying a review)? See
> [0b — Predicting a word vs. classifying](00b-predicting-next-word-vs-classifying.md)
> first: next-word prediction is just classification over the vocabulary, and the
> LSTM memory is what supplies the long-range context to get it right.

Short gaps are fine:

```
"the clouds are in the ___"   → "sky"   (context is 3 words back — easy)
```

Long gaps are where vanilla RNNs fail. LSTM is built for the long gaps.

## Why the vanilla RNN forgets — two views

### View 1: the state gets overwritten every step

A vanilla RNN has **one** state, and it is completely recomputed every step:

```
Oₜ = tanh( xₜ·Wᵢ + Oₜ₋₁·Wₕ )
```

`Oₜ₋₁` is squashed through `tanh` and mixed with the new input **every single
step**. There's no way to say "leave this piece of memory *alone* for a while."
Old information is continuously blended away.

### View 2: the gradient vanishes (from your BPTT notes)

Recall from [backprop through time](../back%20propogation/backpropagation-through-time.md):
the gradient reaching an early step carries a product of factors

```
f′(a) · Wₕ    repeated once per step of distance
```

Over many steps this product either **shrinks to ~0** (vanishing) or **blows up**
(exploding). Vanishing means the early steps receive almost no learning signal —
the network literally *cannot* learn long-range links. In the verified example
there, the **t=1 gradient was ~100× smaller** than the t=3 gradient — over just 3
steps.

## What we actually want

Two abilities the vanilla RNN lacks:

1. **Persist** — keep a piece of information unchanged for many steps ("remember
   we're talking about France").
2. **Selectively update** — decide *per piece* what to keep, what to overwrite,
   and what to ignore, instead of blending everything every step.

## The core intuition of the fix

> Give the network a **separate memory track that it does not have to rewrite
> every step** — one it can choose to *leave alone*, *erase*, or *add to* — and
> control that track with small learnable **gates**.

That separate track is the **cell state**; the controllers are the **gates**.
That's the whole idea of LSTM, and it's the subject of [file 2 →](02-core-idea-cell-state-and-gates.md).

## Key takeaways

- The problem is **long-term dependencies**: needing context from many steps back.
- Vanilla RNN fails because its single state is **overwritten every step**, and
  its gradients **vanish** over distance.
- We want to **persist** information and **selectively update** it.
- LSTM's answer: a **separate cell state** (leave-alone-able memory) governed by
  **gates**. Next: exactly how that works.
