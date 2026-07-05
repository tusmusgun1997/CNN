# Types of RNN — Overview

The same recurrent cell (`Oₜ = f(xₜ·Wᵢ + Oₜ₋₁·Wₕ)`) can be wired into **four
different architectures**, depending on *where inputs enter* and *where outputs
are read*. The architecture is chosen by the **shape of your task**: is the input
a single item or a sequence? Is the output a single item or a sequence?

## The four types at a glance

```
 one-to-one        one-to-many        many-to-one        many-to-many
 (no recurrence)                                        (aligned / seq2seq)

     y                y₁  y₂  y₃           y                y₁  y₂  y₃
     ▲                ▲   ▲   ▲            ▲                ▲   ▲   ▲
   ┌───┐            ┌───┬───┬───┐        ┌───┬───┬───┐    ┌───┬───┬───┐
   │   │            │   →   →   │        │   →   →   │    │   →   →   │
   └───┘            └───┴───┴───┘        └───┴───┴───┘    └───┴───┴───┘
     ▲                ▲                    ▲   ▲   ▲        ▲   ▲   ▲
     x                x                    x₁  x₂  x₃       x₁  x₂  x₃
```

| Type | Input | Output | Classic use case | File |
|------|-------|--------|------------------|------|
| **One-to-one** | single item | single item | plain classification (no sequence) | [01-one-to-one.md](01-one-to-one.md) |
| **One-to-many** | single item | sequence | image → caption, music generation | [02-one-to-many.md](02-one-to-many.md) |
| **Many-to-one** | sequence | single item | **sentiment analysis**, spam detection | [03-many-to-one.md](03-many-to-one.md) |
| **Many-to-many** | sequence | sequence | translation, POS tagging, video labeling | [04-many-to-many.md](04-many-to-many.md) |

## How to pick — two questions

```
              Is the INPUT a sequence?
                 no            yes
               ┌────────────┬─────────────┐
Is the   no    │ one-to-one │ many-to-one │
OUTPUT a       ├────────────┼─────────────┤
sequence? yes  │ one-to-many│ many-to-many│
               └────────────┴─────────────┘
```

## One cell, four wirings

Nothing inside the cell changes between the types — same equations, same weights
Wᵢ, Wₕ, W_o (see the [RNN overview](../overview/rnn-overview.md)). What changes:

- **Where x enters:** every step (many-*) or only the first step (one-to-many).
- **Where ŷ is read:** only the last step (many-to-one), every step
  (many-to-many aligned), or after the whole input is consumed (seq2seq).

In Keras this is mostly one switch:

```python
SimpleRNN(units, return_sequences=False)  # emit ONLY the last Oₜ  → many-to-one
SimpleRNN(units, return_sequences=True)   # emit Oₜ at EVERY step  → many-to-many
```

Read the files in order — each type builds on the previous one, and
[many-to-one](03-many-to-one.md) is the one you have already seen animated in
[the recurrent-layer demo](../overview/rnn-recurrent-layer-animated.html).
