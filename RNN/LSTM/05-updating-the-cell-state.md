# 5. Updating the Cell State (the Heart of LSTM)

This is where the conveyor belt actually moves forward. We now combine the
[forget gate](03-the-forget-gate.md) and the [input gate](04-the-input-gate.md)
into the **single most important equation** in the LSTM.

## The equation

```
C_t = f_t ⊙ C_(t-1)  +  i_t ⊙ C̃_t
      └── forget ──┘    └── input ──┘
      keep old memory   add new memory
```

`⊙` is element-wise (Hadamard) multiplication. Read it as two moves on each
memory slot:

1. **Erase:** scale the old memory by the forget gate → `f_t ⊙ C_(t-1)`.
2. **Write:** add the gated candidate → `i_t ⊙ C̃_t`.

```
            C_(t-1)          f_t          C̃_t          i_t         C_t
 slot 1:     0.90     ×      0.8   +      0.20   ×      0.5    =   0.72 + 0.10 = 0.82
 slot 2:    -0.30     ×      0.1   +      0.60   ×      0.9    =  -0.03 + 0.54 = 0.51
             (old)        (mostly              (new           (write            (new
                           forget slot2)        content)       most of it)      memory)
```

Slot 1 was **kept** (f≈0.8) and lightly topped up. Slot 2 was **wiped**
(f≈0.1) and **overwritten** with new content (i≈0.9). Each slot is edited
**independently** — that's the selective memory a vanilla RNN can't do.

## Why this exact form? (the big "why")

Two design choices here are the whole reason LSTMs work.

### 1. The update is ADDITIVE

The new memory is the old memory **plus** a term:

```
C_t = (something) · C_(t-1)  +  (new stuff)
```

Compare the vanilla RNN, where the state is **replaced** by a squashed mix every
step: `Oₜ = tanh(…Oₜ₋₁…)`. There, `Oₜ₋₁` is buried inside a nonlinearity, so its
influence (and its gradient) decays fast.

In the LSTM, `C_(t-1)` sits in an **addition**, outside any activation function.
When gradients flow backward, `∂C_t/∂C_(t-1) = f_t` — a plain multiply by the
forget gate, **no tanh derivative shrinking it**. If `f_t ≈ 1`, the gradient
passes **almost unchanged** across many steps. This is the famous
**Constant Error Carousel** — the highway that defeats the vanishing gradient
(full analysis in [file 9](09-why-lstm-solves-vanishing-gradient.md)).

### 2. The gates make it SELECTIVE

`f_t` and `i_t` are per-element, so the network can, for each slot, choose any
combination of keep/erase/write:

| f_t | i_t | Effect on the slot |
|-----|-----|--------------------|
| ≈ 1 | ≈ 0 | **hold** — carry the memory unchanged (long-term storage) |
| ≈ 0 | ≈ 1 | **overwrite** — replace with new content |
| ≈ 1 | ≈ 1 | **accumulate** — keep old and add new |
| ≈ 0 | ≈ 0 | **reset** — wipe the slot to ~0 |

One equation gives the LSTM its full vocabulary of memory operations.

## Where we are in the cell

```
   C_(t-1) ──►(⊙ f_t)──►( + )──────────► C_t ───►(to output gate, file 6)
                          ▲
                   (i_t ⊙ C̃_t)
```

The belt has now been edited and carries `C_t` forward — both to the **next time
step** and to the **output gate**, which decides what to reveal.

## Key takeaways

- **The heart of LSTM:** `C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t` = *keep* old memory +
  *write* new memory, per element.
- The update is **additive**, so `∂C_t/∂C_(t-1) = f_t` — gradients cross many
  steps without vanishing (the **constant error carousel**).
- Per-element gates let each slot independently **hold / overwrite / accumulate /
  reset**.
- Next: turn the updated memory into an output — the [output gate](06-the-output-gate.md).
