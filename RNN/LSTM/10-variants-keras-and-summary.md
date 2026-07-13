# 10. Variants, Keras & Summary

The finish line: the common LSTM variants, how it looks in Keras, and a one-page
recap of the whole course.

## GRU — the popular lighter cousin

The **Gated Recurrent Unit (GRU)** keeps the gating idea but is simpler and
faster: it **merges the cell state and hidden state into one**, and uses **two
gates** instead of three.

| | **LSTM** | **GRU** |
|---|----------|---------|
| States | 2 (C_t and h_t) | 1 (h_t only) |
| Gates | 3 (forget, input, output) | 2 (**reset**, **update**) |
| Parameters | more (×4) | fewer (×3) → faster |
| When to prefer | long/complex dependencies, big data | smaller data, speed, similar accuracy |

GRU's **update gate** does the job of LSTM's forget+input together ("how much old
vs. new"), and the **reset gate** controls how much past state influences the new
candidate. Same philosophy — *gated, mostly-additive memory* — fewer moving parts.
In practice LSTM and GRU perform similarly; try both.

## Other variants (good to recognize)

| Variant | Change |
|---------|--------|
| **Peephole LSTM** | gates also look at the cell state `C_(t-1)` directly |
| **Bidirectional LSTM** | one LSTM reads left→right, another right→left; outputs concatenated — great when the *whole* sequence is available (not live) |
| **Stacked / deep LSTM** | multiple LSTM layers on top of each other (`return_sequences=True` feeds the next layer) |
| **Coupled forget/input** | tie `i_t = 1 − f_t` (fewer params) |

## LSTM in Keras

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(vocab_size, 64, mask_zero=True),   # see RNN/encoding/embedding
    LSTM(32),                                     # many-to-one: returns last h_t (dim 32)
    Dense(1, activation="sigmoid"),              # sentiment: pos/neg
])
```

Useful arguments:

| Argument | Effect |
|----------|--------|
| `LSTM(units)` | `units` = hidden/cell size **d** |
| `return_sequences=True` | emit `h_t` at **every** step (many-to-many / stacking) |
| `return_state=True` | also return the final `h_T` and `C_T` (needed for seq2seq encoder→decoder) |
| `Bidirectional(LSTM(...))` | wrap for a bi-LSTM |
| `recurrent_dropout` | dropout on the recurrent connections |

`LSTM(32)` on a 64-dim embedding input has `4·(32·(32+64)+32) = 12,416`
parameters — the **×4** you derived in [file 7](07-the-complete-cell.md).

## The whole course on one page

**The problem** ([file 1](01-why-lstm.md)): vanilla RNNs overwrite their single
state every step and their gradients vanish → they forget long-range context.

**The idea** ([file 2](02-core-idea-cell-state-and-gates.md)): add a **cell state
C_t** (a memory conveyor belt edited only by multiply + add) controlled by
**gates** (sigmoid valves, `gate ⊙ vector`).

**The six equations** ([file 7](07-the-complete-cell.md)):

```
f_t = σ(W_f·[h_(t-1),x_t] + b_f)          forget:   what to erase
i_t = σ(W_i·[h_(t-1),x_t] + b_i)          input:    how much to write
C̃_t = tanh(W_C·[h_(t-1),x_t] + b_C)       candidate: what to write
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t          update:   keep old + add new   ← the heart
o_t = σ(W_o·[h_(t-1),x_t] + b_o)          output:   what to reveal
h_t = o_t ⊙ tanh(C_t)                      hidden:   the output
```

**The why** ([file 9](09-why-lstm-solves-vanishing-gradient.md)): `∂C_t/∂C_(t-1)
= f_t`, a learned multiply near 1 → gradients ride the belt across many steps (the
**Constant Error Carousel**) → long-range learning works.

## Gate cheat sheet

| Gate | Formula | Activation | Job |
|------|---------|-----------|-----|
| **Forget** `f_t` | `σ(W_f·[h,x]+b_f)` | sigmoid (0,1) | erase old memory |
| **Input** `i_t` | `σ(W_i·[h,x]+b_i)` | sigmoid (0,1) | admit new content |
| **Candidate** `C̃_t` | `tanh(W_C·[h,x]+b_C)` | tanh (−1,1) | propose new content |
| **Output** `o_t` | `σ(W_o·[h,x]+b_o)` | sigmoid (0,1) | reveal memory as output |

## Where to go next

- **Attention & Transformers** — LSTMs process sequentially and still have a
  memory bottleneck; attention lets a model look at *all* steps directly, and
  became the Transformer (modern LLMs). This is the natural sequel to the
  [seq2seq bottleneck](../types%20of%20RNN/04-many-to-many.md#the-bottleneck-and-where-the-story-goes-next).
- **Build one:** stack the [encoding pipeline](../encoding/00-overview.md) →
  `Embedding → LSTM → Dense` into a real sentiment model.

## Key takeaways

- **GRU** = lighter LSTM (1 state, 2 gates); similar accuracy, fewer params.
- **Bidirectional / stacked / peephole** are common structural variants.
- Keras: `LSTM(units)`, with `return_sequences` / `return_state` / `Bidirectional`
  for the [RNN types](../types%20of%20RNN/00-overview.md); parameter count carries
  the signature **×4**.
- The whole idea in one line: **a gated, additive memory belt that the network
  controls — so it can remember for as long as it needs.**

← Back to [Start Here](00-start-here.md)
