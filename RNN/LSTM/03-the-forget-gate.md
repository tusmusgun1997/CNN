# 3. The Forget Gate

**Question it answers:** *"Of the memory I'm carrying, what should I throw away?"*

The forget gate is the first edit to the conveyor belt. It looks at the new
situation and decides, **per memory slot**, how much of the old cell state
`C_(t-1)` to keep.

## The equation

```
f_t = σ( W_f · [ h_(t-1) , x_t ] + b_f )
```

Term by term:

| Piece | What it is | Shape (d = hidden size, n = input size) |
|-------|-----------|------------------------------------------|
| `x_t` | current input | (n,) |
| `h_(t-1)` | previous output (short-term context) | (d,) |
| `[h_(t-1), x_t]` | the two **concatenated** into one vector | (d + n,) |
| `W_f` | forget-gate weights (learned) | (d, d + n) |
| `b_f` | forget-gate bias (learned) | (d,) |
| `σ` | sigmoid → squashes each value into **(0, 1)** | — |
| `f_t` | the forget gate — one value per memory slot | (d,) |

So `f_t` is a vector of numbers between 0 and 1, the **same size as the cell
state**.

## How it's used

It multiplies the old memory element-wise (this happens in
[file 5](05-updating-the-cell-state.md), shown here for context):

```
kept_memory = f_t ⊙ C_(t-1)
```

```
f_t element = 1  →  keep that memory slot fully      (remember)
f_t element = 0  →  erase that memory slot completely (forget)
f_t element = 0.7 → keep 70% of it
```

## Why sigmoid? (the "why")

We need a **fraction to keep**, per element — that is exactly what a sigmoid
produces: a smooth, differentiable value in (0, 1). A hard 0/1 switch wouldn't be
trainable by gradient descent; the sigmoid is the soft, learnable version of an
on/off valve.

## Why look at both h_(t-1) and x_t?

The decision "should I forget this?" depends on **both**:

- the **new input** `x_t` ("a new sentence started — drop the old subject"), and
- the **recent context** `h_(t-1)` ("we were mid-thought — keep it").

Concatenating them lets `W_f` learn any mix of the two.

## Intuition example

Tracking grammatical **gender/number** for translation:

```
"The girl … she …"     → keep the "female/singular" memory  (f ≈ 1)
"The girl … The boys …" → the new subject "boys" arrives:
                          forget "female/singular", make room  (f ≈ 0)
```

The forget gate learns to **reset the relevant slots** exactly when the subject
changes, and to **hold them** otherwise.

## The forget-bias = 1 trick (a practical why)

In practice `b_f` is often **initialized to 1** (or 2). Since `σ(large positive)
≈ 1`, the gate starts out **near "keep everything"**. This means that early in
training the LSTM **defaults to remembering**, giving gradients a clear path to
flow while it learns what's actually worth forgetting. (You'll see `f_t ≈ 0.75`
in the [worked example](08-worked-example-with-numbers.md) because of exactly
this bias.)

## Key takeaways

- Forget gate `f_t = σ(W_f·[h_(t-1), x_t] + b_f)` → a (0,1) vector, one value per
  memory slot.
- It decides **how much of the old memory `C_(t-1)` survives** (used as
  `f_t ⊙ C_(t-1)`).
- **Sigmoid** because we need a differentiable *fraction to keep*; it reads both
  the input and the recent context.
- **Bias initialized to 1** so the cell defaults to remembering early in training.
- Next: what new information to write — the [input gate](04-the-input-gate.md).
