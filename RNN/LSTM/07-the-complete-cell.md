# 7. The Complete LSTM Cell

All the pieces in one place: the six equations, the shapes/vectors, and the
parameter count.

## The six equations (in order)

```
1.  f_t  = σ ( W_f · [h_(t-1), x_t] + b_f )        forget gate      (0,1)
2.  i_t  = σ ( W_i · [h_(t-1), x_t] + b_i )        input gate       (0,1)
3.  C̃_t  = tanh( W_C · [h_(t-1), x_t] + b_C )      candidate       (-1,1)
4.  C_t  = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t              cell state (memory)
5.  o_t  = σ ( W_o · [h_(t-1), x_t] + b_o )        output gate      (0,1)
6.  h_t  = o_t ⊙ tanh( C_t )                        hidden state (output)  (-1,1)
```

Read it as a story: **decide what to forget (1), what to write (2–3), update the
memory (4), decide what to reveal (5), produce the output (6).**

## The vectors and their shapes

Let **d = hidden size** (also the cell-state size) and **n = input size**.

| Vector | Meaning | Shape |
|--------|---------|-------|
| `x_t` | input | (n,) |
| `h_(t-1)`, `h_t` | hidden state / output | (d,) |
| `C_(t-1)`, `C_t` | cell state (memory) | (d,) |
| `[h_(t-1), x_t]` | concatenation fed to every gate | (d + n,) |
| `f_t, i_t, o_t` | gates | (d,) |
| `C̃_t` | candidate | (d,) |

| Weight | Shape | Bias | Shape |
|--------|-------|------|-------|
| `W_f, W_i, W_C, W_o` | (d, d + n) | `b_f, b_i, b_C, b_o` | (d,) |

Everything the gates produce is size **d** (matching the cell state), so all the
`⊙` element-wise products line up.

## Split form (how libraries actually write it)

The concatenated `W·[h, x]` is often split into two matrices — one for the input,
one for the recurrent state:

```
W_f · [h_(t-1), x_t]  =  U_f · h_(t-1)  +  W_f · x_t
```

- `W_*` (input kernels): shape (d, n)
- `U_*` (recurrent kernels): shape (d, d)

Same math, just bookkeeping. Keras calls these `kernel` and `recurrent_kernel`.

## Counting the parameters

Four gate computations, each with a weight matrix `(d, d+n)` and a bias `(d,)`:

```
params = 4 × [ d·(d + n)  +  d ]
       = 4 × [ d·d + d·n + d ]
```

Example — hidden size **d = 128**, input size **n = 100**:

```
4 × (128·228 + 128) = 4 × (29,184 + 128) = 4 × 29,312 = 117,248 parameters
```

The "**× 4**" is the signature of an LSTM — four times a vanilla RNN layer of the
same size (one set of weights → four sets, one per gate/candidate).

## The full data flow (one cell)

```
                              ┌──────────────────────────► C_t (to next step)
   C_(t-1) ──►( ⊙ )──►( + )───┤
                │       │      └── tanh ──►( ⊙ )──► h_t ──► output + next step
              f_t    i_t⊙C̃_t                 │
                │       │                     o_t
                └───────┴─────────┬───────────┘
                                  │  all read:
                        [ h_(t-1) , x_t ]
```

## What flows between time steps

Unlike the vanilla RNN (one state), the LSTM passes **two** things forward:

```
   step t-1  ──(  h_(t-1) , C_(t-1)  )──►  step t  ──(  h_t , C_t  )──►  step t+1
```

`h` is the short-term/output state; `C` is the long-term memory belt.

## Key takeaways

- Six equations: **forget, input, candidate, cell update, output, hidden**.
- Every gate reads `[h_(t-1), x_t]`, outputs size **d**, and combines via
  element-wise `⊙`.
- Weights: four `(d, d+n)` matrices + four `(d,)` biases → **params = 4·(d·(d+n)+d)**;
  the "**×4**" vs. a vanilla RNN.
- Two states travel between steps: **h_t** (output) and **C_t** (memory).
- Next: [see all of this with real numbers](08-worked-example-with-numbers.md).
