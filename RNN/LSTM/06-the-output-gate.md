# 6. The Output Gate

**Question it answers:** *"Given my updated memory, what should I actually output
right now?"*

The cell state `C_t` is the full long-term memory — but you don't want to dump
all of it out at every step. The output gate decides **which parts of the memory
to expose** as this step's output `h_t`.

## The two equations

```
o_t = σ( W_o · [ h_(t-1) , x_t ] + b_o )        ← output gate: what to reveal
h_t = o_t ⊙ tanh( C_t )                          ← the actual output
```

| Symbol | Meaning | Activation | Range | Shape |
|--------|---------|-----------|-------|-------|
| `o_t` | output gate ("how much of each slot to reveal") | sigmoid | (0, 1) | (d,) |
| `tanh(C_t)` | the memory, squashed to a bounded range | tanh | (−1, 1) | (d,) |
| `h_t` | **hidden state / output** of this step | — | (−1, 1) | (d,) |

## How it works — two steps

1. **Squash the memory:** `tanh(C_t)` maps the (possibly large) cell state into
   (−1, 1). This keeps the output bounded and well-scaled.
2. **Filter it:** multiply by the output gate `o_t` element-wise — reveal the
   slots the network deems relevant *now*, hide the rest.

```
o_t element = 1  →  fully expose that memory slot in the output
o_t element = 0  →  keep that slot hidden (still remembered in C_t, just not output)
```

## Why keep memory and output separate? (the "why")

This is the payoff of having **two** states:

- `C_t` (cell state) = **everything I remember** — including things I'm not using
  this instant.
- `h_t` (hidden state) = **what's relevant to say/predict right now**.

Example: an LSTM might store "the subject is plural" deep in `C_t` for many steps
without outputting it, then **open the output gate** for that slot exactly when it
reaches the verb that must agree ("the boys **are**"). The information was
*remembered* the whole time but only *revealed* when needed. A vanilla RNN, with
its single state, can't hold something silently — whatever it remembers, it also
outputs.

## Why tanh on C_t (and not on the memory update)?

- The cell state `C_t` can grow fairly large through repeated additions. `tanh`
  **re-bounds** it to (−1, 1) before it leaves the cell, so downstream layers get
  a stable, normalized signal.
- Note this `tanh` is applied **on the way out**, not on the belt itself — so it
  does **not** interfere with the additive memory highway from
  [file 5](05-updating-the-cell-state.md). Memory stays un-squashed; only the
  *readout* is squashed.

## Where h_t goes

`h_t` has two destinations, exactly like the vanilla RNN's `Oₜ`:

```
h_t ──► the prediction for this step (e.g. through W and softmax/sigmoid)
    └─► fed back into the NEXT time step as h_(t-1) (and used by all gates there)
```

## The full cell, now complete

```
   C_(t-1) ─►(⊙f)─►(+)──────────────► C_t ──────────────► (next step)
                    ▲                  │
              (i ⊙ C̃)               tanh
                                        │
   h_(t-1),x_t ─► [f, i, C̃, o gates]   (⊙ o) ─► h_t ─► output + next step
```

All four computations (`f, i, C̃, o`) read `[h_(t-1), x_t]`; the belt update and
the gated readout produce `C_t` and `h_t`.

## Key takeaways

- Output gate `o_t = σ(...)`; output `h_t = o_t ⊙ tanh(C_t)`.
- `tanh(C_t)` **bounds** the memory for readout; `o_t` **filters** which slots are
  exposed.
- Separating `C_t` (remember) from `h_t` (reveal) lets the LSTM **hold
  information silently** and surface it only when needed.
- The output `tanh` is on the **readout**, not the belt, so the additive memory
  highway is preserved.
- Next: [assemble all six equations](07-the-complete-cell.md) with shapes and
  parameter counts.
