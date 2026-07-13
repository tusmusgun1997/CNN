# 9. Why LSTM Beats the Vanishing Gradient

This is the payoff. We saw in [RNN backprop](../back%20propogation/backpropagation-through-time.md)
that vanilla RNNs forget because gradients **vanish** over distance. Here's
exactly why the LSTM's cell state fixes that.

## Recap: why the vanilla RNN's gradient dies

In a vanilla RNN, the gradient of a loss at step T w.r.t. an early state passes
through this factor once per step:

```
∂Oₜ / ∂Oₜ₋₁ = f′(aₜ) · Wₕ
```

Over a distance of *k* steps, you multiply **k** of these together:

```
∏ (f′ · Wₕ)     →   with tanh, f′ ≤ 1, so this product shrinks toward 0
```

Repeatedly multiplying numbers below 1 ⇒ **vanishing gradient** ⇒ early steps get
no learning signal ⇒ the network can't learn long-range dependencies.

## The LSTM's gradient highway

Now look at the LSTM cell-state update ([file 5](05-updating-the-cell-state.md)):

```
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

Differentiate the cell state w.r.t. the previous cell state:

```
∂C_t / ∂C_(t-1) = f_t
```

That's it — **just the forget gate**. No `tanh` derivative, no `Wₕ` matrix buried
in a nonlinearity. The cell state connects to its past by a **plain multiply by
f_t**.

### Why that changes everything

Over *k* steps, the gradient along the cell-state path is:

```
∂C_T / ∂C_(T-k) = f_T ⊙ f_(T-1) ⊙ … ⊙ f_(T-k+1)     (a product of forget gates)
```

If the network **wants to remember**, it learns `f_t ≈ 1` for the relevant slots.
Then this product stays **≈ 1** across many steps — the gradient flows back
**almost undiminished**. The network *controls its own gradient flow* by setting
the forget gates.

Contrast:

| | Vanilla RNN | LSTM (cell state) |
|---|-------------|--------------------|
| Backward factor per step | `f′(a)·Wₕ` (forced ≤ ~1, uncontrolled) | `f_t` (learned, can be ≈ 1) |
| Over k steps | `∏ f′·Wₕ` → shrinks to 0 | `∏ f_t` → stays ≈ 1 if remembering |
| Who decides | nobody — fixed by tanh & weights | the **network**, via the forget gate |

## The Constant Error Carousel (CEC)

The original LSTM paper called this the **Constant Error Carousel**: when
`f_t = 1` and `i_t = 0`, the cell state simply **copies itself** forward
(`C_t = C_(t-1)`), and the error/gradient **rides backward unchanged**, like a
carousel that neither speeds up nor slows down. That un-decaying path is the
"long-term" in Long Short-Term Memory.

```
   error at step T ──►──►──►──►──► reaches step T−k with ≈ full strength
        (rides the cell-state belt, multiplied only by f_t ≈ 1 each step)
```

## Why the additive form is essential

The magic comes from **addition**:

```
C_t = f_t ⊙ C_(t-1)  +  (new stuff)
      └── linear in C_(t-1) ──┘
```

Because `C_(t-1)` enters **linearly** (multiplied by a gate, then added), its
gradient is a clean multiply. In the vanilla RNN, `Oₜ₋₁` enters **inside `tanh`**,
so its gradient always gets an extra `f′ ≤ 1` squeeze every step. LSTM moved the
memory **out of the nonlinearity** and into an additive path — that one structural
change is the fix.

> The output `tanh` in `h_t = o_t ⊙ tanh(C_t)` is on the **readout branch**, not
> on the belt, so it never touches the `C_t → C_(t-1)` gradient path. The highway
> stays clear.

## What about exploding gradients?

The additive path *can* still let gradients grow if forget gates exceed... no —
forget gates are in (0,1), so the cell-state product can't explode. Any remaining
explosion (through the gate weights) is handled in practice by **gradient
clipping**. So LSTM structurally kills *vanishing*, and clipping mops up
*exploding*.

## Key takeaways

- Vanilla RNN backward factor `f′·Wₕ` (< 1) multiplied k times → **vanishes**.
- LSTM cell state: `∂C_t/∂C_(t-1) = f_t` — a **learned** multiply. If the net
  wants to remember, `f_t ≈ 1`, so the gradient survives across many steps.
- This un-decaying path is the **Constant Error Carousel**; it works because
  `C_(t-1)` enters the update **additively**, outside any nonlinearity.
- The network **controls its own memory and gradient flow** via the forget gate —
  the essence of why LSTMs learn long-range dependencies.
- Next: [GRU, Keras, and the summary](10-variants-keras-and-summary.md).
