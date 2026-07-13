# Backpropagation Through Time (BPTT) — Detailed Calculation

Training an RNN means finding gradients of the loss w.r.t. its weights. But an
RNN's weights are **shared across every time step**, which changes how the chain
rule unfolds. The algorithm is **Backpropagation Through Time (BPTT)**.

We use the exact example from the notes: a **3-word input** (`xᵢ₁, xᵢ₂, xᵢ₃`),
one recurrent layer, one output. Because the input has **3 time steps**, the
gradient for the shared weights breaks into **3 levels of differentiation** — the
three summed lines you see on the whiteboard.

> Companion animation: [`bptt-animation.html`](bptt-animation.html)
> All gradient formulas below are verified against TensorFlow autograd.

---

## 0. The forward pass (what we differentiate)

```
O₁ = f( xᵢ₁·Wᵢ + O₀·Wₕ )          O₀ = 0
O₂ = f( xᵢ₂·Wᵢ + O₁·Wₕ )
O₃ = f( xᵢ₃·Wᵢ + O₂·Wₕ )
ŷ  = g( O₃·Wₒ )
L  = loss( ŷ , y )
```

- **f** = hidden activation (tanh), **g** = output activation (sigmoid).
- Three weight matrices to learn — and **each is reused at every step**:
  - **Wᵢ** — input weights (also written `wᵢ`)
  - **Wₕ** — recurrent / hidden weights (the notes call it `wₙ`)
  - **Wₒ** — output weights (`wₒ`)

We need three gradients: **∂L/∂Wₒ, ∂L/∂Wᵢ, ∂L/∂Wₕ**.

## 1. The one idea that makes BPTT different

In a normal feed-forward net each weight is used **once**, so its gradient is a
single chain. Here `Wᵢ` appears inside `O₁`, `O₂` **and** `O₃`. The loss depends
on `Wᵢ` through **three different routes**, so by the multivariable chain rule we
**add the gradient from every route**:

> **Shared weight ⇒ gradient = SUM of the contributions from all time steps.**

That sum is the whole story. Everything below is just writing that sum out.

## 2. The dependency graph (unrolled)

Follow the arrows: to reach `Wᵢ`, the loss can stop at `O₃`, or pass back to
`O₂`, or all the way to `O₁`.

```
                         ┌──────────────► xᵢ₃ ─(Wᵢ)─┐
   L ──► ŷ ──► O₃ ──┤                              ├──► (cell 3)
        (Wₒ)        └──► O₂ ──┬──────────► xᵢ₂ ─(Wᵢ)─┐
                     (Wₕ)     └──► O₁ ──┬──► xᵢ₁ ─(Wᵢ)─┐   ├──► (cell 2)
                               (Wₕ)     └──► O₀        └──► (cell 1)
```

- **O₃** depends on `Wᵢ` (via `xᵢ₃`) **and** on `O₂` (via `Wₕ`).
- **O₂** depends on `Wᵢ` (via `xᵢ₂`) **and** on `O₁` (via `Wₕ`).
- **O₁** depends on `Wᵢ` (via `xᵢ₁`) **and** on `O₀`.

Every time you go one cell further back, you pick up one more `∂Oₜ/∂Oₜ₋₁` factor.

## 3. Warm-up: ∂L/∂Wₒ (output weights — the easy one)

`Wₒ` is used **only once** (in `ŷ`), so it's a plain single chain — no time
unrolling:

```
∂L        ∂L     ∂ŷ
──── =  ──── · ────
∂Wₒ      ∂ŷ    ∂Wₒ
```

With `ŷ = g(O₃·Wₒ)`, the local factor `∂ŷ/∂Wₒ` brings down `O₃`. And if `g` is
sigmoid with binary cross-entropy loss, the two output factors collapse (the same
cancellation as in the [CNN backprop](../../CNN/backpropagation/backpropagation.html)):

```
∂L/∂(O₃·Wₒ) = ŷ − y          ⟹     ∂L/∂Wₒ = (ŷ − y) · O₃
```

## 4. ∂L/∂Wᵢ — THREE levels of differentiation

This is the whiteboard equation. `Wᵢ` lives in `O₁`, `O₂`, `O₃`, so we sum three
paths — one ending at each time step:

```
∂L      ∂L   ∂ŷ   ∂O₃
──── =  ── · ── · ────                                       ◄─ level 1 (via O₃, t=3)
∂Wᵢ     ∂ŷ   ∂O₃  ∂Wᵢ

        ∂L   ∂ŷ   ∂O₃   ∂O₂
      + ── · ── · ─── · ────                                 ◄─ level 2 (via O₂, t=2)
        ∂ŷ   ∂O₃  ∂O₂   ∂Wᵢ

        ∂L   ∂ŷ   ∂O₃   ∂O₂   ∂O₁
      + ── · ── · ─── · ─── · ────                           ◄─ level 3 (via O₁, t=1)
        ∂ŷ   ∂O₃  ∂O₂   ∂O₁   ∂Wᵢ
```

Read each line as a **route back through the unrolled network**:

| Level | Route | Meaning |
|-------|-------|---------|
| **1** | L → ŷ → O₃ → Wᵢ | Wᵢ's effect through the **last** cell only |
| **2** | L → ŷ → O₃ → O₂ → Wᵢ | Wᵢ's effect one step **earlier**, routed forward through O₂→O₃ |
| **3** | L → ŷ → O₃ → O₂ → O₁ → Wᵢ | Wᵢ's effect at the **first** step, routed all the way forward |

The deeper the level, the more `∂Oₜ/∂Oₜ₋₁ = f′·Wₕ` factors it collects. **Three
words ⇒ three levels.** Five words ⇒ five levels, and so on.

### 4a. Each local derivative, evaluated

Let `aₜ = xᵢₜ·Wᵢ + Oₜ₋₁·Wₕ` be the pre-activation, so `Oₜ = f(aₜ)`.

| Local term | Value | Why |
|------------|-------|-----|
| `∂L/∂ŷ` | loss′(ŷ, y) | derivative of the loss |
| `∂ŷ/∂O₃` | `g′(O₃Wₒ) · Wₒ` | ŷ = g(O₃Wₒ) |
| `∂O₃/∂Wᵢ` | `f′(a₃) · xᵢ₃` | direct: a₃ contains `xᵢ₃·Wᵢ` |
| `∂O₃/∂O₂` | `f′(a₃) · Wₕ` | a₃ contains `O₂·Wₕ` |
| `∂O₂/∂Wᵢ` | `f′(a₂) · xᵢ₂` | direct at step 2 |
| `∂O₂/∂O₁` | `f′(a₂) · Wₕ` | a₂ contains `O₁·Wₕ` |
| `∂O₁/∂Wᵢ` | `f′(a₁) · xᵢ₁` | direct at step 1 |

> For **tanh**, `f′(aₜ) = 1 − Oₜ²` — handy, because you already have `Oₜ` from the
> forward pass.

### 4b. It collapses to a clean sum

Define the **error at each hidden state** by pushing the loss back through the
cells:

```
δ₃ = (∂L/∂ŷ)(∂ŷ/∂O₃)                 e₃ = δ₃ · f′(a₃)     ← error entering cell 3
δ₂ = e₃ · Wₕ                          e₂ = δ₂ · f′(a₂)     ← error entering cell 2
δ₁ = e₂ · Wₕ                          e₁ = δ₁ · f′(a₁)     ← error entering cell 1
```

Then the three-level equation is exactly:

```
∂L
──── = e₃·xᵢ₃  +  e₂·xᵢ₂  +  e₁·xᵢ₁     =   Σₜ  eₜ · xᵢₜ
∂Wᵢ    └level 1┘  └level 2┘  └level 3┘
```

Each level is just **one time step's error × that step's input**. That is BPTT.

## 5. ∂L/∂Wₕ — three levels again

`Wₕ` is *also* shared across all steps, so the same three-level structure applies
— only the **last factor** changes, because `Wₕ` multiplies `Oₜ₋₁` (not `xᵢₜ`):

```
∂L      ∂L   ∂ŷ   ∂O₃
──── =  ── · ── · ────                                       (via O₃)
∂Wₕ     ∂ŷ   ∂O₃  ∂Wₕ

        ∂L   ∂ŷ   ∂O₃   ∂O₂
      + ── · ── · ─── · ────                                 (via O₂)
        ∂ŷ   ∂O₃  ∂O₂   ∂Wₕ

        ∂L   ∂ŷ   ∂O₃   ∂O₂   ∂O₁
      + ── · ── · ─── · ─── · ────                           (via O₁)
        ∂ŷ   ∂O₃  ∂O₂   ∂O₁   ∂Wₕ
```

with the local terms `∂Oₜ/∂Wₕ = f′(aₜ) · Oₜ₋₁`. Using the same errors `eₜ`:

```
∂L
──── = e₃·O₂  +  e₂·O₁  +  e₁·O₀     =   Σₜ  eₜ · Oₜ₋₁
∂Wₕ
```

Each level is **one step's error × the previous memory**.

## 6. The general BPTT rule

For a sequence of length **T**:

```
∂L/∂Wₒ =  (∂L/∂ŷ)·(∂ŷ/∂Wₒ)                         (single term)

∂L/∂Wᵢ =  Σ (t = 1 … T)   eₜ · xᵢₜ                  (T terms — one per time step)

∂L/∂Wₕ =  Σ (t = 1 … T)   eₜ · Oₜ₋₁                 (T terms)

   where   eₜ = (∂L/∂Oₜ) · f′(aₜ)     and     ∂L/∂Oₜ = eₜ₊₁ · Wₕ   (recursion)
```

The recursion `∂L/∂Oₜ = eₜ₊₁·Wₕ` is why the error is "carried backward through
time" — the name BPTT.

## 7. Why BPTT gives vanishing / exploding gradients

Look at **level 3** (the earliest step): it contains the product

```
f′(a₃)·Wₕ  ×  f′(a₂)·Wₕ   →   two Wₕ factors and two f′ factors
```

For a length-T sequence, the earliest step carries **(T−1) copies** of `f′·Wₕ`
multiplied together:

- If `|f′·Wₕ| < 1` repeatedly → the product shrinks toward 0 → **vanishing
  gradient**: early steps barely learn, so the RNN **forgets long-range context**.
- If `|f′·Wₕ| > 1` repeatedly → the product blows up → **exploding gradient**
  (fixed with gradient clipping).

> In the verified example, the **t=1 gradient term was ~100× smaller** than the
> t=3 term — the vanishing effect visible over just 3 steps. Over 100 steps it is
> catastrophic. This is the exact problem **LSTM** and **GRU** were invented to
> solve (their gated cell state creates a path where the gradient can flow without
> repeated shrinking).

## 8. Practical note — Truncated BPTT

For long sequences, unrolling all T steps is expensive and unstable. In practice
libraries use **Truncated BPTT**: run forward over the whole sequence but only
back-propagate through the last *k* steps (a sliding window). Keras' RNN layers
handle all of this internally when you call `model.fit(...)`.

## Key takeaways

- RNN weights are **shared across time**, so each weight's gradient is a **sum
  over all time steps** — 3 words ⇒ **3 levels of differentiation**.
- **∂L/∂Wₒ**: single chain (used once). **∂L/∂Wᵢ** and **∂L/∂Wₕ**: three summed
  chains, each reaching one step further back through extra `∂Oₜ/∂Oₜ₋₁ = f′·Wₕ`
  factors.
- Collapsed form: `∂L/∂Wᵢ = Σ eₜ·xᵢₜ`, `∂L/∂Wₕ = Σ eₜ·Oₜ₋₁`, with the error
  `eₜ` carried backward by `∂L/∂Oₜ = eₜ₊₁·Wₕ`.
- The repeated `Wₕ·f′` products cause **vanishing / exploding gradients** →
  motivation for **LSTM / GRU**.
- Watch it run in [`bptt-animation.html`](bptt-animation.html).
