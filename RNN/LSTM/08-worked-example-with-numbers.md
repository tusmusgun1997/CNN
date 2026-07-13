# 8. Worked Example — Real Vectors Through Every Gate

Now we push actual numbers through the [six equations](07-the-complete-cell.md),
**two time steps**, so you can see every gate value and watch the memory carry
forward. *(All values verified with NumPy.)*

## Setup

- Hidden/cell size **d = 2**, input size **n = 2** → concatenation `[h, x]` has
  size 4.
- Start empty: **h₀ = [0, 0]**, **C₀ = [0, 0]**.
- Inputs: **x₁ = [1, 0]**, then **x₂ = [0, 1]**.
- Weights (each gate: a 2×4 matrix + a 2-vector bias). Note **b_f = [1, 1]** — the
  [forget-bias trick](03-the-forget-gate.md):

```
W_f = [ 0.3  -0.2   0.4   0.1 ]   b_f = [1, 1]
      [ 0.1   0.5  -0.3   0.2 ]
W_i = [ 0.2   0.1  -0.1   0.3 ]   b_i = [0, 0]
      [-0.4   0.2   0.5  -0.2 ]
W_C = [ 0.5  -0.3   0.2   0.4 ]   b_C = [0, 0]
      [ 0.1   0.6  -0.2   0.3 ]
W_o = [ 0.4   0.2   0.1  -0.3 ]   b_o = [0, 0]
      [-0.2   0.3   0.4   0.1 ]
```

The columns of each matrix line up with `[h₁, h₂, x₁, x₂]`.

## ⏱ Time step t = 1 — input x₁ = [1, 0]

Concatenate: **z = [h₀, x₁] = [0, 0, 1, 0]**.

```
① forget:    f₁ = σ(W_f·z + b_f)
                = σ([0.4, -0.3] + [1, 1]) = σ([1.4, 0.7])   = [0.8022, 0.6682]
② input:     i₁ = σ(W_i·z + b_i)
                = σ([-0.1, 0.5])                             = [0.4750, 0.6225]
③ candidate: C̃₁ = tanh(W_C·z + b_C)
                = tanh([0.2, -0.2])                          = [0.1974, -0.1974]
④ cell:      C₁ = f₁ ⊙ C₀ + i₁ ⊙ C̃₁
                = [0,0]     + [0.0938, -0.1229]              = [0.0938, -0.1229]
⑤ output:    o₁ = σ(W_o·z + b_o)
                = σ([0.1, 0.4])                              = [0.5250, 0.5987]
⑥ hidden:    h₁ = o₁ ⊙ tanh(C₁)
                = [0.525, 0.5987] ⊙ [0.0935, -0.1222]        = [0.0491, -0.0732]
```

**Read the story of step 1:**

- `f₁ ≈ [0.80, 0.67]` — high (thanks to bias 1): "keep most memory" — but memory
  was empty, so nothing to keep yet.
- `i₁ ≈ [0.48, 0.62]` — moderate: write about half of the candidate.
- `C̃₁ = [0.20, -0.20]` — the proposed content (one slot +, one slot −).
- **`C₁ = [0.094, -0.123]`** — first real memory written.
- `h₁ = [0.049, -0.073]` — the output, a filtered view of that memory.

## ⏱ Time step t = 2 — input x₂ = [0, 1]

Now the previous states feed back: **z = [h₁, x₂] = [0.0491, -0.0732, 0, 1]**.

```
① forget:    f₂ = σ(W_f·z + b_f)                    = [0.7557, 0.7628]
② input:     i₂ = σ(W_i·z + b_i)                    = [0.5751, 0.4417]
③ candidate: C̃₂ = tanh(W_C·z + b_C)                = [0.4190,  0.2552]
④ cell:      C₂ = f₂ ⊙ C₁          + i₂ ⊙ C̃₂
                = [0.0709, -0.0937] + [0.2410, 0.1127]
                = [0.3118,  0.0190]
⑤ output:    o₂ = σ(W_o·z + b_o)                    = [0.4268, 0.5171]
⑥ hidden:    h₂ = o₂ ⊙ tanh(C₂)
                = [0.4268, 0.5171] ⊙ [0.3021, 0.0190]        = [0.1289, 0.0098]
```

**Read the story of step 2 — this is the whole point of LSTM:**

- `f₂ ≈ [0.76, 0.76]` — the forget gate **keeps ~76% of the old memory C₁**. Look
  at line ④: `f₂ ⊙ C₁ = [0.0709, -0.0937]` — the memory from step 1 **survived
  into step 2**, just scaled a bit. That's the conveyor belt carrying information
  forward.
- `i₂ ⊙ C̃₂ = [0.241, 0.113]` — new content from x₂ is **added** on top.
- **`C₂ = [0.312, 0.019]`** = kept-old **+** new — the additive update in action.

## The summary table

| t | x_t | f_t | i_t | C̃_t | **C_t (memory)** | o_t | **h_t (output)** |
|---|-----|-----|-----|-----|------------------|-----|------------------|
| 1 | [1,0] | [0.80, 0.67] | [0.48, 0.62] | [0.20, −0.20] | **[0.094, −0.123]** | [0.53, 0.60] | **[0.049, −0.073]** |
| 2 | [0,1] | [0.76, 0.76] | [0.58, 0.44] | [0.42, 0.26] | **[0.312, 0.019]** | [0.43, 0.52] | **[0.129, 0.010]** |

## What to notice

1. **Gates are fractions in (0,1)**; **candidate is in (−1,1)** — exactly as the
   activations promise.
2. **Memory carried forward:** `f₂ ⊙ C₁` shows step-1 memory persisting into
   step 2 — not overwritten, just gently scaled.
3. **Additive growth:** `C₂ = f₂⊙C₁ + i₂⊙C̃₂` — old + new, the anti-vanishing
   update.
4. **Output ≠ memory:** `h_t = o_t ⊙ tanh(C_t)` — the output is a *filtered view*
   of the memory, not the memory itself.

> Reproduce these exact numbers by running the tiny NumPy script this file was
> generated from (in the course scratch notes) — every value here matched it.

## Key takeaways

- Each step runs the six equations on `z = [h_(t-1), x_t]`, producing gate
  fractions, a candidate, the new memory `C_t`, and the output `h_t`.
- The forget gate (~0.76) **carries C₁ into C₂**; the input term **adds** new
  content — you can literally see memory persist and accumulate.
- Next: *why* this additive memory path defeats the
  [vanishing gradient](09-why-lstm-solves-vanishing-gradient.md).
