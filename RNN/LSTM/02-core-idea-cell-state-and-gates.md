# 2. The Core Idea: Cell State + Gates

This is the most important page in the course. Get this picture and every
equation later is just filling in details.

## Two states, not one

A vanilla RNN carries **one** vector between steps (`Oₜ`). An LSTM carries **two**:

| State | Name | Role | Analogy |
|-------|------|------|---------|
| **C_t** | **cell state** | long-term memory | the **conveyor belt** running through the whole sequence |
| **h_t** | **hidden state** | short-term / output | what you **read off** the belt right now |

The cell state `C_t` is the star. It runs straight along the top of the cell,
with only **tiny, controlled edits** at each step:

```
        C_(t-1) ───────►(× f)───►(+ )────────────► C_t     ← the conveyor belt
   (old memory)          ▲         ▲                          (mostly flows straight through)
                       forget    add new
                        gate      info
```

Because the belt is edited by a **multiply** (to erase) and an **add** (to write)
— *not* by being squashed through a nonlinearity every step — information can ride
it for a long time untouched. **This straight, mostly-additive path is what beats
the vanishing gradient** (full "why" in [file 9](09-why-lstm-solves-vanishing-gradient.md)).

## Gates: learnable valves

A **gate** is a small neural layer that outputs numbers **between 0 and 1**, one
per memory slot, and then **multiplies** them into a vector element-wise:

```
gate value 0.0  →  "let NOTHING through"   (close the valve)
gate value 1.0  →  "let EVERYTHING through" (open the valve)
gate value 0.7  →  "let 70% through"
```

The 0–1 range comes from a **sigmoid** — that's *why* gates use sigmoid: a sigmoid
is a smooth, differentiable "how much to open the valve" knob. Multiplying a
memory vector by a gate vector **selectively keeps or discards each element**.

> A gate is literally: `gate = σ(W·[inputs] + b)`, then `gate ⊙ something`.
> The σ makes the valve; the `⊙` applies it element-by-element.

## The three gates — what each one decides

An LSTM has **three** gates, each answering one question about the conveyor belt:

```
                         ┌──────────────── C_(t-1) (incoming memory)
                         ▼
   1) FORGET gate  f_t : "what should we ERASE from the old memory?"
                         │        C_(t-1) ⊙ f_t
                         ▼
   2) INPUT gate   i_t : "what NEW info should we WRITE to memory?"
                         │        + i_t ⊙ C̃_t
                         ▼
                        C_t  (updated memory continues down the belt)
                         │
   3) OUTPUT gate  o_t : "what should we READ OUT of memory right now?"
                         ▼
                        h_t = o_t ⊙ tanh(C_t)   (the output)
```

| Gate | Question | Controls |
|------|----------|----------|
| **Forget** `f_t` | *What to throw away?* | how much of `C_(t-1)` survives |
| **Input** `i_t` | *What to store?* | how much of the new candidate is added |
| **Output** `o_t` | *What to reveal?* | how much of `C_t` becomes the output `h_t` |

That's it — **erase, write, read**, each with its own learnable valve.

## Why this design is powerful

- **Selective memory:** each element of `C` can be kept for 100 steps (gate ≈ 1)
  or wiped instantly (gate ≈ 0) — *independently*. The vanilla RNN couldn't do
  "keep this, drop that."
- **Separation of concerns:** `C_t` = what I *remember*; `h_t` = what I *say
  right now*. You can hold something in memory without outputting it.
- **Learned control:** the gates are trained like any other weights, so the
  network *learns* what's worth remembering for the task.

## The plan from here

The next four files are just these boxes, one at a time, with their exact math:

- [Forget gate](03-the-forget-gate.md) → [Input gate](04-the-input-gate.md) →
  [Cell update](05-updating-the-cell-state.md) → [Output gate](06-the-output-gate.md).

## Key takeaways

- LSTM has **two** states: **C_t** (long-term memory, the conveyor belt) and
  **h_t** (short-term output).
- The belt is edited only by a **forget-multiply** and an **input-add**, so
  memory can persist — the anti-vanishing trick.
- A **gate** = `σ(...)` → a 0–1 valve applied by element-wise multiply. Sigmoid
  is used *because* it gives a smooth "how much to open" value.
- Three gates: **forget** (erase), **input** (write), **output** (read).
