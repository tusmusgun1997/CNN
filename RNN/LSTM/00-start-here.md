# LSTM — Start Here (Learning Path)

**LSTM (Long Short-Term Memory)** is the upgrade to the vanilla RNN that finally
lets a network **remember things over long sequences**. This folder is a
**study-in-order course**: read the files by number, each one builds on the last.

## Why you're ready for this

You just finished [RNN backpropagation through time](../back%20propogation/backpropagation-through-time.md),
which ended on a problem: the repeated `f′·Wₕ` factors make gradients **vanish**,
so a vanilla RNN forgets the distant past. LSTM is the fix. Everything here is
motivated by that single problem.

## The sequence — study in this order

| # | File | What you'll get |
|---|------|-----------------|
| 0b | [Predicting a word vs. classifying](00b-predicting-next-word-vs-classifying.md) | **Read first** — how "predict the next word" is still classification (labels come free) |
| 1 | [Why LSTM? The problem](01-why-lstm.md) | The long-term-dependency problem, concretely |
| 2 | [Core idea: cell state + gates](02-core-idea-cell-state-and-gates.md) | The one big intuition (a memory conveyor belt + valves) |
| 3 | [The Forget gate](03-the-forget-gate.md) | What to erase from memory — math + why |
| 4 | [The Input gate](04-the-input-gate.md) | What new info to write — math + why |
| 5 | [Updating the cell state](05-updating-the-cell-state.md) | Combining forget + input — the heart of LSTM |
| 6 | [The Output gate](06-the-output-gate.md) | What to expose as the output — math + why |
| 7 | [The complete cell](07-the-complete-cell.md) | All 6 equations, shapes, vectors, parameter count |
| 8 | [Worked example with numbers](08-worked-example-with-numbers.md) | Real vectors flowing through every gate, 2 time steps |
| 9 | [Why LSTM beats vanishing gradients](09-why-lstm-solves-vanishing-gradient.md) | The deep "why" — the gradient highway |
| 10 | [Variants, Keras & summary](10-variants-keras-and-summary.md) | GRU, bidirectional, the Keras API, cheat sheet |

## How to read it

- **If you're wondering "how does predicting the next word even work as a task?"**
  read [0b](00b-predicting-next-word-vs-classifying.md) first — it shows next-word
  prediction is just classification over the vocabulary with free labels.
- **First pass:** files 1 → 2 for pure intuition, no heavy math. Stop and make
  sure the "conveyor belt" picture makes sense.
- **Second pass:** files 3 → 7, one gate at a time. Each gate is a small,
  self-contained idea.
- **Third pass:** file 8 with a calculator/eyes on the numbers, then 9 for the
  payoff (why it all works), then 10 to connect to real code.

## The 30-second summary

> A vanilla RNN has one state that gets rewritten every step, so old information
> is quickly overwritten. An LSTM adds a second state — the **cell state** — that
> flows down the sequence almost untouched, like a **conveyor belt**. Three
> **gates** (small neural valves) decide what to **erase** from the belt, what to
> **add** to it, and what to **read off** it. Because the belt is updated by
> **addition** (not by squashing through a nonlinearity every step), gradients
> can flow across many steps without vanishing — so the network remembers.

Notation used throughout (defined fully in [file 7](07-the-complete-cell.md)):

| Symbol | Meaning |
|--------|---------|
| `x_t` | input at time t |
| `h_t` | hidden state / output at time t (short-term) |
| `C_t` | **cell state** at time t (long-term memory) |
| `f_t, i_t, o_t` | forget / input / output gates (values in 0–1) |
| `C̃_t` | candidate new memory (values in −1…1) |
| `σ` | sigmoid · `⊙` | element-wise multiply |

Turn to [file 1 →](01-why-lstm.md)
