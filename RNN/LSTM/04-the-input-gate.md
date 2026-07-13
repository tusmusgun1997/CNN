# 4. The Input Gate (+ Candidate)

**Question it answers:** *"What new information should I write into memory?"*

Writing to memory takes **two parts** working together:

1. a **candidate** `C̃_t` — the *new information* proposed for storage, and
2. the **input gate** `i_t` — a valve deciding *how much* of that candidate to
   actually write.

## The two equations

```
C̃_t = tanh( W_C · [ h_(t-1) , x_t ] + b_C )      ← candidate: WHAT could be added
i_t = σ( W_i · [ h_(t-1) , x_t ] + b_i )          ← input gate: HOW MUCH to add
```

Both read the same concatenated `[h_(t-1), x_t]`, but through **different learned
weights** and **different activations**.

| Symbol | Meaning | Activation | Range | Shape |
|--------|---------|-----------|-------|-------|
| `C̃_t` | candidate new memory ("write proposal") | **tanh** | **(−1, 1)** | (d,) |
| `i_t` | input gate ("how much of it to keep") | **sigmoid** | **(0, 1)** | (d,) |
| `W_C, W_i` | learned weights | — | — | (d, d+n) |
| `b_C, b_i` | learned biases | — | — | (d,) |

## How they're used

They combine by element-wise multiply to form the **actual amount written**
(the addition happens in [file 5](05-updating-the-cell-state.md)):

```
written = i_t ⊙ C̃_t
```

- `C̃_t` says *what* the new content is (can be positive or negative).
- `i_t` scales it: `i = 1` write it fully, `i = 0` write nothing, `i = 0.5`
  write half.

## Why two separate pieces? (the "why")

Splitting "**what**" from "**how much**" is deliberate:

- The **candidate** `C̃_t` proposes content regardless of relevance.
- The **gate** `i_t` independently judges relevance/importance.

This lets the LSTM compute a rich candidate but **hold it back** if the situation
doesn't call for storing it — the network can "have a thought" without committing
it to long-term memory.

## Why tanh for the candidate? (the "why")

- **Range (−1, 1):** the candidate can **add** to a memory slot (positive) **or
  subtract** from it (negative). A sigmoid candidate (0..1) could only ever push
  memory up.
- **Zero-centered:** values around 0 keep the memory stable and gradients healthy.

So the pairing is: **tanh for content** (signed values to add), **sigmoid for the
valve** (a fraction to admit). That contrast — *content vs. amount* — is the
recurring pattern in LSTM.

## Intuition example

Reading a new fact worth storing:

```
"… I grew up in France …"
   candidate C̃_t : encodes "language = French"      (the content)
   input gate i_t : ≈ 1 for the language slots       (yes, store this — it's important)
                    ≈ 0 for irrelevant slots          (don't touch those)
   written = i_t ⊙ C̃_t : "French" gets written into the language slots
```

Later, the [output/prediction](06-the-output-gate.md) can retrieve "French" to
fill the blank — even 50 words later — because the forget gate kept it and this
input gate wrote it.

## Key takeaways

- Writing new memory = **candidate `C̃_t`** (what) × **input gate `i_t`** (how
  much), combined as `i_t ⊙ C̃_t`.
- `C̃_t = tanh(...)` → **(−1,1)** so memory can go **up or down**;
  `i_t = σ(...)` → **(0,1)**, the write valve.
- Separating **content from amount** lets the LSTM compute a candidate yet choose
  not to store it.
- Next: put forget + input together to actually [update the cell state](05-updating-the-cell-state.md).
