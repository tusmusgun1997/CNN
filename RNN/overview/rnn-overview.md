# RNN — Overview & Forward Propagation (Recurrent Layer, Step by Step)

A **Recurrent Neural Network (RNN)** is a neural network built for **sequences** —
text, speech, time-series, video — where the *order* of the data matters and each
item depends on what came before. This overview explains what an RNN is, why it
exists, its architecture, and works out the **forward pass of a recurrent layer
one time-step at a time** with real numbers.

> Notation here matches the standard "simplified representation":
> **Oₜ = f(xₜ·Wᵢ + Oₜ₋₁·Wₕ)** and **ŷ = g(Oₜ·W_o)**.

---

## Table of contents
1. [Why RNNs? (the problem with feed-forward nets)](#1-why-rnns)
2. [Sequence data & the core idea: memory](#2-sequence-data--the-core-idea-memory)
3. [The recurrent cell — simplified representation](#3-the-recurrent-cell)
4. [Notation & symbols](#4-notation--symbols)
5. [Forward-propagation equations](#5-forward-propagation-equations)
6. [Shapes & parameters](#6-shapes--parameters)
7. [Unfolding through time](#7-unfolding-through-time)
8. [Full worked example — every step of a recurrent layer](#8-full-worked-example)
9. [Counting the parameters](#9-counting-the-parameters)
10. [RNN architectures (one-to-many, many-to-one, …)](#10-rnn-architectures)
11. [Activations](#11-activations)
12. [Limitations → LSTM / GRU](#12-limitations)
13. [RNN vs CNN vs feed-forward](#13-rnn-vs-cnn-vs-feed-forward)
14. [Key takeaways](#14-key-takeaways)

---

## 1. Why RNNs?

A normal feed-forward network (and a CNN) takes a **fixed-size input** and treats
every input independently. That breaks for language:

- Sentences have **different lengths** ("good" vs "not good at all").
- **Order matters**: "the movie was not good" ≠ "good, the movie was not".
- A word's meaning depends on **previous words** (context).

An RNN solves this by processing the sequence **one item at a time**, while
carrying a **memory** of everything it has seen so far. The same small network is
reused at every step — so it handles any length and shares what it learns across
all positions.

## 2. Sequence data & the core idea: memory

Take **sentiment analysis**: read a review, output positive/negative.

```
Review:  "movie  was  great"
Tokens:   x₁     x₂    x₃        ← fed in ONE BY ONE (t = 1, 2, 3)
Label:    positive (1)
```

Each word is first turned into a **vector** (e.g. one-hot or an embedding). The
RNN reads x₁, then x₂, then x₃, updating an internal **hidden state (memory)**
each time. The final memory summarizes the whole sentence and is used to predict
the sentiment.

The one idea that defines an RNN:

> **The output of the cell at time t is fed back in as an input at time t+1.**
> That feedback loop is the network's memory.

## 3. The recurrent cell

Here is the cell that is reused at every time step (the "simplified
representation"):

```
                              ŷ  (final output, e.g. sentiment)
                              ▲
                            [ g ]        g = sigmoid / softmax
                              ▲
                              │  W_o
                              │
        Oₜ₋₁ ──►(× Wₕ)──►┌────┴────┐
       (memory in)       │   (+)   │──► Oₜ  (memory out → next step & up)
        xₜ  ──►(× Wᵢ)──► │  then f │
       (input in)        └─────────┘        f = tanh / ReLU
```

Two things flow **into** the cell and get combined:

1. the **current input** xₜ, scaled by input weights **Wᵢ**, and
2. the **previous memory** Oₜ₋₁, scaled by recurrent weights **Wₕ**.

They are **added**, passed through an activation **f**, and produce the new
memory **Oₜ**. Optionally Oₜ is passed through output weights **W_o** and a second
activation **g** to produce the prediction **ŷ**.

## 4. Notation & symbols

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| **xₜ** (a.k.a. xᵢₜ) | input vector at time step *t* | (1, d) — d = input size |
| **Oₜ** | cell output / **hidden state / memory** at step *t* (some books call it hₜ) | (1, h) — h = hidden size |
| **O₀** | initial memory (usually all zeros) | (1, h) |
| **Wᵢ** | input weights (input → hidden) | (d, h) |
| **Wₕ** | recurrent weights (hidden → hidden) | (h, h) |
| **W_o** | output weights (hidden → output) | (h, o) — o = output size |
| **bₕ, b_o** | biases | (1, h), (1, o) |
| **f** | hidden activation | tanh or ReLU |
| **g** | output activation | sigmoid or softmax |
| **i** | row # / which feature | — |
| **t** | time step | 1, 2, 3, … |

> Reading the subscripts: **xᵢₜ** = the *i*-th input row at time step *t*. In the
> worked example we drop *i* and just write xₜ for the vector at step *t*.

## 5. Forward-propagation equations

The whole recurrent layer is just these two equations, applied at every step:

```
   Hidden state (memory):   Oₜ = f( xₜ · Wᵢ  +  Oₜ₋₁ · Wₕ  +  bₕ )
   Output (optional):       ŷₜ = g( Oₜ · W_o  +  b_o )
```

Key points:

- **Same Wᵢ, Wₕ, W_o at every time step** — this weight-sharing is what lets one
  small network handle a sequence of any length (analogous to how a CNN reuses one
  filter across all positions of an image).
- Oₜ depends on Oₜ₋₁, which depends on Oₜ₋₂, … all the way back to O₀. So Oₜ
  carries information from **every earlier step** — the memory.
- For a "many-to-one" task (like sentiment), you often only take the **last**
  output: ŷ = g(O_T · W_o).

## 6. Shapes & parameters

Using the review example from the notes — vocabulary/embedding size **d = 5**,
hidden size **h = 3**:

```
xₜ        : (1, 5)        one word as a 5-dim vector
Wᵢ        : (5, 3)        (input 5  →  hidden 3)
Oₜ₋₁, Oₜ  : (1, 3)
Wₕ        : (3, 3)        (hidden 3 →  hidden 3)
W_o       : (3, o)        (hidden 3 →  output o)
```

Check the matrix multiply shapes line up:

```
xₜ · Wᵢ     = (1,5)·(5,3) = (1,3)   ✔
Oₜ₋₁ · Wₕ   = (1,3)·(3,3) = (1,3)   ✔
add them    → (1,3)  → tanh → Oₜ (1,3)   ✔
```

Both terms produce a **(1,3)** vector, so they can be added — that's why Wᵢ is
(d,h) and Wₕ is (h,h).

## 7. Unfolding through time

Although it is **one** cell with a loop, we "unfold" it to see the sequence — one
copy per time step, all sharing the same weights:

```
   t=1              t=2              t=3
  ┌─────┐  O₁      ┌─────┐  O₂      ┌─────┐  O₃
  │cell │ ───────► │cell │ ───────► │cell │ ───────►  ŷ = g(O₃·W_o)
  └─────┘   Wₕ     └─────┘   Wₕ     └─────┘
     ▲ Wᵢ             ▲ Wᵢ             ▲ Wᵢ
     │                │                │
    x₁ (t=1)         x₂ (t=2)         x₃ (t=3)
```

- Memory flows **left → right** through Wₕ.
- Each input enters **bottom → up** through Wᵢ.
- **Wᵢ, Wₕ are identical in every box** (shared).

## 8. Full worked example

Let's compute a whole recurrent layer **by hand**, step by step. Small numbers so
you can follow every operation. *(These values are verified against NumPy.)*

### Setup

- Vocabulary of 3 words → **one-hot, d = 3**. Hidden size **h = 2**. Output **o = 1**.
- Sequence (3 words): x₁ = [1,0,0], x₂ = [0,1,0], x₃ = [0,0,1]
- Biases = 0 (for a clean example; in general there is a bias term).

```
Wᵢ (3×2) = [ 0.5  -0.3 ]      Wₕ (2×2) = [ 0.3   0.1 ]      W_o (2×1) = [ 0.6 ]
           [ 0.1   0.4 ]                 [-0.2   0.5 ]                  [-0.4 ]
           [-0.2   0.2 ]
```

Initial memory: **O₀ = [0, 0]**.

> 💡 Because the inputs are **one-hot**, `xₜ · Wᵢ` simply **selects one row of Wᵢ**.
> e.g. x₁ = [1,0,0] picks row 0 of Wᵢ = [0.5, −0.3].

### ⏱ Time step t = 1 — input x₁ = [1,0,0]

```
1) input term:   x₁ · Wᵢ  = row 0 of Wᵢ           = [ 0.5, -0.3 ]
2) memory term:  O₀ · Wₕ  = [0,0]·Wₕ              = [ 0.0,  0.0 ]
3) add:          a₁ = x₁Wᵢ + O₀Wₕ                 = [ 0.5, -0.3 ]
4) activate:     O₁ = tanh(a₁) = [tanh(0.5), tanh(-0.3)]
                    = [ 0.4621, -0.2913 ]
```

**O₁ = [0.4621, −0.2913]**  ← the memory after reading word 1.

### ⏱ Time step t = 2 — input x₂ = [0,1,0]

```
1) input term:   x₂ · Wᵢ  = row 1 of Wᵢ           = [ 0.1,  0.4 ]
2) memory term:  O₁ · Wₕ  = [0.4621, -0.2913]·Wₕ
     component 1 = 0.4621·0.3 + (-0.2913)·(-0.2)  =  0.1386 + 0.0583 =  0.1969
     component 2 = 0.4621·0.1 + (-0.2913)·( 0.5)  =  0.0462 - 0.1457 = -0.0994
                = [ 0.1969, -0.0994 ]
3) add:          a₂ = [0.1+0.1969, 0.4-0.0994]    = [ 0.2969,  0.3006 ]
4) activate:     O₂ = tanh(a₂)                    = [ 0.2885,  0.2918 ]
```

**O₂ = [0.2885, 0.2918]**  ← memory now blends word 1 (via O₁) **and** word 2.

### ⏱ Time step t = 3 — input x₃ = [0,0,1]

```
1) input term:   x₃ · Wᵢ  = row 2 of Wᵢ           = [-0.2,  0.2 ]
2) memory term:  O₂ · Wₕ  = [0.2885, 0.2918]·Wₕ
     component 1 = 0.2885·0.3 + 0.2918·(-0.2)     =  0.0865 - 0.0583 =  0.0282
     component 2 = 0.2885·0.1 + 0.2918·( 0.5)     =  0.0288 + 0.1459 =  0.1747
                = [ 0.0282,  0.1748 ]
3) add:          a₃ = [-0.2+0.0282, 0.2+0.1748]   = [-0.1718,  0.3748 ]
4) activate:     O₃ = tanh(a₃)                    = [-0.1702,  0.3581 ]
```

**O₃ = [−0.1702, 0.3581]**  ← final memory: a summary of the whole 3-word sequence.

### 🎯 Output layer (many-to-one)

Use the **last** memory O₃ to make the prediction:

```
z  = O₃ · W_o = (-0.1702)(0.6) + (0.3581)(-0.4)
             = -0.1021 - 0.1432 = -0.2453
ŷ  = sigmoid(z) = 1 / (1 + e^0.2453) = 0.439
```

**ŷ ≈ 0.44** → the model leans slightly "negative" (below 0.5). During training,
backprop-through-time would adjust Wᵢ, Wₕ, W_o to push this toward the true label.

### The whole layer at a glance

| t | input xₜ | xₜ·Wᵢ | Oₜ₋₁·Wₕ | aₜ (sum) | **Oₜ = tanh(aₜ)** |
|---|----------|-------|---------|----------|-------------------|
| 1 | [1,0,0] | [0.5, −0.3] | [0, 0] | [0.5, −0.3] | **[0.4621, −0.2913]** |
| 2 | [0,1,0] | [0.1, 0.4] | [0.1969, −0.0994] | [0.2969, 0.3006] | **[0.2885, 0.2918]** |
| 3 | [0,0,1] | [−0.2, 0.2] | [0.0282, 0.1748] | [−0.1718, 0.3748] | **[−0.1702, 0.3581]** |

Then ŷ = sigmoid(O₃·W_o) = **0.44**.

## 9. Counting the parameters

An RNN's parameters do **not** grow with sequence length (weights are shared). For
input size *d*, hidden size *h*, output size *o*:

```
Wᵢ : d·h        Wₕ : h·h        bₕ : h
W_o: h·o        b_o: o

Total = d·h + h·h + h + h·o + o
```

For the worked example (d=3, h=2, o=1):

```
Wᵢ = 3·2 = 6   |   Wₕ = 2·2 = 4   |   W_o = 2·1 = 2   →  12 weights (+3 biases = 15)
```

A 100-word review uses the **same** 15 numbers as a 3-word one — that's the power
of weight sharing across time.

## 10. RNN architectures

By choosing where inputs enter and outputs come out, the same cell covers many
tasks:

| Type | Shape | Example |
|------|-------|---------|
| **One-to-one** | 1 in → 1 out | plain classification (barely "recurrent") |
| **One-to-many** | 1 in → sequence out | image → caption |
| **Many-to-one** | sequence in → 1 out | **sentiment analysis** (our example) |
| **Many-to-many (aligned)** | seq → seq, same length | part-of-speech tagging |
| **Many-to-many (seq2seq)** | seq → seq, different length | machine translation |

```
one-to-many        many-to-one         many-to-many
   □                □ □ □               □ □ □
   ↓ ↘ ↘            ↘ ↓ ↙               ↓ ↓ ↓
   □ □ □              □                 □ □ □
```

## 11. Activations

- **Hidden activation f:** usually **tanh** (keeps values in [−1, 1], centered) —
  the notes also mention **ReLU**. tanh is the classic default for vanilla RNNs.
- **Output activation g:**
  - **sigmoid** → one probability (binary tasks like sentiment),
  - **softmax** → a probability distribution (multi-class / next-word prediction).

## 12. Limitations → LSTM / GRU

Vanilla RNNs struggle with **long sequences**:

- **Vanishing gradients:** signals shrink as they propagate back through many time
  steps (repeated multiplication by small numbers), so the network **forgets**
  distant context.
- **Exploding gradients:** the opposite — values blow up (fixed with gradient
  clipping).

The fix is gated cells that learn what to keep and forget:

- **LSTM** (Long Short-Term Memory) — adds a **cell state** and input/forget/output
  gates.
- **GRU** (Gated Recurrent Unit) — a simpler, faster variant with reset/update
  gates.

These keep the same "read one step at a time, carry memory" idea, just with a
smarter memory. *(Topics for later folders.)*

## 13. RNN vs CNN vs feed-forward

| | Feed-forward / MLP | CNN | **RNN** |
|---|--------------------|-----|---------|
| Built for | fixed vectors | grids (images) | **sequences** |
| Key trick | full connections | **spatial** weight sharing (filters) | **temporal** weight sharing (across time) |
| Handles variable length? | ❌ | ❌ (fixed input) | ✅ |
| Has memory of order? | ❌ | ❌ | ✅ (hidden state) |
| Shares weights over | — | positions in space | **positions in time** |

Notice CNNs and RNNs share the *same* deep idea — **reuse one small set of weights
across the whole input** — CNNs across space, RNNs across time.

## 14. Key takeaways

- An RNN processes a **sequence one step at a time**, carrying a **hidden state
  (memory) Oₜ**.
- Core equations: **Oₜ = f(xₜ·Wᵢ + Oₜ₋₁·Wₕ + bₕ)** and **ŷ = g(Oₜ·W_o + b_o)**.
- The **same Wᵢ, Wₕ, W_o are reused at every time step** (weight sharing across
  time), so parameter count is independent of sequence length.
- The two terms `xₜ·Wᵢ` and `Oₜ₋₁·Wₕ` are both **(1, h)**, added, then activated —
  see the [worked example](#8-full-worked-example) computing O₁→O₂→O₃ and ŷ.
- **tanh** for the hidden state, **sigmoid/softmax** for the output.
- Vanilla RNNs forget long-range context (**vanishing gradients**) → **LSTM/GRU**.
- RNNs are the **temporal cousin** of CNNs — both are built on weight sharing.

---

### Related in this repo
- [rnn-recurrent-layer-animated.html](rnn-recurrent-layer-animated.html) — **animated companion**: watch the feedback loop run on 3 zero-padded reviews
- [../../CNN/basics/01-what-is-a-cnn.md](../../CNN/basics/01-what-is-a-cnn.md) — weight sharing in CNNs (the spatial analogue)
- [../../CNN/backpropagation/backpropagation.html](../../CNN/backpropagation/backpropagation.html) — chain-rule backprop (RNN training uses "backprop through time")
