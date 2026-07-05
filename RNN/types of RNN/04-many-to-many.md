# 4. Many-to-Many

**A sequence of inputs → a sequence of outputs.** The most general type — and it
comes in **two importantly different variants**, depending on whether input and
output lengths match.

## Variant A — aligned (same length, output at every step)

Every input step immediately produces an output step: T in → T out, position by
position.

```
        y₁        y₂        y₃          one output PER input step
        ▲         ▲         ▲
     ┌─────┐   ┌─────┐   ┌─────┐
     │cell │──►│cell │──►│cell │        ŷₜ = g(Oₜ·W_o)  at every t
     └─────┘   └─────┘   └─────┘
        ▲         ▲         ▲
        x₁        x₂        x₃
      "time"   "flies"   "fast"
        ↓         ↓         ↓
       NOUN      VERB      ADV          e.g. part-of-speech tagging
```

Equations — same cell, but now **read the output at every step**:

```
Oₜ = f(xₜ·Wᵢ + Oₜ₋₁·Wₕ)         (unchanged)
ŷₜ = g(Oₜ·W_o)                   for EVERY t, not just the last
```

### Use cases (aligned)

| Task | Input step | Output step |
|------|-----------|-------------|
| **Part-of-speech tagging** | word | its POS tag (noun/verb/…) |
| **Named-entity recognition (NER)** | word | entity tag (person/place/none) |
| **Per-frame video labeling** | video frame | activity label |
| **Music transcription** | audio frame | note being played |

### Keras sketch (aligned)

```python
model = Sequential([
    Embedding(vocab_size, 64, mask_zero=True),
    SimpleRNN(32, return_sequences=True),           # True = emit Oₜ at EVERY step
    TimeDistributed(Dense(num_tags, activation="softmax")),  # a tag per step
])
```

`return_sequences=True` is the switch; `TimeDistributed` applies the same output
layer independently at each time step.

## Variant B — encoder–decoder / seq2seq (different lengths)

Translation breaks the aligned pattern: "movie was great" (3 words) →
"la película fue genial" (4 words), and word 2 of the output may depend on word 3
of the input. The fix: **two RNNs glued by a single vector**.

```
        ENCODER (many-to-one)          DECODER (one-to-many)
   ┌─────┐  ┌─────┐  ┌─────┐        ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
   │cell │─►│cell │─►│cell │───c───►│cell │─►│cell │─►│cell │─►│cell │
   └─────┘  └─────┘  └─────┘        └─────┘  └─────┘  └─────┘  └─────┘
      ▲        ▲        ▲              ▲        ▲        ▲        ▲
      x₁       x₂       x₃          <START>    ŷ₁       ŷ₂       ŷ₃
   "movie"   "was"   "great"           │        └──feeds──┘ (each output
                                       ▼                     becomes next input)
                              "la  película  fue  genial"  <END>
```

- The **encoder** reads the whole input and compresses it into the final state
  **c** (the *context vector*) — it is literally a [many-to-one](03-many-to-one.md).
- The **decoder** starts from c and generates the output sequence — it is
  literally a [one-to-many](02-one-to-many.md), feeding each produced word back
  in, until it emits `<END>`.

> **Many-to-many (B) = many-to-one + one-to-many.** The two earlier types are the
> building blocks.

### Use cases (seq2seq)

| Task | Input sequence | Output sequence |
|------|----------------|-----------------|
| **Machine translation** | English sentence | Hindi/Spanish sentence |
| **Text summarization** | long article | short summary |
| **Chatbots / dialogue** | user's message | reply |
| **Speech-to-text** | audio frames | words |
| **Question answering** | question | answer |

### The bottleneck (and where the story goes next)

The entire input must squeeze through the **single vector c** — for long
sentences that's a serious information bottleneck. Fixing it led to
**attention** (let the decoder peek back at *all* encoder states, not just c),
and attention grew into the **Transformer** ("Attention Is All You Need") —
the architecture behind modern LLMs. That story starts exactly here.

## A vs B — quick comparison

| | **A: aligned** | **B: seq2seq** |
|---|----------------|-----------------|
| Lengths | output length = input length | independent lengths |
| Output timing | immediately, every step | only after reading everything |
| Structure | one RNN | two RNNs (encoder + decoder) |
| Example | POS tagging | translation |
| Keras core | `return_sequences=True` + `TimeDistributed` | encoder state → decoder `initial_state` |

## Key takeaways

- Many-to-many **A (aligned)**: T in → T out, `ŷₜ = g(Oₜ·W_o)` at every step;
  Keras `return_sequences=True`. Use for tagging tasks.
- Many-to-many **B (seq2seq)**: encoder (many-to-one) compresses to a context
  vector **c**; decoder (one-to-many) unrolls the answer. Use for translation,
  summarization, chat.
- B's **bottleneck** at c motivated **attention → Transformers**.
- ← Back to the [types overview](00-overview.md).
