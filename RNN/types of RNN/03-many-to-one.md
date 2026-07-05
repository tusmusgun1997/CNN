# 3. Many-to-One

**A sequence of inputs → one output.** The network reads the whole sequence step
by step, accumulating everything into its memory, and only **at the end** produces
a single verdict.

This is the architecture used throughout this course — the
[worked example](../overview/rnn-overview.md#8-full-worked-example) and the
[animated demo](../overview/rnn-recurrent-layer-animated.html) are both
many-to-one.

## The diagram

```
                                    ŷ          ONE output, at the END
                                    ▲
                                  [ g ]        g = sigmoid / softmax
                                    ▲  W_o
      ┌─────┐   ┌─────┐   ┌─────┐   │
      │cell │──►│cell │──►│cell │───┘          intermediate outputs O₁, O₂
      └─────┘   └─────┘   └─────┘              exist but are NOT read out —
         ▲         ▲         ▲                 they only feed the memory
         x₁        x₂        x₃
       "movie"   "was"    "great"
```

## The equations (nothing new — just where you read)

```
O₁ = f(x₁·Wᵢ + O₀·Wₕ)
O₂ = f(x₂·Wᵢ + O₁·Wₕ)
O₃ = f(x₃·Wᵢ + O₂·Wₕ)      ← O₃ has "seen" the entire sequence
ŷ  = g(O₃·W_o)              ← read the LAST memory only
```

The final hidden state is a **summary vector of the whole sequence** — the RNN's
equivalent of "having read the review". Everything before it exists only to build
that summary.

## Use cases

| Task | The many inputs | The one output |
|------|-----------------|----------------|
| **Sentiment analysis** | review, word by word | positive / negative |
| **Spam detection** | email, word by word | spam / not spam |
| **Topic classification** | article, word by word | topic label |
| **Speaker / language ID** | audio frames | who / which language |
| **Time-series forecasting (next value)** | last N sensor readings | the next value |
| **Fraud detection** | a sequence of transactions | fraudulent / normal |

The common shape: *lots of ordered evidence → one decision*.

## Why order matters here (the "not great" test)

A [bag-of-words](../encoding/bag%20of%20words/bag-of-words.md) model sees
identical vectors for "movie was not great" and any shuffle of those words.
Many-to-one RNN doesn't: the "not" at t=3 changes the memory that greets "great"
at t=4, flipping the verdict — exactly what Review 3 shows in the
[animated demo](../overview/rnn-recurrent-layer-animated.html) (ŷ ≈ 0.47).

## Keras sketch

```python
model = Sequential([
    Embedding(vocab_size, 64, mask_zero=True),   # see encoding/embedding
    SimpleRNN(32, return_sequences=False),       # False = emit ONLY the last O
    Dense(1, activation="sigmoid"),              # the one output
])
```

`return_sequences=False` (the default) **is** the many-to-one switch: the layer
throws away O₁…O_{T−1} and hands only O_T to the next layer.

## Key takeaways

- Many-to-one = **read everything, answer once**; the last hidden state is a
  learned **summary of the whole sequence**.
- Poster use case: **sentiment analysis** (this course's running example).
- Keras: `return_sequences=False`.
- Beats bag-of-words precisely because it **remembers order** ("not great").
- Next: outputs at every step — [many-to-many](04-many-to-many.md).
