# Integer Encoding (Tokenization + Indexing)

**Integer encoding** is the first — and mandatory — step of every text pipeline:
give every word in the vocabulary a **unique integer index**, then rewrite each
review as a **sequence of those integers**.

> Companion code: [`integer_encoding_keras.py`](integer_encoding_keras.py)

## 1. The idea

```
Vocabulary (built from the training texts, sorted by word frequency):
   was → 2     movie → 3     great → 4     not → 5     boring → 6
   (0 is reserved for PAD, 1 for unknown words [UNK])

R1  "movie was great"      →  [3, 2, 4]
R2  "movie was boring"     →  [3, 2, 6]
R3  "movie was not great"  →  [3, 2, 5, 4]
```

> Keras orders the vocabulary by **how often each word appears** (most frequent
> first) — "was"/"movie" appear in all 3 reviews, so they get the lowest indices.

Text becomes numbers, **word order is preserved**, and each review is now a
sequence an RNN can step through.

## 2. The two reserved indices

Keras' `TextVectorization` reserves the first two slots:

| Index | Token | Purpose |
|-------|-------|---------|
| **0** | `''` (PAD) | padding shorter sequences to a common length |
| **1** | `[UNK]` | any word not seen when the vocabulary was built (out-of-vocabulary) |

So a 5-word vocabulary actually produces indices 0–6 (5 words + PAD + UNK).
The **UNK** token matters in the real world: at test time you *will* meet words
that weren't in training data ("film was excellent" → `[1, 3, 1]`).

## 3. Zero padding

Reviews have lengths 3, 3, 4 — but a batch must be one rectangular tensor. So
shorter sequences are padded with **0** up to the longest (or a fixed) length:

```
[3, 2, 4]       →  [3, 2, 4, 0]
[3, 2, 6]       →  [3, 2, 6, 0]
[3, 2, 5, 4]    →  [3, 2, 5, 4]      ← already max length

Batch tensor shape: (3 reviews, 4 time steps)
```

- **post-padding** (zeros at the end) vs **pre-padding** (zeros at the front) —
  both exist; `pad_sequences` defaults to **pre**, `TextVectorization` does **post**.
- Downstream layers can be told to **ignore** the zeros (masking) — see the
  [embedding folder](../embedding/embedding.md).

## 4. Why you can't feed these ints straight into a network

The indices are **labels, not quantities**. `great = 4` and `boring = 5` does
**not** mean boring is "one more than" great, and `was(3) = movie(2) + 1` is
meaningless. A network doing arithmetic on raw indices would learn these fake
relationships. That's why integer encoding is always followed by:

- **one-hot encoding** (each index → its own axis, no fake order), or
- an **Embedding layer** (each index → a learned dense vector).

## 5. Keras API (the encoding part only)

```python
from tensorflow.keras.layers import TextVectorization

vectorizer = TextVectorization(output_mode="int")   # word → integer index
vectorizer.adapt(reviews)          # builds the vocabulary from the texts
encoded = vectorizer(reviews)      # (3, 4) int tensor, zero-padded

# manual padding control:
from tensorflow.keras.utils import pad_sequences
pad_sequences(seqs, maxlen=4, padding="post")
```

Run [`integer_encoding_keras.py`](integer_encoding_keras.py) to see the built
vocabulary, the encoded reviews, the padding, and an out-of-vocabulary example.

## Key takeaways

- Integer encoding = **vocabulary + index per word**; every review becomes a
  sequence of ints. Order is preserved.
- Keras reserves **0 = PAD** and **1 = [UNK]**.
- **Zero padding** makes all sequences the same length (post or pre).
- Raw indices carry **false ordinal meaning** — never feed them directly to
  dense/recurrent math; follow with [one-hot](../one%20hot%20encoding/one-hot-encoding.md)
  or an [embedding](../embedding/embedding.md).
