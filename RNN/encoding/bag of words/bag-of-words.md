# Bag of Words (Multi-Hot / Count / TF-IDF)

**Bag of words (BoW)** collapses a whole review into **one single vector** — it
records *which* words appear (and maybe how often), but **throws the order away**.
Imagine cutting a sentence into individual words and shaking them in a bag: you
know what's inside, but not the sequence.

> Companion code: [`bag_of_words_keras.py`](bag_of_words_keras.py)

## 1. The three flavours

With vocabulary (movie, was, great, not, boring) and review R3
*"movie was not great"*:

| Flavour | Records | R3 vector `(movie, was, great, not, boring)` |
|---------|---------|-----------------------------------------------|
| **multi-hot** (binary) | word present? 1/0 | `[1, 1, 1, 1, 0]` |
| **count** | how many times | `[1, 1, 1, 1, 0]` (all appear once here) |
| **TF-IDF** | count, down-weighted if the word is common in *all* documents | `[0.85, 0.85, 1.0, 1.18, 0]`-ish — rare words ("not") score higher |

Every review — 3 words or 300 — becomes **one vector of length vocab-size**:

```
"movie was great"      →  [1, 1, 1, 0, 0]
"movie was boring"     →  [1, 1, 0, 0, 1]
"movie was not great"  →  [1, 1, 1, 1, 0]

Batch shape: (3 reviews, 5)   ← no time dimension at all!
```

## 2. The fatal flaw: order is gone

Compare these two sentences:

```
"movie was not great"  (negative)     →  [1, 1, 1, 1, 0]
"movie was great, not boring"-style
reorderings with the same words        →  [1, 1, 1, 1, 0]   ← SAME vector!
```

Any two texts with the same words get the **identical** encoding, whatever the
meaning. The "not" can no longer cancel the "great" because the model can't see
that "not" came *before* "great" — the exact interaction the
[animated RNN demo](../../overview/rnn-recurrent-layer-animated.html) showed the
recurrent loop handling.

**This is why BoW is not used as RNN input.** An RNN's whole point is stepping
through time; BoW deletes time. It exists here as the contrast that shows *why*
sequences matter.

## 3. When BoW is actually fine

- **Simple/fast baselines:** BoW + logistic regression or a small dense net is a
  classic, surprisingly strong text classifier.
- **Topic-ish tasks** where keywords alone carry the signal (spam filtering,
  topic tagging).
- **TF-IDF + classical ML** is still everywhere in industry for search/ranking.

## 4. TF-IDF in one paragraph

Counts overweight words that appear everywhere ("movie", "was" appear in all 3
reviews — they discriminate nothing). **TF-IDF** = *term frequency ×
inverse document frequency*: multiply the count by a penalty that shrinks with
how many documents contain the word. Words that appear in **every** review get
pushed toward 0; distinctive words ("not", "boring") stand out.

## 5. Keras API — one layer, three modes

```python
from tensorflow.keras.layers import TextVectorization

TextVectorization(output_mode="multi_hot")  # presence 1/0
TextVectorization(output_mode="count")      # occurrence counts
TextVectorization(output_mode="tf_idf")     # counts × IDF weighting
```

Run [`bag_of_words_keras.py`](bag_of_words_keras.py) to see all three on the
reviews, plus the order-blindness demonstration.

## Key takeaways

- BoW: whole document → **one vector** (multi-hot / count / TF-IDF); shape
  `(reviews, vocab)` — the time dimension disappears.
- **Word order is destroyed** — "not great" is indistinguishable from any
  shuffle of the same words → unsuitable as RNN input.
- Great cheap **baseline** for keyword-driven tasks; **TF-IDF** highlights
  distinctive words.
- For sequence models, use [integer encoding](../integer%20encoding/integer-encoding.md)
  → [embedding](../embedding/embedding.md) instead.
