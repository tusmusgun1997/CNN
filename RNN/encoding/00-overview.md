# Text Encoding — Overview

Neural networks can't read words — they only eat **numbers**. **Encoding** is the
step that turns raw text into numeric tensors an RNN (or any model) can consume.
This folder covers the four standard encodings, each in its own sub-folder with an
explanation (`.md`) and a runnable Keras implementation (`.py`) — **encoding only,
no model training**.

All four use the same running example from the [RNN overview](../overview/rnn-overview.md):

```
R1: "movie was great"       → y = 1 (positive)
R2: "movie was boring"      → y = 0 (negative)
R3: "movie was not great"   → y = 0 (negative)

Vocabulary (5 words): movie, was, great, not, boring
```

## The four encodings

| # | Folder | Idea | Output per review | Keeps word order? |
|---|--------|------|-------------------|-------------------|
| 1 | [integer encoding](integer%20encoding/integer-encoding.md) | word → index number | sequence of ints, e.g. `[2, 3, 4, 0]` | ✅ |
| 2 | [one hot encoding](one%20hot%20encoding/one-hot-encoding.md) | word → binary vector | matrix, e.g. `4 × 5` | ✅ |
| 3 | [bag of words](bag%20of%20words/bag-of-words.md) | whole review → one count vector | single vector, e.g. `1 × 5` | ❌ |
| 4 | [embedding](embedding/embedding.md) | word → dense **learned** vector | matrix, e.g. `4 × d` | ✅ |

## How they build on each other

```
raw text ──► INTEGER ENCODING (tokenize + index)  ← always the first step
                    │
        ┌───────────┼──────────────┐
        ▼           ▼              ▼
    ONE-HOT     EMBEDDING      BAG OF WORDS
   (sparse,     (dense,        (order thrown
    exact)       learned)       away)
```

- **Integer encoding** is the foundation — every other encoding starts from it.
- **One-hot** is the teaching-friendly version (it's what the
  [animated RNN demo](../overview/rnn-recurrent-layer-animated.html) uses).
- **Embedding** is what real RNNs use in practice.
- **Bag of words** deliberately discards order — fine for simple classifiers,
  wrong for RNNs (that's the point of comparing it).

## Which one for an RNN?

| Situation | Use |
|-----------|-----|
| Tiny teaching example / small vocab | one-hot |
| Any real NLP task | **integer encoding → Embedding layer** |
| Baseline without sequence info | bag of words + dense classifier |

> **Zero padding** shows up in every sequence encoding: reviews have different
> lengths, so shorter ones are padded (with index `0` / the zero vector) to a
> common length — exactly as animated in the
> [recurrent-layer demo](../overview/rnn-recurrent-layer-animated.html).
