# Word Embeddings (the Embedding Layer)

An **embedding** maps each word to a **small, dense vector of real numbers** that
is **learned during training**. It is the encoding every real-world RNN (and
transformer) uses.

> Companion code: [`embedding_keras.py`](embedding_keras.py)

## 1. The idea

Instead of a huge sparse one-hot, each word gets, say, **4 numbers**:

```
              d₁     d₂     d₃     d₄        (d = embedding dimension)
movie   →  [ 0.21, -0.03,  0.14,  0.08 ]
great   →  [ 0.90,  0.75, -0.12,  0.30 ]
boring  →  [-0.88,  0.70, -0.09,  0.25 ]      ← values LEARNED, not designed
```

A review becomes a `(time steps, d)` matrix — same layout as one-hot, but dense
and tiny:

| | one-hot (50k vocab) | embedding (d=128) |
|---|---------------------|--------------------|
| vector per word | 50,000 numbers (one 1) | 128 numbers (all used) |
| review of 200 words | 200 × 50,000 | 200 × 128 |
| similar words close together? | ❌ never | ✅ after training |
| values | fixed 0/1 | **trainable parameters** |

## 2. It's just a lookup table

The Embedding layer is literally a weight matrix **E of shape (vocab_size, d)** —
one row per word. "Encoding" a word = **fetching its row**:

```
            E (7 × 4)                       embedding("great")
  index 0 → [ .01  .02 ... ]  ← PAD row      = row 4 of E
  index 1 → [ ... ]           ← UNK row
  index 2 → [ ... ]           ← movie
  ...
  index 4 → [ ... ]           ← great   ◄────┘
```

Connection to one-hot (from the [one-hot folder](../one%20hot%20encoding/one-hot-encoding.md)):
a one-hot vector times a matrix **selects a row** — so

> **Embedding(x) ≡ one_hot(x) · E** — an embedding layer is one-hot encoding
> followed by a dense layer, implemented as a fast lookup so the giant one-hot
> is never materialised.

## 3. Learned = meaningful geometry

The rows of E start **random** and are updated by backprop like any weight.
Because words used in similar contexts receive similar gradients, training pulls
them together:

```
after training (conceptually):
   great ≈ excellent ≈ awesome        (clustered)
   boring ≈ dull                      (another cluster)
   distance(great, boring) = large    ← the geometry encodes meaning
```

This is exactly what one-hot could never do (all words equidistant), and it's why
embeddings dominate: the encoding itself carries **semantics**.

## 4. Padding and masking

Sequences are still zero-padded to a common length. With `mask_zero=True`, the
Embedding layer flags index **0 (PAD)** so downstream recurrent layers **skip**
those steps instead of processing a meaningless row:

```
Embedding(input_dim=vocab, output_dim=d, mask_zero=True)
```

(Compare the animated demo, where the PAD step contributed `xW = [0,0,0]` but the
memory still churned — masking is the cleaner, production answer.)

## 5. Keras API (the encoding part only)

```python
from tensorflow.keras.layers import TextVectorization, Embedding

vectorizer = TextVectorization(output_mode="int")   # words -> integer indices
vectorizer.adapt(reviews)
ids = vectorizer(reviews)                            # (3, 4) ints, zero-padded

emb = Embedding(input_dim=len(vectorizer.get_vocabulary()),   # vocab size
                output_dim=4,                                  # d
                mask_zero=True)
vectors = emb(ids)                                   # (3, 4, 4) dense tensor
```

Run [`embedding_keras.py`](embedding_keras.py) to see the lookup-table property
verified (`output == E[index]`), the `(3, 4, 4)` review tensor, and the PAD mask.

## Key takeaways

- Embedding: word index → **dense learned vector** (a row of the weight matrix E).
- Equivalent to **one-hot × dense layer**, implemented as a lookup.
- **Trainable** → similar words end up **near each other**; the encoding carries
  meaning, unlike one-hot's all-equidistant vectors.
- Scales: `d ≈ 50–300` regardless of vocabulary size.
- `mask_zero=True` makes RNNs **skip the zero padding**.
- Standard real-world pipeline: **integer encoding → Embedding → RNN**.
