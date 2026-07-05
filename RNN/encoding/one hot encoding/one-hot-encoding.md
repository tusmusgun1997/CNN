# One-Hot Encoding

**One-hot encoding** turns each word into a **binary vector** as long as the
vocabulary: all zeros except a single **1** at the word's index. This is the
encoding used in the [animated RNN demo](../../overview/rnn-recurrent-layer-animated.html)
— `movie = [1,0,0,0,0]`.

> Companion code: [`one_hot_encoding_keras.py`](one_hot_encoding_keras.py)

## 1. The idea

With our 5-word vocabulary (movie=0, was=1, great=2, not=3, boring=4):

```
            movie  was  great  not  boring
movie   →  [  1     0     0     0     0  ]
was     →  [  0     1     0     0     0  ]
great   →  [  0     0     1     0     0  ]
not     →  [  0     0     0     1     0  ]
boring  →  [  0     0     0     0     1  ]
```

A review becomes a **matrix** — one one-hot row per time step:

```
"movie was great" + PAD  →   [1 0 0 0 0]     shape (4, 5)
                             [0 1 0 0 0]     = (time steps, vocab size)
                             [0 0 1 0 0]
                             [0 0 0 0 0]  ← zero PAD: NOT a one-hot of any word!
```

Note the padding row: the **all-zero vector** — this is exactly the
"zero padding" from the RNN overview. It's the one row with no 1 anywhere.

## 2. Why one-hot fixes the integer problem

Integer encoding said `great = 2, boring = 4`, which fakes an order/distance.
One-hot puts every word on **its own axis**:

- No word is "bigger" than another.
- Every pair of words is **equally distant** (all vectors are orthogonal).
- `x·Wᵢ` with a one-hot `x` simply **selects one row of Wᵢ** — the matrix
  multiply becomes a clean lookup (this is the "one-hot trick" the animated
  demo exploits, and it's also secretly how the Embedding layer works).

## 3. The costs

| Problem | Why it hurts |
|---------|--------------|
| **Size explodes** | real vocab = 50,000 words → each word is a 50,000-dim vector; a 200-word review is a 200 × 50,000 matrix (mostly zeros) |
| **No similarity** | "great" and "excellent" are as far apart as "great" and "boring" — orthogonality means the encoding carries **zero meaning** |
| **Sparse & wasteful** | 99.99…% of every vector is 0 |

These are exactly the problems the [embedding](../embedding/embedding.md)
encoding solves — dense, small, and *learned* so similar words end up close.

## 4. Keras APIs (three ways, all shown in the script)

```python
# A) classic: integer-encode first, then to_categorical
from tensorflow.keras.utils import to_categorical
onehot = to_categorical([0, 1, 2], num_classes=5)          # (3, 5)

# B) StringLookup straight from strings, output_mode="one_hot"
from tensorflow.keras.layers import StringLookup
lookup = StringLookup(vocabulary=["movie","was","great","not","boring"],
                      num_oov_indices=0, output_mode="one_hot")

# C) CategoryEncoding on integer indices
from tensorflow.keras.layers import CategoryEncoding
enc = CategoryEncoding(num_tokens=5, output_mode="one_hot")
```

Run [`one_hot_encoding_keras.py`](one_hot_encoding_keras.py) to see all three
produce the same matrices, including the zero-padded review tensors of shape
`(3, 4, 5)` = (reviews, time steps, vocab).

## Key takeaways

- One word → one binary vector with a single 1; a review → a `(steps, vocab)`
  matrix; the **PAD row is all zeros**.
- Fixes integer encoding's **fake ordering**: all words orthogonal / equidistant.
- One-hot `x·W` = **row selection** — the bridge to understanding embeddings.
- Doesn't scale (huge, sparse) and encodes **no similarity** → use
  [embeddings](../embedding/embedding.md) for real tasks.
