"""
One-hot encoding with Keras — the encoding part only (no model, no training).

Shows three equivalent ways to one-hot the example reviews:
  A) to_categorical on integer indices  (the classic)
  B) StringLookup(output_mode="one_hot") straight from strings
  C) CategoryEncoding on integer indices
and builds the final zero-padded review tensor of shape (3, 4, 5).

Run:
    pip install tensorflow
    python one_hot_encoding_keras.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import StringLookup, CategoryEncoding

np.set_printoptions(precision=0, suppress=True)

VOCAB = ["movie", "was", "great", "not", "boring"]        # 5 words, indices 0..4
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}
REVIEWS = [
    ["movie", "was", "great"],          # y = 1
    ["movie", "was", "boring"],         # y = 0
    ["movie", "was", "not", "great"],   # y = 0
]
MAXLEN = 4

print("TensorFlow:", tf.__version__)
print("\nVocabulary:", {w: i for i, w in enumerate(VOCAB)})

# ---------------------------------------------------------------------------
# A) integer indices -> to_categorical
# ---------------------------------------------------------------------------
print("\n--- A) to_categorical -------------------------------------------")
idx = [WORD2IDX[w] for w in REVIEWS[0]]                 # [0, 1, 2]
oh = to_categorical(idx, num_classes=len(VOCAB))         # (3, 5)
print("indices", idx, "->")
print(oh)

# ---------------------------------------------------------------------------
# B) StringLookup: strings -> one-hot directly (exactly 5 dims, no PAD/UNK slots)
# ---------------------------------------------------------------------------
print("\n--- B) StringLookup(output_mode='one_hot') ----------------------")
lookup = StringLookup(vocabulary=VOCAB, num_oov_indices=0, output_mode="one_hot")
oh_b = lookup(tf.constant(REVIEWS[0]))                   # (3, 5)
print("words", REVIEWS[0], "->")
print(oh_b.numpy())

# ---------------------------------------------------------------------------
# C) CategoryEncoding on integer indices
# ---------------------------------------------------------------------------
print("\n--- C) CategoryEncoding(output_mode='one_hot') ------------------")
enc = CategoryEncoding(num_tokens=len(VOCAB), output_mode="one_hot")
oh_c = enc(tf.constant(idx))
print("indices", idx, "->")
print(oh_c.numpy())

same = np.allclose(oh, oh_b.numpy()) and np.allclose(oh, oh_c.numpy())
print("\nAll three methods identical:", same)

# ---------------------------------------------------------------------------
# Full batch: one-hot every review, ZERO-PAD to 4 time steps -> (3, 4, 5)
# ---------------------------------------------------------------------------
print("\n--- Zero-padded batch tensor ------------------------------------")
batch = np.zeros((len(REVIEWS), MAXLEN, len(VOCAB)), dtype="float32")
for r, words in enumerate(REVIEWS):
    for t, w in enumerate(words):
        batch[r, t] = to_categorical(WORD2IDX[w], num_classes=len(VOCAB))
    # rows beyond len(words) stay all-zero  <- the zero padding

for r, words in enumerate(REVIEWS):
    print(f"\nR{r+1} {' '.join(words)!r}  (padded to {MAXLEN} steps):")
    for t in range(MAXLEN):
        tag = VOCAB[np.argmax(batch[r, t])] if batch[r, t].any() else "PAD (all zeros)"
        print(f"   t={t+1}: {batch[r, t]}   <- {tag}")

print("\nBatch shape:", batch.shape, " = (reviews, time steps, vocab size)")
print("\n[OK] This (3, 4, 5) tensor is exactly what the animated RNN demo feeds")
print("     one row at a time: O_t = tanh(x_t.Wi + O_(t-1).Wh)")
