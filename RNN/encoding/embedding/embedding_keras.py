"""
Word embeddings with Keras — the encoding part only (no model training).

Pipeline demonstrated:
    text --TextVectorization--> integer ids --Embedding--> dense vectors

Shows:
  1. the (vocab, d) embedding matrix E (random until trained),
  2. that encoding is a pure LOOKUP: output row == E[index],
  3. the full review tensor (3 reviews, 4 steps, d) with zero padding,
  4. the padding mask from mask_zero=True.

Run:
    pip install tensorflow
    python embedding_keras.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding

np.set_printoptions(precision=4, suppress=True)
tf.random.set_seed(42)

REVIEWS = [
    "movie was great",       # y = 1
    "movie was boring",      # y = 0
    "movie was not great",   # y = 0
]
EMB_DIM = 4

print("TensorFlow:", tf.__version__, "\n")

# ---------------------------------------------------------------------------
# 1. text -> integer ids (see the 'integer encoding' folder)
# ---------------------------------------------------------------------------
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(REVIEWS)
vocab = vectorizer.get_vocabulary()
ids = vectorizer(REVIEWS)                     # (3, 4) zero-padded
print("Vocabulary:", list(enumerate(vocab)))
print("Integer ids (3, 4):")
print(ids.numpy(), "\n")

# ---------------------------------------------------------------------------
# 2. the Embedding layer = a lookup table E of shape (vocab, d)
# ---------------------------------------------------------------------------
emb = Embedding(input_dim=len(vocab), output_dim=EMB_DIM, mask_zero=True)
_ = emb(ids)                                   # build the layer
E = emb.get_weights()[0]                       # the table itself

print(f"Embedding matrix E: shape {E.shape} = (vocab size, d={EMB_DIM})")
print("(values are RANDOM now - training would tune them like any weight)")
for i, row in enumerate(E):
    print(f"  E[{i}] {vocab[i]!r:9s} -> {row}")

# ---------------------------------------------------------------------------
# 3. encoding = row lookup, verified
# ---------------------------------------------------------------------------
print("\n--- lookup property: Embedding(id) == E[id] ---------------------")
gid = int(vectorizer(["great"]).numpy()[0][0])          # id of 'great'
out = emb(tf.constant([[gid]])).numpy()[0, 0]
print(f"id('great') = {gid}")
print("emb(great) :", out)
print("E[id]      :", E[gid])
print("identical  :", bool(np.allclose(out, E[gid])))

# ---------------------------------------------------------------------------
# 4. the full encoded batch + padding mask
# ---------------------------------------------------------------------------
vectors = emb(ids)                             # (3, 4, EMB_DIM)
print("\nEncoded batch shape:", tuple(vectors.shape), "= (reviews, time steps, d)")
print("\nR1 'movie was great' + PAD, step by step:")
toks = ["movie", "was", "great", "PAD(0)"]
for t in range(4):
    print(f"  t={t+1} {toks[t]:8s} -> {vectors.numpy()[0, t]}")

mask = emb.compute_mask(ids)
print("\nmask_zero=True padding mask (True = real word, False = PAD, skipped by RNNs):")
print(mask.numpy())

print("\n[OK] text -> ids -> dense vectors. This (3, 4, d) tensor is what a real")
print("     RNN consumes: Embedding replaces the one-hot rows with learned rows.")
