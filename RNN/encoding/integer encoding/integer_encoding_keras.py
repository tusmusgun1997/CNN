"""
Integer encoding with Keras — the encoding part only (no model, no training).

Turns the 3 example reviews into zero-padded integer sequences using
TextVectorization, shows the vocabulary (with PAD/UNK), and demonstrates
manual padding control with pad_sequences plus an out-of-vocabulary case.

Run:
    pip install tensorflow
    python integer_encoding_keras.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences

REVIEWS = [
    "movie was great",       # y = 1
    "movie was boring",      # y = 0
    "movie was not great",   # y = 0
]

print("TensorFlow:", tf.__version__, "\n")
print("Raw reviews:")
for r in REVIEWS:
    print("  -", r)

# ---------------------------------------------------------------------------
# 1. Build the vocabulary and encode  (word -> integer index)
# ---------------------------------------------------------------------------
vectorizer = TextVectorization(output_mode="int")  # integer indices out
vectorizer.adapt(REVIEWS)                          # learn vocab from the texts

vocab = vectorizer.get_vocabulary()
print("\nVocabulary (index -> token):")
for idx, tok in enumerate(vocab):
    label = {0: " <- PAD (reserved)", 1: " <- UNK (reserved, out-of-vocabulary)"}.get(idx, "")
    print(f"  {idx}: {tok!r}{label}")

encoded = vectorizer(REVIEWS)   # a (3, 4) tensor - padded to the longest review
print("\nEncoded reviews (auto zero-POST-padded to the longest, length 4):")
for text, row in zip(REVIEWS, encoded.numpy()):
    print(f"  {text!r:26s} -> {row}")
print("Batch tensor shape:", tuple(encoded.shape), " (reviews, time steps)")

# ---------------------------------------------------------------------------
# 2. Manual padding control with pad_sequences
# ---------------------------------------------------------------------------
# the unpadded sequences (the ids from step 1, before padding):
raw_seqs = [[3, 2, 4], [3, 2, 6], [3, 2, 5, 4]]

post = pad_sequences(raw_seqs, maxlen=4, padding="post")   # zeros at the END
pre  = pad_sequences(raw_seqs, maxlen=4, padding="pre")    # zeros at the FRONT (default)

print("\npad_sequences(padding='post'):  [zeros appended]")
print(post)
print("pad_sequences(padding='pre'):   [zeros prepended - the default]")
print(pre)

# ---------------------------------------------------------------------------
# 3. Out-of-vocabulary words map to index 1 ([UNK])
# ---------------------------------------------------------------------------
new_text = ["film was excellent"]          # 'film' and 'excellent' were never seen
oov = vectorizer(new_text)
print(f"\nUnseen text {new_text[0]!r} -> {oov.numpy()[0]}   (1 = [UNK])")

print("\n[OK] Integer encoding done - order preserved, everything is numbers now.")
print("Next step in a real pipeline: one-hot or an Embedding layer on top.")
