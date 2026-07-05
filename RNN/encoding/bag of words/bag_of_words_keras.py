"""
Bag-of-words encodings with Keras — the encoding part only (no model).

One TextVectorization layer, three output modes on the same 3 reviews:
  multi_hot -> word present? (1/0)
  count     -> how many times?
  tf_idf    -> counts down-weighted for words common to all reviews
Plus the key demonstration: bag-of-words is BLIND to word order.

Run:
    pip install tensorflow
    python bag_of_words_keras.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

np.set_printoptions(precision=3, suppress=True)

REVIEWS = [
    "movie was great",       # y = 1
    "movie was boring",      # y = 0
    "movie was not great",   # y = 0
]

print("TensorFlow:", tf.__version__, "\n")

def show(mode):
    vec = TextVectorization(output_mode=mode)
    vec.adapt(REVIEWS)
    vocab = vec.get_vocabulary()          # index 0 is [UNK] in these modes
    out = vec(REVIEWS).numpy()
    print(f"--- output_mode='{mode}' ---")
    print("columns:", vocab)
    for text, row in zip(REVIEWS, out):
        print(f"  {text!r:26s} -> {row}")
    print("batch shape:", out.shape, " (reviews, vocab) - NO time dimension\n")
    return vec

show("multi_hot")
show("count")
vec_tfidf = show("tf_idf")

print("note (tf_idf): 'movie'/'was' appear in ALL reviews -> low weight;")
print("'not'/'boring' appear in only one review -> highest weight.\n")

# ---------------------------------------------------------------------------
# The fatal flaw: word order is gone
# ---------------------------------------------------------------------------
print("--- order blindness -------------------------------------------------")
vec = TextVectorization(output_mode="multi_hot")
vec.adapt(REVIEWS)
a = vec(["movie was not great"]).numpy()[0]     # negative meaning
b = vec(["not was great movie"]).numpy()[0]     # scrambled nonsense
print("'movie was not great' ->", a)
print("'not was great movie' ->", b)
print("identical encodings?  ", bool(np.allclose(a, b)))
print("\n[OK] Same words => same vector, regardless of order.")
print("     That's why bag-of-words cannot feed an RNN - the sequence is lost.")
