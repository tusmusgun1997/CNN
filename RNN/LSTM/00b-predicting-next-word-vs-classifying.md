# 0b. Predicting the Next Word — It's Still Classification

**Read this before the gates.** It answers a question that confuses almost
everyone crossing over from sentiment analysis:

> "For reviews we **classified** — train on labeled good/bad reviews, then at test
> time say good or bad. But now we want to **predict the next word**. That's not a
> label we chose… how does the network even *do* that?"

The whole confusion melts once you see one thing:

> ## Predicting the next word **is** classification.
> It's the exact same machinery — just with more classes, and with labels that
> come for **free**.

## 1. Sentiment vs. next-word — the same shape

| | **Sentiment (what you know)** | **Next-word prediction** |
|---|-------------------------------|--------------------------|
| Input | a review (sequence of words) | the text so far (sequence of words) |
| Output | 1 of **2** classes: 👍 / 👎 | 1 of **V** classes: *which word comes next* |
| Output layer | `Dense(2)` + softmax (or `Dense(1)` + sigmoid) | `Dense(V)` + softmax, V = vocabulary size |
| Loss | cross-entropy | **the same** cross-entropy |
| Train | on (review → label) pairs | on (text-so-far → next-word) pairs |
| Test/use | input review → pick highest-prob class | input text → pick highest-prob **word** |

The **only real differences**:

1. **Number of classes.** Sentiment has 2. Next-word has **V** (e.g. 50,000 — one
   class per word in the vocabulary). "Which word comes next?" is just "which of
   the 50,000 classes?"
2. **Where the labels come from** (the big one — section 3).

Everything else — softmax, cross-entropy, backprop — is identical to the
classifier you already understand (and to the [CNN softmax head](../../CNN/backpropagation/backpropagation.html)).

## 2. What the network actually outputs

For next-word prediction, the final layer has **one neuron per vocabulary word**.
Softmax turns those into a **probability distribution over the whole vocabulary**:

```
Context: "the cat sat on the ___"

  model outputs a probability for EVERY word:
     P(mat)   = 0.61   ◄─ highest
     P(floor) = 0.18
     P(roof)  = 0.06
     P(sofa)  = 0.05
     P(dog)   = 0.002
     …            (all V words, summing to 1.0)

  prediction = the highest-probability word  →  "mat"
```

So "predicting a word" = **classify the context into one of V word-classes**. The
network isn't doing anything mysterious — it outputs a score for each possible
word and picks the best, exactly like picking "cat vs dog" but with a bigger menu.

## 3. The key insight: the labels are FREE (self-supervised)

This is what feels missing. In sentiment, a **human** had to label each review
👍/👎 — expensive, limited data. So where do next-word "labels" come from?

> **The next word in the text IS the label.** No human labeling needed. The text
> labels itself.

Take one sentence and slide along it — every position gives a training pair
where the **target is simply the actual next word**:

```
Sentence:  "the   cat   sat   on   the   mat"

 input →  target (the real next word)
 ─────────────────────────────────────
 "the"                    → "cat"
 "the cat"                → "sat"
 "the cat sat"            → "on"
 "the cat sat on"         → "the"
 "the cat sat on the"     → "mat"
 "the cat sat on the mat" → <END>
```

One 6-word sentence = **6 free training examples**. A book = millions. The
internet = trillions. That's why language models can be trained on enormous data
with **no human labels** — the answer key was inside the text all along. This is
called **self-supervised learning**.

## 4. How training works (you already know this part)

At each position, it's a plain classification step:

```
1. Feed the context words through the LSTM  → hidden state h (a summary so far)
2. Dense(V) + softmax on h                  → predicted distribution over words
3. Compare to the TRUE next word (its one-hot)  via cross-entropy loss
4. Backprop, nudge the weights so the true word gets higher probability
```

Over millions of examples, the network learns **which words tend to follow which
contexts**. During training we always know the true next word (it's just the text
— this is called *teacher forcing*).

> This is identical to training the sentiment classifier: forward → softmax →
> cross-entropy vs the true label → backprop. The "label" is now "the next word"
> instead of "positive/negative."

## 5. How it generates text (the part that feels like magic)

Prediction gives you **one** next word. To generate a whole sentence, you just
**do it repeatedly, feeding each prediction back in**:

```
seed:  "the cat sat on the"
  → predict → "mat"          append it
  "the cat sat on the mat"
  → predict → "."            append it
  "the cat sat on the mat ."
  → predict → <END>          stop
```

That feedback loop is the **one-to-many / autoregressive** wiring from
[types of RNN → one-to-many](../types%20of%20RNN/02-one-to-many.md). Predict a
word, glue it on, predict again. **This is literally how ChatGPT writes** — one
word (token) at a time, each conditioned on everything before it.

- **Argmax** (always pick the top word) → safe, repetitive text.
- **Sampling** (pick randomly according to the probabilities) → creative, varied
  text. The "temperature" knob controls how random.

## 6. So where does the LSTM memory come in?

To assign a good probability to the next word, the network needs **context** — and
often context from **far back**:

```
"I grew up in France … [40 words] … so I speak fluent ___"
                                                        ▲
        the LSTM's cell state carried "France" this whole way,
        so P(French) is high  →  the memory is what makes the
        classification correct.
```

The classification machinery (softmax over the vocabulary) is *how* it predicts;
the **LSTM memory is what gives it the context to predict well**. That's exactly
why we're studying LSTM — [file 1](01-why-lstm.md) picks up right here.

## Key takeaways

- **Next-word prediction = classification over the vocabulary.** Same softmax +
  cross-entropy + backprop as sentiment; just **V classes** instead of 2.
- The output is a **probability distribution over all words**; the prediction is
  the highest-probability word (or a sample from the distribution).
- **Labels are free** — the actual next word in the text is the target
  (**self-supervised**), so no human labeling is needed and data is unlimited.
- **Generation = repeat prediction, feeding each word back in** (autoregressive —
  how LLMs write).
- The **LSTM memory supplies the long-range context** that makes each
  classification accurate. → continue to [file 1: Why LSTM](01-why-lstm.md).
