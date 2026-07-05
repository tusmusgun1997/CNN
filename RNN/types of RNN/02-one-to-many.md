# 2. One-to-Many

**One input → a sequence of outputs.** The network receives a single item once,
then **unrolls a sequence out of it**, one step at a time — producing a new
output at every step while carrying its memory forward.

## The diagram

```
         y₁      y₂      y₃      ...        outputs at EVERY step
         ▲       ▲       ▲
      ┌─────┐ ┌─────┐ ┌─────┐
      │cell │→│cell │→│cell │→ ...          memory Oₜ flows right
      └─────┘ └─────┘ └─────┘
         ▲
         x                                  input enters ONCE (t = 1)
```

Two common wiring variants:

1. **Input at the first step only** — x kicks off the memory, then the network
   free-runs: `O₁ = f(x·Wᵢ)`, `Oₜ = f(Oₜ₋₁·Wₕ)` for t > 1.
2. **Input repeated at every step** (Keras `RepeatVector`) — the same x is fed
   at each step alongside the memory. Easier to train, very common in practice.

Often the output of step t is also **fed back as the input of step t+1**
(e.g. the word just generated becomes the next input) — that's how generation
keeps itself coherent.

## Use cases

| Task | The one input | The many outputs |
|------|---------------|------------------|
| **Image captioning** | one image (CNN features) | "a cat sitting on a mat" — word by word |
| **Music generation** | one seed note / genre vector | a melody, note by note |
| **Text generation from a topic** | one topic/seed embedding | a sentence, word by word |
| **Molecule generation** | one property vector | a SMILES string, character by character |

### Image captioning — the classic pipeline

```
image ──► CNN (feature extractor) ──► feature vector x ──► one-to-many RNN
                                                            │
                                       "a" → "cat" → "on" → "a" → "mat" → <END>
```

This is also the classic **CNN + RNN marriage**: the CNN answers *what is in the
image* (one vector), the RNN unrolls that into *a sentence* (a sequence). Note
the special `<END>` token — since the output length isn't fixed, the network
itself learns to say when to stop.

## How the sequence ends

A one-to-many generator needs a stopping rule, usually one of:

- emit a special **`<END>` token** (learned, most common), or
- generate a **fixed number** of steps.

## Keras sketch (shape of the idea)

```python
# variant 2: repeat the single input at every step
model = Sequential([
    RepeatVector(T),                      # x (1 vector)  -> (T, features)
    SimpleRNN(64, return_sequences=True), # memory unrolls over T steps
    TimeDistributed(Dense(vocab, activation="softmax")),  # one word per step
])
```

## Key takeaways

- One-to-many = **single input, sequence output**; input enters once (or is
  repeated), and the memory unrolls the sequence.
- Generated outputs are often **fed back** as the next step's input.
- Needs a **stopping mechanism** (`<END>` token or fixed length).
- Poster use case: **image captioning** (CNN gives the "one", RNN gives the "many").
- Next: the mirror image — [many-to-one](03-many-to-one.md).
